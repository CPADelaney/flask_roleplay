# lore/systems/dynamics.py

import logging
import json
import random
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

from agents import Agent, Runner
from agents.run_context import RunContextWrapper

from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

from embedding.vector_store import generate_embedding

from lore.core.base_manager import BaseLoreManager
from lore.managers.geopolitical import GeopoliticalSystemManager
from lore.utils.theming import MatriarchalThemingUtils

class LoreDynamicsSystem(BaseLoreManager):
    """
    Consolidated system for evolving world lore, generating emergent events,
    expanding content, and managing how the world changes over time.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.faith_system = None  # Will be initialized later if needed
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
        self.cache_namespace = "lore_dynamics"
    
    async def ensure_initialized(self):
        """Ensure system is initialized"""
        if not self.initialized:
            await super().ensure_initialized()
            await self.initialize_tables()
    
    async def initialize_tables(self):
        """Ensure required tables exist"""
        table_definitions = {
            "LoreChangeHistory": """
                CREATE TABLE LoreChangeHistory (
                    id SERIAL PRIMARY KEY,
                    lore_type TEXT NOT NULL,
                    lore_id TEXT NOT NULL,
                    previous_description TEXT NOT NULL,
                    new_description TEXT NOT NULL,
                    change_reason TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """,
            "WorldState": """
                CREATE TABLE WorldState (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    stability_index INTEGER CHECK (stability_index BETWEEN 1 AND 10),
                    narrative_tone TEXT,
                    power_dynamics TEXT,
                    power_hierarchy JSONB,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (user_id, conversation_id)
                );
            """
        }
        
        await self.initialize_tables_for_class(table_definitions)
    
    #===========================================================================
    # CORE LORE EVOLUTION METHODS
    #===========================================================================
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_lore_with_event",
        action_description="Evolving lore with event",
        id_from_context=lambda ctx: "lore_dynamics"
    )
    async def evolve_lore_with_event(self, event_description: str) -> Dict[str, Any]:
        """
        Evolve world lore based on a narrative event
        
        Args:
            event_description: Description of the event
            
        Returns:
            Dictionary with evolution results
        """
        # Ensure initialized
        await self.ensure_initialized()
        
        # Check permissions
        permission = await self.check_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="evolve_lore_with_event",
            action_details={"event_description": event_description}
        )
        
        if not permission["approved"]:
            logging.warning(f"Lore evolution not approved: {permission.get('reasoning')}")
            return {"error": permission.get("reasoning"), "approved": False}
        
        # 1. Identify affected lore elements
        affected_elements = await self._identify_affected_lore(event_description)
        
        # 2. Generate modifications for each affected element
        updates = await self._generate_lore_updates(affected_elements, event_description)
        
        # 3. Apply updates to database
        await self._apply_lore_updates(updates)
        
        # 4. Generate potential new lore elements triggered by this event
        new_elements = await self._generate_consequential_lore(event_description, affected_elements)
        
        # 5. Report to governance system
        await self.report_action(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "evolve_lore_with_event",
                "description": f"Evolved lore with event: {event_description[:50]}"
            },
            result={
                "affected_elements": len(affected_elements),
                "new_elements_created": len(new_elements)
            }
        )
        
        return {
            "affected_elements": affected_elements,
            "updates": updates,
            "new_elements": new_elements
        }
    
    async def _identify_affected_lore(self, event_description: str) -> List[Dict[str, Any]]:
        """
        Identify lore elements that would be affected by the event
        
        Args:
            event_description: Description of the event
            
        Returns:
            List of affected lore elements with relevance scores
        """
        # Generate embedding for the event
        event_embedding = await generate_embedding(event_description)
        
        # Search all lore types for potentially affected elements
        affected_elements = []
        
        # List of lore types to check
        lore_types = [
            "WorldLore", 
            "Factions", 
            "CulturalElements", 
            "HistoricalEvents", 
            "GeographicRegions", 
            "LocationLore",
            "UrbanMyths",
            "LocalHistories",
            "Landmarks",
            "NotableFigures"
        ]
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                for lore_type in lore_types:
                    try:
                        # Check if table exists
                        table_exists = await conn.fetchval(f"""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = '{lore_type.lower()}'
                            );
                        """)
                        
                        if not table_exists:
                            continue
                            
                        # Not all tables may have embedding - handle that case
                        has_embedding = await conn.fetchval(f"""
                            SELECT EXISTS (
                                SELECT FROM information_schema.columns 
                                WHERE table_name = '{lore_type.lower()}' AND column_name = 'embedding'
                            );
                        """)
                        
                        if not has_embedding:
                            continue
                        
                        # Perform similarity search
                        id_field = 'id'
                        if lore_type == 'LocationLore':
                            id_field = 'location_id'
                            
                        rows = await conn.fetch(f"""
                            SELECT {id_field} as id, name, description, 
                                   1 - (embedding <=> $1) as relevance
                            FROM {lore_type}
                            WHERE 1 - (embedding <=> $1) > 0.6
                            ORDER BY relevance DESC
                            LIMIT 5
                        """, event_embedding)
                        
                        for row in rows:
                            affected_elements.append({
                                'lore_type': lore_type,
                                'lore_id': row['id'],
                                'name': row['name'],
                                'description': row['description'],
                                'relevance': row['relevance']
                            })
                    except Exception as e:
                        logging.error(f"Error checking {lore_type} for affected elements: {e}")
        
        # Sort by relevance
        affected_elements.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Filter to most relevant (if too many)
        if len(affected_elements) > 15:
            affected_elements = affected_elements[:15]
            
        return affected_elements
    
    async def _generate_lore_updates(
        self, 
        affected_elements: List[Dict[str, Any]], 
        event_description: str
    ) -> List[Dict[str, Any]]:
        """
        Generate specific updates for affected lore elements
        
        Args:
            affected_elements: List of affected lore elements
            event_description: Description of the event
            
        Returns:
            List of updates to apply
        """
        updates = []
        
        # Get current world state for context
        world_state = await self._fetch_world_state()
        
        # Calculate societal impact of the event
        societal_impact = await self._calculate_societal_impact(
            event_description, 
            world_state.get('stability_index', 8),
            world_state.get('power_hierarchy', {})
        )
        
        # Use an LLM to generate updates for each element
        for element in affected_elements:
            # Create the run context
            run_ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
            # Get relationship network for this element
            related_elements = await self._fetch_related_elements(element.get('lore_id', ''))
            
            # Determine element's position in power hierarchy
            hierarchy_position = await self._get_hierarchy_position(element)
            
            # Get update history
            update_history = await self._fetch_element_update_history(element.get('lore_id', ''))
            
            # Build enhanced prompt for the LLM
            prompt = await self._build_lore_update_prompt(
                element=element,
                event_description=event_description,
                societal_impact=societal_impact,
                related_elements=related_elements,
                hierarchy_position=hierarchy_position,
                update_history=update_history
            )
            
            # Create an agent for lore updates
            lore_update_agent = Agent(
                name="LoreUpdateAgent",
                instructions="You update existing lore elements based on narrative events while maintaining thematic consistency.",
                model="o3-mini"
            )
            
            # Get the response
            result = await Runner.run(lore_update_agent, prompt, context=run_ctx.context)
            response_text = result.final_output
            
            try:
                # Parse the response
                update_data = await self._parse_lore_update_response(response_text, element)
                
                # Add the update
                updates.append({
                    'lore_type': element['lore_type'],
                    'lore_id': element['lore_id'],
                    'name': element['name'],
                    'old_description': element['description'],
                    'new_description': update_data['new_description'],
                    'update_reason': update_data['update_reason'],
                    'impact_level': update_data['impact_level'],
                    'timestamp': datetime.datetime.now().isoformat()
                })
                
            except Exception as e:
                logging.error(f"Failed to parse LLM response or update element: {e}")
                # Add fallback update if parsing fails
                updates.append({
                    'lore_type': element['lore_type'],
                    'lore_id': element['lore_id'],
                    'name': element['name'],
                    'old_description': element['description'],
                    'new_description': element['description'] + f"\n\nRecent developments: {event_description}",
                    'update_reason': "Event impact (automatic fallback)",
                    'impact_level': 3,
                    'timestamp': datetime.datetime.now().isoformat()
                })
        
        return updates
    
    async def _apply_lore_updates(self, updates: List[Dict[str, Any]]) -> None:
        """
        Apply the generated updates to the database
        
        Args:
            updates: List of updates to apply
        """
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                for update in updates:
                    lore_type = update['lore_type']
                    lore_id = update['lore_id']
                    new_description = update['new_description']
                    
                    # Generate new embedding for the updated content
                    # Determine ID field name
                    id_field = 'id'
                    if lore_type == 'LocationLore':
                        id_field = 'location_id'
                    
                    try:
                        # Update the description first
                        await conn.execute(f"""
                            UPDATE {lore_type}
                            SET description = $1
                            WHERE {id_field} = $2
                        """, new_description, lore_id)
                        
                        # Generate and store new embedding
                        item_name = update.get('name', 'Unknown')
                        embedding_text = f"{item_name} {new_description}"
                        await self.generate_and_store_embedding(embedding_text, conn, lore_type, id_field, lore_id)
                        
                        # Record the update in history
                        await conn.execute("""
                            INSERT INTO LoreChangeHistory 
                            (lore_type, lore_id, previous_description, new_description, change_reason)
                            VALUES ($1, $2, $3, $4, $5)
                        """, lore_type, lore_id, update['old_description'], new_description, update['update_reason'])
                    except Exception as e:
                        logging.error(f"Error updating {lore_type} ID {lore_id}: {e}")
                        
                    # Clear relevant cache
                    self.invalidate_cache_pattern(f"{lore_type}_{lore_id}")
    
    async def _generate_consequential_lore(
        self, 
        event_description: str, 
        affected_elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate new lore elements that might emerge as a consequence of the event
        
        Args:
            event_description: Description of the event
            affected_elements: List of affected lore elements
            
        Returns:
            List of new lore elements
        """
        # Create the run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # Extract names of affected elements for context
        affected_names = [f"{e['name']} ({e['lore_type']})" for e in affected_elements[:5]]
        affected_names_text = ", ".join(affected_names)
        
        # Create a prompt for the LLM
        prompt = f"""
        Based on the following event and affected lore elements, generate 1-3 new lore elements that would emerge as consequences.
        
        EVENT:
        {event_description}
        
        AFFECTED ELEMENTS:
        {affected_names_text}
        
        Generate between 1 and 3 new lore elements of different types. Consider:
        - New urban myths that might emerge
        - Changes to local history or folklore
        - New traditions or cultural practices
        - New historical events that this triggers
        - New notable figures who rise to prominence
        - New organizations or factions that form
        
        For each new element, provide:
        1. "lore_type": The type of lore element (WorldLore, Factions, CulturalElements, UrbanMyths, etc.)
        2. "name": A name for the element
        3. "description": A detailed description
        4. "connection": How it relates to the event and other lore
        5. "significance": A number from 1-10 indicating importance
        
        Format your response as a JSON array containing these objects.
        IMPORTANT: Maintain the matriarchal/femdom power dynamics in all new lore.
        """
        
        # Create an agent for new lore generation
        lore_creation_agent = Agent(
            name="LoreCreationAgent",
            instructions="You create new lore elements that emerge from significant events.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(lore_creation_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        # Parse the JSON response
        try:
            new_elements = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(new_elements, list):
                if isinstance(new_elements, dict):
                    new_elements = [new_elements]
                else:
                    new_elements = []
                    
            # Save the new lore elements to appropriate tables
            await self._save_new_lore_elements(new_elements, event_description)
                
            return new_elements
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for new lore: {response_text}")
            return []
    
    async def _save_new_lore_elements(
        self, 
        new_elements: List[Dict[str, Any]], 
        event_description: str
    ) -> None:
        """
        Save newly generated lore elements to appropriate tables
        
        Args:
            new_elements: List of new lore elements
            event_description: Original event description
        """
        for element in new_elements:
            lore_type = element.get('lore_type')
            name = element.get('name')
            description = element.get('description')
            significance = element.get('significance', 5)
            
            if not all([lore_type, name, description]):
                continue
                
            # Apply matriarchal theming
            description = MatriarchalThemingUtils.apply_matriarchal_theme(lore_type.lower(), description)
                
            # Save based on type
            try:
                if lore_type == "WorldLore":
                    await self.lore_manager.add_world_lore(
                        name=name,
                        category=element.get('category', 'event_consequence'),
                        description=description,
                        significance=significance,
                        tags=element.get('tags', ['event_consequence', 'new_lore'])
                    )
                elif lore_type == "Factions":
                    await self.lore_manager.add_faction(
                        name=name,
                        faction_type=element.get('faction_type', 'event_consequence'),
                        description=description,
                        values=element.get('values', ['power', 'authority']),
                        goals=element.get('goals', ['stability', 'influence']),
                        headquarters=element.get('headquarters'),
                        founding_story=f"Founded in response to: {event_description}"
                    )
                elif lore_type == "CulturalElements":
                    await self.lore_manager.add_cultural_element(
                        name=name,
                        element_type=element.get('element_type', 'tradition'),
                        description=description,
                        practiced_by=element.get('practiced_by', ['society']),
                        significance=significance,
                        historical_origin=f"Emerged from: {event_description}"
                    )
                elif lore_type == "HistoricalEvents":
                    await self.lore_manager.add_historical_event(
                        name=name,
                        description=description,
                        date_description=element.get('date_description', 'Recently'),
                        significance=significance,
                        participating_factions=element.get('participating_factions', []),
                        consequences=element.get('consequences', ['Still unfolding'])
                    )
                elif lore_type == "UrbanMyths":
                    # Fall back to world lore
                    await self.lore_manager.add_world_lore(
                        name=name,
                        category='urban_myth',
                        description=description,
                        significance=significance,
                        tags=['urban_myth', 'event_consequence']
                    )
                elif lore_type == "NotableFigures":
                    # Fall back to world lore
                    await self.lore_manager.add_world_lore(
                        name=name,
                        category='notable_figure',
                        description=description,
                        significance=significance,
                        tags=['notable_figure', 'event_consequence']
                    )
                else:
                    # Generic fallback for unknown types
                    await self.lore_manager.add_world_lore(
                        name=name,
                        category=lore_type.lower(),
                        description=description,
                        significance=significance,
                        tags=[lore_type.lower(), 'event_consequence']
                    )
            except Exception as e:
                logging.error(f"Error saving new {lore_type} '{name}': {e}")
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_emergent_event",
        action_description="Generating emergent world event",
        id_from_context=lambda ctx: "lore_dynamics"
    )
    async def generate_emergent_event(self, ctx) -> Dict[str, Any]:
        """
        Generate a random emergent event in the world with governance oversight.
        
        Returns:
            Event details and lore impact
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get current world state for context
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get active factions
                factions = await conn.fetch("""
                    SELECT id, name, type, description
                    FROM Factions
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                # Get some nations
                nations = await conn.fetch("""
                    SELECT id, name, government_type
                    FROM Nations
                    LIMIT 3
                """)
                
                # Get some locations
                locations = await conn.fetch("""
                    SELECT id, location_name, description
                    FROM Locations
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                # Convert to lists
                faction_data = [dict(faction) for faction in factions]
                nation_data = [dict(nation) for nation in nations]
                location_data = [dict(location) for location in locations]
        
        # Determine event type
        event_types = [
            "political_shift", "military_conflict", "natural_disaster",
            "cultural_development", "technological_advancement", "religious_event",
            "economic_change", "diplomatic_incident"
        ]
        
        # Choose event type with weights based on available data
        event_type = self._choose_event_type(event_types, faction_data, nation_data, location_data)
        
        # Create a prompt based on event type
        prompt = self._create_event_prompt(event_type, faction_data, nation_data, location_data)
        
        # Create an agent for event generation
        event_agent = Agent(
            name="EmergentEventAgent",
            instructions="You create emergent world events for fantasy settings.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(event_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            event_data = json.loads(response_text)
            
            # Create a unified event description for lore evolution
            event_description = event_data.get("description", "")
            event_name = event_data.get("event_name", "Unnamed Event")
            
            # Apply matriarchal theming
            themed_description = MatriarchalThemingUtils.apply_matriarchal_theme("event", event_description)
            event_data["description"] = themed_description
            
            # Use lore evolution functionality to evolve lore based on this event
            lore_updates = await self.evolve_lore_with_event(
                f"{event_name}: {themed_description}"
            )
            
            # Add the lore updates to the event data
            event_data["lore_updates"] = lore_updates
            
            # Return the combined data
            return event_data
            
        except Exception as e:
            logging.error(f"Error generating emergent event: {e}")
            return {"error": str(e)}
    
    def _choose_event_type(self, event_types, faction_data, nation_data, location_data):
        """Choose appropriate event type based on available context data"""
        weights = [1] * len(event_types)
        
        # More likely to generate faction-related events if we have factions
        if faction_data:
            faction_event_indices = [0, 1, 3, 6, 7]  # political, military, cultural, economic, diplomatic
            for idx in faction_event_indices:
                if idx < len(weights):
                    weights[idx] += 1
        
        # More likely to generate nation-related events if we have nations
        if nation_data:
            nation_event_indices = [0, 1, 7]  # political, military, diplomatic
            for idx in nation_event_indices:
                if idx < len(weights):
                    weights[idx] += 1
        
        # More likely to generate location-related events if we have locations
        if location_data:
            location_event_indices = [2, 3, 5]  # natural, cultural, religious
            for idx in location_event_indices:
                if idx < len(weights):
                    weights[idx] += 1
        
        # Select event type based on weights
        return random.choices(event_types, weights=weights, k=1)[0]
    
    def _create_event_prompt(self, event_type, faction_data, nation_data, location_data):
        """
        Create a tailored prompt for generating a specific type of event
        
        Args:
            event_type: Type of event to generate
            faction_data: Faction context data
            nation_data: Nation context data
            location_data: Location context data
            
        Returns:
            Prompt string
        """
        if event_type == "political_shift":
            entities = faction_data if faction_data else nation_data
            return f"""
            Generate a political shift event for this world:
            
            POTENTIAL ENTITIES INVOLVED:
            {json.dumps(entities, indent=2)}
            
            Create a significant political shift event that:
            1. Changes power dynamics in a meaningful way
            2. Is specific and detailed enough to impact the world
            3. Reinforces or challenges matriarchal power structures
            4. Could lead to interesting narrative developments
            
            Return a JSON object with:
            - event_name: Name of the event
            - event_type: "political_shift"
            - description: Detailed description
            - entities_involved: Array of entity names involved
            - instigator: The primary instigating entity
            - immediate_consequences: Array of immediate consequences
            - potential_long_term_effects: Array of potential long-term effects
            - affected_lore_categories: Array of lore categories this would affect
            """
        elif event_type == "military_conflict":
            return f"""
            Generate a military conflict event for this world:
            
            POTENTIAL NATIONS INVOLVED:
            {json.dumps(nation_data, indent=2)}
            
            POTENTIAL FACTIONS INVOLVED:
            {json.dumps(faction_data, indent=2)}
            
            POTENTIAL LOCATIONS:
            {json.dumps(location_data, indent=2)}
            
            Create a military conflict that:
            1. Involves clear aggressor and defender sides
            2. Has a specific cause and objective
            3. Creates opportunities for heroism and character development
            4. Shows how warfare operates in a matriarchal society
            
            Return a JSON object with:
            - event_name: Name of the conflict
            - event_type: "military_conflict"
            - description: Detailed description
            - aggressor: Entity that initiated the conflict
            - defender: Entity defending against aggression
            - location: Where the conflict is primarily occurring
            - cause: What caused the conflict
            - scale: Scale of the conflict (skirmish, battle, war, etc.)
            - current_status: Current status of the conflict
            - casualties: Description of casualties
            - affected_lore_categories: Array of lore categories this would affect
            """
        elif event_type == "natural_disaster":
            return f"""
            Generate a natural disaster event for this world:
            
            POTENTIAL LOCATIONS AFFECTED:
            {json.dumps(location_data, indent=2)}
            
            Create a natural disaster that:
            1. Has a significant impact on the affected area
            2. Could have supernatural or magical elements
            3. Creates opportunities for societal response that highlights values
            4. Changes the physical landscape in some way
            
            Return a JSON object with:
            - event_name: Name of the disaster
            - event_type: "natural_disaster"
            - description: Detailed description
            - disaster_type: Type of disaster (earthquake, flood, magical storm, etc.)
            - primary_location: Primary location affected
            - secondary_locations: Array of secondarily affected locations
            - severity: Severity level (1-10)
            - immediate_impact: Description of immediate impact
            - response: How communities and leadership are responding
            - supernatural_elements: Any supernatural aspects (if applicable)
            - affected_lore_categories: Array of lore categories this would affect
            """
        elif event_type == "cultural_development":
            return f"""
            Generate a cultural development event for this world:
            
            POTENTIAL LOCATIONS:
            {json.dumps(location_data, indent=2)}
            
            POTENTIAL FACTIONS:
            {json.dumps(faction_data, indent=2)}
            
            Create a cultural development that:
            1. Introduces a new tradition, art form, or cultural practice
            2. Comes from a specific community or demographic
            3. Has meaning within the matriarchal power structure
            4. Could spread to other communities
            
            Return a JSON object with:
            - event_name: Name of the cultural development
            - event_type: "cultural_development"
            - description: Detailed description
            - development_type: Type of development (tradition, art form, practice, etc.)
            - originating_group: Group where it originated
            - significance: Social significance
            - symbolism: Symbolic meaning
            - reception: How different groups have received it
            - spread_potential: Likelihood of spreading (1-10)
            - affected_lore_categories: Array of lore categories this would affect
            """
        elif event_type == "technological_advancement":
            return f"""
            Generate a technological or magical advancement event for this world:
            
            POTENTIAL LOCATIONS:
            {json.dumps(location_data, indent=2)}
            
            POTENTIAL FACTIONS:
            {json.dumps(faction_data, indent=2)}
            
            Create a technological or magical advancement that:
            1. Changes how something is done in the world
            2. Was developed by specific individuals or groups
            3. Has both benefits and potential drawbacks
            4. Reinforces matriarchal control of knowledge or resources
            
            Return a JSON object with:
            - event_name: Name of the advancement
            - event_type: "technological_advancement"
            - description: Detailed description
            - advancement_type: Type (technological, magical, alchemical, etc.)
            - creator: Who created or discovered it
            - applications: Potential applications
            - limitations: Current limitations
            - control: Who controls access to this advancement
            - societal_impact: How it impacts society
            - affected_lore_categories: Array of lore categories this would affect
            """
        elif event_type == "religious_event":
            return f"""
            Generate a religious event for this world:
            
            POTENTIAL LOCATIONS:
            {json.dumps(location_data, indent=2)}
            
            Create a religious event that:
            1. Has significance within an existing faith system
            2. Could be interpreted as divine intervention or revelation
            3. Affects religious practices or beliefs
            4. Reinforces the feminine divine nature of the world
            
            Return a JSON object with:
            - event_name: Name of the religious event
            - event_type: "religious_event"
            - description: Detailed description
            - event_category: Type of event (miracle, prophecy, divine manifestation, etc.)
            - location: Where it occurred
            - witnesses: Who witnessed it
            - religious_significance: Significance to believers
            - skeptic_explanation: How skeptics explain it
            - faith_impact: How it impacts faith practices
            - affected_lore_categories: Array of lore categories this would affect
            """
        elif event_type == "economic_change":
            return f"""
            Generate an economic change event for this world:
            
            POTENTIAL NATIONS:
            {json.dumps(nation_data, indent=2)}
            
            POTENTIAL FACTIONS:
            {json.dumps(faction_data, indent=2)}
            
            Create an economic change that:
            1. Shifts resource distribution or trade patterns
            2. Affects multiple groups or regions
            3. Creates winners and losers
            4. Shows how economic power relates to matriarchal control
            
            Return a JSON object with:
            - event_name: Name of the economic change
            - event_type: "economic_change"
            - description: Detailed description
            - change_type: Type of change (trade shift, resource discovery, currency change, etc.)
            - primary_causes: What caused the change
            - beneficiaries: Who benefits
            - disadvantaged: Who is disadvantaged
            - resource_involved: Primary resource or commodity involved
            - wealth_redistribution: How wealth is being redistributed
            - affected_lore_categories: Array of lore categories this would affect
            """
        else:  # diplomatic_incident
            return f"""
            Generate a diplomatic incident for this world:
            
            POTENTIAL NATIONS:
            {json.dumps(nation_data, indent=2)}
            
            POTENTIAL FACTIONS:
            {json.dumps(faction_data, indent=2)}
            
            Create a diplomatic incident that:
            1. Creates tension between two or more groups
            2. Stems from a specific action or misunderstanding
            3. Requires diplomatic resolution
            4. Highlights cultural differences or values
            
            Return a JSON object with:
            - event_name: Name of the diplomatic incident
            - event_type: "diplomatic_incident"
            - description: Detailed description
            - parties_involved: Array of involved parties
            - instigating_action: What triggered the incident
            - cultural_factors: Cultural factors at play
            - severity: Severity level (1-10)
            - potential_resolutions: Possible ways to resolve it
            - current_status: Current diplomatic status
            - affected_lore_categories: Array of lore categories this would affect
            """
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="mature_lore_over_time",
        action_description="Maturing lore over time",
        id_from_context=lambda ctx: "lore_dynamics"
    )
    async def mature_lore_over_time(self, days_passed: int = 7) -> Dict[str, Any]:
        """
        Natural evolution of lore over time, simulating how history and culture develop
        
        Args:
            days_passed: Number of in-game days that have passed
            
        Returns:
            Summary of maturation changes
        """
        # Check permissions with governance system
        await self.ensure_initialized()
        
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="mature_lore_over_time",
            action_details={"days_passed": days_passed}
        )
        
        if not permission["approved"]:
            logging.warning(f"Lore maturation not approved: {permission.get('reasoning')}")
            return {"error": permission.get("reasoning"), "approved": False}
        
        # Calculate probabilities based on days passed
        # More time = more changes
        base_probability = min(0.5, days_passed * 0.02)  # Cap at 50%
        
        maturation_summary = {
            "myth_evolution": [],
            "culture_development": [],
            "geopolitical_shifts": [],
            "reputation_changes": []
        }
        
        # 1. Urban Myths Evolution (highest chance of change)
        if random.random() < (base_probability * 1.5):
            myth_changes = await self._evolve_urban_myths()
            maturation_summary["myth_evolution"] = myth_changes
        
        # 2. Cultural Elements Development
        if random.random() < base_probability:
            culture_changes = await self._develop_cultural_elements()
            maturation_summary["culture_development"] = culture_changes
        
        # 3. Geopolitical Landscape Shifts (rarer)
        if random.random() < (base_probability * 0.7):
            geopolitical_changes = await self._shift_geopolitical_landscape()
            maturation_summary["geopolitical_shifts"] = geopolitical_changes
        
        # 4. Notable Figure Reputation Changes
        if random.random() < (base_probability * 1.2):
            reputation_changes = await self._evolve_notable_figures()
            maturation_summary["reputation_changes"] = reputation_changes
        
        # Report to governance system
        changes_count = sum(len(changes) for changes in maturation_summary.values())
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "mature_lore_over_time",
                "description": f"Matured lore over {days_passed} days"
            },
            result={
                "changes_applied": changes_count
            }
        )
        
        return {
            "days_passed": days_passed,
            "changes_applied": changes_count,
            "maturation_summary": maturation_summary
        }
    
    #===========================================================================
    # HELPER METHODS
    #===========================================================================
    
    async def _fetch_world_state(self) -> Dict[str, Any]:
        """Fetch current world state from database"""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Query the WorldState table
                world_state = await conn.fetchrow("""
                    SELECT * FROM WorldState 
                    WHERE user_id = $1 AND conversation_id = $2
                    LIMIT 1
                """, self.user_id, self.conversation_id)
                
                if world_state:
                    result = dict(world_state)
                    # Parse JSON fields
                    if 'power_hierarchy' in result and result['power_hierarchy']:
                        try:
                            result['power_hierarchy'] = json.loads(result['power_hierarchy'])
                        except:
                            result['power_hierarchy'] = {}
                    return result
                else:
                    # Return default values if no world state found
                    return {
                        'stability_index': 8,
                        'narrative_tone': 'dramatic',
                        'power_dynamics': 'strict_hierarchy',
                        'power_hierarchy': {}
                    }
    
    async def _fetch_related_elements(self, lore_id: str) -> List[Dict[str, Any]]:
        """Fetch elements related to the given lore ID"""
        if not lore_id:
            return []
            
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                try:
                    # Check if relationships table exists
                    table_exists = await conn.fetchval("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'lorerelationships'
                        );
                    """)
                    
                    if not table_exists:
                        return []
                    
                    # Query relationships
                    related = await conn.fetch("""
                        SELECT e.lore_id, e.name, e.lore_type, r.relationship_type, r.relationship_strength 
                        FROM LoreElements e
                        JOIN LoreRelationships r ON e.lore_id = r.target_id
                        WHERE r.source_id = $1
                    """, lore_id)
                    
                    return [dict(rel) for rel in related]
                except Exception as e:
                    logging.error(f"Error fetching related elements for {lore_id}: {e}")
                    return []
    
    async def _get_hierarchy_position(self, element: Dict[str, Any]) -> int:
        """Determine element's position in the power hierarchy"""
        # Implementation would calculate or retrieve hierarchy position
        # For now, return a default based on lore_type
        if element.get('lore_type', '') == 'character':
            # Check if character has a stored hierarchy value
            if 'hierarchy_position' in element:
                return element['hierarchy_position']
            else:
                # Default based on name keywords
                name = element.get('name', '').lower()
                if any(title in name for title in ['queen', 'empress', 'matriarch', 'high', 'supreme']):
                    return 1
                elif any(title in name for title in ['princess', 'duchess', 'lady', 'noble']):
                    return 3
                elif any(title in name for title in ['advisor', 'minister', 'council']):
                    return 5
                else:
                    return 8
        elif element.get('lore_type', '') == 'faction':
            # Check faction importance
            if 'importance' in element:
                return max(1, 10 - element['importance'])
            else:
                return 4  # Default for factions
        elif element.get('lore_type', '') == 'location':
            # Check location significance
            if 'significance' in element:
                return max(1, 10 - element['significance'])
            else:
                return 6  # Default for locations
        else:
            return 5  # Default middle position
    
    async def _fetch_element_update_history(self, lore_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch recent update history for an element"""
        if not lore_id:
            return []
            
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                try:
                    # Check if history table exists
                    table_exists = await conn.fetchval("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'lorechangehistory'
                        );
                    """)
                    
                    if not table_exists:
                        return []
                    
                    # Query history
                    history = await conn.fetch("""
                        SELECT lore_type, lore_id, change_reason, timestamp
                        FROM LoreChangeHistory
                        WHERE lore_id = $1
                        ORDER BY timestamp DESC
                        LIMIT $2
                    """, lore_id, limit)
                    
                    return [dict(update) for update in history]
                except Exception as e:
                    logging.error(f"Error fetching update history for {lore_id}: {e}")
                    return []
    
    async def _build_lore_update_prompt(
        self,
        element: Dict[str, Any],
        event_description: str,
        societal_impact: Dict[str, Any],
        related_elements: List[Dict[str, Any]],
        hierarchy_position: int,
        update_history: List[Dict[str, Any]]
    ) -> str:
        """
        Build a prompt for updating lore based on an event
        
        Args:
            element: The lore element to update
            event_description: Description of the event
            societal_impact: Impact assessment on society
            related_elements: Elements connected to this one
            hierarchy_position: Position in power hierarchy
            update_history: Recent update history
            
        Returns:
            A detailed prompt for the LLM
        """
        # Format update history as context
        history_context = ""
        if update_history:
            history_items = []
            for update in update_history:
                history_items.append(f"- {update['timestamp']}: {update['change_reason']}")
            history_context = "UPDATE HISTORY:\n" + "\n".join(history_items)
        
        # Format related elements as context
        relationships_context = ""
        if related_elements:
            rel_items = []
            for rel in related_elements:
                rel_items.append(f"- {rel['name']} ({rel['lore_type']}): {rel['relationship_type']} - {rel['relationship_strength']}/10")
            relationships_context = "RELATIONSHIPS:\n" + "\n".join(rel_items)
        
        # Determine hierarchy-appropriate directive
        if hierarchy_position <= 2:
            hierarchy_directive = """
            DIRECTIVE: This element represents a highest-tier authority figure. 
            Their decisions significantly impact the world. 
            They rarely change their core principles but may adjust strategies.
            They maintain control and authority in all situations.
            """
        elif hierarchy_position <= 4:
            hierarchy_directive = """
            DIRECTIVE: This element represents high authority.
            They have significant influence but answer to the highest tier.
            They strongly maintain the established order while pursuing their ambitions.
            They assert dominance in their domain but show deference to higher authority.
            """
        elif hierarchy_position <= 7:
            hierarchy_directive = """
            DIRECTIVE: This element has mid-level authority.
            They implement the will of higher authorities while managing those below.
            They may have personal aspirations but function within established boundaries.
            They balance compliance with higher authority against control of subordinates.
            """
        else:
            hierarchy_directive = """
            DIRECTIVE: This element has low authority in the hierarchy.
            They follow directives from above and have limited autonomy.
            They may seek to improve their position but must navigate carefully.
            They show appropriate deference to those of higher status.
            """
        
        # Build the complete prompt
        prompt = f"""
        The following lore element in our matriarchal-themed RPG world requires updating based on recent events:
        
        LORE ELEMENT:
        Type: {element['lore_type']}
        Name: {element['name']}
        Current Description: {element['description']}
        Position in Hierarchy: {hierarchy_position}/10 (lower number = higher authority)
        
        {relationships_context}
        
        {history_context}
        
        EVENT THAT OCCURRED:
        {event_description}
        
        SOCIETAL IMPACT ASSESSMENT:
        Stability Impact: {societal_impact['stability_impact']}/10
        Power Structure Change: {societal_impact['power_structure_change']}
        Public Perception Shift: {societal_impact['public_perception']}
        
        {hierarchy_directive}
        
        Generate a sophisticated update for this lore element that incorporates the impact of this event.
        The update should maintain narrative consistency while allowing for meaningful development.
        
        Return your response as a JSON object with:
        {{
            "new_description": "The updated description that reflects event impact",
            "update_reason": "Detailed explanation of why this update makes sense",
            "impact_level": A number from 1-10 indicating how significantly this event affects this element
        }}
        """
        
        return prompt
    
    async def _parse_lore_update_response(self, response_text: str, element: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the LLM response with error handling
        
        Args:
            response_text: Raw response from the LLM
            element: Original lore element
            
        Returns:
            Parsed update data
        """
        try:
            # First try to parse as JSON
            update_data = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['new_description', 'update_reason', 'impact_level']
            for field in required_fields:
                if field not in update_data:
                    raise ValueError(f"Missing required field: {field}")
            
            return update_data
            
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse JSON response for {element['name']}")
            
            # Try regex extraction for common patterns
            patterns = {
                'new_description': r'"new_description"\s*:\s*"([^"]+)"',
                'update_reason': r'"update_reason"\s*:\s*"([^"]+)"',
                'impact_level': r'"impact_level"\s*:\s*(\d+)'
            }
            
            extracted_data = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, response_text, re.DOTALL)
                if match:
                    if key == 'impact_level':
                        extracted_data[key] = int(match.group(1))
                    else:
                        extracted_data[key] = match.group(1)
            
            # Fill in missing required fields with defaults
            if 'new_description' not in extracted_data:
                # Find the longest paragraph as a fallback description
                paragraphs = re.split(r'\n\n+', response_text)
                paragraphs = [p for p in paragraphs if len(p) > 50]
                if paragraphs:
                    extracted_data['new_description'] = max(paragraphs, key=len)
                else:
                    extracted_data['new_description'] = element['description']
            
            if 'update_reason' not in extracted_data:
                extracted_data['update_reason'] = "Event impact (extracted from unstructured response)"
                
            if 'impact_level' not in extracted_data:
                # Look for numbers in text that might indicate impact level
                numbers = re.findall(r'\b([1-9]|10)\b', response_text)
                if numbers:
                    extracted_data['impact_level'] = int(numbers[0])
                else:
                    extracted_data['impact_level'] = 5
            
            return extracted_data
    
    async def _calculate_societal_impact(
        self,
        event_description: str,
        stability_index: int,
        power_hierarchy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate the societal impact of an event
        
        Args:
            event_description: Description of the event
            stability_index: Current stability of society (1-10)
            power_hierarchy: Current power structure data
            
        Returns:
            Dictionary of societal impact metrics
        """
        # Analyze event text for impact keywords
        impact_keywords = {
            'high_impact': [
                'overthrown', 'revolution', 'usurped', 'conquered', 'rebellion',
                'assassination', 'coup', 'catastrophe', 'disaster'
            ],
            'medium_impact': [
                'challenge', 'conflict', 'dispute', 'tension', 'unrest',
                'scandal', 'controversy', 'uprising', 'demonstration'
            ],
            'low_impact': [
                'minor', 'small', 'limited', 'isolated', 'contained',
                'private', 'personal', 'individual', 'trivial'
            ]
        }
        
        # Count keyword occurrences
        high_count = sum(1 for word in impact_keywords['high_impact'] if word.lower() in event_description.lower())
        medium_count = sum(1 for word in impact_keywords['medium_impact'] if word.lower() in event_description.lower())
        low_count = sum(1 for word in impact_keywords['low_impact'] if word.lower() in event_description.lower())
        
        # Calculate base stability impact
        if high_count > 0:
            base_stability_impact = 7 + min(high_count, 3)  # Max 10
        elif medium_count > 0:
            base_stability_impact = 4 + min(medium_count, 3)  # Max 7
        elif low_count > 0:
            base_stability_impact = 2 + min(low_count, 2)  # Max 4
        else:
            base_stability_impact = 3  # Default moderate impact
        
        # Adjust for current stability
        # Higher stability means events have less impact
        stability_modifier = (10 - stability_index) / 10
        adjusted_impact = base_stability_impact * (0.5 + stability_modifier)
        
        # Determine power structure change
        if adjusted_impact >= 8:
            power_change = "significant realignment of authority"
        elif adjusted_impact >= 6:
            power_change = "moderate shift in power dynamics"
        elif adjusted_impact >= 4:
            power_change = "subtle adjustments to authority structures"
        else:
            power_change = "minimal change to established order"
        
        # Determine public perception
        if adjusted_impact >= 7:
            if "rebellion" in event_description.lower() or "uprising" in event_description.lower():
                perception = "widespread questioning of authority"
            else:
                perception = "significant public concern"
        elif adjusted_impact >= 5:
            perception = "notable public interest and discussion"
        else:
            perception = "limited public awareness or interest"
        
        return {
            'stability_impact': round(adjusted_impact),
            'power_structure_change': power_change,
            'public_perception': perception
        }
    
    #===========================================================================
    # LORE MATURATION METHODS
    #===========================================================================
    
    async def _evolve_urban_myths(self) -> List[Dict[str, Any]]:
        """Evolve urban myths over time"""
        changes = []
        
        # Choose a random sample of urban myths to evolve
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if we have a dedicated urban myths table
                has_urban_myths_table = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'urbanmyths'
                    );
                """)
                
                if has_urban_myths_table:
                    # Get myths from dedicated table
                    rows = await conn.fetch("""
                        SELECT id, name, description, believability, spread_rate
                        FROM UrbanMyths
                        ORDER BY RANDOM()
                        LIMIT 3
                    """)
                else:
                    # Fall back to world lore with urban_myth category
                    rows = await conn.fetch("""
                        SELECT id, name, description
                        FROM WorldLore
                        WHERE category = 'urban_myth'
                        ORDER BY RANDOM()
                        LIMIT 3
                    """)
                
                for row in rows:
                    myth_id = row['id']
                    myth_name = row['name']
                    old_description = row['description']
                    
                    # Determine change type (grow, evolve, or fade)
                    change_type = random.choice(["grow", "evolve", "fade"])
                    
                    # Create a context for the agent
                    run_ctx = RunContextWrapper(context={
                        "user_id": self.user_id,
                        "conversation_id": self.conversation_id
                    })
                    
                    # Create an appropriate prompt based on change type
                    if change_type == "grow":
                        prompt = f"""
                        This urban myth is growing in popularity and becoming more elaborate:
                        
                        MYTH: {myth_name}
                        CURRENT DESCRIPTION: {old_description}
                        
                        Expand this myth to include new details, aspects or variations that have emerged.
                        The myth should become more widespread and elaborate, but maintain its core essence.
                        
                        Return only the new expanded description, written in the same style as the original.
                        """
                        
                        change_description = "Myth has grown in popularity and become more elaborate."
                        new_believability = min(10, row.get('believability', 5) + random.randint(1, 2))
                        new_spread = min(10, row.get('spread_rate', 5) + random.randint(1, 2))
                        
                    elif change_type == "evolve":
                        prompt = f"""
                        This urban myth is evolving with new variations:
                        
                        MYTH: {myth_name}
                        CURRENT DESCRIPTION: {old_description}
                        
                        Evolve this myth by changing some details while keeping its core recognizable.
                        Perhaps add a twist, change the outcome, or incorporate a new element.
                        
                        Return only the new evolved description, written in the same style as the original.
                        """
                        
                        change_description = "Myth has evolved with new variations or interpretations."
                        # Keep same values for believability and spread rate
                        new_believability = row.get('believability', 5)
                        new_spread = row.get('spread_rate', 5)
                        
                    else:  # fade
                        prompt = f"""
                        This urban myth is fading from public consciousness:
                        
                        MYTH: {myth_name}
                        CURRENT DESCRIPTION: {old_description}
                        
                        Modify this description to indicate the myth is becoming less believed or remembered.
                        Add phrases like "once widely believed" or "few now remember" to show its decline.
                        
                        Return only the new faded description, written in the same style as the original.
                        """
                        
                        change_description = "Myth is fading from public consciousness or becoming less believed."
                        new_believability = max(1, row.get('believability', 5) - random.randint(1, 2))
                        new_spread = max(1, row.get('spread_rate', 5) - random.randint(1, 2))
                    
                    # Create agent and get updated description
                    myth_agent = Agent(
                        name="MythEvolutionAgent",
                        instructions="You develop and evolve urban myths over time.",
                        model="o3-mini"
                    )
                    
                    result = await Runner.run(myth_agent, prompt, context=run_ctx.context)
                    new_description = result.final_output
                    
                    # Apply matriarchal theming to the new description
                    new_description = MatriarchalThemingUtils.apply_matriarchal_theme("myth", new_description, emphasis_level=1)
                    
                    # Apply the update to the database
                    try:
                        if has_urban_myths_table:
                            # Update in dedicated table
                            await conn.execute("""
                                UPDATE UrbanMyths
                                SET description = $1,
                                    believability = $2,
                                    spread_rate = $3
                                WHERE id = $4
                            """, new_description, new_believability, new_spread, myth_id)
                        else:
                            # Update in WorldLore
                            # Generate new embedding
                            embedding_text = f"{myth_name} {new_description}"
                            new_embedding = await generate_embedding(embedding_text)
                            
                            await conn.execute("""
                                UPDATE WorldLore
                                SET description = $1, embedding = $2
                                WHERE id = $3
                            """, new_description, new_embedding, myth_id)
                        
                        # Record the change
                        changes.append({
                            "myth_id": myth_id,
                            "name": myth_name,
                            "change_type": change_type,
                            "old_description": old_description,
                            "new_description": new_description,
                            "change_description": change_description
                        })
                        
                        # Clear cache
                        if has_urban_myths_table:
                            self.invalidate_cache_pattern(f"UrbanMyths_{myth_id}")
                        else:
                            self.invalidate_cache_pattern(f"WorldLore_{myth_id}")
                    except Exception as e:
                        logging.error(f"Error updating myth {myth_id}: {e}")
        
        return changes
    
    async def _develop_cultural_elements(self) -> List[Dict[str, Any]]:
        """Develop cultural elements over time"""
        changes = []
        
        # Choose a random sample of cultural elements to develop
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if the table exists
                table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'culturalelements'
                    );
                """)
                
                if not table_exists:
                    return changes
                
                rows = await conn.fetch("""
                    SELECT id, name, type, description, significance, practiced_by
                    FROM CulturalElements
                    ORDER BY RANDOM()
                    LIMIT 2
                """)
                
                for row in rows:
                    element_id = row['id']
                    element_name = row['name']
                    element_type = row['type']
                    old_description = row['description']
                    old_significance = row['significance']
                    practiced_by = row['practiced_by'] if row['practiced_by'] else []
                    
                    # Determine change type
                    change_type = random.choice(["formalize", "adapt", "spread", "codify"])
                    
                    # Create the run context
                    run_ctx = RunContextWrapper(context={
                        "user_id": self.user_id,
                        "conversation_id": self.conversation_id
                    })
                    
                    # Create appropriate prompt based on change type
                    if change_type == "formalize":
                        prompt = f"""
                        This cultural element is becoming more formalized and structured:
                        
                        ELEMENT: {element_name} ({element_type})
                        CURRENT DESCRIPTION: {old_description}
                        PRACTICED BY: {', '.join(practiced_by) if practiced_by else 'Various groups'}
                        
                        Update this description to show how this cultural element has become more formalized.
                        Add details about specific rules, structures, or ceremonies that have developed.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        new_significance = min(10, old_significance + 1)
                        change_description = "Element has become more formal and structured."
                        
                    elif change_type == "adapt":
                        prompt = f"""
                        This cultural element is adapting to changing social circumstances:
                        
                        ELEMENT: {element_name} ({element_type})
                        CURRENT DESCRIPTION: {old_description}
                        PRACTICED BY: {', '.join(practiced_by) if practiced_by else 'Various groups'}
                        
                        Update this description to show how this cultural element has adapted to changing times.
                        Show how it maintains its essence while evolving to remain relevant.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        new_significance = old_significance  # Remains the same
                        change_description = "Element has adapted to changing social circumstances."
                        
                    elif change_type == "spread":
                        new_groups = ["neighboring regions", "younger generations", "urban centers", "rural communities"]
                        added_groups = random.sample(new_groups, 1)
                        
                        # Don't add groups that are already listed
                        if practiced_by:
                            added_groups = [g for g in added_groups if g not in practiced_by]
                        
                        prompt = f"""
                        This cultural element is spreading to new groups, specifically: {', '.join(added_groups)}
                        
                        ELEMENT: {element_name} ({element_type})
                        CURRENT DESCRIPTION: {old_description}
                        CURRENTLY PRACTICED BY: {', '.join(practiced_by) if practiced_by else 'Various groups'}
                        
                        Update this description to show how this cultural element has spread to these new groups.
                        How might they adapt or interpret it slightly differently?
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        new_significance = min(10, old_significance + 1)
                        change_description = f"Element has spread to new groups: {', '.join(added_groups)}"
                        
                        # Update the practiced_by list
                        if practiced_by:
                            practiced_by.extend(added_groups)
                        else:
                            practiced_by = added_groups
                            
                    else:  # codify
                        prompt = f"""
                        This cultural element is becoming codified into formal rules, laws, or texts:
                        
                        ELEMENT: {element_name} ({element_type})
                        CURRENT DESCRIPTION: {old_description}
                        PRACTICED BY: {', '.join(practiced_by) if practiced_by else 'Various groups'}
                        
                        Update this description to show how this practice has become codified.
                        Add details about written texts, legal codes, or formal teachings that preserve it.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        new_significance = min(10, old_significance + 2)
                        change_description = "Element has been codified into formal rules, laws, or texts."
                    
                    # Create agent and get updated description
                    culture_agent = Agent(
                        name="CultureEvolutionAgent",
                        instructions="You develop and evolve cultural elements over time.",
                        model="o3-mini"
                    )
                    
                    result = await Runner.run(culture_agent, prompt, context=run_ctx.context)
                    new_description = result.final_output
                    
                    # Apply matriarchal theming
                    new_description = MatriarchalThemingUtils.apply_matriarchal_theme("culture", new_description, emphasis_level=1)
                    
                    # Apply the update to the database
                    try:
                        # Generate new embedding
                        embedding_text = f"{element_name} {element_type} {new_description}"
                        new_embedding = await generate_embedding(embedding_text)
                        
                        await conn.execute("""
                            UPDATE CulturalElements
                            SET description = $1, 
                                significance = $2, 
                                practiced_by = $3,
                                embedding = $4
                            WHERE id = $5
                        """, new_description, new_significance, practiced_by, new_embedding, element_id)
                        
                        # Record the change
                        changes.append({
                            "element_id": element_id,
                            "name": element_name,
                            "type": element_type,
                            "change_type": change_type,
                            "old_description": old_description,
                            "new_description": new_description,
                            "change_description": change_description
                        })
                        
                        # Clear cache
                        self.invalidate_cache_pattern(f"CulturalElements_{element_id}")
                    except Exception as e:
                        logging.error(f"Error updating cultural element {element_id}: {e}")
        
        return changes
    
    async def _shift_geopolitical_landscape(self) -> List[Dict[str, Any]]:
        """Evolve the geopolitical landscape with subtle shifts"""
        changes = []
        
        # Load faction data to work with
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if tables exist
                factions_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'factions'
                    );
                """)
                
                if not factions_exist:
                    return changes
                
                # Get a sample of factions
                factions = await conn.fetch("""
                    SELECT id, name, type, description, territory, rivals, allies
                    FROM Factions
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                if not factions:
                    return changes
                
                # Determine what kind of shift to generate
                shift_types = ["alliance_change", "territory_dispute", "influence_growth"]
                
                # Regions exist? Add "regional_governance" if we have them
                regions_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'geographicregions'
                    );
                """)
                
                if regions_exist:
                    shift_types.append("regional_governance")
                    
                    # Get a random region
                    regions = await conn.fetch("""
                        SELECT id, name, description, governing_faction
                        FROM GeographicRegions
                        ORDER BY RANDOM()
                        LIMIT 1
                    """)
                    
                    region = regions[0] if regions else None
                
                # Pick a random shift type
                shift_type = random.choice(shift_types)
                
                # Create the run context
                run_ctx = RunContextWrapper(context={
                    "user_id": self.user_id,
                    "conversation_id": self.conversation_id
                })
                
                # Handle the specific shift type
                if shift_type == "alliance_change" and len(factions) >= 2:
                    # Two factions change their relationship
                    faction1 = dict(factions[0])
                    faction2 = dict(factions[1])
                    
                    # Determine current relationship
                    current_allies1 = faction1['allies'] if faction1['allies'] else []
                    current_rivals1 = faction1['rivals'] if faction1['rivals'] else []
                    
                    # Set up the appropriate prompt based on current relationship
                    if faction2['name'] in current_allies1:
                        new_relationship = "rivalry"
                        if faction2['name'] in current_allies1:
                            current_allies1.remove(faction2['name'])
                        if faction2['name'] not in current_rivals1:
                            current_rivals1.append(faction2['name'])
                        
                        prompt = f"""
                        Two factions that were once allies are now becoming rivals:
                        
                        FACTION 1: {faction1['name']} ({faction1['type']})
                        Description: {faction1['description']}
                        
                        FACTION 2: {faction2['name']} ({faction2['type']})
                        Description: {faction2['description']}
                        
                        Update the description of Faction 1 to reflect this change in relationship.
                        Include information about why they've become rivals and what tensions exist now.
                        Maintain the matriarchal power dynamics in your description.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                    elif faction2['name'] in current_rivals1:
                        new_relationship = "alliance"
                        if faction2['name'] in current_rivals1:
                            current_rivals1.remove(faction2['name'])
                        if faction2['name'] not in current_allies1:
                            current_allies1.append(faction2['name'])
                        
                        prompt = f"""
                        Two factions that were once rivals are now becoming allies:
                        
                        FACTION 1: {faction1['name']} ({faction1['type']})
                        Description: {faction1['description']}
                        
                        FACTION 2: {faction2['name']} ({faction2['type']})
                        Description: {faction2['description']}
                        
                        Update the description of Faction 1 to reflect this change in relationship.
                        Include information about why they've become allies and what common interests they now share.
                        Maintain the matriarchal power dynamics in your description.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                    else:
                        new_relationship = "new alliance"
                        if faction2['name'] not in current_allies1:
                            current_allies1.append(faction2['name'])
                        
                        prompt = f"""
                        Two factions are forming a new alliance:
                        
                        FACTION 1: {faction1['name']} ({faction1['type']})
                        Description: {faction1['description']}
                        
                        FACTION 2: {faction2['name']} ({faction2['type']})
                        Description: {faction2['description']}
                        
                        Update the description of Faction 1 to reflect this new alliance.
                        Include information about why they've become allies and what mutual benefits exist.
                        Maintain the matriarchal power dynamics in your description.
                        
                        Return only the new description, written in the same style as the original.
                        """
                    
                    # Create agent and get updated description
                    geo_agent = Agent(
                        name="GeopoliticsAgent",
                        instructions="You evolve geopolitical relationships between factions.",
                        model="o3-mini"
                    )
                    
                    result = await Runner.run(geo_agent, prompt, context=run_ctx.context)
                    new_description = result.final_output
                    
                    # Apply matriarchal theming
                    new_description = MatriarchalThemingUtils.apply_matriarchal_theme("faction", new_description, emphasis_level=1)
                    
                    # Apply the update to the database
                    try:
                        # Generate new embedding
                        embedding_text = f"{faction1['name']} {faction1['type']} {new_description}"
                        new_embedding = await generate_embedding(embedding_text)
                        
                        await conn.execute("""
                            UPDATE Factions
                            SET description = $1, 
                                allies = $2,
                                rivals = $3,
                                embedding = $4
                            WHERE id = $5
                        """, new_description, current_allies1, current_rivals1, new_embedding, faction1['id'])
                        
                        # Need to also update faction 2's allies/rivals
                        current_allies2 = faction2['allies'] if faction2['allies'] else []
                        current_rivals2 = faction2['rivals'] if faction2['rivals'] else []
                        
                        if new_relationship == "alliance" or new_relationship == "new alliance":
                            if faction1['name'] in current_rivals2:
                                current_rivals2.remove(faction1['name'])
                            if faction1['name'] not in current_allies2:
                                current_allies2.append(faction1['name'])
                        else:  # rivalry
                            if faction1['name'] in current_allies2:
                                current_allies2.remove(faction1['name'])
                            if faction1['name'] not in current_rivals2:
                                current_rivals2.append(faction1['name'])
                        
                        await conn.execute("""
                            UPDATE Factions
                            SET allies = $1,
                                rivals = $2
                            WHERE id = $3
                        """, current_allies2, current_rivals2, faction2['id'])
                        
                        # Record the change
                        changes.append({
                            "change_type": "alliance_change",
                            "faction1_id": faction1['id'],
                            "faction1_name": faction1['name'],
                            "faction2_id": faction2['id'],
                            "faction2_name": faction2['name'],
                            "new_relationship": new_relationship,
                            "old_description": faction1['description'],
                            "new_description": new_description
                        })
                        
                        # Clear cache
                        self.invalidate_cache_pattern(f"Factions_{faction1['id']}")
                        self.invalidate_cache_pattern(f"Factions_{faction2['id']}")
                    except Exception as e:
                        logging.error(f"Error updating faction relationship: {e}")
                
                elif shift_type == "territory_dispute":
                    # A faction is expanding its territory
                    faction = dict(factions[0])
                    
                    prompt = f"""
                    This faction is expanding its territorial claims or influence:
                    
                    FACTION: {faction['name']} ({faction['type']})
                    Current Description: {faction['description']}
                    Current Territory: {faction['territory'] if faction['territory'] else "Unspecified areas"}
                    
                    Update the description to show how this faction is claiming new territory or expanding influence.
                    Include what areas they're moving into and any resistance or consequences this creates.
                    Maintain the matriarchal power dynamics in your description.
                    
                    Return only the new description, written in the same style as the original.
                    """
                    
                    # Create agent and get updated description
                    geo_agent = Agent(
                        name="GeopoliticsAgent",
                        instructions="You evolve geopolitical relationships between factions.",
                        model="o3-mini"
                    )
                    
                    result = await Runner.run(geo_agent, prompt, context=run_ctx.context)
                    new_description = result.final_output
                    
                    # Apply matriarchal theming
                    new_description = MatriarchalThemingUtils.apply_matriarchal_theme("faction", new_description, emphasis_level=1)
                    
                    # Apply the update to the database
                    try:
                        # Generate new embedding
                        embedding_text = f"{faction['name']} {faction['type']} {new_description}"
                        new_embedding = await generate_embedding(embedding_text)
                        
                        # Also update territory to include expansion
                        new_territory = faction['territory'] if faction['territory'] else []
                        new_areas = ["expanding into neighboring regions", "claiming new settlements"]
                        
                        if not isinstance(new_territory, list):
                            # Handle string case
                            new_territory = [new_territory, new_areas[0]]
                        else:
                            new_territory.append(random.choice(new_areas))
                        
                        await conn.execute("""
                            UPDATE Factions
                            SET description = $1,
                                territory = $2,
                                embedding = $3
                            WHERE id = $4
                        """, new_description, new_territory, new_embedding, faction['id'])
                        
                        # Record the change
                        changes.append({
                            "change_type": "territory_dispute",
                            "faction_id": faction['id'],
                            "faction_name": faction['name'],
                            "old_description": faction['description'],
                            "new_description": new_description,
                            "new_territory": new_territory
                        })
                        
                        # Clear cache
                        self.invalidate_cache_pattern(f"Factions_{faction['id']}")
                    except Exception as e:
                        logging.error(f"Error updating faction territory: {e}")
                
                elif shift_type == "regional_governance" and regions_exist and region:
                    # A region changes its governing faction
                    faction = dict(factions[0])
                    region = dict(region)
                    
                    # Only proceed if this is a different faction than current governance
                    if region['governing_faction'] != faction['name']:
                        prompt = f"""
                        A geographic region is changing its governing faction:
                        
                        REGION: {region['name']}
                        Current Description: {region['description']}
                        Previous Governing Faction: {region['governing_faction'] if region['governing_faction'] else "None"}
                        New Governing Faction: {faction['name']} ({faction['type']})
                        
                        Update the region's description to reflect this change in governance.
                        Include how this transition occurred and what changes the new faction is implementing.
                        Maintain the matriarchal power dynamics in your description.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        # Create agent and get updated description
                        geo_agent = Agent(
                            name="GeopoliticsAgent",
                            instructions="You evolve geopolitical relationships between factions.",
                            model="o3-mini"
                        )
                        
                        result = await Runner.run(geo_agent, prompt, context=run_ctx.context)
                        new_description = result.final_output
                        
                        # Apply matriarchal theming
                        new_description = MatriarchalThemingUtils.apply_matriarchal_theme("region", new_description, emphasis_level=1)
                        
                        # Apply the update to the database
                        try:
                            # Generate new embedding
                            embedding_text = f"{region['name']} {new_description}"
                            new_embedding = await generate_embedding(embedding_text)
                            
                            await conn.execute("""
                                UPDATE GeographicRegions
                                SET description = $1,
                                    governing_faction = $2,
                                    embedding = $3
                                WHERE id = $4
                            """, new_description, faction['name'], new_embedding, region['id'])
                            
                            # Record the change
                            changes.append({
                                "change_type": "regional_governance",
                                "region_id": region['id'],
                                "region_name": region['name'],
                                "old_governing_faction": region['governing_faction'] if region['governing_faction'] else "None",
                                "new_governing_faction": faction['name'],
                                "old_description": region['description'],
                                "new_description": new_description
                            })
                            
                            # Clear cache
                            self.invalidate_cache_pattern(f"GeographicRegions_{region['id']}")
                        except Exception as e:
                            logging.error(f"Error updating region governance: {e}")
                
                else:  # influence_growth
                    # A faction is growing in influence
                    faction = dict(factions[0])
                    
                    prompt = f"""
                    This faction is experiencing a significant growth in influence and power:
                    
                    FACTION: {faction['name']} ({faction['type']})
                    Current Description: {faction['description']}
                    
                    Update the description to show how this faction is becoming more influential.
                    Include new strategies they're employing, resources they're acquiring, or alliances they're leveraging.
                    Maintain the matriarchal power dynamics in your description.
                    
                    Return only the new description, written in the same style as the original.
                    """
                    
                    # Create agent and get updated description
                    geo_agent = Agent(
                        name="GeopoliticsAgent",
                        instructions="You evolve geopolitical relationships between factions.",
                        model="o3-mini"
                    )
                    
                    result = await Runner.run(geo_agent, prompt, context=run_ctx.context)
                    new_description = result.final_output
                    
                    # Apply matriarchal theming
                    new_description = MatriarchalThemingUtils.apply_matriarchal_theme("faction", new_description, emphasis_level=1)
                    
                    # Apply the update to the database
                    try:
                        # Generate new embedding
                        embedding_text = f"{faction['name']} {faction['type']} {new_description}"
                        new_embedding = await generate_embedding(embedding_text)
                        
                        await conn.execute("""
                            UPDATE Factions
                            SET description = $1,
                                embedding = $2
                            WHERE id = $3
                        """, new_description, new_embedding, faction['id'])
                        
                        # Record the change
                        changes.append({
                            "change_type": "influence_growth",
                            "faction_id": faction['id'],
                            "faction_name": faction['name'],
                            "old_description": faction['description'],
                            "new_description": new_description
                        })
                        
                        # Clear cache
                        self.invalidate_cache_pattern(f"Factions_{faction['id']}")
                    except Exception as e:
                        logging.error(f"Error updating faction influence: {e}")
        
        return changes
    
    async def _evolve_notable_figures(self) -> List[Dict[str, Any]]:
        """Evolve reputations and stories of notable figures"""
        changes = []
        
        # Check if we have a dedicated notable figures table
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                has_notable_figures_table = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'notablefigures'
                    );
                """)
                
                if has_notable_figures_table:
                    # Get figures from dedicated table
                    rows = await conn.fetch("""
                        SELECT id, name, description, significance, reputation
                        FROM NotableFigures
                        ORDER BY RANDOM()
                        LIMIT 2
                    """)
                else:
                    # Fall back to world lore with notable_figure category
                    rows = await conn.fetch("""
                        SELECT id, name, description, significance
                        FROM WorldLore
                        WHERE category = 'notable_figure'
                        ORDER BY RANDOM()
                        LIMIT 2
                    """)
                
                # Process each notable figure
                for row in rows:
                    figure_id = row['id']
                    figure_name = row['name']
                    old_description = row['description']
                    old_significance = row['significance'] if 'significance' in row else 5
                    
                    # Determine change type
                    change_options = ["reputation_rise", "reputation_fall", "scandal", "achievement", "reform"]
                    
                    # If we have reputation field, use it to influence likely changes
                    reputation = row.get('reputation', 5)
                    
                    # Create weights based on current reputation
                    if reputation and reputation >= 7:
                        # High reputation more likely to fall or have scandal
                        weights = [1, 3, 2, 1, 1]
                    elif reputation and reputation <= 3:
                        # Low reputation more likely to rise or reform
                        weights = [3, 1, 1, 2, 2]
                    else:
                        # Mid reputation equal chances
                        weights = [1, 1, 1, 1, 1]
                    
                    # Weighted choice
                    change_type = random.choices(
                        change_options, 
                        weights=weights,
                        k=1
                    )[0]
                    
                    # Create the run context
                    run_ctx = RunContextWrapper(context={
                        "user_id": self.user_id,
                        "conversation_id": self.conversation_id
                    })
                    
                    # Create appropriate prompt based on change type
                    if change_type == "reputation_rise":
                        prompt = f"""
                        This notable figure is experiencing a rise in reputation and influence:
                        
                        FIGURE: {figure_name}
                        Current Description: {old_description}
                        
                        Update the description to show how this figure's reputation has improved.
                        Include their recent positive actions, public perception changes, or new supporters.
                        Maintain the matriarchal power dynamics in your description.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        new_reputation = min(10, (reputation if reputation else 5) + random.randint(1, 2))
                        reputation_change = "improved"
                        
                    elif change_type == "reputation_fall":
                        prompt = f"""
                        This notable figure is experiencing a decline in reputation and influence:
                        
                        FIGURE: {figure_name}
                        Current Description: {old_description}
                        
                        Update the description to show how this figure's reputation has declined.
                        Include their recent missteps, public perception changes, or lost support.
                        Maintain the matriarchal power dynamics in your description.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        new_reputation = max(1, (reputation if reputation else 5) - random.randint(1, 2))
                        reputation_change = "declined"
                        
                    elif change_type == "scandal":
                        prompt = f"""
                        This notable figure is involved in a recent scandal:
                        
                        FIGURE: {figure_name}
                        Current Description: {old_description}
                        
                        Update the description to incorporate a scandal this figure is facing.
                        Include the nature of the scandal and how they're responding to it.
                        Maintain the matriarchal power dynamics in your description.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        new_reputation = max(1, (reputation if reputation else 5) - random.randint(2, 3))
                        reputation_change = "significantly damaged"
                        
                    elif change_type == "achievement":
                        prompt = f"""
                        This notable figure has recently achieved something significant:
                        
                        FIGURE: {figure_name}
                        Current Description: {old_description}
                        
                        Update the description to incorporate a major achievement by this figure.
                        Include what they accomplished and how it has affected their standing.
                        Maintain the matriarchal power dynamics in your description.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        new_reputation = min(10, (reputation if reputation else 5) + random.randint(2, 3))
                        reputation_change = "significantly improved"
                        
                    else:  # reform
                        prompt = f"""
                        This notable figure has undergone a personal or political reform:
                        
                        FIGURE: {figure_name}
                        Current Description: {old_description}
                        
                        Update the description to show how this figure has changed their stance, approach, or values.
                        Include what prompted this change and how others have reacted to it.
                        Maintain the matriarchal power dynamics in your description.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        # Reform could go either way
                        direction = random.choice([-1, 1])
                        new_reputation = max(1, min(10, (reputation if reputation else 5) + direction * random.randint(1, 2)))
                        reputation_change = "transformed"
                    
                    # Create agent and get updated description
                    figure_agent = Agent(
                        name="NotableFigureAgent",
                        instructions="You evolve the stories and reputations of notable figures.",
                        model="o3-mini"
                    )
                    
                    result = await Runner.run(figure_agent, prompt, context=run_ctx.context)
                    new_description = result.final_output
                    
                    # Apply matriarchal theming
                    new_description = MatriarchalThemingUtils.apply_matriarchal_theme("character", new_description, emphasis_level=1)
                    
                    # Apply the update to the database
                    try:
                        # Generate new embedding
                        embedding_text = f"{figure_name} {new_description}"
                        new_embedding = await generate_embedding(embedding_text)
                        
                        if has_notable_figures_table:
                            # Update in dedicated table
                            await conn.execute("""
                                UPDATE NotableFigures
                                SET description = $1,
                                    reputation = $2,
                                    embedding = $3
                                WHERE id = $4
                            """, new_description, new_reputation, new_embedding, figure_id)
                        else:
                            # Update in WorldLore
                            await conn.execute("""
                                UPDATE WorldLore
                                SET description = $1,
                                    embedding = $2
                                WHERE id = $3
                            """, new_description, new_embedding, figure_id)
                        
                        # Record the change
                        changes.append({
                            "figure_id": figure_id,
                            "name": figure_name,
                            "change_type": change_type,
                            "old_description": old_description,
                            "new_description": new_description,
                            "reputation_change": reputation_change
                        })
                        
                        # Clear cache
                        if has_notable_figures_table:
                            self.invalidate_cache_pattern(f"NotableFigures_{figure_id}")
                        else:
                            self.invalidate_cache_pattern(f"WorldLore_{figure_id}")
                    except Exception as e:
                        logging.error(f"Error updating notable figure {figure_id}: {e}")
        
        return changes
    
    #===========================================================================
    # WORLD EVOLUTION METHODS
    #===========================================================================
    
    @with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            action_type="evolve_world_over_time",
            action_description="Evolving world over time",
            id_from_context=lambda ctx: "lore_dynamics"
        )
        async def evolve_world_over_time(self, ctx, days_passed: int = 30) -> Dict[str, Any]:
            """
            Evolve the world over time, simulating the passage of days or weeks.
            
            Args:
                days_passed: Number of days to simulate
                
            Returns:
                Dictionary of evolution results
            """
            # Create run context
            run_ctx = RunContextWrapper(context=ctx.context)
            
            evolution_results = {}
            
            # 1. Mature lore naturally over time
            try:
                maturation = await self.mature_lore_over_time(days_passed)
                evolution_results["lore_maturation"] = maturation
                logging.info(f"Matured lore over {days_passed} days with {maturation.get('changes_applied', 0)} changes")
            except Exception as e:
                logging.error(f"Error maturing lore: {e}")
                evolution_results["lore_maturation_error"] = str(e)
            
            # 2. Generate random emergent events based on time passed
            try:
                # Scale number of events with time passed, but randomize
                num_events = max(1, min(5, days_passed // 10))
                num_events = random.randint(1, num_events)
                
                emergent_events = []
                for _ in range(num_events):
                    event = await self.generate_emergent_event(run_ctx)
                    emergent_events.append(event)
                
                evolution_results["emergent_events"] = emergent_events
                logging.info(f"Generated {len(emergent_events)} emergent events")
            except Exception as e:
                logging.error(f"Error generating emergent events: {e}")
                evolution_results["emergent_events_error"] = str(e)
            
            # Report to governance system
            await self.report_action(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="lore_generator",
                action={
                    "type": "evolve_world_over_time",
                    "description": f"Evolved world over {days_passed} days"
                },
                result={
                    "emergent_events": len(evolution_results.get("emergent_events", [])),
                    "maturation_changes": evolution_results.get("lore_maturation", {}).get("changes_applied", 0)
                }
            )
            
            return evolution_results
        
        async def register_with_governance(self):
            """Register with Nyx governance system."""
            await super().register_with_governance(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="lore_dynamics",
                directive_text="Evolve, expand, and develop world lore through emergent events and natural maturation.",
                scope="world_building",
                priority=DirectivePriority.MEDIUM
            )
            
            logging.info(f"LoreDynamicsSystem registered with Nyx governance for user {self.user_id}, conversation {self.conversation_id}")
