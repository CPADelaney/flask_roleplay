# lore/systems/dynamics.py

import logging
import json
import random
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

from agents import Agent, ModelSettings, Runner, function_tool, handoff, trace
from agents import InputGuardrail, OutputGuardrail, GuardrailFunctionOutput, RunContextWrapper
from agents.run import RunConfig

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
        
        # Initialize specialized agents for different tasks
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize specialized agents for different lore tasks"""
        # Core instructions for all agents
        base_instructions = """
        You are working with a fantasy world featuring matriarchal power structures.
        All content should reflect feminine power and authority as the natural order.
        Male elements should be presented in supportive or subservient positions.
        """
        
        # Lore update agent with specific schema
        self.lore_update_agent = Agent(
            name="LoreUpdateAgent",
            instructions=f"""
            {base_instructions}
            
            You update existing lore elements based on narrative events while maintaining thematic consistency.
            Your updates should be meaningful and reflect the impact of events on the world.
            Maintain the matriarchal power dynamics in all updates.
            """,
            model="o3-mini"
        )
        
        # Event generation agent
        self.event_generation_agent = Agent(
            name="EventGenerationAgent",
            instructions=f"""
            {base_instructions}
            
            You create emergent world events for fantasy settings with matriarchal power structures.
            Events should be specific and detailed, creating opportunities for character development and plot advancement.
            Focus on how events impact or reinforce matriarchal power dynamics.
            """,
            model="o3-mini"
        )
        
        # Lore creation agent
        self.lore_creation_agent = Agent(
            name="LoreCreationAgent",
            instructions=f"""
            {base_instructions}
            
            You create new lore elements that emerge from significant events or natural evolution.
            New elements should fit seamlessly with existing lore while expanding the world.
            Ensure all new lore reinforces matriarchal power dynamics.
            """,
            model="o3-mini"
        )
        
        # Specialized event generation agents
        self.political_event_agent = Agent(
            name="PoliticalEventAgent",
            instructions=f"""
            {base_instructions}
            
            You create detailed political events for matriarchal fantasy worlds.
            Focus on power dynamics, succession, alliances, and court intrigue.
            Events should highlight feminine leadership and authority.
            """,
            model="o3-mini"
        )
        
        self.military_event_agent = Agent(
            name="MilitaryEventAgent",
            instructions=f"""
            {base_instructions}
            
            You create detailed military conflicts for matriarchal fantasy worlds.
            Focus on strategy, leadership, and consequences of warfare.
            Events should highlight feminine military command structures.
            """,
            model="o3-mini"
        )
        
        self.cultural_event_agent = Agent(
            name="CulturalEventAgent",
            instructions=f"""
            {base_instructions}
            
            You create detailed cultural developments for matriarchal fantasy worlds.
            Focus on traditions, arts, festivals, and social changes.
            Events should highlight feminine cultural influence and values.
            """,
            model="o3-mini"
        )
    
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
    
    # Create schema models for structured output
    class EventValidation(BaseModel):
        """Validation model for event descriptions"""
        is_valid: bool
        reasoning: str

    class LoreUpdate(BaseModel):
        """Model for lore updates"""
        lore_id: str
        lore_type: str
        name: str
        old_description: str
        new_description: str
        update_reason: str
        impact_level: int = Field(..., ge=1, le=10)
    
    class LoreElement(BaseModel):
        """Model for new lore elements"""
        lore_type: str
        name: str
        description: str
        connection: str
        significance: int = Field(..., ge=1, le=10)
        
    # Guardrail function for event validation
    async def _validate_event_description(self, ctx, agent, input_data: str) -> GuardrailFunctionOutput:
        """Validate that the event description is appropriate for lore evolution"""
        # Create validation agent with structured output
        validation_agent = Agent(
            name="EventValidationAgent",
            instructions="Determine if the event description is appropriate for world lore evolution.",
            model="o3-mini",
            output_type=self.EventValidation
        )
        
        # Create run context
        run_ctx = self.create_run_context(ctx)
        
        # Create validation prompt
        prompt = f"""
        Evaluate if this event description is appropriate for evolving world lore:
        
        EVENT: {input_data}
        
        Consider:
        1. Is it specific enough to cause meaningful lore changes?
        2. Is it consistent with a matriarchal fantasy setting?
        3. Is it free from inappropriate, offensive, or out-of-setting content?
        
        Return a structured response with:
        - is_valid: true/false
        - reasoning: Explanation of your decision
        """
        
        # Run validation
        result = await Runner.run(validation_agent, prompt, context=run_ctx.context)
        validation = result.final_output
        
        # Return guardrail output
        return GuardrailFunctionOutput(
            output_info=validation.dict(),
            tripwire_triggered=not validation.is_valid
        )
    
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
        # Create a trace for the entire evolution process
        with trace(
            "LoreEvolutionWorkflow", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "event": event_description[:50],
                "action": "evolve_lore"
            }
        ):
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
            
            # Create run context
            run_ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
            # Add input guardrail for event validation
            input_guardrail = InputGuardrail(guardrail_function=self._validate_event_description)
            
            # Configure the evolution agent with tools
            evolution_agent = Agent(
                name="LoreEvolutionAgent",
                instructions="""
                You guide the evolution of world lore based on significant events.
                Identify affected elements, generate updates, and create new lore elements as appropriate.
                Maintain thematic consistency and reinforce matriarchal power dynamics in all updates.
                """,
                model="o3-mini",
                input_guardrails=[input_guardrail],
                tools=[
                    function_tool(self._identify_affected_lore),
                    function_tool(self._generate_lore_updates),
                    function_tool(self._apply_lore_updates),
                    function_tool(self._generate_consequential_lore)
                ]
            )
            
            # Set up run configuration for tracing
            run_config = RunConfig(
                workflow_name="LoreEvolution",
                trace_metadata={
                    "user_id": str(self.user_id),
                    "conversation_id": str(self.conversation_id),
                    "event_type": "lore_evolution"
                }
            )
            
            # Create prompt for the evolution agent
            prompt = f"""
            A significant event has occurred in the world that requires lore evolution:
            
            EVENT DESCRIPTION:
            {event_description}
            
            Please guide the evolution of world lore by:
            1. Identifying affected lore elements using the _identify_affected_lore function
            2. Generating appropriate updates for each element using _generate_lore_updates
            3. Applying the updates to the database using _apply_lore_updates
            4. Generating potential new lore elements triggered by this event using _generate_consequential_lore
            
            Provide a summary of the changes made and their significance to the world.
            """
            
            # Run the evolution process
            result = await Runner.run(
                evolution_agent,
                prompt,
                context=run_ctx.context,
                run_config=run_config
            )
            
            # Return the result with structured data
            try:
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
                    "new_elements": new_elements,
                    "summary": result.final_output
                }
            except Exception as e:
                logging.error(f"Error in lore evolution process: {e}")
                return {
                    "error": str(e),
                    "event_description": event_description,
                    "agent_output": result.final_output
                }
    
    @function_tool
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
    
    @function_tool
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
        
        # Create a trace for the update generation process
        with trace(
            "LoreUpdateGeneration", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "elements_count": len(affected_elements),
                "event": event_description[:50]
            }
        ):
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
                
                # Configure the update agent with structured output
                update_agent = self.lore_update_agent.clone(
                    output_type=self.LoreUpdate
                )
                
                # Get the response with structured output
                result = await Runner.run(update_agent, prompt, context=run_ctx.context)
                update_data = result.final_output
                
                # Add the update
                updates.append(update_data.dict())
            
        return updates
    
    @function_tool
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
    
    @function_tool
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
        # Create a trace for the consequential lore generation
        with trace(
            "ConsequentialLoreGeneration", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "event": event_description[:50]
            }
        ):
            # Create the run context
            run_ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
            # Extract names of affected elements for context
            affected_names = [f"{e['name']} ({e['lore_type']})" for e in affected_elements[:5]]
            affected_names_text = ", ".join(affected_names)
            
            # Configure the lore creation agent with structured output
            lore_creation_agent = self.lore_creation_agent.clone(
                output_type=List[self.LoreElement]
            )
            
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
            - "lore_type": The type of lore element (WorldLore, Factions, CulturalElements, UrbanMyths, etc.)
            - "name": A name for the element
            - "description": A detailed description
            - "connection": How it relates to the event and other lore
            - "significance": A number from 1-10 indicating importance
            
            Return a structured list of LoreElement objects.
            IMPORTANT: Maintain the matriarchal/femdom power dynamics in all new lore.
            """
            
            # Get the response with structured output
            result = await Runner.run(lore_creation_agent, prompt, context=run_ctx.context)
            new_elements = result.final_output
            
            # Process and save the new elements
            processed_elements = []
            for element in new_elements:
                # Apply matriarchal theming
                element.description = MatriarchalThemingUtils.apply_matriarchal_theme(
                    element.lore_type.lower(), element.description
                )
                
                # Save the element
                await self._save_new_lore_element(element, event_description)
                
                # Add to processed list
                processed_elements.append(element.dict())
                
            return processed_elements
    
    async def _save_new_lore_element(
        self, 
        element: LoreElement, 
        event_description: str
    ) -> None:
        """
        Save a newly generated lore element to the appropriate table
        
        Args:
            element: The lore element to save
            event_description: Original event description
        """
        try:
            lore_type = element.lore_type
            name = element.name
            description = element.description
            significance = element.significance
            
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    # Save based on type
                    if lore_type == "WorldLore":
                        await conn.execute("""
                            INSERT INTO WorldLore 
                            (name, category, description, significance, tags)
                            VALUES ($1, $2, $3, $4, $5)
                        """, name, 'event_consequence', description, significance, ['event_consequence', 'new_lore'])
                    elif lore_type == "Factions":
                        await conn.execute("""
                            INSERT INTO Factions
                            (name, type, description, values, goals, founding_story)
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """, name, 'event_consequence', description, 
                             ['power', 'authority'], ['stability', 'influence'], 
                             f"Founded in response to: {event_description}")
                    elif lore_type == "CulturalElements":
                        await conn.execute("""
                            INSERT INTO CulturalElements
                            (name, element_type, description, practiced_by, significance, historical_origin)
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """, name, 'tradition', description, 
                             ['society'], significance, 
                             f"Emerged from: {event_description}")
                    elif lore_type == "HistoricalEvents":
                        await conn.execute("""
                            INSERT INTO HistoricalEvents
                            (name, description, date_description, significance, consequences)
                            VALUES ($1, $2, $3, $4, $5)
                        """, name, description, 'Recently', significance, ['Still unfolding'])
                    else:
                        # Generic fallback for unknown types
                        await conn.execute("""
                            INSERT INTO WorldLore 
                            (name, category, description, significance, tags)
                            VALUES ($1, $2, $3, $4, $5)
                        """, name, lore_type.lower(), description, significance, [lore_type.lower(), 'event_consequence'])
                        
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
        # Create a trace for the event generation process
        with trace(
            "EmergentEventGeneration", 
            group_id=self.trace_group_id,
            metadata=self.trace_metadata
        ):
            # Create the run context
            run_ctx = self.create_run_context(ctx)
            
            # Get context data from database
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
            
            # Setup specialized agents as handoffs
            # Define handlers for appropriate event types
            event_orchestrator_agent = Agent(
                name="EventOrchestratorAgent",
                instructions="""
                You orchestrate the generation of emergent events in the world.
                First determine what type of event should occur, then delegate to specialized agents.
                Ensure all events reflect matriarchal power dynamics.
                """,
                model="o3-mini",
                handoffs=[
                    handoff(self.political_event_agent,
                           tool_name_override="transfer_to_political_agent",
                           tool_description_override="Transfer to political event agent for political events"),
                    handoff(self.military_event_agent,
                           tool_name_override="transfer_to_military_agent",
                           tool_description_override="Transfer to military event agent for combat events"),
                    handoff(self.cultural_event_agent,
                           tool_name_override="transfer_to_cultural_agent",
                           tool_description_override="Transfer to cultural event agent for cultural events")
                ]
            )
            
            # Run configuration for tracing
            run_config = RunConfig(
                workflow_name="EmergentEventGeneration",
                trace_metadata={
                    "user_id": str(self.user_id),
                    "conversation_id": str(self.conversation_id),
                    "event_type": "emergent_event"
                }
            )
            
            # Determine event type
            event_types = [
                "political_shift", "military_conflict", "natural_disaster",
                "cultural_development", "technological_advancement", "religious_event",
                "economic_change", "diplomatic_incident"
            ]
            
            # Choose event type with weights based on available data
            event_type = self._choose_event_type(event_types, faction_data, nation_data, location_data)
            
            # Create a prompt for the event orchestrator
            prompt = f"""
            Generate an emergent world event for this fantasy world. Based on the available data, 
            I've determined that a {event_type.replace('_', ' ')} event would be most appropriate.
            
            AVAILABLE WORLD DATA:
            
            Factions:
            {json.dumps(faction_data, indent=2)}
            
            Nations:
            {json.dumps(nation_data, indent=2)}
            
            Locations:
            {json.dumps(location_data, indent=2)}
            
            Please determine the appropriate specialized agent to handle this event type:
            
            - Use transfer_to_political_agent for political_shift, diplomatic_incident, or economic_change events
            - Use transfer_to_military_agent for military_conflict events
            - Use transfer_to_cultural_agent for cultural_development, religious_event, natural_disaster, or technological_advancement events
            
            The specialized agent will generate a detailed event description that can be used to evolve the world lore.
            """
            
            # Run the event generation process with handoffs
            result = await Runner.run(
                event_orchestrator_agent,
                prompt,
                context=run_ctx.context,
                run_config=run_config
            )
            
            # Process the result
            try:
                response_text = result.final_output
                event_data = {}
                
                # Try to parse as JSON first
                try:
                    event_data = json.loads(response_text)
                except json.JSONDecodeError:
                    # If it's not JSON, create a simple event object
                    event_data = {
                        "event_type": event_type,
                        "description": response_text
                    }
                
                # Generate a name if none provided
                if "event_name" not in event_data:
                    event_data["event_name"] = f"{event_type.replace('_', ' ').title()} Event"
                
                # Apply matriarchal theming
                if "description" in event_data:
                    event_data["description"] = MatriarchalThemingUtils.apply_matriarchal_theme(
                        "event", event_data["description"]
                    )
                
                # Create a unified event description for lore evolution
                event_description = f"{event_data.get('event_name', 'Unnamed Event')}: {event_data.get('description', '')}"
                
                # Use lore evolution functionality to evolve lore based on this event
                lore_updates = await self.evolve_lore_with_event(event_description)
                
                # Add the lore updates to the event data
                event_data["lore_updates"] = lore_updates
                
                # Return the combined data
                return event_data
                
            except Exception as e:
                logging.error(f"Error generating emergent event: {e}")
                return {"error": str(e), "raw_output": result.final_output}
    
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
        # Create a trace for the maturation process
        with trace(
            "LoreMaturationProcess", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "days_passed": days_passed
            }
        ):
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
        
        Return a structured LoreUpdate object with:
        - lore_id: "{element['lore_id']}"
        - lore_type: "{element['lore_type']}"
        - name: "{element['name']}"
        - old_description: The current description (provided above)
        - new_description: The updated description that reflects event impact
        - update_reason: Detailed explanation of why this update makes sense
        - impact_level: A number from 1-10 indicating how significantly this event affects this element
        """
        
        return prompt
    
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
                    
                    # Create tracing configuration
                    trace_config = RunConfig(
                        workflow_name="MythEvolution",
                        trace_metadata={
                            **self.trace_metadata,
                            "myth_id": str(myth_id),
                            "change_type": change_type
                        }
                    )
                    
                    result = await Runner.run(
                        myth_agent, 
                        prompt, 
                        context=run_ctx.context,
                        run_config=trace_config
                    )
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
                    
                    # Create tracing configuration
                    trace_config = RunConfig(
                        workflow_name="CultureEvolution",
                        trace_metadata={
                            **self.trace_metadata,
                            "element_id": str(element_id),
                            "change_type": change_type
                        }
                    )
                    
                    result = await Runner.run(
                        culture_agent, 
                        prompt, 
                        context=run_ctx.context,
                        run_config=trace_config
                    )
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
        # Create a trace for the entire evolution process
        with trace(
            "WorldEvolutionWorkflow", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "days_passed": days_passed
            }
        ):
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
