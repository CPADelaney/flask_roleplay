# lore/systems/dynamics.py

import logging
import json
import random
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

# ------------------ AGENTS SDK IMPORTS ------------------
from agents import (
    Agent,
    ModelSettings,
    Runner,
    function_tool,
    handoff,
    trace,
    InputGuardrail,
    OutputGuardrail,
    GuardrailFunctionOutput,
    RunContextWrapper
)
from agents.run import RunConfig

# ------------------ NYX/GOVERNANCE IMPORTS ------------------
from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

# ------------------ PROJECT IMPORTS ------------------
from embedding.vector_store import generate_embedding
from lore.core.base_manager import BaseLoreManager
from lore.managers.geopolitical import GeopoliticalSystemManager
from lore.utils.theming import MatriarchalThemingUtils

class LoreDynamicsSystem(BaseLoreManager):
    """
    Consolidated system for evolving world lore, generating emergent events,
    expanding content, and managing how the world changes over time.
    
    NOW AGENT-IFIED: we replace internal "random or code-based logic" with 
    dynamic LLM calls to ensure all changes are interesting, unique, and 
    thematically consistent.
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
        base_instructions = (
            "You are working with a fantasy world featuring matriarchal power structures. "
            "All content should reflect feminine power and authority as the natural order. "
            "Male elements should be presented in supportive or subservient positions."
        )
        
        # Lore update agent
        self.lore_update_agent = Agent(
            name="LoreUpdateAgent",
            instructions=(
                f"{base_instructions}\n\n"
                "You update existing lore elements based on narrative events while maintaining thematic consistency. "
                "Your updates should be meaningful and reflect the impact of events on the world. "
                "Maintain the matriarchal power dynamics in all updates."
            ),
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        # Event generation agent
        self.event_generation_agent = Agent(
            name="EventGenerationAgent",
            instructions=(
                f"{base_instructions}\n\n"
                "You create emergent world events for fantasy settings with matriarchal power structures. "
                "Events should be specific and detailed, creating opportunities for character development "
                "and plot advancement. Focus on how events impact or reinforce matriarchal power dynamics."
            ),
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        # Lore creation agent
        self.lore_creation_agent = Agent(
            name="LoreCreationAgent",
            instructions=(
                f"{base_instructions}\n\n"
                "You create new lore elements that emerge from significant events or natural evolution. "
                "New elements should fit seamlessly with existing lore while expanding the world. "
                "Ensure all new lore reinforces matriarchal power dynamics."
            ),
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        # Political event agent
        self.political_event_agent = Agent(
            name="PoliticalEventAgent",
            instructions=(
                f"{base_instructions}\n\n"
                "You create detailed political events for matriarchal fantasy worlds. "
                "Focus on power dynamics, succession, alliances, and court intrigue. "
                "Events should highlight feminine leadership and authority."
            ),
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        # Military event agent
        self.military_event_agent = Agent(
            name="MilitaryEventAgent",
            instructions=(
                f"{base_instructions}\n\n"
                "You create detailed military conflicts for matriarchal fantasy worlds. "
                "Focus on strategy, leadership, and the consequences of warfare. "
                "Events should highlight feminine military command structures."
            ),
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        # Cultural event agent
        self.cultural_event_agent = Agent(
            name="CulturalEventAgent",
            instructions=(
                f"{base_instructions}\n\n"
                "You create detailed cultural developments for matriarchal fantasy worlds. "
                "Focus on traditions, arts, festivals, and social changes. "
                "Events should highlight feminine cultural influence and values."
            ),
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.9)
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
    # SCHEMA MODELS
    #===========================================================================
    
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
    
    #===========================================================================
    # EVENT VALIDATION GUARDRAIL
    #===========================================================================
    
    async def _validate_event_description(self, ctx, agent, input_data: str) -> GuardrailFunctionOutput:
        """
        Validate that the event description is appropriate for lore evolution
        using an LLM-based validation agent that outputs EventValidation.
        """
        validation_agent = Agent(
            name="EventValidationAgent",
            instructions=(
                "Determine if the event description is appropriate for world lore evolution. "
                "Return a structured response with is_valid=True/False and reasoning."
            ),
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.8),
            output_type=self.EventValidation
        )
        
        run_ctx = self.create_run_context(ctx)
        
        prompt = f"""
        Evaluate if this event description is appropriate for evolving world lore:
        
        EVENT: {input_data}
        
        Consider:
        1. Is it specific enough to cause meaningful lore changes?
        2. Is it consistent with a matriarchal fantasy setting?
        3. Is it free from inappropriate or out-of-setting content?
        
        Output JSON with:
        - is_valid
        - reasoning
        """
        
        result = await Runner.run(validation_agent, prompt, context=run_ctx.context)
        validation = result.final_output
        
        return GuardrailFunctionOutput(
            output_info=validation.dict(),
            tripwire_triggered=not validation.is_valid
        )
    
    #===========================================================================
    # CORE LORE EVOLUTION
    #===========================================================================
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_lore_with_event",
        action_description="Evolving lore with event",
        id_from_context=lambda ctx: "lore_dynamics"
    )
    async def evolve_lore_with_event(self, event_description: str) -> Dict[str, Any]:
        """
        Evolve world lore based on a narrative event, using a specialized LLM-based
        agent to coordinate steps (identify, generate updates, apply changes, create new lore).
        """
        with trace(
            "LoreEvolutionWorkflow", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "event": event_description[:50],
                "action": "evolve_lore"
            }
        ):
            await self.ensure_initialized()
            
            permission = await self.check_permission(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="lore_generator",
                action_type="evolve_lore_with_event",
                action_details={"event_description": event_description}
            )
            if not permission["approved"]:
                logging.warning(f"Lore evolution not approved: {permission.get('reasoning')}")
                return {"error": permission.get("reasoning"), "approved": False}
            
            run_ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
            # Input guardrail for event validation
            input_guardrail = InputGuardrail(guardrail_function=self._validate_event_description)
            
            # Evolving agent that orchestrates the steps
            evolution_agent = Agent(
                name="LoreEvolutionAgent",
                instructions=(
                    "You guide the evolution of world lore based on significant events. "
                    "Identify affected elements, generate updates, apply them, and create new lore. "
                    "Maintain matriarchal power dynamics in all transformations."
                ),
                model="o3-mini",
                model_settings=ModelSettings(temperature=0.9),
                input_guardrails=[input_guardrail],
                tools=[
                    function_tool(self._identify_affected_lore),
                    function_tool(self._generate_lore_updates),
                    function_tool(self._apply_lore_updates),
                    function_tool(self._generate_consequential_lore)
                ]
            )
            
            run_config = RunConfig(
                workflow_name="LoreEvolution",
                trace_metadata={
                    "user_id": str(self.user_id),
                    "conversation_id": str(self.conversation_id),
                    "event_type": "lore_evolution"
                }
            )
            
            prompt = f"""
            A significant event has occurred that requires lore evolution:
            
            EVENT DESCRIPTION:
            {event_description}
            
            Please:
            1) Identify affected lore elements with _identify_affected_lore
            2) Generate updates for each with _generate_lore_updates
            3) Apply them via _apply_lore_updates
            4) Generate potential new lore elements with _generate_consequential_lore
            
            Finally, provide a summary of changes made and their significance.
            """
            
            # Let the agent orchestrate
            result = await Runner.run(
                evolution_agent,
                prompt,
                context=run_ctx.context,
                run_config=run_config
            )
            
            # In parallel (or afterwards), we manually call the steps ourselves
            # just to ensure we have structured data to return. The agent usage above
            # might do partial calls, but let's ensure we finalize them here.
            try:
                affected_elements = await self._identify_affected_lore(event_description)
                updates = await self._generate_lore_updates(affected_elements, event_description)
                await self._apply_lore_updates(updates)
                new_elements = await self._generate_consequential_lore(event_description, affected_elements)
                
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
    
    #===========================================================================
    # IDENTIFY AFFECTED LORE
    #===========================================================================
    @function_tool
    async def _identify_affected_lore(self, event_description: str) -> List[Dict[str, Any]]:
        """
        Identify lore elements that might be impacted by an event.
        We keep the DB logic, but rely on embeddings for similarity. 
        This is mostly 'technical'. 
        """
        event_embedding = await generate_embedding(event_description)
        affected_elements = []
        
        lore_types = [
            "WorldLore", "Factions", "CulturalElements", "HistoricalEvents",
            "GeographicRegions", "LocationLore", "UrbanMyths", "LocalHistories",
            "Landmarks", "NotableFigures"
        ]
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                for lore_type in lore_types:
                    try:
                        table_exists = await conn.fetchval(f"""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = '{lore_type.lower()}'
                            );
                        """)
                        if not table_exists:
                            continue
                        
                        has_embedding = await conn.fetchval(f"""
                            SELECT EXISTS (
                                SELECT FROM information_schema.columns 
                                WHERE table_name = '{lore_type.lower()}' AND column_name = 'embedding'
                            );
                        """)
                        if not has_embedding:
                            continue
                        
                        id_field = 'id'
                        if lore_type == 'LocationLore':
                            id_field = 'location_id'
                        
                        rows = await conn.fetch(f"""
                            SELECT {id_field} as id, name, description, 
                                   1 - (embedding <=> $1) as relevance
                            FROM {lore_type}
                            WHERE embedding IS NOT NULL
                            ORDER BY embedding <=> $1
                            LIMIT 5
                        """, event_embedding)
                        
                        # Filter only those with decent relevance
                        for row in rows:
                            if row['relevance'] >= 0.6:
                                affected_elements.append({
                                    'lore_type': lore_type,
                                    'lore_id': row['id'],
                                    'name': row['name'],
                                    'description': row['description'],
                                    'relevance': row['relevance']
                                })
                    except Exception as e:
                        logging.error(f"Error checking {lore_type} for affected elements: {e}")
        
        # Sort by descending relevance, limit to top 15
        affected_elements.sort(key=lambda x: x['relevance'], reverse=True)
        if len(affected_elements) > 15:
            affected_elements = affected_elements[:15]
        
        return affected_elements
    
    #===========================================================================
    # GENERATE LORE UPDATES
    #===========================================================================
    @function_tool
    async def _generate_lore_updates(
        self, 
        affected_elements: List[Dict[str, Any]], 
        event_description: str
    ) -> List[Dict[str, Any]]:
        """
        For each affected element, we agent-ify how we generate updates, 
        relying on lore_update_agent with an *LLM-based approach* rather than 
        direct random or code-based transformations.
        """
        updates = []
        world_state = await self._fetch_world_state()
        
        # Agentify the "societal impact" approach
        # Instead of code-based keywords, we ask the agent for a JSON specifying:
        # { stability_impact: int, power_structure_change: str, public_perception: str }
        societal_impact = await self._agent_calculate_societal_impact(
            event_description,
            world_state.get('stability_index', 8),
            world_state.get('power_hierarchy', {})
        )
        
        with trace(
            "LoreUpdateGeneration", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "elements_count": len(affected_elements),
                "event": event_description[:50]
            }
        ):
            for element in affected_elements:
                run_ctx = RunContextWrapper(context={
                    "user_id": self.user_id,
                    "conversation_id": self.conversation_id
                })
                
                related_elements = await self._fetch_related_elements(element.get('lore_id', ''))
                hierarchy_position = await self._get_hierarchy_position(element)
                update_history = await self._fetch_element_update_history(element.get('lore_id', ''))
                
                # We'll build the prompt, but the logic to finalize the "LoreUpdate" object 
                # is in the lore_update_agent. We pass them as context in the prompt.
                prompt = await self._build_lore_update_prompt(
                    element=element,
                    event_description=event_description,
                    societal_impact=societal_impact,
                    related_elements=related_elements,
                    hierarchy_position=hierarchy_position,
                    update_history=update_history
                )
                
                update_agent = self.lore_update_agent.clone(
                    output_type=self.LoreUpdate
                )
                
                result = await Runner.run(update_agent, prompt, context=run_ctx.context)
                update_data = result.final_output
                updates.append(update_data.dict())
        
        return updates
    
    #------------------------------------------------------------------------
    # AGENT-BASED SOCIETAL IMPACT INSTEAD OF KEYWORD SEARCH
    #------------------------------------------------------------------------
    async def _agent_calculate_societal_impact(
        self,
        event_description: str,
        stability_index: int,
        power_hierarchy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use an LLM to produce the 'societal impact' of an event, 
        returning { 'stability_impact': int, 'power_structure_change': str, 'public_perception': str }
        in JSON format.
        """
        agent = Agent(
            name="SocietalImpactAgent",
            instructions=(
                "Given an event description, current stability index (1-10), "
                "and a JSON representation of the power_hierarchy, determine:\n"
                "1) stability_impact: integer 1-10\n"
                "2) power_structure_change: short text describing how power might shift\n"
                "3) public_perception: short text describing how people see or react to the event\n\n"
                "Output valid JSON with these keys."
            ),
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        prompt = f"""
        EVENT DESCRIPTION: {event_description}
        CURRENT STABILITY INDEX: {stability_index}
        POWER HIERARCHY: {json.dumps(power_hierarchy, indent=2)}

        Provide a JSON object with these fields:
        {{
          "stability_impact": <integer 1-10>,
          "power_structure_change": "<string>",
          "public_perception": "<string>"
        }}
        """
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        run_config = RunConfig(
            workflow_name="CalculateSocietalImpact",
            trace_metadata=self.trace_metadata
        )
        
        result = await Runner.run(agent, prompt, context=run_ctx.context, run_config=run_config)
        
        try:
            data = json.loads(result.final_output)
            # Basic validation
            if (
                isinstance(data, dict) and 
                "stability_impact" in data and 
                "power_structure_change" in data and
                "public_perception" in data
            ):
                return data
            return {"stability_impact": 5, "power_structure_change": "minor", "public_perception": "neutral"}
        except json.JSONDecodeError:
            return {"stability_impact": 5, "power_structure_change": "minor", "public_perception": "neutral"}
    
    #===========================================================================
    # APPLY LORE UPDATES
    #===========================================================================
    @function_tool
    async def _apply_lore_updates(self, updates: List[Dict[str, Any]]) -> None:
        """
        Apply the agent-generated updates to the database.
        """
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                for update in updates:
                    lore_type = update['lore_type']
                    lore_id = update['lore_id']
                    new_description = update['new_description']
                    old_description = update['old_description']
                    
                    # Generate a new embedding
                    id_field = 'id'
                    if lore_type == 'LocationLore':
                        id_field = 'location_id'
                    
                    try:
                        await conn.execute(f"""
                            UPDATE {lore_type}
                            SET description = $1
                            WHERE {id_field} = $2
                        """, new_description, lore_id)
                        
                        # Generate embedding
                        item_name = update.get('name', 'Unknown')
                        embedding_text = f"{item_name} {new_description}"
                        await self.generate_and_store_embedding(embedding_text, conn, lore_type, id_field, lore_id)
                        
                        # Insert into LoreChangeHistory
                        await conn.execute("""
                            INSERT INTO LoreChangeHistory 
                            (lore_type, lore_id, previous_description, new_description, change_reason)
                            VALUES ($1, $2, $3, $4, $5)
                        """, lore_type, lore_id, old_description, new_description, update['update_reason'])
                    except Exception as e:
                        logging.error(f"Error updating {lore_type} ID {lore_id}: {e}")
                    
                    self.invalidate_cache_pattern(f"{lore_type}_{lore_id}")
    
    #===========================================================================
    # GENERATE CONSEQUENTIAL LORE
    #===========================================================================
    @function_tool
    async def _generate_consequential_lore(
        self, 
        event_description: str, 
        affected_elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Creates new lore elements triggered by the event.
        Uses a LoreCreationAgent returning a list of LoreElement objects.
        """
        with trace(
            "ConsequentialLoreGeneration", 
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "event": event_description[:50]}
        ):
            run_ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
            # Summarize up to 5 affected elements
            summary_list = [
                f"{e['name']} ({e['lore_type']})" for e in affected_elements[:5]
            ]
            context_info = ", ".join(summary_list)
            
            # Clone lore_creation_agent for structured output
            creation_agent = self.lore_creation_agent.clone(
                output_type=List[self.LoreElement]
            )
            
            prompt = f"""
            A major event occurred, impacting these elements: {context_info}
            
            EVENT:
            {event_description}
            
            Generate 1-3 new lore elements that arise as a consequence of this event. 
            Each element must specify:
              - 'lore_type': string,
              - 'name': string,
              - 'description': string,
              - 'connection': how it ties to the event or existing elements,
              - 'significance': integer 1-10
            Return strictly valid JSON array with LoreElement objects, reinforcing matriarchal themes.
            """
            
            result = await Runner.run(creation_agent, prompt, context=run_ctx.context)
            new_elements = result.final_output
            
            processed = []
            for element in new_elements:
                # Apply matriarchal theming
                element.description = MatriarchalThemingUtils.apply_matriarchal_theme(
                    element.lore_type.lower(), element.description
                )
                
                await self._save_new_lore_element(element, event_description)
                processed.append(element.dict())
            
            return processed
    
    async def _save_new_lore_element(
        self, 
        element: LoreElement, 
        event_description: str
    ) -> None:
        """
        Persist newly generated lore elements to the appropriate table in DB.
        """
        lore_type = element.lore_type
        name = element.name
        description = element.description
        significance = element.significance
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                try:
                    if lore_type == "WorldLore":
                        await conn.execute("""
                            INSERT INTO WorldLore (name, category, description, significance, tags)
                            VALUES ($1, $2, $3, $4, $5)
                        """, name, 'event_consequence', description, significance, ['event_consequence', 'new_lore'])
                    elif lore_type == "Factions":
                        await conn.execute("""
                            INSERT INTO Factions
                            (name, type, description, values, goals, founding_story)
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """, name, 'event_consequence', description, ['power','authority'],
                             ['stability','influence'], f"Founded due to: {event_description}")
                    elif lore_type == "CulturalElements":
                        await conn.execute("""
                            INSERT INTO CulturalElements
                            (name, element_type, description, practiced_by, significance, historical_origin)
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """, name, 'tradition', description, ['society'], significance, f"From {event_description}")
                    elif lore_type == "HistoricalEvents":
                        await conn.execute("""
                            INSERT INTO HistoricalEvents
                            (name, description, date_description, significance, consequences)
                            VALUES ($1, $2, 'Recently', $3, $4)
                        """, name, description, significance, ['Still unfolding'])
                    else:
                        # Fallback
                        await conn.execute("""
                            INSERT INTO WorldLore 
                            (name, category, description, significance, tags)
                            VALUES ($1, $2, $3, $4, $5)
                        """, name, lore_type.lower(), description, significance, [lore_type.lower(), 'event_consequence'])
                except Exception as e:
                    logging.error(f"Error saving new {lore_type} '{name}': {e}")
    
    #===========================================================================
    # AGENT-IFY OUR EVENT TYPE DECISION
    #===========================================================================
    def _choose_event_type_fallback(self, event_types):
        """If the agent fails to provide a valid event type, fallback to random choice."""
        return random.choice(event_types)

    async def _agent_determine_event_type(
        self, 
        event_types: List[str],
        faction_data: List[Dict[str, Any]],
        nation_data: List[Dict[str, Any]],
        location_data: List[Dict[str, Any]]
    ) -> str:
        """
        Agent-based approach to deciding which event type is appropriate
        given the context: factions, nations, locations, etc.
        
        Returns a single string from event_types, or uses fallback if it fails.
        """
        type_agent = Agent(
            name="EventTypeSelectorAgent",
            instructions=(
                "You will receive a set of possible event types in an array, along with some data about the world. "
                "Pick exactly one event type from that array that best fits the context. "
                "Return JSON with a single field: {'event_type': 'the chosen type'}"
            ),
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.8)
        )
        
        prompt_data = {
            "possible_types": event_types,
            "factions": faction_data,
            "nations": nation_data,
            "locations": location_data
        }
        
        prompt = (
            "Determine which single event type from 'possible_types' best suits the given context. "
            "Return JSON with 'event_type' only.\n\n"
            f"CONTEXT:\n{json.dumps(prompt_data, indent=2)}\n"
        )
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        run_config = RunConfig(
            workflow_name="DetermineEventType",
            trace_metadata=self.trace_metadata
        )
        
        result = await Runner.run(type_agent, prompt, context=run_ctx.context, run_config=run_config)
        
        try:
            data = json.loads(result.final_output)
            if isinstance(data, dict) and "event_type" in data and data["event_type"] in event_types:
                return data["event_type"]
            else:
                # fallback
                return self._choose_event_type_fallback(event_types)
        except json.JSONDecodeError:
            # fallback
            return self._choose_event_type_fallback(event_types)
    
    #===========================================================================
    # EMERGENT EVENT GENERATION
    #===========================================================================
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_emergent_event",
        action_description="Generating emergent world event",
        id_from_context=lambda ctx: "lore_dynamics"
    )
    async def generate_emergent_event(self, ctx) -> Dict[str, Any]:
        """
        Generate a random emergent event in the world with governance oversight,
        fully agent-ified for event type decisions and handoffs to specialized event agents.
        """
        with trace(
            "EmergentEventGeneration", 
            group_id=self.trace_group_id,
            metadata=self.trace_metadata
        ):
            run_ctx = self.create_run_context(ctx)
            
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    factions = await conn.fetch("""
                        SELECT id, name, type, description
                        FROM Factions
                        ORDER BY RANDOM()
                        LIMIT 3
                    """)
                    nations = await conn.fetch("""
                        SELECT id, name, government_type
                        FROM Nations
                        LIMIT 3
                    """)
                    locations = await conn.fetch("""
                        SELECT id, location_name, description
                        FROM Locations
                        ORDER BY RANDOM()
                        LIMIT 3
                    """)
                    
            faction_data = [dict(f) for f in factions]
            nation_data = [dict(n) for n in nations]
            location_data = [dict(l) for l in locations]
            
            event_types = [
                "political_shift", "military_conflict", "natural_disaster",
                "cultural_development", "technological_advancement",
                "religious_event", "economic_change", "diplomatic_incident"
            ]
            
            # Agent-based selection
            event_type = await self._agent_determine_event_type(
                event_types, faction_data, nation_data, location_data
            )
            
            # Orchestration agent with specialized handoffs
            event_orchestrator_agent = Agent(
                name="EventOrchestratorAgent",
                instructions=(
                    "You orchestrate emergent events. Decide which specialized agent to hand off to. "
                    "If the event is 'political_shift', 'diplomatic_incident', or 'economic_change', hand off to 'transfer_to_political_agent'. "
                    "If it's 'military_conflict', hand off to 'transfer_to_military_agent'. "
                    "Otherwise, hand off to 'transfer_to_cultural_agent'."
                ),
                model="o3-mini",
                model_settings=ModelSettings(temperature=0.8),
                handoffs=[
                    handoff(
                        self.political_event_agent,
                        tool_name_override="transfer_to_political_agent",
                        tool_description_override="Transfer to political event agent for politics/diplomacy/economics events"
                    ),
                    handoff(
                        self.military_event_agent,
                        tool_name_override="transfer_to_military_agent",
                        tool_description_override="Transfer to military event agent for combat events"
                    ),
                    handoff(
                        self.cultural_event_agent,
                        tool_name_override="transfer_to_cultural_agent",
                        tool_description_override="Transfer to cultural event agent for cultural/natural/technological events"
                    )
                ]
            )
            
            run_config = RunConfig(
                workflow_name="EmergentEventGeneration",
                trace_metadata={
                    "user_id": str(self.user_id),
                    "conversation_id": str(self.conversation_id),
                    "event_type": "emergent_event"
                }
            )
            
            prompt_data = {
                "event_type": event_type,
                "factions": faction_data,
                "nations": nation_data,
                "locations": location_data
            }
            prompt = (
                "We have chosen the event_type: {event_type}. "
                "Use the specialized agent best suited for this event. "
                "Return a JSON or textual description of the event."
            ).format(**prompt_data)
            
            result = await Runner.run(
                event_orchestrator_agent,
                prompt,
                context=run_ctx.context,
                run_config=run_config
            )
            
            try:
                response_text = result.final_output
                event_data = {}
                try:
                    event_data = json.loads(response_text)
                except json.JSONDecodeError:
                    event_data = {
                        "event_type": event_type,
                        "description": response_text
                    }
                
                if "event_name" not in event_data:
                    event_data["event_name"] = f"{event_type.replace('_', ' ').title()} Event"
                
                if "description" in event_data:
                    event_data["description"] = MatriarchalThemingUtils.apply_matriarchal_theme(
                        "event", event_data["description"]
                    )
                
                # Use event_data["description"] to evolve lore
                event_description = f"{event_data.get('event_name','Unnamed Event')}: {event_data.get('description','')}"
                lore_updates = await self.evolve_lore_with_event(event_description)
                event_data["lore_updates"] = lore_updates
                
                return event_data
            except Exception as e:
                logging.error(f"Error generating emergent event: {e}")
                return {"error": str(e), "raw_output": result.final_output}
    
    #===========================================================================
    # LORE MATURATION
    #===========================================================================
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="mature_lore_over_time",
        action_description="Maturing lore over time",
        id_from_context=lambda ctx: "lore_dynamics"
    )
    async def mature_lore_over_time(self, days_passed: int = 7) -> Dict[str, Any]:
        """
        Natural evolution of lore over time, simulating how history and culture develop.
        Instead of code-based random logic, we can keep partial queries but 
        rely on specialized agents for transformations.
        """
        with trace(
            "LoreMaturationProcess", 
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "days_passed": days_passed}
        ):
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
            
            # We'll simply try a few different evolutions, each using an agent
            # or specialized approach. We'll store results in a dict.
            base_probability = min(0.5, days_passed * 0.02)
            # Summaries
            maturation_summary = {
                "myth_evolution": [],
                "culture_development": [],
                "geopolitical_shifts": [],
                "reputation_changes": []
            }
            
            # 1. Evolve urban myths
            if random.random() < (base_probability * 1.5):
                myth_changes = await self._evolve_urban_myths()
                maturation_summary["myth_evolution"] = myth_changes
            
            # 2. Cultural elements
            if random.random() < base_probability:
                culture_changes = await self._develop_cultural_elements()
                maturation_summary["culture_development"] = culture_changes
            
            # 3. Geopolitical shifts
            if random.random() < (base_probability * 0.7):
                geopolitical_changes = await self._shift_geopolitical_landscape()
                maturation_summary["geopolitical_shifts"] = geopolitical_changes
            
            # 4. Notable figure reputations
            if random.random() < (base_probability * 1.2):
                reputation_changes = await self._evolve_notable_figures()
                maturation_summary["reputation_changes"] = reputation_changes
            
            changes_count = sum(len(x) for x in maturation_summary.values())
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
        """Fetch current WorldState from DB or return defaults."""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                world_state = await conn.fetchrow("""
                    SELECT * FROM WorldState 
                    WHERE user_id = $1 AND conversation_id = $2
                    LIMIT 1
                """, self.user_id, self.conversation_id)
                if world_state:
                    result = dict(world_state)
                    if 'power_hierarchy' in result and result['power_hierarchy']:
                        try:
                            result['power_hierarchy'] = json.loads(result['power_hierarchy'])
                        except:
                            result['power_hierarchy'] = {}
                    return result
                else:
                    return {
                        'stability_index': 8,
                        'narrative_tone': 'dramatic',
                        'power_dynamics': 'strict_hierarchy',
                        'power_hierarchy': {}
                    }
    
    async def _fetch_related_elements(self, lore_id: str) -> List[Dict[str, Any]]:
        """Fetch relationships from a LoreRelationships table if it exists."""
        if not lore_id:
            return []
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                try:
                    table_exists = await conn.fetchval("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'lorerelationships'
                        );
                    """)
                    if not table_exists:
                        return []
                    
                    related = await conn.fetch("""
                        SELECT e.lore_id, e.name, e.lore_type, r.relationship_type, r.relationship_strength 
                        FROM LoreElements e
                        JOIN LoreRelationships r ON e.lore_id = r.target_id
                        WHERE r.source_id = $1
                    """, lore_id)
                    return [dict(r) for r in related]
                except Exception as e:
                    logging.error(f"Error fetching related elements for {lore_id}: {e}")
                    return []
    
    async def _get_hierarchy_position(self, element: Dict[str, Any]) -> int:
        """
        Determine an element's position in the power hierarchy. 
        For now, keep some fallback logic, but you could also agent-ify it.
        """
        if element.get('lore_type', '').lower() == 'character':
            if 'hierarchy_position' in element:
                return element['hierarchy_position']
            name = element.get('name','').lower()
            if any(title in name for title in ['queen','empress','matriarch','high','supreme']):
                return 1
            elif any(title in name for title in ['princess','duchess','lady','noble']):
                return 3
            elif any(title in name for title in ['advisor','minister','council']):
                return 5
            else:
                return 8
        elif element.get('lore_type','').lower() == 'faction':
            if 'importance' in element:
                return max(1, 10 - element['importance'])
            else:
                return 4
        elif element.get('lore_type','').lower() == 'location':
            if 'significance' in element:
                return max(1, 10 - element['significance'])
            else:
                return 6
        else:
            return 5
    
    async def _fetch_element_update_history(self, lore_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch recent changes from LoreChangeHistory."""
        if not lore_id:
            return []
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                try:
                    table_exists = await conn.fetchval("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'lorechangehistory'
                        );
                    """)
                    if not table_exists:
                        return []
                    
                    history = await conn.fetch("""
                        SELECT lore_type, lore_id, change_reason, timestamp
                        FROM LoreChangeHistory
                        WHERE lore_id = $1
                        ORDER BY timestamp DESC
                        LIMIT $2
                    """, lore_id, limit)
                    return [dict(h) for h in history]
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
        """Construct the text prompt for the lore update agent, referencing context data."""
        history_lines = []
        for h in update_history:
            history_lines.append(f"- {h['timestamp']}: {h['change_reason']}")
        history_context = "UPDATE HISTORY:\n" + "\n".join(history_lines) if history_lines else ""
        
        relationship_lines = []
        for rel in related_elements:
            relationship_lines.append(
                f"- {rel['name']} ({rel['lore_type']}): {rel['relationship_type']} at strength {rel['relationship_strength']}"
            )
        relationship_context = "RELATIONSHIPS:\n" + "\n".join(relationship_lines) if relationship_lines else ""
        
        if hierarchy_position <= 2:
            directive = (
                "This is a top-tier authority figure or element. Their changes dramatically affect the world. "
                "They seldom shift their core stance but may adjust strategies significantly."
            )
        elif hierarchy_position <= 4:
            directive = (
                "This element wields high authority but is still subordinate to the apex. "
                "They maintain strong control in their domain while deferring to the higher power."
            )
        elif hierarchy_position <= 7:
            directive = (
                "This element holds mid-level authority, implementing will from above while managing subordinates. "
                "They have moderate influence and must balance compliance and ambition."
            )
        else:
            directive = (
                "This element is relatively low in the power hierarchy. "
                "They have limited autonomy and must carefully navigate the structures above them."
            )
        
        prompt = f"""
        The following lore element requires an update based on a recent event:

        LORE ELEMENT:
        Type: {element['lore_type']}
        Name: {element['name']}
        Current Description: {element['description']}
        Hierarchy Position: {hierarchy_position}/10 (lower = higher authority)

        {relationship_context}

        {history_context}

        EVENT:
        {event_description}

        SOCIETAL IMPACT:
        - stability_impact: {societal_impact.get('stability_impact',5)}/10
        - power_structure_change: {societal_impact.get('power_structure_change','unknown')}
        - public_perception: {societal_impact.get('public_perception','quiet')}
        
        DIRECTIVE:
        {directive}

        Generate a LoreUpdate object with:
        - lore_id: "{element['lore_id']}"
        - lore_type: "{element['lore_type']}"
        - name: "{element['name']}"
        - old_description: (the current description)
        - new_description: (updated text reflecting the event)
        - update_reason: (why this update is happening)
        - impact_level: (1-10 representing significance)

        Maintain strong matriarchal themes and internal consistency.
        """
        return prompt
    
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

    async def _agentify_urban_myth_changes(self) -> List[Dict[str, Any]]:
        """
        Example approach: unify the creation of new myth descriptions in a single agent,
        or call an agent for each myth. Returns a list of changes.
        """
        changes = []
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check for a table of urban myths, fallback to WorldLore
                has_urban_myths = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'urbanmyths'
                    );
                """)
                
                if has_urban_myths:
                    rows = await conn.fetch("""
                        SELECT id, name, description, believability, spread_rate
                        FROM UrbanMyths
                        ORDER BY RANDOM()
                        LIMIT 3
                    """)
                    table_name = "UrbanMyths"
                    field_map = {
                        "id": "id",
                        "desc_field": "description",
                        "believability": "believability",
                        "spread_rate": "spread_rate"
                    }
                else:
                    rows = await conn.fetch("""
                        SELECT id, name, description
                        FROM WorldLore
                        WHERE category = 'urban_myth'
                        ORDER BY RANDOM()
                        LIMIT 3
                    """)
                    table_name = "WorldLore"
                    field_map = {
                        "id": "id",
                        "desc_field": "description",
                        "believability": None,
                        "spread_rate": None
                    }
                
                myth_agent = Agent(
                    name="UrbanMythEvolutionAgent",
                    instructions=(
                        "You update the text of an urban myth in a matriarchal fantasy world. "
                        "Possible changes: it grows, evolves, or fades. Return only the new text."
                    ),
                    model="o3-mini",
                    model_settings=ModelSettings(temperature=0.8)
                )
                
                for row in rows:
                    myth_id = row[field_map["id"]]
                    myth_name = row["name"]
                    old_description = row[field_map["desc_field"]]
                    
                    # Weighted random selection or an agent-based approach. 
                    # We'll do a simple random approach for the type, 
                    # but the text transformation is agent-based:
                    change_type = random.choice(["grow","evolve","fade"])
                    
                    prompt = f"""
                    URBAN MYTH: {myth_name}
                    CURRENT DESCRIPTION: {old_description}
                    
                    CHANGE TYPE: {change_type}

                    If 'grow', the myth becomes more elaborate and widely believed.
                    If 'evolve', the myth changes details while retaining core essence.
                    If 'fade', it declines in belief or is partially forgotten.

                    Return the new myth description in a matriarchal, consistent style.
                    """
                    
                    # LLM call
                    result = await Runner.run(myth_agent, prompt, context={})
                    new_description = result.final_output
                    new_description = MatriarchalThemingUtils.apply_matriarchal_theme("myth", new_description)
                    
                    # Attempt to update DB
                    try:
                        # Possibly update believability/spread
                        new_bel = row.get("believability", 5)
                        new_spread = row.get("spread_rate", 5)
                        if change_type == "grow":
                            new_bel = min(10, new_bel + 1)
                            new_spread = min(10, new_spread + 1)
                        elif change_type == "fade":
                            new_bel = max(1, new_bel - 1)
                            new_spread = max(1, new_spread - 1)
                        
                        if table_name == "UrbanMyths":
                            await conn.execute(f"""
                                UPDATE UrbanMyths
                                SET description = $1,
                                    believability = $2,
                                    spread_rate = $3
                                WHERE id = $4
                            """, new_description, new_bel, new_spread, myth_id)
                        else:
                            # WorldLore
                            new_embedding = await generate_embedding(f"{myth_name} {new_description}")
                            await conn.execute("""
                                UPDATE WorldLore
                                SET description = $1,
                                    embedding = $2
                                WHERE id = $3
                            """, new_description, new_embedding, myth_id)
                        
                        changes.append({
                            "myth_id": myth_id,
                            "name": myth_name,
                            "change_type": change_type,
                            "old_description": old_description,
                            "new_description": new_description
                        })
                        
                        self.invalidate_cache_pattern(f"{table_name}_{myth_id}")
                        
                    except Exception as e:
                        logging.error(f"Error updating myth {myth_id} in {table_name}: {e}")
        return changes
    
    async def _develop_cultural_elements(self) -> List[Dict[str, Any]]:
        """
        Evolve or adapt cultural elements over time using an agent-based approach.
        We fetch random cultural elements, choose a 'change_type' for each,
        and let an agent rewrite the description to reflect that change.
        """
        changes = []
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if the CulturalElements table exists
                table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'culturalelements'
                    );
                """)
                if not table_exists:
                    return changes
                
                # Fetch a small random sample of cultural elements
                rows = await conn.fetch("""
                    SELECT id, name, element_type, description, practiced_by, significance
                    FROM CulturalElements
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                if not rows:
                    return changes
                
                # Create a dedicated agent for cultural evolution
                culture_agent = Agent(
                    name="CultureEvolutionAgent",
                    instructions=(
                        "You evolve cultural elements in a fantasy matriarchal setting. "
                        "Given the current description and a chosen 'change_type', rewrite it to "
                        "reflect how the element transforms (formalizes, adapts, spreads, codifies, etc.). "
                        "Return only the new, updated description in a cohesive style."
                    ),
                    model="o3-mini",
                    model_settings=ModelSettings(temperature=0.8)
                )
                
                # Potential change types
                possible_changes = ["formalize", "adapt", "spread", "codify"]
                
                for row in rows:
                    element_id = row["id"]
                    element_name = row["name"]
                    element_type = row["element_type"] or "unspecified"
                    old_description = row["description"]
                    practiced_by = row["practiced_by"] if row["practiced_by"] else []
                    significance = row["significance"] if row["significance"] else 5
                    
                    change_type = random.choice(possible_changes)
                    
                    # Build a prompt describing the existing data and the desired evolution
                    prompt = f"""
                    CULTURAL ELEMENT: {element_name} ({element_type})
                    CURRENT DESCRIPTION: {old_description}
                    PRACTICED BY: {', '.join(practiced_by) if practiced_by else 'unknown groups'}
                    CURRENT SIGNIFICANCE: {significance}/10
    
                    CHANGE TYPE: {change_type}
    
                    If 'formalize', it becomes more structured (codified ceremonies, rules).
                    If 'adapt', it changes to remain relevant in shifting circumstances.
                    If 'spread', it gains new adherents or practitioners.
                    If 'codify', it is written into law, scripture, or cultural canon.
    
                    Rewrite the description to reflect this transformation, while preserving the core identity.
                    Maintain strong matriarchal themes.
                    Return only the new description text.
                    """
                    
                    # Run the agent
                    result = await Runner.run(culture_agent, prompt, context={})
                    new_description = result.final_output
                    
                    # Apply matriarchal theming
                    new_description = MatriarchalThemingUtils.apply_matriarchal_theme(
                        "culture", new_description, emphasis_level=1
                    )
                    
                    # Decide how significance changes (example: formalize => +1, spread => +1, etc.)
                    new_significance = significance
                    if change_type == "formalize":
                        new_significance = min(10, significance + 1)
                    elif change_type == "spread":
                        new_significance = min(10, significance + 1)
                    # 'adapt' or 'codify' might remain the same or shift by 1
                    elif change_type == "codify":
                        new_significance = min(10, significance + 2)
                    
                    # For 'spread', maybe we also add a new group to practiced_by
                    if change_type == "spread":
                        possible_new_groups = ["border tribes", "urban elite", "remote villages"]
                        addition = random.choice(possible_new_groups)
                        if addition not in practiced_by:
                            practiced_by.append(addition)
                    
                    # Generate an embedding
                    embedding_text = f"{element_name} {new_description}"
                    new_embedding = await generate_embedding(embedding_text)
                    
                    # Update the DB
                    try:
                        await conn.execute("""
                            UPDATE CulturalElements
                            SET description = $1,
                                significance = $2,
                                practiced_by = $3,
                                embedding = $4
                            WHERE id = $5
                        """, new_description, new_significance, practiced_by, new_embedding, element_id)
                        
                        changes.append({
                            "element_id": element_id,
                            "name": element_name,
                            "type": element_type,
                            "change_type": change_type,
                            "old_description": old_description,
                            "new_description": new_description,
                            "significance_before": significance,
                            "significance_after": new_significance
                        })
                        
                        self.invalidate_cache_pattern(f"CulturalElements_{element_id}")
                    except Exception as e:
                        logging.error(f"Error updating cultural element {element_id}: {e}")
        
        return changes
    
    # ---------------------------------------------------------------------------
    # (2) _shift_geopolitical_landscape
    # ---------------------------------------------------------------------------
    async def _shift_geopolitical_landscape(self) -> List[Dict[str, Any]]:
        """
        Evolve geopolitics (alliances, territory, governance) in a matriarchal world.
        We fetch random factions or regions, choose a 'shift_type', then rely on
        an LLM agent to rewrite the relevant descriptions.
        """
        changes = []
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if the Factions table exists
                factions_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'factions'
                    );
                """)
                if not factions_exist:
                    return changes
                
                # Fetch random factions
                factions = await conn.fetch("""
                    SELECT id, name, type, description, territory, rivals, allies
                    FROM Factions
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                if not factions:
                    return changes
                
                # Also check if GeographicRegions exist
                regions_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'geographicregions'
                    );
                """)
                region = None
                if regions_exist:
                    region_row = await conn.fetch("""
                        SELECT id, name, description, governing_faction
                        FROM GeographicRegions
                        ORDER BY RANDOM()
                        LIMIT 1
                    """)
                    region = dict(region_row[0]) if region_row else None
                
                # Potential shift types
                shift_types = ["alliance_change", "territory_dispute", "influence_growth"]
                if region:
                    shift_types.append("regional_governance")
                
                # We'll do 1 or 2 shifts randomly
                num_shifts = random.randint(1, 2)
                for _ in range(num_shifts):
                    if not factions:
                        break
                    shift_type = random.choice(shift_types)
                    
                    # Create agent
                    geo_agent = Agent(
                        name="GeopoliticsEvolutionAgent",
                        instructions=(
                            "You rewrite faction or region descriptions to reflect changes in alliances, "
                            "territory, or governance. Keep the matriarchal theme strong."
                        ),
                        model="o3-mini",
                        model_settings=ModelSettings(temperature=0.8)
                    )
                    
                    # We'll pick at least one random faction
                    faction = dict(random.choice(factions))
                    
                    if shift_type == "alliance_change" and len(factions) > 1:
                        # Identify another faction
                        others = [f for f in factions if f["id"] != faction["id"]]
                        if not others:
                            continue
                        other_faction = dict(random.choice(others))
                        
                        # Decide if they become allies or rivals
                        new_relationship = random.choice(["alliance", "rivalry"])
                        
                        prompt = f"""
                        FACTION 1: {faction['name']} (type: {faction['type']})
                        DESCRIPTION: {faction['description']}
    
                        FACTION 2: {other_faction['name']} (type: {other_faction['type']})
    
                        SHIFT TYPE: {new_relationship}
    
                        Rewrite the description of Faction 1 to reflect this new {new_relationship}.
                        Include how they relate to {other_faction['name']} now.
                        """
                        
                        result = await Runner.run(geo_agent, prompt, context={})
                        new_description = result.final_output
                        new_description = MatriarchalThemingUtils.apply_matriarchal_theme("faction", new_description)
                        
                        # Update DB
                        try:
                            embedding_text = f"{faction['name']} {new_description}"
                            new_embedding = await generate_embedding(embedding_text)
                            
                            # Update faction1
                            allies = faction['allies'] if faction['allies'] else []
                            rivals = faction['rivals'] if faction['rivals'] else []
                            if new_relationship == "alliance":
                                if other_faction['name'] in rivals:
                                    rivals.remove(other_faction['name'])
                                if other_faction['name'] not in allies:
                                    allies.append(other_faction['name'])
                            else:  # rivalry
                                if other_faction['name'] in allies:
                                    allies.remove(other_faction['name'])
                                if other_faction['name'] not in rivals:
                                    rivals.append(other_faction['name'])
                            
                            await conn.execute("""
                                UPDATE Factions
                                SET description = $1,
                                    allies = $2,
                                    rivals = $3,
                                    embedding = $4
                                WHERE id = $5
                            """, new_description, allies, rivals, new_embedding, faction["id"])
                            
                            # Also reflect changes in the other faction if needed
                            # (You can do a symmetrical update if you want.)
                            
                            changes.append({
                                "change_type": "alliance_change",
                                "new_relationship": new_relationship,
                                "faction1_id": faction["id"],
                                "faction2_id": other_faction["id"],
                                "old_description": faction["description"],
                                "new_description": new_description
                            })
                            self.invalidate_cache_pattern(f"Factions_{faction['id']}")
                        except Exception as e:
                            logging.error(f"Error updating faction {faction['id']} for alliance_change: {e}")
                    
                    elif shift_type == "territory_dispute":
                        # The faction expands or disputes territory
                        territory = faction["territory"] if faction["territory"] else []
                        prompt = f"""
                        FACTION: {faction['name']} (type: {faction['type']})
                        CURRENT DESCRIPTION: {faction['description']}
                        CURRENT TERRITORY: {territory}
    
                        SHIFT TYPE: territory_dispute
    
                        Rewrite the description to show that this faction is embroiled in a dispute
                        or expansion of territory. Indicate who or what they're challenging.
                        """
                        
                        result = await Runner.run(geo_agent, prompt, context={})
                        new_description = result.final_output
                        new_description = MatriarchalThemingUtils.apply_matriarchal_theme("faction", new_description)
                        
                        new_embedding = await generate_embedding(f"{faction['name']} {new_description}")
                        
                        # Possibly append new territory
                        new_areas = ["borderlands", "resource-rich hills", "disputed farmland"]
                        chosen_area = random.choice(new_areas)
                        if isinstance(territory, list):
                            territory.append(chosen_area)
                        else:
                            territory = [territory, chosen_area]
                        
                        try:
                            await conn.execute("""
                                UPDATE Factions
                                SET description = $1,
                                    territory = $2,
                                    embedding = $3
                                WHERE id = $4
                            """, new_description, territory, new_embedding, faction["id"])
                            
                            changes.append({
                                "change_type": "territory_dispute",
                                "faction_id": faction["id"],
                                "old_description": faction["description"],
                                "new_description": new_description,
                                "added_territory": chosen_area
                            })
                            
                            self.invalidate_cache_pattern(f"Factions_{faction['id']}")
                        except Exception as e:
                            logging.error(f"Error updating territory for faction {faction['id']}: {e}")
                    
                    elif shift_type == "regional_governance" and region:
                        # The region changes governing_faction to or from the current one
                        prompt = f"""
                        REGION: {region['name']}
                        CURRENT DESCRIPTION: {region['description']}
                        CURRENT GOVERNING FACTION: {region['governing_faction'] or 'None'}
    
                        NEW GOVERNING FACTION: {faction['name']}
    
                        SHIFT TYPE: regional_governance
    
                        Rewrite the region's description to reflect this new governance. 
                        Indicate how authority transitions, and how it impacts local life.
                        """
                        
                        result = await Runner.run(geo_agent, prompt, context={})
                        new_description = result.final_output
                        new_description = MatriarchalThemingUtils.apply_matriarchal_theme("region", new_description)
                        
                        try:
                            region_embedding = await generate_embedding(f"{region['name']} {new_description}")
                            await conn.execute("""
                                UPDATE GeographicRegions
                                SET description = $1,
                                    governing_faction = $2,
                                    embedding = $3
                                WHERE id = $4
                            """, new_description, faction["name"], region_embedding, region["id"])
                            
                            changes.append({
                                "change_type": "regional_governance",
                                "region_id": region["id"],
                                "old_governing_faction": region["governing_faction"],
                                "new_governing_faction": faction["name"],
                                "old_description": region["description"],
                                "new_description": new_description
                            })
                            
                            self.invalidate_cache_pattern(f"GeographicRegions_{region['id']}")
                        except Exception as e:
                            logging.error(f"Error updating region {region['id']}: {e}")
                    
                    else:  # "influence_growth"
                        prompt = f"""
                        FACTION: {faction['name']} (type: {faction['type']})
                        CURRENT DESCRIPTION: {faction['description']}
    
                        SHIFT TYPE: influence_growth
    
                        Rewrite the description to show how this faction is expanding its influence or power. 
                        Mention new alliances, resources, or strategies. Maintain matriarchal themes.
                        """
                        
                        result = await Runner.run(geo_agent, prompt, context={})
                        new_description = result.final_output
                        new_description = MatriarchalThemingUtils.apply_matriarchal_theme("faction", new_description)
                        
                        new_embedding = await generate_embedding(f"{faction['name']} {new_description}")
                        
                        try:
                            await conn.execute("""
                                UPDATE Factions
                                SET description = $1,
                                    embedding = $2
                                WHERE id = $3
                            """, new_description, new_embedding, faction["id"])
                            
                            changes.append({
                                "change_type": "influence_growth",
                                "faction_id": faction["id"],
                                "old_description": faction["description"],
                                "new_description": new_description
                            })
                            self.invalidate_cache_pattern(f"Factions_{faction['id']}")
                        except Exception as e:
                            logging.error(f"Error updating faction influence: {e}")
        
        return changes
    
    # ---------------------------------------------------------------------------
    # (3) _evolve_notable_figures
    # ---------------------------------------------------------------------------
    async def _evolve_notable_figures(self) -> List[Dict[str, Any]]:
        """
        Use an agent to evolve the reputations or stories of notable figures
        in a matriarchal setting. We fetch random figures (NotableFigures table or
        fallback from WorldLore), pick a 'change_type', and ask the agent for a 
        new description that matches the transformation.
        """
        changes = []
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                has_notable_figures_table = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'notablefigures'
                    );
                """)
                
                if has_notable_figures_table:
                    rows = await conn.fetch("""
                        SELECT id, name, description, significance, reputation
                        FROM NotableFigures
                        ORDER BY RANDOM()
                        LIMIT 2
                    """)
                    table_name = "NotableFigures"
                    field_map = {
                        "id": "id",
                        "desc_field": "description",
                        "rep_field": "reputation"
                    }
                else:
                    # fallback to world lore with category='notable_figure'
                    rows = await conn.fetch("""
                        SELECT id, name, description, significance
                        FROM WorldLore
                        WHERE category = 'notable_figure'
                        ORDER BY RANDOM()
                        LIMIT 2
                    """)
                    table_name = "WorldLore"
                    field_map = {
                        "id": "id",
                        "desc_field": "description",
                        "rep_field": None  # no direct reputation field
                    }
                
                if not rows:
                    return changes
                
                figure_agent = Agent(
                    name="NotableFigureEvolutionAgent",
                    instructions=(
                        "You evolve the reputations or stories of notable figures in a matriarchal setting. "
                        "Given a change_type, rewrite their description to reflect that new condition."
                    ),
                    model="o3-mini",
                    model_settings=ModelSettings(temperature=0.8)
                )
                
                # Potential changes
                options = ["reputation_rise", "reputation_fall", "scandal", "achievement", "reform"]
                
                for row in rows:
                    figure_id = row[field_map["id"]]
                    figure_name = row["name"]
                    old_description = row[field_map["desc_field"]]
                    
                    significance = row.get("significance", 5)
                    if field_map["rep_field"]:
                        reputation = row.get(field_map["rep_field"], 5)
                    else:
                        reputation = 5  # fallback
                    
                    change_type = random.choice(options)
                    prompt = f"""
                    FIGURE: {figure_name}
                    CURRENT DESCRIPTION: {old_description}
                    CURRENT REPUTATION: {reputation} (1-10)
                    SIGNIFICANCE: {significance}
    
                    CHANGE TYPE: {change_type}
    
                    If 'reputation_rise', they gain prestige.
                    If 'reputation_fall', they lose standing.
                    If 'scandal', a negative event tarnishes them.
                    If 'achievement', they've done something noteworthy.
                    If 'reform', they've changed their approach or stance.
    
                    Rewrite the description to reflect this new reality, 
                    maintaining the matriarchal power context.
                    Return only the updated description text.
                    """
                    
                    result = await Runner.run(figure_agent, prompt, context={})
                    new_description = result.final_output
                    new_description = MatriarchalThemingUtils.apply_matriarchal_theme("character", new_description)
                    
                    # Adjust reputation
                    new_rep = reputation
                    if change_type == "reputation_rise":
                        new_rep = min(10, new_rep + random.randint(1, 2))
                    elif change_type == "reputation_fall":
                        new_rep = max(1, new_rep - random.randint(1, 2))
                    elif change_type == "scandal":
                        new_rep = max(1, new_rep - random.randint(2, 3))
                    elif change_type == "achievement":
                        new_rep = min(10, new_rep + random.randint(2, 3))
                    elif change_type == "reform":
                        direction = random.choice([-1, 1])
                        new_rep = max(1, min(10, new_rep + direction * random.randint(1, 2)))
                    
                    # Generate embedding
                    embedding_text = f"{figure_name} {new_description}"
                    new_embedding = await generate_embedding(embedding_text)
                    
                    # Apply DB updates
                    try:
                        if table_name == "NotableFigures":
                            await conn.execute(f"""
                                UPDATE NotableFigures
                                SET description = $1,
                                    reputation = $2,
                                    embedding = $3
                                WHERE id = $4
                            """, new_description, new_rep, new_embedding, figure_id)
                        else:
                            # WorldLore fallback
                            await conn.execute("""
                                UPDATE WorldLore
                                SET description = $1,
                                    embedding = $2
                                WHERE id = $3
                            """, new_description, new_embedding, figure_id)
                        
                        changes.append({
                            "figure_id": figure_id,
                            "name": figure_name,
                            "old_description": old_description,
                            "new_description": new_description,
                            "change_type": change_type,
                            "old_reputation": reputation,
                            "new_reputation": new_rep
                        })
                        
                        self.invalidate_cache_pattern(f"{table_name}_{figure_id}")
                    except Exception as e:
                        logging.error(f"Error updating notable figure {figure_id}: {e}")
        
        return changes
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_world_over_time",
        action_description="Evolving world over time",
        id_from_context=lambda ctx: "lore_dynamics"
    )
    async def evolve_world_over_time(self, ctx, days_passed: int = 30) -> Dict[str, Any]:
        """
        Evolve the entire world across a specified number of days, 
        calling sub-methods for lore maturity and emergent events. 
        """
        with trace(
            "WorldEvolutionWorkflow", 
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "days_passed": days_passed}
        ):
            run_ctx = RunContextWrapper(context=ctx.context)
            evolution_results = {}
            
            # 1) Mature existing lore
            try:
                maturation = await self.mature_lore_over_time(days_passed)
                evolution_results["lore_maturation"] = maturation
            except Exception as e:
                logging.error(f"Error maturing lore: {e}")
                evolution_results["lore_maturation_error"] = str(e)
            
            # 2) Generate emergent events
            try:
                num_events = max(1, min(5, days_passed // 10))
                num_events = random.randint(1, num_events)
                
                emergent_events = []
                for _ in range(num_events):
                    event = await self.generate_emergent_event(run_ctx)
                    emergent_events.append(event)
                evolution_results["emergent_events"] = emergent_events
            except Exception as e:
                logging.error(f"Error generating emergent events: {e}")
                evolution_results["emergent_events_error"] = str(e)
            
            # Log final
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
    
    #===========================================================================
    # REGISTER WITH GOVERNANCE
    #===========================================================================
    async def register_with_governance(self):
        """Register with Nyx governance."""
        await super().register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_dynamics",
            directive_text=(
                "Evolve, expand, and develop world lore through emergent events "
                "and natural maturation, in a matriarchal setting."
            ),
            scope="world_building",
            priority=DirectivePriority.MEDIUM
        )
        
        logging.info(f"LoreDynamicsSystem registered with Nyx governance for user {self.user_id}, conversation {self.conversation_id}")
