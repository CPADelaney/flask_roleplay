# lore/systems/dynamics.py

import logging
import json
import random
import re
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
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
from lore.managers.base_manager import BaseLoreManager
from lore.managers.geopolitical import GeopoliticalSystemManager
from lore.utils.theming import MatriarchalThemingUtils

# ===========================================================================
# PYDANTIC MODELS FOR STRUCTURED DATA
# ===========================================================================

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

class SocietalImpact(BaseModel):
    """Model for societal impact analysis"""
    stability_impact: int = Field(..., ge=1, le=10)
    power_structure_change: str
    public_perception: str

class EventType(BaseModel):
    """Model for event type selection"""
    event_type: str

class MythEvolution(BaseModel):
    """Model for myth evolution"""
    myth_id: str
    name: str
    change_type: str
    old_description: str
    new_description: str
    change_description: str
    new_believability: Optional[int] = None
    new_spread: Optional[int] = None

class CulturalEvolution(BaseModel):
    """Model for cultural element evolution"""
    element_id: str
    name: str
    element_type: str
    change_type: str
    old_description: str
    new_description: str
    significance_before: int
    significance_after: int

class GeopoliticalShift(BaseModel):
    """Model for geopolitical changes"""
    change_type: str
    faction_id: Optional[str] = None
    faction_name: Optional[str] = None
    old_description: Optional[str] = None
    new_description: Optional[str] = None
    additional_data: Dict[str, Any] = {}

class FigureEvolution(BaseModel):
    """Model for notable figure evolution"""
    figure_id: str
    name: str
    change_type: str
    old_description: str
    new_description: str
    old_reputation: int
    new_reputation: int

class EventCandidate(BaseModel):
    """Model for event candidates in evolutionary selection"""
    event_type: str
    event_name: str
    description: str
    location: Optional[str] = None
    factions_involved: List[str] = []
    impact_level: int = Field(..., ge=1, le=10)
    generation_index: int
    selection_reasoning: Optional[str] = None
    evaluation: Optional[Dict[str, Any]] = None

class MutationDirective(BaseModel):
    """Model for mutation directives"""
    aspect: str
    directive: str

class MutationDirectives(BaseModel):
    """Container for mutation directives"""
    directives: List[MutationDirective]

class EventSelection(BaseModel):
    """Model for event selection results"""
    selected_index: int
    evaluation: Dict[str, Any]
    reasoning: str

class NarrativeEvaluation(BaseModel):
    """Model for narrative evaluation"""
    scores: Dict[str, int]
    overall_score: int
    feedback: List[str]
    suggestions: List[str]

class ImprovementSuggestions(BaseModel):
    """Model for improvement suggestions"""
    criteria: Dict[str, Dict[str, Any]]

class PlanStep(BaseModel):
    """Model for multi-step plan steps"""
    title: str
    type: str
    goal: str
    required_actions: List[str]
    dependencies: List[int]
    potential_branches: List[str]
    expected_impact: Dict[str, Any]
    status: str = "pending"
    outcome: Optional[Dict[str, Any]] = None

class NarrativePlan(BaseModel):
    """Model for narrative plans"""
    overview: str
    steps: List[PlanStep]

class WorldChangePhase(BaseModel):
    """Model for world change phases"""
    phase: str
    content: str

class EvolutionScenarioYear(BaseModel):
    """Model for evolution scenario yearly data"""
    year: int
    content: str

# ===========================================================================
# MAIN LORE DYNAMICS SYSTEM CLASS
# ===========================================================================

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
        self._agents_initialized = False
        self._agents = {}
    
    def _get_agent(self, agent_name: str) -> Agent:
        """Lazy-load agents on demand to avoid initialization overhead"""
        if not self._agents_initialized:
            self._initialize_agents()
            self._agents_initialized = True
        return self._agents.get(agent_name)
    
    def _initialize_agents(self):
        """Initialize specialized agents for different tasks"""
        base_instructions = (
            "You are working with a fantasy world featuring matriarchal power structures. "
            "All content should reflect feminine power and authority as the natural order. "
            "Male elements should be presented in supportive or subservient positions."
        )
        
        model_settings = ModelSettings(temperature=0.9)
        
        agent_configs = {
            "lore_update": {
                "name": "LoreUpdateAgent",
                "instructions": (
                    f"{base_instructions}\n\n"
                    "You update existing lore elements based on narrative events while maintaining thematic consistency. "
                    "Your updates should be meaningful and reflect the impact of events on the world. "
                    "Maintain the matriarchal power dynamics in all updates."
                ),
                "model": "gpt-4.1-nano",
                "settings": model_settings
            },
            "event_generation": {
                "name": "EventGenerationAgent",
                "instructions": (
                    f"{base_instructions}\n\n"
                    "You create emergent world events for fantasy settings with matriarchal power structures. "
                    "Events should be specific and detailed, creating opportunities for character development "
                    "and plot advancement. Focus on how events impact or reinforce matriarchal power dynamics."
                ),
                "model": "gpt-4.1-nano",
                "settings": model_settings
            },
            "lore_creation": {
                "name": "LoreCreationAgent",
                "instructions": (
                    f"{base_instructions}\n\n"
                    "You create new lore elements that emerge from significant events or natural evolution. "
                    "New elements should fit seamlessly with existing lore while expanding the world. "
                    "Ensure all new lore reinforces matriarchal power dynamics."
                ),
                "model": "gpt-4.1-nano",
                "settings": model_settings
            },
            "political_event": {
                "name": "PoliticalEventAgent",
                "instructions": (
                    f"{base_instructions}\n\n"
                    "You create detailed political events for matriarchal fantasy worlds. "
                    "Focus on power dynamics, succession, alliances, and court intrigue. "
                    "Events should highlight feminine leadership and authority."
                ),
                "model": "gpt-4.1-nano",
                "settings": model_settings
            },
            "military_event": {
                "name": "MilitaryEventAgent",
                "instructions": (
                    f"{base_instructions}\n\n"
                    "You create detailed military conflicts for matriarchal fantasy worlds. "
                    "Focus on strategy, leadership, and the consequences of warfare. "
                    "Events should highlight feminine military command structures."
                ),
                "model": "gpt-4.1-nano",
                "settings": model_settings
            },
            "cultural_event": {
                "name": "CulturalEventAgent",
                "instructions": (
                    f"{base_instructions}\n\n"
                    "You create detailed cultural developments for matriarchal fantasy worlds. "
                    "Focus on traditions, arts, festivals, and social changes. "
                    "Events should highlight feminine cultural influence and values."
                ),
                "model": "gpt-4.1-nano",
                "settings": model_settings
            }
        }
        
        # Create agents
        for key, config in agent_configs.items():
            self._agents[key] = Agent(
                name=config["name"],
                instructions=config["instructions"],
                model=config["model"],
                model_settings=config["settings"]
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
                CREATE TABLE IF NOT EXISTS LoreChangeHistory (
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
                CREATE TABLE IF NOT EXISTS WorldState (
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
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8),
            output_type=EventValidation
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
    async def evolve_lore_with_event(self, ctx, event_description: str) -> Dict[str, Any]:
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
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.9),
                input_guardrails=[input_guardrail],
                tools=[
                    self._identify_affected_lore, 
                    self._generate_lore_updates,
                    self._apply_lore_updates,
                    self._generate_consequential_lore
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
    @function_tool(strict_mode=False)
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
    @function_tool(strict_mode=False)
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
                
                update_agent = self._get_agent("lore_update").clone(
                    output_type=LoreUpdate
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
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9),
            output_type=SocietalImpact
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
        return result.final_output.dict()
    
    #===========================================================================
    # APPLY LORE UPDATES
    #===========================================================================
    @function_tool(strict_mode=False)
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
    @function_tool(strict_mode=False)
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
            creation_agent = self._get_agent("lore_creation").clone(
                output_type=List[LoreElement]
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
    
    async def _save_new_lore_element(self, element: LoreElement, event_description: str) -> None:
        """
        Persist newly generated lore elements through the canon system.
        """
        lore_type = element.lore_type
        name = element.name
        description = element.description
        significance = element.significance
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                try:
                    # Create context for canon operations
                    from agents import RunContextWrapper
                    ctx = RunContextWrapper(context={
                        "user_id": self.user_id,
                        "conversation_id": self.conversation_id
                    })
                    
                    if lore_type == "WorldLore":
                        # WorldLore uses direct insert as it's a catch-all category
                        await conn.execute("""
                            INSERT INTO WorldLore (user_id, conversation_id, name, category, description, significance, tags, embedding)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """, self.user_id, self.conversation_id, name, 'event_consequence', description, 
                            significance, ['event_consequence', 'new_lore'], 
                            await generate_embedding(f"{name} {description}"))
                        
                    elif lore_type == "Factions":
                        await canon.find_or_create_faction(
                            ctx, conn, name,
                            type='event_consequence',
                            description=description,
                            values=['power', 'authority'],
                            goals=['stability', 'influence'],
                            founding_story=f"Founded due to: {event_description}"
                        )
                        
                    elif lore_type == "CulturalElements":
                        await canon.find_or_create_cultural_element(
                            ctx, conn, name,
                            element_type='tradition',
                            description=description,
                            practiced_by=['society'],
                            significance=significance,
                            historical_origin=f"From {event_description}"
                        )
                        
                    elif lore_type == "HistoricalEvents":
                        await canon.find_or_create_historical_event(
                            ctx, conn, name,
                            description=description,
                            date_description='Recently',
                            significance=significance,
                            consequences=['Still unfolding'],
                            event_type='emergent'
                        )
                        
                    elif lore_type == "GeographicRegions":
                        await canon.find_or_create_geographic_region(
                            ctx, conn, name,
                            region_type='discovered',
                            description=description,
                            climate='varied',
                            strategic_value=significance
                        )
                        
                    else:
                        # Fallback to WorldLore for unknown types
                        await conn.execute("""
                            INSERT INTO WorldLore (user_id, conversation_id, name, category, description, significance, tags, embedding)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """, self.user_id, self.conversation_id, name, lore_type.lower(), description, 
                            significance, [lore_type.lower(), 'event_consequence'],
                            await generate_embedding(f"{name} {description}"))
                            
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
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8),
            output_type=EventType
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
        event_type_data = result.final_output
        
        if event_type_data.event_type in event_types:
            return event_type_data.event_type
        else:
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
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.8),
                handoffs=[
                    handoff(
                        self._get_agent("political_event"),
                        tool_name_override="transfer_to_political_agent",
                        tool_description_override="Transfer to political event agent for politics/diplomacy/economics events"
                    ),
                    handoff(
                        self._get_agent("military_event"),
                        tool_name_override="transfer_to_military_agent",
                        tool_description_override="Transfer to military event agent for combat events"
                    ),
                    handoff(
                        self._get_agent("cultural_event"),
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
                lore_updates = await self.evolve_lore_with_event(ctx, event_description)
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

    async def check_permission(self, agent_type, agent_id, action_type, action_details):
        """Check permission with governance system."""
        if not self.governor:
            # If no governor, assume permission granted
            return {"approved": True}
        
        return await self.governor.check_action_permission(
            agent_type=agent_type,
            agent_id=agent_id,
            action_type=action_type,
            action_details=action_details
        )
        
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
        """Evolve urban myths over time using canon system"""
        changes = []
        
        # Create context for canon operations
        from agents import RunContextWrapper
        ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
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
                        WHERE user_id = $1 AND conversation_id = $2
                        ORDER BY RANDOM()
                        LIMIT 3
                    """, self.user_id, self.conversation_id)
                else:
                    # Fall back to world lore with urban_myth category
                    rows = await conn.fetch("""
                        SELECT id, name, description
                        FROM WorldLore
                        WHERE user_id = $1 AND conversation_id = $2 
                        AND category = 'urban_myth'
                        ORDER BY RANDOM()
                        LIMIT 3
                    """, self.user_id, self.conversation_id)
                
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
                        model="gpt-4.1-nano"
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
                    
                    # Update through canon system
                    if has_urban_myths_table:
                        await canon.update_entity_canonically(
                            ctx, conn, "UrbanMyths", myth_id,
                            {
                                'description': new_description,
                                'believability': new_believability,
                                'spread_rate': new_spread
                            },
                            f"Urban myth '{myth_name}' {change_type}ing naturally over time"
                        )
                    else:
                        # For WorldLore, update directly as it's less structured
                        await conn.execute("""
                            UPDATE WorldLore
                            SET description = $1, embedding = $2
                            WHERE id = $3 AND user_id = $4 AND conversation_id = $5
                        """, new_description, await generate_embedding(f"{myth_name} {new_description}"), 
                            myth_id, self.user_id, self.conversation_id)
                    
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
        
        return changes
    
    async def _develop_cultural_elements(self) -> List[Dict[str, Any]]:
        """
        Evolve or adapt cultural elements over time using canon system.
        """
        changes = []
        
        # Create context for canon operations
        from agents import RunContextWrapper
        ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
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
                    WHERE user_id = $1 AND conversation_id = $2
                    ORDER BY RANDOM()
                    LIMIT 3
                """, self.user_id, self.conversation_id)
                
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
                    model="gpt-4.1-nano",
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
                    
                    # Decide how significance changes
                    new_significance = significance
                    if change_type == "formalize":
                        new_significance = min(10, significance + 1)
                    elif change_type == "spread":
                        new_significance = min(10, significance + 1)
                    elif change_type == "codify":
                        new_significance = min(10, significance + 2)
                    
                    # For 'spread', add new practitioners
                    new_practiced_by = practiced_by.copy()
                    if change_type == "spread":
                        possible_new_groups = ["border tribes", "urban elite", "remote villages", "merchant guilds", "warrior societies"]
                        addition = random.choice(possible_new_groups)
                        if addition not in new_practiced_by:
                            new_practiced_by.append(addition)
                    
                    # Update through canon system
                    await canon.update_entity_canonically(
                        ctx, conn, "CulturalElements", element_id,
                        {
                            'description': new_description,
                            'significance': new_significance,
                            'practiced_by': new_practiced_by,
                            'embedding': await generate_embedding(f"{element_name} {new_description}")
                        },
                        f"Cultural element '{element_name}' {change_type}d naturally over time"
                    )
                    
                    changes.append({
                        "element_id": element_id,
                        "name": element_name,
                        "element_type": element_type,
                        "change_type": change_type,
                        "old_description": old_description,
                        "new_description": new_description,
                        "significance_before": significance,
                        "significance_after": new_significance
                    })
                    
                    self.invalidate_cache_pattern(f"CulturalElements_{element_id}")
        
        return changes
    
    async def _shift_geopolitical_landscape(self) -> List[Dict[str, Any]]:
        """
        Evolve geopolitics using canon system for all updates.
        """
        changes = []
        
        # Create context for canon operations
        from agents import RunContextWrapper
        ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
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
                    WHERE user_id = $1 AND conversation_id = $2
                    ORDER BY RANDOM()
                    LIMIT 3
                """, self.user_id, self.conversation_id)
                
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
                        WHERE user_id = $1 AND conversation_id = $2
                        ORDER BY RANDOM()
                        LIMIT 1
                    """, self.user_id, self.conversation_id)
                    region = dict(region_row[0]) if region_row else None
                
                # Potential shift types
                shift_types = ["alliance_change", "territory_dispute", "influence_growth"]
                if region:
                    shift_types.append("regional_governance")
                
                # Create agent
                geo_agent = Agent(
                    name="GeopoliticsEvolutionAgent",
                    instructions=(
                        "You rewrite faction or region descriptions to reflect changes in alliances, "
                        "territory, or governance. Keep the matriarchal theme strong."
                    ),
                    model="gpt-4.1-nano",
                    model_settings=ModelSettings(temperature=0.8)
                )
                
                # Do 1 or 2 shifts randomly
                num_shifts = random.randint(1, 2)
                for _ in range(num_shifts):
                    if not factions:
                        break
                        
                    shift_type = random.choice(shift_types)
                    faction = dict(random.choice(factions))
                    
                    if shift_type == "alliance_change" and len(factions) > 1:
                        # Alliance change between factions
                        others = [f for f in factions if f["id"] != faction["id"]]
                        if not others:
                            continue
                            
                        other_faction = dict(random.choice(others))
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
                        
                        # Update relationships
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
                        
                        # Update through canon system
                        await canon.update_entity_canonically(
                            ctx, conn, "Factions", faction["id"],
                            {
                                'description': new_description,
                                'allies': allies,
                                'rivals': rivals,
                                'embedding': await generate_embedding(f"{faction['name']} {new_description}")
                            },
                            f"Faction '{faction['name']}' formed {new_relationship} with {other_faction['name']}"
                        )
                        
                        changes.append({
                            "change_type": "alliance_change",
                            "faction_id": faction["id"],
                            "faction_name": faction['name'],
                            "new_relationship": new_relationship,
                            "other_faction": other_faction['name'],
                            "old_description": faction["description"],
                            "new_description": new_description
                        })
                        
                    elif shift_type == "territory_dispute":
                        # Territory expansion/dispute
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
                        
                        # Add new territory
                        new_areas = ["borderlands", "resource-rich hills", "disputed farmland", "coastal regions", "mountain passes"]
                        chosen_area = random.choice(new_areas)
                        
                        if isinstance(territory, list):
                            new_territory = territory + [chosen_area]
                        else:
                            new_territory = [territory, chosen_area]
                        
                        # Update through canon system
                        await canon.update_entity_canonically(
                            ctx, conn, "Factions", faction["id"],
                            {
                                'description': new_description,
                                'territory': new_territory,
                                'embedding': await generate_embedding(f"{faction['name']} {new_description}")
                            },
                            f"Faction '{faction['name']}' engaged in territorial dispute over {chosen_area}"
                        )
                        
                        changes.append({
                            "change_type": "territory_dispute",
                            "faction_id": faction["id"],
                            "faction_name": faction['name'],
                            "old_description": faction["description"],
                            "new_description": new_description,
                            "added_territory": chosen_area
                        })
                        
                    elif shift_type == "regional_governance" and region:
                        # Change regional governance
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
                        
                        # Update through canon system
                        await canon.update_entity_canonically(
                            ctx, conn, "GeographicRegions", region["id"],
                            {
                                'description': new_description,
                                'governing_faction': faction["name"],
                                'embedding': await generate_embedding(f"{region['name']} {new_description}")
                            },
                            f"Region '{region['name']}' governance shifted to faction '{faction['name']}'"
                        )
                        
                        changes.append({
                            "change_type": "regional_governance",
                            "region_id": region["id"],
                            "region_name": region['name'],
                            "old_governing_faction": region["governing_faction"],
                            "new_governing_faction": faction["name"],
                            "old_description": region["description"],
                            "new_description": new_description
                        })
                        
                    else:  # influence_growth
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
                        
                        # Update through canon system
                        await canon.update_entity_canonically(
                            ctx, conn, "Factions", faction["id"],
                            {
                                'description': new_description,
                                'embedding': await generate_embedding(f"{faction['name']} {new_description}")
                            },
                            f"Faction '{faction['name']}' expanded its influence and power"
                        )
                        
                        changes.append({
                            "change_type": "influence_growth",
                            "faction_id": faction["id"],
                            "faction_name": faction['name'],
                            "old_description": faction["description"],
                            "new_description": new_description
                        })
                    
                    self.invalidate_cache_pattern(f"Factions_{faction['id']}")
        
        return changes
    
    async def _evolve_notable_figures(self) -> List[Dict[str, Any]]:
        """
        Use canon system to evolve notable figures' reputations and stories.
        """
        changes = []
        
        # Create context for canon operations
        from agents import RunContextWrapper
        ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
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
                        WHERE user_id = $1 AND conversation_id = $2
                        ORDER BY RANDOM()
                        LIMIT 2
                    """, self.user_id, self.conversation_id)
                    table_name = "NotableFigures"
                else:
                    # fallback to world lore with category='notable_figure'
                    rows = await conn.fetch("""
                        SELECT id, name, description, significance
                        FROM WorldLore
                        WHERE user_id = $1 AND conversation_id = $2
                        AND category = 'notable_figure'
                        ORDER BY RANDOM()
                        LIMIT 2
                    """, self.user_id, self.conversation_id)
                    table_name = "WorldLore"
                
                if not rows:
                    return changes
                
                figure_agent = Agent(
                    name="NotableFigureEvolutionAgent",
                    instructions=(
                        "You evolve the reputations or stories of notable figures in a matriarchal setting. "
                        "Given a change_type, rewrite their description to reflect that new condition."
                    ),
                    model="gpt-4.1-nano",
                    model_settings=ModelSettings(temperature=0.8)
                )
                
                # Potential changes
                options = ["reputation_rise", "reputation_fall", "scandal", "achievement", "reform"]
                
                for row in rows:
                    figure_id = row['id']
                    figure_name = row['name']
                    old_description = row['description']
                    
                    significance = row.get('significance', 5)
                    reputation = row.get('reputation', 50) if table_name == "NotableFigures" else 50
                    
                    change_type = random.choice(options)
                    prompt = f"""
                    FIGURE: {figure_name}
                    CURRENT DESCRIPTION: {old_description}
                    CURRENT REPUTATION: {reputation} (1-100)
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
                        new_rep = min(100, new_rep + random.randint(5, 15))
                    elif change_type == "reputation_fall":
                        new_rep = max(0, new_rep - random.randint(5, 15))
                    elif change_type == "scandal":
                        new_rep = max(0, new_rep - random.randint(10, 25))
                    elif change_type == "achievement":
                        new_rep = min(100, new_rep + random.randint(10, 20))
                    elif change_type == "reform":
                        direction = random.choice([-1, 1])
                        new_rep = max(0, min(100, new_rep + direction * random.randint(5, 15)))
                    
                    # Update through canon system
                    if table_name == "NotableFigures":
                        await canon.update_entity_canonically(
                            ctx, conn, "NotableFigures", figure_id,
                            {
                                'description': new_description,
                                'reputation': new_rep,
                                'embedding': await generate_embedding(f"{figure_name} {new_description}")
                            },
                            f"Notable figure '{figure_name}' experienced {change_type}"
                        )
                    else:
                        # WorldLore update
                        await conn.execute("""
                            UPDATE WorldLore
                            SET description = $1, embedding = $2
                            WHERE id = $3 AND user_id = $4 AND conversation_id = $5
                        """, new_description, await generate_embedding(f"{figure_name} {new_description}"),
                            figure_id, self.user_id, self.conversation_id)
                    
                    changes.append({
                        "figure_id": figure_id,
                        "name": figure_name,
                        "change_type": change_type,
                        "old_description": old_description,
                        "new_description": new_description,
                        "old_reputation": reputation,
                        "new_reputation": new_rep
                    })
                    
                    self.invalidate_cache_pattern(f"{table_name}_{figure_id}")
        
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

class MultiStepPlanner:
    """
    Handles multi-step planning for complex lore evolution with dependent agent execution.
    """
    
    def __init__(self, dynamics_system):
        self.dynamics_system = dynamics_system
        self.planning_agent = Agent(
            name="NarrativePlanningAgent",
            instructions="""
            You create multi-step narrative plans for fantasy world evolution.
            Break complex narratives into interdependent steps, identifying prerequisites,
            expected outcomes, and potential narrative branches.
            Maintain matriarchal themes throughout the planning process.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8),
            output_type=NarrativePlan
        )
    
    async def create_evolution_plan(self, initial_prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a multi-step plan for evolving the world based on an initial prompt."""
        run_ctx = RunContextWrapper(context=context)
        
        # Get world state
        world_state = await self.dynamics_system._fetch_world_state()
        
        # Create planning prompt
        prompt = f"""
        Create a multi-step plan for evolving this world based on the following prompt:
        
        PROMPT: {initial_prompt}
        
        CURRENT WORLD STATE:
        {json.dumps(world_state, indent=2)}
        
        Break this down into 3-5 sequential steps, each with:
        1. A goal/outcome
        2. Required actions
        3. Dependencies on previous steps
        4. Potential narrative branches
        5. Expected impact on world state
        
        Return a JSON plan with these steps, ensuring matriarchal themes remain central.
        """
        
        result = await Runner.run(self.planning_agent, prompt, context=run_ctx.context)
        plan = result.final_output
        
        # Store the plan
        plan_id = await self._store_plan(plan, initial_prompt)
        plan_dict = plan.dict()
        plan_dict["id"] = plan_id
        return plan_dict
    
    async def execute_plan_step(self, plan_id: str, step_index: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific step in a multi-step plan."""
        # Get the plan
        plan = await self._get_plan(plan_id)
        if not plan:
            return {"error": "Plan not found"}
        
        # Check step dependencies
        if not self._check_dependencies(plan, step_index):
            return {"error": "Dependencies for this step have not been completed"}
        
        # Get the step
        steps = plan.get("steps", [])
        if step_index >= len(steps):
            return {"error": f"Step index {step_index} out of range"}
        
        step = steps[step_index]
        
        # Determine the appropriate agent for this step
        agent_type = self._determine_agent_type(step)
        executor_agent = self._get_executor_agent(agent_type)
        
        # Create execution prompt
        prompt = f"""
        Execute this narrative step:
        
        STEP:
        {json.dumps(step, indent=2)}
        
        PLAN CONTEXT:
        {json.dumps({"prompt": plan.get("prompt"), "overview": plan.get("overview")}, indent=2)}
        
        PREVIOUS OUTCOMES:
        {json.dumps(self._get_previous_outcomes(plan, step_index), indent=2)}
        
        Produce the specific lore changes, events, or developments that accomplish this step.
        Return JSON with detailed outcomes that can be applied to the world state.
        """
        
        run_ctx = RunContextWrapper(context=context)
        result = await Runner.run(executor_agent, prompt, context=run_ctx.context)
        
        try:
            outcome = json.loads(result.final_output)
            
            # Apply the outcome to the world
            applied_outcome = await self._apply_step_outcome(step, outcome)
            
            # Update the plan with the outcome
            await self._update_plan_step(plan_id, step_index, applied_outcome)
            
            return applied_outcome
        except json.JSONDecodeError:
            return {"error": "Failed to parse step outcome", "raw_output": result.final_output}
    
    async def _store_plan(self, plan: NarrativePlan, prompt: str) -> str:
        """Store a narrative plan."""
        # Implementation would depend on database structure
        plan_id = f"plan_{uuid.uuid4()}"
        plan_dict = plan.dict()
        plan_dict["id"] = plan_id
        plan_dict["prompt"] = prompt
        plan_dict["created_at"] = datetime.now().isoformat()
        plan_dict["status"] = "created"
        
        # Mark all steps as pending
        for step in plan_dict.get("steps", []):
            step["status"] = "pending"
            step["outcome"] = None
        
        # TODO: Store in database
        
        return plan_id
    
    async def _get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get a stored narrative plan."""
        # Implementation would depend on database structure
        # This is a placeholder
        return None
    
    def _check_dependencies(self, plan: Dict[str, Any], step_index: int) -> bool:
        """Check if dependencies for a step have been completed."""
        steps = plan.get("steps", [])
        current_step = steps[step_index] if step_index < len(steps) else None
        
        if not current_step:
            return False
        
        dependencies = current_step.get("dependencies", [])
        
        for dep_index in dependencies:
            if dep_index >= len(steps):
                return False
            
            dep_step = steps[dep_index]
            if dep_step.get("status") != "completed":
                return False
        
        return True
    
    def _determine_agent_type(self, step: Dict[str, Any]) -> str:
        """Determine the appropriate agent type for a step."""
        step_type = step.get("type", "").lower()
        
        if "political" in step_type or "diplomatic" in step_type:
            return "political"
        elif "military" in step_type or "conflict" in step_type:
            return "military"
        elif "cultural" in step_type or "social" in step_type:
            return "cultural"
        else:
            return "general"
    
    def _get_executor_agent(self, agent_type: str) -> Agent:
        """Get the appropriate executor agent for a step type."""
        if agent_type == "political":
            return self.dynamics_system._get_agent("political_event")
        elif agent_type == "military":
            return self.dynamics_system._get_agent("military_event")
        elif agent_type == "cultural":
            return self.dynamics_system._get_agent("cultural_event")
        else:
            return self.dynamics_system._get_agent("event_generation")
    
    def _get_previous_outcomes(self, plan: Dict[str, Any], current_index: int) -> List[Dict[str, Any]]:
        """Get outcomes of previous steps that this step depends on."""
        steps = plan.get("steps", [])
        current_step = steps[current_index] if current_index < len(steps) else None
        
        if not current_step:
            return []
        
        dependencies = current_step.get("dependencies", [])
        outcomes = []
        
        for dep_index in dependencies:
            if dep_index < len(steps):
                dep_step = steps[dep_index]
                if dep_step.get("status") == "completed" and dep_step.get("outcome"):
                    outcomes.append({
                        "step_index": dep_index,
                        "step_title": dep_step.get("title"),
                        "outcome": dep_step.get("outcome")
                    })
        
        return outcomes
    
    async def _apply_step_outcome(self, step: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the outcome of a step to the world state."""
        # Implementation would depend on what changes need to be applied
        # This could involve calling various methods on the dynamics_system
        
        if "event" in outcome:
            # Apply an event
            event_result = await self.dynamics_system.evolve_lore_with_event(outcome["event"])
            outcome["applied_changes"] = event_result
        
        if "cultural_changes" in outcome:
            # Apply cultural changes
            cultural_results = await self._apply_cultural_changes(outcome["cultural_changes"])
            outcome["applied_cultural_changes"] = cultural_results
        
        if "political_changes" in outcome:
            # Apply political changes
            political_results = await self._apply_political_changes(outcome["political_changes"])
            outcome["applied_political_changes"] = political_results
        
        return outcome
    
    async def _update_plan_step(self, plan_id: str, step_index: int, outcome: Dict[str, Any]) -> None:
        """Update a plan step with its outcome."""
        # Implementation would depend on database structure
        pass
    
    async def _apply_cultural_changes(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply cultural changes to the world."""
        # Placeholder implementation
        return {"status": "applied", "count": len(changes)}
    
    async def _apply_political_changes(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply political changes to the world."""
        # Placeholder implementation
        return {"status": "applied", "count": len(changes)}

class NarrativeEvaluator:
    """
    Evaluates the quality of generated narrative elements and provides feedback
    to improve future generations.
    """
    
    def __init__(self, dynamics_system):
        self.dynamics_system = dynamics_system
        self.evaluation_agent = Agent(
            name="NarrativeQualityAgent",
            instructions="""
            You evaluate fantasy world narrative elements for quality, coherence, and interest.
            Provide specific, constructive feedback on how to improve future generations.
            Consider themes, character motivations, plot development, and matriarchal elements.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.7),
            output_type=NarrativeEvaluation
        )
        self.feedback_history = []
    
    async def evaluate_narrative(self, narrative_element: Dict[str, Any], element_type: str) -> Dict[str, Any]:
        """Evaluate a narrative element's quality and provide feedback."""
        with trace(
            "NarrativeEvaluation", 
            group_id=self.dynamics_system.trace_group_id,
            metadata={"element_type": element_type}
        ):
            # Get criteria based on element type
            criteria = self._get_evaluation_criteria(element_type)
            
            prompt = f"""
            Evaluate this {element_type} narrative element for quality:
            
            NARRATIVE ELEMENT:
            {json.dumps(narrative_element, indent=2)}
            
            EVALUATION CRITERIA:
            {json.dumps(criteria, indent=2)}
            
            Provide detailed evaluation and feedback on each criterion.
            Score each criterion 1-10 and provide an overall score.
            Include specific suggestions for improvement.
            """
            
            result = await Runner.run(self.evaluation_agent, prompt, context={})
            evaluation = result.final_output
            
            # Store the feedback for learning
            self.feedback_history.append({
                "element_type": element_type,
                "evaluation": evaluation.dict(),
                "timestamp": datetime.now().isoformat()
            })
            
            # If we have collected enough feedback, update generation parameters
            if len(self.feedback_history) >= 5:
                await self._update_generation_parameters()
            
            return evaluation.dict()
    
    def _get_evaluation_criteria(self, element_type: str) -> Dict[str, Any]:
        """Get evaluation criteria based on element type."""
        base_criteria = {
            "coherence": "How logically consistent is this element?",
            "interest": "How engaging and interesting is this element?",
            "matriarchal_themes": "How well does it incorporate matriarchal power structures?",
            "originality": "How original and creative is this element?"
        }
        
        type_specific = {
            "event": {
                "plausibility": "Does this event seem plausible given the world context?",
                "impact": "Does this event have meaningful impact on the world?",
                "character_motivations": "Are character motivations clear and believable?"
            },
            "cultural_development": {
                "anthropological_realism": "Does this cultural element follow realistic development patterns?",
                "cultural_specificity": "Is this element specific and unique to the culture?",
                "internal_consistency": "Is this element consistent with the culture's other aspects?"
            },
            "political_shift": {
                "power_dynamics": "Does this shift reflect realistic power dynamics?",
                "faction_consistency": "Is it consistent with established faction motivations?",
                "consequences": "Are the consequences meaningful and far-reaching?"
            }
        }
        
        criteria = base_criteria.copy()
        criteria.update(type_specific.get(element_type, {}))
        
        return criteria
    
    async def _update_generation_parameters(self) -> None:
        """Update generation parameters based on feedback history."""
        # Average scores by criteria
        criteria_scores = {}
        criteria_counts = {}
        
        for feedback in self.feedback_history:
            evaluation = feedback.get("evaluation", {})
            scores = evaluation.get("scores", {})
            
            for criterion, score in scores.items():
                if criterion not in criteria_scores:
                    criteria_scores[criterion] = 0
                    criteria_counts[criterion] = 0
                
                criteria_scores[criterion] += score
                criteria_counts[criterion] += 1
        
        avg_scores = {c: criteria_scores[c] / criteria_counts[c] for c in criteria_scores}
        
        # Find lowest scoring criteria
        sorted_scores = sorted(avg_scores.items(), key=lambda x: x[1])
        lowest_criteria = [c for c, s in sorted_scores[:2]]
        
        # Get improvement suggestions for lowest criteria
        improvement_suggestions = await self._generate_improvement_suggestions(lowest_criteria)
        
        # Apply suggestions to generation parameters
        await self._apply_improvement_suggestions(improvement_suggestions)
        
        # Clear history after update
        self.feedback_history = []
    
    async def _generate_improvement_suggestions(self, criteria: List[str]) -> Dict[str, Any]:
        """Generate suggestions for improving specific criteria."""
        history_samples = random.sample(self.feedback_history, min(3, len(self.feedback_history)))
        
        suggestion_agent = Agent(
            name="ImprovementSuggestionAgent",
            instructions="Generate specific improvement suggestions for narrative generation based on evaluation feedback.",
            model="gpt-4.1-nano",
            output_type=ImprovementSuggestions
        )
        
        prompt = f"""
        Generate improvement suggestions for these narrative criteria:
        {criteria}
        
        Based on these evaluation samples:
        {json.dumps(history_samples, indent=2)}
        
        For each criterion, provide:
        1. Specific changes to generation parameters
        2. Thematic elements to emphasize
        3. Patterns to avoid
        
        Return JSON with actionable suggestions.
        """
        
        result = await Runner.run(suggestion_agent, prompt, context={})
        return result.final_output.dict()
    
    async def _apply_improvement_suggestions(self, suggestions: Dict[str, Any]) -> None:
        """Apply improvement suggestions to generation parameters."""
        # Implementation would depend on how generation parameters are stored
        # This is a placeholder
        pass

class NarrativeEvolutionSystem:
    """
    Implements evolutionary selection for compelling narrative elements.
    """
    
    def __init__(self, dynamics_system):
        self.dynamics_system = dynamics_system
        self.selection_agent = Agent(
            name="NarrativeSelectionAgent",
            instructions="""
            You evaluate and select the most compelling narrative elements from a pool of candidates.
            Select elements that enhance the overall narrative, maintain consistency, and advance matriarchal themes.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.7),
            output_type=EventSelection
        )
        self.mutation_agent = Agent(
            name="NarrativeMutationAgent",
            instructions="""
            You modify narrative elements to improve their quality, interest, and consistency.
            Enhance matriarchal themes and ensure coherent integration with the world.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9),
            output_type=MutationDirectives
        )
    
    async def generate_candidate_pool(self, element_type: str, context: Dict[str, Any], pool_size: int = 3) -> List[Dict[str, Any]]:
        """Generate a pool of candidate narrative elements."""
        candidates = []
        
        # Select appropriate agent based on element type
        if element_type == "political_event":
            generator_agent = self.dynamics_system._get_agent("political_event")
        elif element_type == "military_event":
            generator_agent = self.dynamics_system._get_agent("military_event")
        elif element_type == "cultural_event":
            generator_agent = self.dynamics_system._get_agent("cultural_event")
        else:
            generator_agent = self.dynamics_system._get_agent("event_generation")
        
        # Configure agent for structured output
        generator_agent = generator_agent.clone(output_type=EventCandidate)
        
        # Build base prompt
        base_prompt = f"""
        Generate a {element_type} for a matriarchal fantasy world.
        
        CONTEXT:
        {json.dumps(context, indent=2)}
        
        Ensure it has:
        1. Clear causes and consequences
        2. Connections to existing elements
        3. Strong feminine leadership and agency
        4. Realistic motivations and actions
        
        Return a detailed JSON description.
        """
        
        # Generate multiple candidates with variations
        run_ctx = RunContextWrapper(context={})
        for i in range(pool_size):
            # Add variation directive to make candidates distinct
            variation_prompt = f"""
            {base_prompt}
            
            VARIATION DIRECTIVE:
            For this variation, emphasize {self._get_variation_emphasis(element_type, i)}.
            """
            
            result = await Runner.run(generator_agent, variation_prompt, context=run_ctx.context)
            candidate = result.final_output
            candidate.generation_index = i
            candidates.append(candidate.dict())
        
        return candidates
    
    async def select_best_candidate(self, candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best candidate from the pool based on quality criteria."""
        if not candidates:
            return {"error": "Empty candidate pool"}
        
        # If there's only one candidate, no need to select
        if len(candidates) == 1:
            return candidates[0]
        
        # Filter out candidates with errors
        valid_candidates = [c for c in candidates if "error" not in c]
        
        if not valid_candidates:
            return {"error": "No valid candidates in pool"}
        
        prompt = f"""
        Select the best narrative element from these candidates:
        
        CANDIDATES:
        {json.dumps(valid_candidates, indent=2)}
        
        CONTEXT:
        {json.dumps(context, indent=2)}
        
        For each candidate, evaluate:
        1. Narrative quality and interest
        2. Consistency with existing lore
        3. Strength of matriarchal elements
        4. Potential for future development
        
        Return JSON with:
        - selected_index: the index of the best candidate
        - evaluation: your assessment of each candidate
        - reasoning: why you selected this one
        """
        
        result = await Runner.run(self.selection_agent, prompt, context={})
        selection = result.final_output
        
        selected_index = selection.selected_index
        if 0 <= selected_index < len(candidates):
            selected = candidates[selected_index]
            selected["selection_reasoning"] = selection.reasoning
            selected["evaluation"] = selection.evaluation
            return selected
        else:
            # Default to first valid candidate if selection fails
            return valid_candidates[0]
    
    async def mutate_element(self, element: Dict[str, Any], mutation_directives: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate a narrative element to improve it based on directives."""
        prompt = f"""
        Improve this narrative element by applying these mutation directives:
        
        ELEMENT:
        {json.dumps(element, indent=2)}
        
        MUTATION DIRECTIVES:
        {json.dumps(mutation_directives, indent=2)}
        
        Enhance the element while preserving its core identity.
        Return the improved JSON with all original fields plus any enhancements.
        """
        
        result = await Runner.run(self.mutation_agent, prompt, context={})
        
        try:
            mutated = json.loads(result.final_output)
            mutated["mutation_applied"] = True
            mutated["original_id"] = element.get("id")
            return mutated
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse mutated element",
                "original": element,
                "raw_output": result.final_output
            }
    
    async def evolve_narrative_element(self, element_type: str, context: Dict[str, Any], initial_element: Optional[Dict[str, Any]] = None, generations: int = 3) -> Dict[str, Any]:
        """Evolve a narrative element through multiple generations of selection and mutation."""
        current_element = initial_element
        
        for generation in range(generations):
            # Generate candidate pool
            if current_element:
                # Use the current element as a seed
                context["seed_element"] = current_element
            
            candidates = await self.generate_candidate_pool(element_type, context)
            
            # Select the best candidate
            best_candidate = await self.select_best_candidate(candidates, context)
            
            if "error" in best_candidate:
                return best_candidate
            
            # For the final generation, return as is
            if generation == generations - 1:
                best_candidate["final_generation"] = True
                best_candidate["evolution_history"] = {
                    "generations": generation + 1,
                    "initial_element": initial_element is not None
                }
                return best_candidate
            
            # Otherwise, generate mutation directives
            mutation_directives = await self._generate_mutation_directives(best_candidate, context)
            
            # Apply mutation
            current_element = await self.mutate_element(best_candidate, mutation_directives)
            
            if "error" in current_element:
                # If mutation fails, return the unmutated best candidate
                best_candidate["mutation_failed"] = True
                return best_candidate
        
        # This should not be reached, but just in case
        return current_element
    
    def _get_variation_emphasis(self, element_type: str, index: int) -> str:
        """Get emphasis for variations to ensure diversity in the candidate pool."""
        political_emphases = [
            "power struggles and ambition",
            "diplomatic negotiations and alliances",
            "resource conflicts and economic factors"
        ]
        
        military_emphases = [
            "strategic brilliance and planning",
            "individual heroism and valor",
            "technological advantages and innovations"
        ]
        
        cultural_emphases = [
            "religious and spiritual dimensions",
            "artistic and intellectual developments",
            "social customs and everyday life"
        ]
        
        general_emphases = [
            "character relationships and motivations",
            "historical significance and legacy",
            "unexpected consequences and dramatic twists"
        ]
        
        if element_type == "political_event":
            emphases = political_emphases
        elif element_type == "military_event":
            emphases = military_emphases
        elif element_type == "cultural_event":
            emphases = cultural_emphases
        else:
            emphases = general_emphases
        
        # Cycle through emphases based on index
        return emphases[index % len(emphases)]
    
    async def _generate_mutation_directives(self, element: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate directives for mutating a narrative element."""
        prompt = f"""
        Generate mutation directives to improve this narrative element:
        
        ELEMENT:
        {json.dumps(element, indent=2)}
        
        CONTEXT:
        {json.dumps(context, indent=2)}
        
        Identify 2-3 specific aspects that could be improved, such as:
        - Character motivations
        - Narrative tension and stakes
        - Integration with world lore
        - Matriarchal thematic elements
        
        For each aspect, provide a clear directive for improvement.
        Return JSON with mutation directives.
        """
        
        result = await Runner.run(self.mutation_agent, prompt, context={})
        return result.final_output.dict()

class WorldStateStreamer:
    """
    Implements progressive streaming of world state changes.
    """
    
    def __init__(self, dynamics_system):
        self.dynamics_system = dynamics_system
        self.streamer_agent = Agent(
            name="WorldStateStreamingAgent",
            instructions="""
            You generate detailed progressive updates about world state changes.
            Narrate how the world evolves, highlighting key developments, reactions,
            and emerging patterns. Focus on matriarchal power dynamics.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8)
        )
    
    async def stream_world_changes(self, event_data: Dict[str, Any], affected_elements: List[Dict[str, Any]]) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream progressive updates about world changes resulting from an event."""
        with trace(
            "WorldChangeStreaming", 
            group_id=self.dynamics_system.trace_group_id,
            metadata={"event": event_data.get("name", "Unknown Event")}
        ):
            # Sort affected elements by impact level
            affected_elements.sort(key=lambda x: x.get("impact_level", 0), reverse=True)
            
            # Prepare the prompt
            prompt = f"""
            Stream detailed updates about how this event changes the world over time:
            
            EVENT:
            {json.dumps(event_data, indent=2)}
            
            AFFECTED ELEMENTS:
            {json.dumps(affected_elements[:5], indent=2)}
            
            Provide a progressive narrative of the changes, divided into:
            1. Immediate aftermath (hours/days)
            2. Short-term developments (weeks)
            3. Medium-term consequences (months)
            4. Long-term transformations (years)
            
            Focus on how matriarchal power structures respond and evolve.
            """
            
            # Set up the streaming result
            # Note: The SDK might not support streaming directly, so we'll simulate it
            # In a real implementation, you'd use the streaming API if available
            
            result = await Runner.run(self.streamer_agent, prompt, context={})
            content = result.final_output
            
            # Process the content into phases
            phase_indicators = {
                "Immediate aftermath": "immediate",
                "Short-term developments": "short_term",
                "Medium-term consequences": "medium_term",
                "Long-term transformations": "long_term"
            }
            
            # Split content into sections and yield progressively
            sections = content.split('\n\n')
            current_phase = "immediate"
            
            for section in sections:
                # Check for phase indicators
                for indicator, phase in phase_indicators.items():
                    if indicator.lower() in section.lower():
                        current_phase = phase
                        break
                
                if section.strip():
                    yield {
                        "phase": current_phase,
                        "content": section.strip()
                    }
    
    async def stream_evolution_scenario(self, initial_state: Dict[str, Any], years: int = 10) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a progressive evolution scenario over multiple years."""
        with trace(
            "EvolutionScenarioStreaming", 
            group_id=self.dynamics_system.trace_group_id,
            metadata={"years": years}
        ):
            # Get the current world state
            if not initial_state:
                initial_state = await self.dynamics_system._fetch_world_state()
            
            # Prepare the prompt
            prompt = f"""
            Stream a progressive evolution scenario of this world over {years} years:
            
            INITIAL STATE:
            {json.dumps(initial_state, indent=2)}
            
            Describe how the world evolves year by year, focusing on:
            - Political developments
            - Cultural evolution
            - Power shifts and conflicts
            - Religious changes
            - Technological developments
            
            Maintain matriarchal power structures throughout but show complexity and change.
            
            Format as:
            Year 1: [developments]
            Year 2: [developments]
            etc.
            """
            
            # Get the full result
            result = await Runner.run(self.streamer_agent, prompt, context={})
            content = result.final_output
            
            # Parse and yield year by year
            year_pattern = re.compile(r"Year (\d+):(.*?)(?=Year \d+:|$)", re.DOTALL)
            matches = year_pattern.findall(content)
            
            for year_str, year_content in matches:
                year = int(year_str)
                if year <= years:
                    yield {
                        "year": year,
                        "content": year_content.strip()
                    }
