# lore/managers/education.py

import logging
import json
from typing import Dict, List, Any, Optional
import random

# Agents SDK imports
from agents import Agent, function_tool, Runner
from agents.run_context import RunContextWrapper
from agents.run import RunConfig
from agents.models import ModelSettings

# Governance & others
from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

# Project imports
from embedding.vector_store import generate_embedding
from lore.core.base_manager import BaseLoreManager

logger = logging.getLogger(__name__)

# We optionally create an agent that can decide how many or which educational systems 
# we might generate. Or we can keep it as is. We'll show an agent-based approach.
distribution_agent = Agent(
    name="EducationDistributionAgent",
    instructions=(
        "You decide how many educational systems or knowledge traditions to create, "
        "and what shape or variety they should have. Return valid JSON with a 'count' field "
        "and maybe additional info. Example:\n"
        "{\n"
        '  "count": 4,\n'
        '  "notes": "Focus on advanced vs basic teaching..."\n'
        "}"
    ),
    model="o3-mini",
    model_settings=ModelSettings(temperature=0.8)
)

class EducationalSystemManager(BaseLoreManager):
    """
    Manages how knowledge is taught and passed down across generations
    within the matriarchal society, including formal and informal systems.
    """

    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)

    async def ensure_initialized(self):
        """Ensure system is initialized."""
        if not self.initialized:
            await super().ensure_initialized()
            await self.initialize_tables()

    async def initialize_tables(self):
        """Ensure educational system tables exist."""
        table_definitions = {
            "EducationalSystems": """
                CREATE TABLE EducationalSystems (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    system_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    target_demographics TEXT[],
                    controlled_by TEXT,
                    core_teachings TEXT[],
                    teaching_methods TEXT[],
                    coming_of_age_rituals TEXT,
                    knowledge_restrictions TEXT,
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_educationalsystems_embedding 
                ON EducationalSystems USING ivfflat (embedding vector_cosine_ops);
            """,
            "KnowledgeTraditions": """
                CREATE TABLE KnowledgeTraditions (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    tradition_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    knowledge_domain TEXT NOT NULL,
                    preservation_method TEXT,
                    access_requirements TEXT,
                    associated_group TEXT,
                    examples TEXT[],
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_knowledgetraditions_embedding 
                ON KnowledgeTraditions USING ivfflat (embedding vector_cosine_ops);
            """
        }
        await self.initialize_tables_for_class(table_definitions)

    # ------------------------------------------------------------------------
    # 1) Adding educational system (CRUD)
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_educational_system",
        action_description="Adding educational system: {name}",
        id_from_context=lambda ctx: "educational_system_manager"
    )
    @function_tool
    async def add_educational_system(
        self,
        ctx,
        name: str,
        system_type: str,
        description: str,
        target_demographics: List[str],
        controlled_by: str,
        core_teachings: List[str],
        teaching_methods: List[str],
        coming_of_age_rituals: str = None,
        knowledge_restrictions: str = None
    ) -> int:
        """
        Add an educational system to the database as a function tool, 
        so an orchestrator agent can call it directly if needed.
        """
        # Ensure tables exist
        await self.initialize_tables()

        # Generate embedding
        embedding_text = f"{name} {system_type} {description} {' '.join(core_teachings)}"
        embedding = await generate_embedding(embedding_text)

        # Store in DB
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                system_id = await conn.fetchval("""
                    INSERT INTO EducationalSystems (
                        name, system_type, description, target_demographics,
                        controlled_by, core_teachings, teaching_methods, 
                        coming_of_age_rituals, knowledge_restrictions, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING id
                """,
                name, system_type, description, target_demographics,
                controlled_by, core_teachings, teaching_methods,
                coming_of_age_rituals, knowledge_restrictions, embedding)

                return system_id

    # ------------------------------------------------------------------------
    # 2) Adding knowledge tradition (CRUD)
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_knowledge_tradition",
        action_description="Adding knowledge tradition: {name}",
        id_from_context=lambda ctx: "educational_system_manager"
    )
    @function_tool
    async def add_knowledge_tradition(
        self,
        ctx,
        name: str,
        tradition_type: str,
        description: str,
        knowledge_domain: str,
        preservation_method: str = None,
        access_requirements: str = None,
        associated_group: str = None,
        examples: List[str] = None
    ) -> int:
        """
        Add a knowledge tradition to the database as a function tool.
        """
        await self.initialize_tables()

        examples = examples or []

        # Generate embedding
        embedding_text = f"{name} {tradition_type} {description} {knowledge_domain}"
        embedding = await generate_embedding(embedding_text)

        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                tradition_id = await conn.fetchval("""
                    INSERT INTO KnowledgeTraditions (
                        name, tradition_type, description, knowledge_domain,
                        preservation_method, access_requirements,
                        associated_group, examples, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                """,
                name, tradition_type, description, knowledge_domain,
                preservation_method, access_requirements,
                associated_group, examples, embedding)

                return tradition_id

    # ------------------------------------------------------------------------
    # 3) Generate educational systems dynamically
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_educational_systems",
        action_description="Generating educational systems for the setting",
        id_from_context=lambda ctx: "educational_system_manager"
    )
    async def generate_educational_systems(self, ctx) -> List[Dict[str, Any]]:
        """
        Use an LLM to generate a set of educational systems for the matriarchal setting,
        possibly letting the LLM decide how many to generate.
        """
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })

        # Let an agent decide how many systems to generate
        distribution_prompt = (
            "We want to create some educational systems for a matriarchal society. "
            "Propose how many we should generate (1-6) and any notes about them in JSON. "
            "Example:\n"
            "{\n"
            '  "count": 4,\n'
            '  "notes": "Focus on different classes..."'
            "\n}"
        )
        dist_config = RunConfig(workflow_name="EduSystemCount")
        dist_result = await Runner.run(distribution_agent, distribution_prompt, context=run_ctx.context, run_config=dist_config)

        try:
            dist_data = json.loads(dist_result.final_output)
            count = dist_data.get("count", 3)  # default if not provided
        except json.JSONDecodeError:
            count = 3  # fallback

        # Next, fetch some context from DB for the prompt
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                factions = await conn.fetch("""
                    SELECT name, type FROM Factions
                    LIMIT 5
                """)
                faction_names = [f"{f['name']} ({f['type']})" for f in factions]

                elements = await conn.fetch("""
                    SELECT name, type FROM CulturalElements
                    LIMIT 5
                """)
                element_names = [f"{e['name']} ({e['type']})" for e in elements]

        prompt = f"""
        Generate {count} educational systems for a matriarchal society.

        SOCIETAL CONTEXT:
        Factions: {', '.join(faction_names)}
        Cultural Elements: {', '.join(element_names)}

        Ensure each system has:
        - "name"
        - "system_type" (formal, apprenticeship, mentorship, etc.)
        - "description"
        - "target_demographics"
        - "controlled_by"
        - "core_teachings"
        - "teaching_methods"
        - "coming_of_age_rituals"
        - "knowledge_restrictions"

        Return a JSON array of objects.
        """
        education_agent = Agent(
            name="EducationalSystemAgent",
            instructions="You create educational systems for fictional matriarchal societies.",
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.9)
        )

        run_config = RunConfig(workflow_name="EducationalSystemGeneration")
        result = await Runner.run(education_agent, prompt, context=run_ctx.context, run_config=run_config)
        response_text = result.final_output

        try:
            systems = json.loads(response_text)
            if not isinstance(systems, list):
                if isinstance(systems, dict):
                    systems = [systems]
                else:
                    systems = []
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response for educational systems: {response_text}")
            return []

        saved_systems = []
        for system in systems:
            # Extract details
            name = system.get('name')
            system_type = system.get('system_type', 'formal')
            description = system.get('description')
            target_demographics = system.get('target_demographics', ['general population'])
            controlled_by = system.get('controlled_by', 'various factions')
            core_teachings = system.get('core_teachings', ['general knowledge'])
            teaching_methods = system.get('teaching_methods', ['lecture', 'practice'])
            coming_of_age_rituals = system.get('coming_of_age_rituals')
            knowledge_restrictions = system.get('knowledge_restrictions')

            if not name or not description:
                continue

            try:
                # Use our "add_educational_system" tool
                system_id = await self.add_educational_system(
                    run_ctx,
                    name=name,
                    system_type=system_type,
                    description=description,
                    target_demographics=target_demographics,
                    controlled_by=controlled_by,
                    core_teachings=core_teachings,
                    teaching_methods=teaching_methods,
                    coming_of_age_rituals=coming_of_age_rituals,
                    knowledge_restrictions=knowledge_restrictions
                )
                system['id'] = system_id
                saved_systems.append(system)
            except Exception as e:
                logger.error(f"Error saving educational system '{name}': {e}")

        return saved_systems

    # ------------------------------------------------------------------------
    # 4) Generate knowledge traditions
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_knowledge_traditions",
        action_description="Generating knowledge traditions for the setting",
        id_from_context=lambda ctx: "educational_system_manager"
    )
    async def generate_knowledge_traditions(self, ctx) -> List[Dict[str, Any]]:
        """
        Generate knowledge traditions that represent how knowledge is
        passed down across generations in informal ways, with LLM-based logic.
        """
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })

        # Let the LLM also decide how many traditions to generate
        dist_prompt = (
            "We want to create a set of knowledge traditions for a matriarchal setting. "
            "How many should we make? Return JSON with a 'count' field. Example: {\"count\": 4}"
        )
        dist_config = RunConfig(workflow_name="KnowledgeTraditionCount")
        dist_result = await Runner.run(distribution_agent, dist_prompt, context=run_ctx.context, run_config=dist_config)

        try:
            dist_data = json.loads(dist_result.final_output)
            count = dist_data.get("count", 4)
        except json.JSONDecodeError:
            count = 4

        # Get some context from DB
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                elements = await conn.fetch("""
                    SELECT name, type, description 
                    FROM CulturalElements
                    LIMIT 3
                """)
                culture_context = "\n".join([
                    f"- {e['name']} ({e['type']}): {e['description'][:80]}..."
                    for e in elements
                ])

                systems = await conn.fetch("""
                    SELECT name, system_type 
                    FROM EducationalSystems
                    LIMIT 2
                """)
                systems_context = ", ".join([
                    f"{s['name']} ({s['system_type']})"
                    for s in systems
                ]) if systems else "No formal systems yet"

        prompt = f"""
        Generate {count} knowledge traditions for a matriarchal society.

        CULTURAL CONTEXT:
        {culture_context}

        EDUCATIONAL SYSTEMS:
        {systems_context}

        For each tradition, return fields:
        - "name"
        - "tradition_type"
        - "description"
        - "knowledge_domain"
        - "preservation_method"
        - "access_requirements"
        - "associated_group"
        - "examples"
        Return a JSON array.
        """
        tradition_agent = Agent(
            name="KnowledgeTraditionAgent",
            instructions="You create knowledge transmission traditions for matriarchal societies.",
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.9)
        )
        run_config = RunConfig(workflow_name="KnowledgeTraditionGeneration")
        result = await Runner.run(tradition_agent, prompt, context=run_ctx.context, run_config=run_config)

        response_text = result.final_output

        try:
            traditions = json.loads(response_text)
            if not isinstance(traditions, list):
                if isinstance(traditions, dict):
                    traditions = [traditions]
                else:
                    traditions = []
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response for knowledge traditions: {response_text}")
            return []

        saved_traditions = []
        for trad in traditions:
            name = trad.get('name')
            tradition_type = trad.get('tradition_type', 'oral')
            description = trad.get('description')
            knowledge_domain = trad.get('knowledge_domain', 'general knowledge')
            preservation_method = trad.get('preservation_method')
            access_requirements = trad.get('access_requirements')
            associated_group = trad.get('associated_group')
            examples = trad.get('examples', [])

            if not name or not description:
                continue

            try:
                tradition_id = await self.add_knowledge_tradition(
                    run_ctx,
                    name=name,
                    tradition_type=tradition_type,
                    description=description,
                    knowledge_domain=knowledge_domain,
                    preservation_method=preservation_method,
                    access_requirements=access_requirements,
                    associated_group=associated_group,
                    examples=examples
                )
                trad['id'] = tradition_id
                saved_traditions.append(trad)
            except Exception as e:
                logger.error(f"Error saving knowledge tradition '{name}': {e}")

        return saved_traditions
