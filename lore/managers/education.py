# lore/managers/education.py

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from pydantic import BaseModel, Field

# OpenAI Agents SDK imports
from agents import (
    Agent, function_tool, Runner, trace,
    GuardrailFunctionOutput, InputGuardrail, handoff
)
from agents.run_context import RunContextWrapper
from agents.run import RunConfig
from agents import ModelSettings

# Governance & others
from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

# Project imports
from embedding.vector_store import generate_embedding
from lore.managers.base_manager import BaseLoreManager

logger = logging.getLogger(__name__)

# Define Pydantic models for structured outputs
class EducationalSystem(BaseModel):
    """Model for educational system structure."""
    name: str
    system_type: str
    description: str
    target_demographics: List[str]
    controlled_by: str
    core_teachings: List[str]
    teaching_methods: List[str]
    coming_of_age_rituals: Optional[str] = None
    knowledge_restrictions: Optional[str] = None
    
    # Matriarchal elements
    female_leadership_roles: List[str]
    male_roles: List[str]
    gender_specific_teachings: Dict[str, List[str]]
    
    # Content restrictions
    taboo_subjects: List[str]
    censorship_level: int = Field(..., ge=1, le=10)
    censorship_enforcement: str

class KnowledgeTradition(BaseModel):
    """Model for knowledge tradition structure."""
    name: str
    tradition_type: str
    description: str
    knowledge_domain: str
    preservation_method: Optional[str] = None
    access_requirements: Optional[str] = None
    associated_group: Optional[str] = None
    examples: List[str] = []
    
    # Matriarchal elements
    female_gatekeepers: bool
    gendered_access: Optional[Dict[str, str]] = None
    matriarchal_reinforcement: str

class TeachingContent(BaseModel):
    """Model for educational content."""
    title: str
    content_type: str
    subject_area: str
    description: str
    target_age_group: str
    key_points: List[str]
    examples: List[str] = []
    exercises: List[Dict[str, Any]] = []
    restricted: bool = False
    restriction_reason: Optional[str] = None

class CensorshipGuardrail(InputGuardrail):
    """Guardrail to detect and prevent inappropriate educational content."""
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
    
    async def __call__(self, ctx, agent, input_data: str) -> GuardrailFunctionOutput:
        """Check if educational content violates taboo subjects or includes inappropriate material."""
        # Simplified for demonstration - in practice you'd have a more sophisticated check
        forbidden_words = [
            "explicit", "graphic", "inappropriate", "harmful",
            "dangerous", "violent", "weapon"
        ]
        
        # More restricted in strict mode
        if self.strict_mode:
            forbidden_words.extend([
                "rebellion", "revolution", "overthrow", "resist", 
                "challenge authority", "disobey"
            ])
        
        # Check for forbidden content
        lower_input = input_data.lower()
        found_words = [word for word in forbidden_words if word in lower_input]
        
        if found_words:
            return GuardrailFunctionOutput(
                output_info={
                    "is_appropriate": False,
                    "forbidden_words": found_words,
                    "recommendation": "Remove or rephrase content with flagged terms."
                },
                tripwire_triggered=True
            )
        
        return GuardrailFunctionOutput(
            output_info={"is_appropriate": True},
            tripwire_triggered=False
        )

class EducationalSystemManager(BaseLoreManager):
    """
    Enhanced manager for educational systems with streaming support,
    agent-to-agent knowledge exchange, structured outputs, and censorship guardrails.
    """

    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        
        # Initialize specialized agents
        self._init_specialized_agents()
        
        # Set up guardrails
        self.censorship_guardrail = CensorshipGuardrail()
        self.strict_censorship_guardrail = CensorshipGuardrail(strict_mode=True)
        

    def _init_specialized_agents(self):
        """Initialize specialized agents for different educational domains."""
        # Distribution agent
        self.distribution_agent = Agent(
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
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8)
        )
        
        # Formal education specialist
        self.formal_education_agent = Agent(
            name="FormalEducationAgent",
            instructions=(
                "You specialize in designing formal educational systems like schools and academies. "
                "Create hierarchical, structured learning programs with clear progression paths. "
                "Emphasize matriarchal authority in educational leadership and teaching methods."
            ),
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        # Apprenticeship specialist
        self.apprenticeship_agent = Agent(
            name="ApprenticeshipAgent",
            instructions=(
                "You specialize in designing apprenticeship and mentorship-based educational systems. "
                "Create learning programs based on direct transmission of skills from expert to novice. "
                "Emphasize matriarchal authority with female masters and gender-based access."
            ),
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        # Religious education specialist
        self.religious_education_agent = Agent(
            name="ReligiousEducationAgent",
            instructions=(
                "You specialize in designing religious educational systems like seminaries. "
                "Create programs that transmit spiritual knowledge, rituals, and traditions. "
                "Emphasize female religious authority and gendered access to sacred knowledge."
            ),
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        # Set up handoffs between agents
        self.education_agent = Agent(
            name="EducationalSystemAgent",
            instructions="You create educational systems for fictional matriarchal societies.",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9),
            handoffs=[
                handoff(
                    self.formal_education_agent,
                    tool_name_override="design_formal_education",
                    tool_description_override="Design formal educational systems like schools and academies"
                ),
                handoff(
                    self.apprenticeship_agent,
                    tool_name_override="design_apprenticeship",
                    tool_description_override="Design apprenticeship and mentorship-based educational systems"
                ),
                handoff(
                    self.religious_education_agent,
                    tool_name_override="design_religious_education",
                    tool_description_override="Design religious educational systems like seminaries"
                )
            ]
        )
        
        # Knowledge exchange agent
        self.knowledge_exchange_agent = Agent(
            name="KnowledgeExchangeAgent",
            instructions=(
                "You facilitate knowledge exchange between educational systems. "
                "Analyze what knowledge can be shared, adapted, or must remain separate. "
                "Consider power dynamics, accessibility, and cultural context."
            ),
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.7)
        )

    async def ensure_initialized(self):
        """Ensure system is initialized."""
        if not self.initialized:
            await super().ensure_initialized()
            await self._initialize_tables()

    async def _initialize_tables(self):
        """Ensure educational system tables exist."""
        table_definitions = {
            "EducationalSystems": """
                CREATE TABLE IF NOT EXISTS EducationalSystems (
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
                    female_leadership_roles TEXT[],
                    male_roles TEXT[],
                    gender_specific_teachings JSONB,
                    taboo_subjects TEXT[],
                    censorship_level INTEGER CHECK (censorship_level BETWEEN 1 AND 10),
                    censorship_enforcement TEXT,
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_educationalsystems_embedding 
                ON EducationalSystems USING ivfflat (embedding vector_cosine_ops);
            """,
            "KnowledgeTraditions": """
                CREATE TABLE IF NOT EXISTS KnowledgeTraditions (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    tradition_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    knowledge_domain TEXT NOT NULL,
                    preservation_method TEXT,
                    access_requirements TEXT,
                    associated_group TEXT,
                    examples TEXT[],
                    female_gatekeepers BOOLEAN DEFAULT TRUE,
                    gendered_access JSONB,
                    matriarchal_reinforcement TEXT,
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_knowledgetraditions_embedding 
                ON KnowledgeTraditions USING ivfflat (embedding vector_cosine_ops);
            """,
            "TeachingContents": """
                CREATE TABLE IF NOT EXISTS TeachingContents (
                    id SERIAL PRIMARY KEY,
                    system_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    subject_area TEXT NOT NULL,
                    description TEXT NOT NULL,
                    target_age_group TEXT,
                    key_points TEXT[],
                    examples TEXT[],
                    exercises JSONB,
                    restricted BOOLEAN DEFAULT FALSE,
                    restriction_reason TEXT,
                    embedding VECTOR(1536),
                    FOREIGN KEY (system_id) REFERENCES EducationalSystems(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_teachingcontents_embedding 
                ON TeachingContents USING ivfflat (embedding vector_cosine_ops);
            """
        }
        await self._initialize_tables_for_class_impl(table_definitions)

    # ------------------------------------------------------------------------
    # 1) Adding educational system (CRUD) with structured output
    # ------------------------------------------------------------------------
    async def _add_educational_system_impl(
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
        knowledge_restrictions: str = None,
        female_leadership_roles: List[str] = None,
        male_roles: List[str] = None,
        gender_specific_teachings: Dict[str, List[str]] = None,
        taboo_subjects: List[str] = None,
        censorship_level: int = 5,
        censorship_enforcement: str = None
    ) -> int:
        """
        Add an educational system to the database with matriarchal elements and censorship controls.
        Implementation method.
        """
        with trace(
            "AddEducationalSystem", 
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "system_name": name}
        ):
            # Ensure tables exist
            await self.ensure_initialized()
            
            # Set defaults for optional matriarchal elements if not provided
            female_leadership_roles = female_leadership_roles or ["Headmistress", "Teacher", "Mentor"]
            male_roles = male_roles or ["Assistant", "Aide", "Custodian"]
            gender_specific_teachings = gender_specific_teachings or {
                "female": ["Leadership", "Authority", "Decision-making"],
                "male": ["Service", "Support", "Compliance"]
            }
            taboo_subjects = taboo_subjects or ["Challenging feminine authority", "Male independence"]
            censorship_enforcement = censorship_enforcement or "Monitored by female leadership"

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
                            coming_of_age_rituals, knowledge_restrictions, embedding,
                            female_leadership_roles, male_roles, gender_specific_teachings,
                            taboo_subjects, censorship_level, censorship_enforcement
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                        RETURNING id
                    """,
                    name, system_type, description, target_demographics,
                    controlled_by, core_teachings, teaching_methods,
                    coming_of_age_rituals, knowledge_restrictions, embedding,
                    female_leadership_roles, male_roles, json.dumps(gender_specific_teachings),
                    taboo_subjects, censorship_level, censorship_enforcement)

                    return system_id

    @staticmethod
    @function_tool
    async def add_educational_system(
        ctx: RunContextWrapper,
        name: str,
        system_type: str,
        description: str,
        target_demographics: List[str],
        controlled_by: str,
        core_teachings: List[str],
        teaching_methods: List[str],
        coming_of_age_rituals: str = None,
        knowledge_restrictions: str = None,
        female_leadership_roles: List[str] = None,
        male_roles: List[str] = None,
        gender_specific_teachings: Dict[str, List[str]] = None,
        taboo_subjects: List[str] = None,
        censorship_level: int = 5,
        censorship_enforcement: str = None
    ) -> int:
        """
        Add an educational system to the database with matriarchal elements and censorship controls.
        
        Args:
            ctx: Context object
            name: Name of the educational system
            system_type: Type of system (formal, apprenticeship, etc.)
            description: Detailed description
            target_demographics: Who the system serves
            controlled_by: Who controls the system
            core_teachings: Main subjects or concepts taught
            teaching_methods: How knowledge is transmitted
            coming_of_age_rituals: Optional rituals for reaching maturity
            knowledge_restrictions: Optional restrictions on knowledge access
            female_leadership_roles: Roles reserved for women in the system
            male_roles: Roles allowed for men in the system
            gender_specific_teachings: Knowledge segregated by gender
            taboo_subjects: Topics that are forbidden or restricted
            censorship_level: Level of information control (1-10)
            censorship_enforcement: How restrictions are enforced
            
        Returns:
            ID of the created educational system
        """
        # Apply governance if available
        try:
            from nyx.nyx_governance import NyxUnifiedGovernor
            governor = await NyxUnifiedGovernor(ctx.context.get("user_id"), ctx.context.get("conversation_id")).initialize()
            permission = await governor.check_action_permission(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="educational_system_manager",
                action_type="add_educational_system",
                action_details={"name": name, "type": system_type}
            )
            
            if not permission.get("approved", True):
                return {"error": permission.get("reasoning", "Action not permitted by governance")}
        except (ImportError, Exception):
            # Governance optional - continue if not available
            pass
            
        return await self._add_educational_system_impl(
            ctx, name, system_type, description, target_demographics, 
            controlled_by, core_teachings, teaching_methods,
            coming_of_age_rituals, knowledge_restrictions,
            female_leadership_roles, male_roles, gender_specific_teachings,
            taboo_subjects, censorship_level, censorship_enforcement
        )

    # ------------------------------------------------------------------------
    # 2) Adding knowledge tradition (CRUD) with structured output
    # ------------------------------------------------------------------------
    async def _add_knowledge_tradition_impl(
        self,
        ctx,
        name: str,
        tradition_type: str,
        description: str,
        knowledge_domain: str,
        preservation_method: str = None,
        access_requirements: str = None,
        associated_group: str = None,
        examples: List[str] = None,
        female_gatekeepers: bool = True,
        gendered_access: Dict[str, str] = None,
        matriarchal_reinforcement: str = None
    ) -> int:
        """
        Add a knowledge tradition to the database.
        Implementation method.
        """
        with trace(
            "AddKnowledgeTradition", 
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "tradition_name": name}
        ):
            await self.ensure_initialized()
    
            examples = examples or []
            
            # Set defaults for optional matriarchal elements if not provided
            gendered_access = gendered_access or {
                "female": "Full access with advancement opportunities",
                "male": "Limited access under supervision"
            }
            matriarchal_reinforcement = matriarchal_reinforcement or "Emphasizes female wisdom and authority"
    
            # Generate embedding
            embedding_text = f"{name} {tradition_type} {description} {knowledge_domain}"
            embedding = await generate_embedding(embedding_text)
    
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    tradition_id = await conn.fetchval("""
                        INSERT INTO KnowledgeTraditions (
                            name, tradition_type, description, knowledge_domain,
                            preservation_method, access_requirements,
                            associated_group, examples, embedding,
                            female_gatekeepers, gendered_access, matriarchal_reinforcement
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        RETURNING id
                    """,
                    name, tradition_type, description, knowledge_domain,
                    preservation_method, access_requirements,
                    associated_group, examples, embedding,
                    female_gatekeepers, json.dumps(gendered_access), matriarchal_reinforcement)
    
                    return tradition_id

    @staticmethod
    @function_tool
    async def add_knowledge_tradition(
        ctx: RunContextWrapper,
        name: str,
        tradition_type: str,
        description: str,
        knowledge_domain: str,
        preservation_method: str = None,
        access_requirements: str = None,
        associated_group: str = None,
        examples: List[str] = None,
        female_gatekeepers: bool = True,
        gendered_access: Dict[str, str] = None,
        matriarchal_reinforcement: str = None
    ) -> int:
        """
        Add a knowledge tradition to the database.
        
        Args:
            ctx: Context object
            name: Name of the tradition
            tradition_type: Type of tradition (oral, written, ritual, etc.)
            description: Detailed description
            knowledge_domain: Domain or field of knowledge
            preservation_method: How knowledge is preserved
            access_requirements: Requirements to access knowledge
            associated_group: Group associated with tradition
            examples: Examples of knowledge in tradition
            female_gatekeepers: Whether women control access to knowledge
            gendered_access: Access levels by gender
            matriarchal_reinforcement: How tradition reinforces matriarchy
            
        Returns:
            ID of the created knowledge tradition
        """
        # Apply governance if available
        try:
            from nyx.nyx_governance import NyxUnifiedGovernor
            governor = await NyxUnifiedGovernor(ctx.context.get("user_id"), ctx.context.get("conversation_id")).initialize()
            permission = await governor.check_action_permission(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="educational_system_manager",
                action_type="add_knowledge_tradition",
                action_details={"name": name, "type": tradition_type}
            )
            
            if not permission.get("approved", True):
                return {"error": permission.get("reasoning", "Action not permitted by governance")}
        except (ImportError, Exception):
            # Governance optional - continue if not available
            pass
            
        return await self._add_knowledge_tradition_impl(
            ctx, name, tradition_type, description, knowledge_domain,
            preservation_method, access_requirements, associated_group,
            examples, female_gatekeepers, gendered_access, matriarchal_reinforcement
        )

    # ------------------------------------------------------------------------
    # 3) Stream educational system generation with progressive updates
    # ------------------------------------------------------------------------
    @staticmethod
    @function_tool
    async def stream_educational_development(
        ctx: RunContextWrapper,
        system_name: str,
        system_type: str,
        matriarchy_level: int = 8
    ) -> AsyncGenerator[str, None]:
        """
        Stream the development of a complete educational system with live updates.
        
        Args:
            ctx: Context object
            system_name: Name for the educational system
            system_type: Type of system (formal, apprenticeship, religious)
            matriarchy_level: Level of matriarchal control (1-10)
            
        Yields:
            Updates as the educational system is developed
        """
        # Apply governance if available
        try:
            from nyx.nyx_governance import NyxUnifiedGovernor
            governor = await NyxUnifiedGovernor(ctx.context.get("user_id"), ctx.context.get("conversation_id")).initialize()
            permission = await governor.check_action_permission(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="educational_system_manager",
                action_type="stream_educational_development",
                action_details={"system_name": system_name, "system_type": system_type}
            )
            
            if not permission.get("approved", True):
                yield f"Error: {permission.get('reasoning', 'Action not permitted by governance')}"
                return
        except (ImportError, Exception):
            # Governance optional - continue if not available
            pass
            
        with trace(
            "StreamEducationalDevelopment", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata, 
                "system_name": system_name,
                "system_type": system_type,
                "matriarchy_level": matriarchy_level
            }
        ):
            run_ctx = self.create_run_context(ctx)
            
            # Determine which specialized agent to use based on system_type
            if system_type.lower() in ["formal", "school", "academy"]:
                specialist_agent = self.formal_education_agent
                system_type = "formal"
            elif system_type.lower() in ["apprenticeship", "mentorship", "guild"]:
                specialist_agent = self.apprenticeship_agent
                system_type = "apprenticeship"
            elif system_type.lower() in ["religious", "seminary", "spiritual"]:
                specialist_agent = self.religious_education_agent
                system_type = "religious"
            else:
                # Default to the general education agent
                specialist_agent = self.education_agent
            
            # 1. Stream basic system outline
            yield f"Developing {system_name} - a {system_type} educational system..."
            
            # Create basic prompt
            basic_prompt = f"""
            Create an outline for a {system_type} educational system named '{system_name}' with 
            a matriarchy level of {matriarchy_level}/10. Include:
            
            1. Core purpose
            2. Target demographics
            3. Main features
            4. Matriarchal authority structure
            """
            
            # Instead of using StreamingResponse, use the regular run method for this step
            basic_result = await Runner.run(specialist_agent, basic_prompt, context=run_ctx.context)
            basic_outline = basic_result.final_output
            
            yield f"\nBasic System Outline:\n{basic_outline}\n"
            await asyncio.sleep(0.5)  # Small delay for better streaming experience
            
            # 2. Stream leadership structure
            yield "\nDeveloping leadership structure..."
            
            leadership_prompt = f"""
            For the {system_type} educational system '{system_name}':
            
            Create a detailed leadership structure that emphasizes feminine authority 
            with a matriarchy level of {matriarchy_level}/10.
            
            Define:
            1. Female leadership roles
            2. Male supportive roles
            3. Power dynamics
            4. Decision-making processes
            """
            
            # For this step, let's use the streaming capability to demonstrate how it works
            stream_config = RunConfig(
                workflow_name="LeadershipStructureGeneration",
                trace_metadata={"component": "leadership_structure"}
            )
            
            streaming_result = Runner.run_streamed(
                specialist_agent, 
                leadership_prompt, 
                context=run_ctx.context,
                run_config=stream_config
            )
            
            # Process the leadership structure streaming events
            leadership_content = []
            async for event in streaming_result.stream_events():
                # Handle different types of streaming events
                if event.type == "run_item_stream_event":
                    if event.item.type == "message_output_item":
                        # Extract text from the message output
                        from agents.items import ItemHelpers
                        message_text = ItemHelpers.text_message_output(event.item)
                        leadership_content.append(message_text)
            
            # Combine the collected content
            leadership_structure = "".join(leadership_content) if leadership_content else streaming_result.final_output
            
            yield f"\nLeadership Structure:\n{leadership_structure}\n"
            await asyncio.sleep(0.5)
            
            # 3. Stream curriculum and teaching methods - using regular run method again
            yield "\nDeveloping curriculum and teaching methods..."
            
            curriculum_prompt = f"""
            For the {system_type} educational system '{system_name}':
            
            Create a curriculum outline with teaching methods that reinforce matriarchal values.
            Matriarchy level: {matriarchy_level}/10
            
            Include:
            1. Core subjects/skills taught
            2. Teaching methodologies
            3. Gender-specific tracks or teachings
            4. Evaluation methods
            """
            
            curriculum_result = await Runner.run(specialist_agent, curriculum_prompt, context=run_ctx.context)
            curriculum = curriculum_result.final_output
            
            yield f"\nCurriculum and Teaching Methods:\n{curriculum}\n"
            await asyncio.sleep(0.5)
            
            # 4. Stream knowledge restrictions and taboos
            yield "\nDeveloping knowledge restrictions and taboo subjects..."
            
            restrictions_prompt = f"""
            For the {system_type} educational system '{system_name}':
            
            Define knowledge restrictions, censorship practices, and taboo subjects:
            Matriarchy level: {matriarchy_level}/10
            
            Include:
            1. Subjects/knowledge restricted by gender
            2. Completely taboo topics
            3. Censorship enforcement mechanisms
            4. Consequences for violations
            """
            
            # Apply the censorship guardrail to this prompt
            restrictions_agent = specialist_agent.clone(
                input_guardrails=[self.censorship_guardrail]
            )
            
            restrictions_result = await Runner.run(restrictions_agent, restrictions_prompt, context=run_ctx.context)
            restrictions = restrictions_result.final_output
            
            yield f"\nKnowledge Restrictions and Taboos:\n{restrictions}\n"
            await asyncio.sleep(0.5)
            
            # 5. Create a complete system definition in structured format
            yield "\nFinalizing educational system definition..."
            
            # Compile all the information into a structured prompt
            complete_prompt = f"""
            Based on all the previous information, create a complete, structured definition
            for the {system_type} educational system '{system_name}'.
            
            SYSTEM OUTLINE:
            {basic_outline}
            
            LEADERSHIP STRUCTURE:
            {leadership_structure}
            
            CURRICULUM:
            {curriculum}
            
            RESTRICTIONS:
            {restrictions}
            
            Return an EducationalSystem object with all required fields.
            """
            
            # Create a system definition agent with structured output
            system_definition_agent = specialist_agent.clone(
                output_type=EducationalSystem
            )
            
            definition_result = await Runner.run(system_definition_agent, complete_prompt, context=run_ctx.context)
            system_definition = definition_result.final_output
            
            # 6. Save the educational system to the database
            try:
                # Convert Pydantic model to parameters
                system_id = await self._add_educational_system_impl(
                    run_ctx,
                    name=system_definition.name,
                    system_type=system_definition.system_type,
                    description=system_definition.description,
                    target_demographics=system_definition.target_demographics,
                    controlled_by=system_definition.controlled_by,
                    core_teachings=system_definition.core_teachings,
                    teaching_methods=system_definition.teaching_methods,
                    coming_of_age_rituals=system_definition.coming_of_age_rituals,
                    knowledge_restrictions=system_definition.knowledge_restrictions,
                    female_leadership_roles=system_definition.female_leadership_roles,
                    male_roles=system_definition.male_roles,
                    gender_specific_teachings=system_definition.gender_specific_teachings,
                    taboo_subjects=system_definition.taboo_subjects,
                    censorship_level=system_definition.censorship_level,
                    censorship_enforcement=system_definition.censorship_enforcement
                )
                
                yield f"\nEducational system saved to database with ID: {system_id}"
            except Exception as e:
                yield f"\nError saving educational system: {str(e)}"
            
            # 7. Final summary
            yield f"\nCompleted development of {system_name} - a {system_definition.system_type} educational system."

    # ------------------------------------------------------------------------
    # 4) Agent-to-agent knowledge exchange
    # ------------------------------------------------------------------------
    @staticmethod
    @function_tool
    async def exchange_knowledge_between_systems(
        ctx: RunContextWrapper,
        source_system_id: int,
        target_system_id: int,
        knowledge_domain: str
    ) -> Dict[str, Any]:
        """
        Facilitate knowledge exchange between two educational systems.
        
        Args:
            ctx: Context object
            source_system_id: ID of the system providing knowledge
            target_system_id: ID of the system receiving knowledge
            knowledge_domain: Domain of knowledge to exchange
            
        Returns:
            Dictionary with results of the knowledge exchange
        """
        # Apply governance if available
        try:
            from nyx.nyx_governance import NyxUnifiedGovernor
            governor = await NyxUnifiedGovernor(ctx.context.get("user_id"), ctx.context.get("conversation_id")).initialize()
            permission = await governor.check_action_permission(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="educational_system_manager",
                action_type="exchange_knowledge_between_systems",
                action_details={"source_id": source_system_id, "target_id": target_system_id}
            )
            
            if not permission.get("approved", True):
                return {"error": permission.get("reasoning", "Action not permitted by governance")}
        except (ImportError, Exception):
            # Governance optional - continue if not available
            pass
            
        with trace(
            "KnowledgeExchange", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata, 
                "source_id": source_system_id,
                "target_id": target_system_id,
                "domain": knowledge_domain
            }
        ):
            run_ctx = self.create_run_context(ctx)
            
            # Fetch source and target system information
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    source_system = await conn.fetchrow("""
                        SELECT * FROM EducationalSystems WHERE id = $1
                    """, source_system_id)
                    
                    target_system = await conn.fetchrow("""
                        SELECT * FROM EducationalSystems WHERE id = $1
                    """, target_system_id)
                    
                    if not source_system or not target_system:
                        return {"error": "One or both systems not found"}
                    
                    # Get teaching contents from source system in this domain
                    contents = await conn.fetch("""
                        SELECT * FROM TeachingContents 
                        WHERE system_id = $1 AND subject_area = $2
                    """, source_system_id, knowledge_domain)
                    
                    source_contents = [dict(c) for c in contents]
            
            # Prepare data for decision-making
            source_data = dict(source_system)
            target_data = dict(target_system)
            
            # Parse JSON fields
            for data in [source_data, target_data]:
                if "gender_specific_teachings" in data and data["gender_specific_teachings"]:
                    try:
                        data["gender_specific_teachings"] = json.loads(data["gender_specific_teachings"])
                    except:
                        data["gender_specific_teachings"] = {}
            
            # Knowledge exchange prompt
            exchange_prompt = f"""
            Facilitate knowledge exchange between these two educational systems:
            
            SOURCE SYSTEM:
            {json.dumps(source_data, indent=2)}
            
            TARGET SYSTEM:
            {json.dumps(target_data, indent=2)}
            
            KNOWLEDGE DOMAIN:
            {knowledge_domain}
            
            AVAILABLE CONTENT:
            {json.dumps(source_contents, indent=2)}
            
            Determine:
            1. Which knowledge can be transferred directly
            2. Which knowledge needs adaptation
            3. Which knowledge is incompatible or restricted
            4. Necessary modifications to make compatible
            
            Return detailed JSON with these categories and explanations.
            """
            
            result = await Runner.run(self.knowledge_exchange_agent, exchange_prompt, context=run_ctx.context)
            
            try:
                exchange_results = json.loads(result.final_output)
            except json.JSONDecodeError:
                exchange_results = {"raw_output": result.final_output}
            
            # Implement the knowledge exchange based on results
            transferred_count = 0
            adapted_count = 0
            restricted_count = 0
            
            # For this example, we'll just simulate the result
            # In a real implementation, you would actually create new TeachingContent entries
            
            return {
                "source_system": source_data["name"],
                "target_system": target_data["name"],
                "knowledge_domain": knowledge_domain,
                "exchange_results": exchange_results,
                "transferred_count": transferred_count,
                "adapted_count": adapted_count,
                "restricted_count": restricted_count
            }

    # ------------------------------------------------------------------------
    # 5) Generate educational systems (with structured output)
    # ------------------------------------------------------------------------
    @staticmethod
    @function_tool
    async def generate_educational_systems(ctx: RunContextWrapper) -> List[Dict[str, Any]]:
        """
        Use an LLM to generate a set of educational systems for the matriarchal setting,
        with structured output and specialized agents.
        
        Args:
            ctx: Context object
        
        Returns:
            List of generated educational systems
        """
        # Apply governance if available
        try:
            from nyx.nyx_governance import NyxUnifiedGovernor
            governor = await NyxUnifiedGovernor(ctx.context.get("user_id"), ctx.context.get("conversation_id")).initialize()
            permission = await governor.check_action_permission(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="educational_system_manager",
                action_type="generate_educational_systems",
                action_details={}
            )
            
            if not permission.get("approved", True):
                return [{"error": permission.get("reasoning", "Action not permitted by governance")}]
        except (ImportError, Exception):
            # Governance optional - continue if not available
            pass
            
        with trace(
            "GenerateEducationalSystems", 
            group_id=self.trace_group_id,
            metadata=self.trace_metadata
        ):
            run_ctx = self.create_run_context(ctx)

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
            dist_result = await Runner.run(
                self.distribution_agent, 
                distribution_prompt, 
                context=run_ctx.context, 
                run_config=dist_config
            )

            try:
                dist_data = json.loads(dist_result.final_output)
                count = dist_data.get("count", 3)  # default if not provided
            except json.JSONDecodeError:
                count = 3  # fallback

            # Next, fetch some context from DB for the prompt
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    # Try to fetch faction info if available
                    try:
                        factions = await conn.fetch("""
                            SELECT name, type FROM Factions
                            LIMIT 5
                        """)
                        faction_names = [f"{f['name']} ({f['type']})" for f in factions]
                    except:
                        faction_names = ["Sisterhood of the Moon", "Matriarchal Council", "Forest Tribe"]
                    
                    # Try to fetch cultural elements if available  
                    try:
                        elements = await conn.fetch("""
                            SELECT name, type FROM CulturalElements
                            LIMIT 5
                        """)
                        element_names = [f"{e['name']} ({e['type']})" for e in elements]
                    except:
                        element_names = ["Rite of Womanhood", "Maternal Authority", "Feminine Leadership"]

            prompt = f"""
            Generate {count} educational systems for a matriarchal society.

            SOCIETAL CONTEXT:
            Factions: {', '.join(faction_names)}
            Cultural Elements: {', '.join(element_names)}

            For each system:
            1. Select an appropriate type (formal, apprenticeship, religious, etc.)
            2. Create a cohesive design with all required fields
            3. Ensure strong matriarchal themes
            4. Define appropriate gender dynamics
            
            Return {count} EducationalSystem objects with all required fields.
            """
            
            # Use the education agent with structured output
            education_definition_agent = self.education_agent.clone(
                output_type=List[EducationalSystem]
            )

            run_config = RunConfig(workflow_name="EducationalSystemGeneration")
            result = await Runner.run(
                education_definition_agent, 
                prompt, 
                context=run_ctx.context, 
                run_config=run_config
            )
            
            systems = result.final_output
            
            saved_systems = []
            for system in systems:
                try:
                    # Add the system to the database
                    system_id = await self._add_educational_system_impl(
                        run_ctx,
                        name=system.name,
                        system_type=system.system_type,
                        description=system.description,
                        target_demographics=system.target_demographics,
                        controlled_by=system.controlled_by,
                        core_teachings=system.core_teachings,
                        teaching_methods=system.teaching_methods,
                        coming_of_age_rituals=system.coming_of_age_rituals,
                        knowledge_restrictions=system.knowledge_restrictions,
                        female_leadership_roles=system.female_leadership_roles,
                        male_roles=system.male_roles,
                        gender_specific_teachings=system.gender_specific_teachings,
                        taboo_subjects=system.taboo_subjects,
                        censorship_level=system.censorship_level,
                        censorship_enforcement=system.censorship_enforcement
                    )
                    
                    # Add system to the result
                    system_dict = system.dict()
                    system_dict["id"] = system_id
                    saved_systems.append(system_dict)
                    
                except Exception as e:
                    logger.error(f"Error saving educational system '{system.name}': {e}")

            return saved_systems

    # ------------------------------------------------------------------------
    # 6) Generate knowledge traditions (with structured output)
    # ------------------------------------------------------------------------
    @staticmethod
    @function_tool
    async def generate_knowledge_traditions(ctx: RunContextWrapper) -> List[Dict[str, Any]]:
        """
        Generate knowledge traditions that represent how knowledge is
        passed down across generations in informal ways, with structured output.
        
        Args:
            ctx: Context object
            
        Returns:
            List of generated knowledge traditions
        """
        # Apply governance if available
        try:
            from nyx.nyx_governance import NyxUnifiedGovernor
            governor = await NyxUnifiedGovernor(ctx.context.get("user_id"), ctx.context.get("conversation_id")).initialize()
            permission = await governor.check_action_permission(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="educational_system_manager",
                action_type="generate_knowledge_traditions",
                action_details={}
            )
            
            if not permission.get("approved", True):
                return [{"error": permission.get("reasoning", "Action not permitted by governance")}]
        except (ImportError, Exception):
            # Governance optional - continue if not available
            pass
            
        with trace(
            "GenerateKnowledgeTraditions", 
            group_id=self.trace_group_id,
            metadata=self.trace_metadata
        ):
            run_ctx = self.create_run_context(ctx)

            # Let the LLM also decide how many traditions to generate
            dist_prompt = (
                "We want to create a set of knowledge traditions for a matriarchal setting. "
                "How many should we make? Return JSON with a 'count' field. Example: {\"count\": 4}"
            )
            dist_config = RunConfig(workflow_name="KnowledgeTraditionCount")
            dist_result = await Runner.run(
                self.distribution_agent, 
                dist_prompt, 
                context=run_ctx.context, 
                run_config=dist_config
            )

            try:
                dist_data = json.loads(dist_result.final_output)
                count = dist_data.get("count", 4)
            except json.JSONDecodeError:
                count = 4

            # Get some context from DB
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    # Try to get cultural elements if available
                    try:
                        elements = await conn.fetch("""
                            SELECT name, type, description 
                            FROM CulturalElements
                            LIMIT 3
                        """)
                        culture_context = "\n".join([
                            f"- {e['name']} ({e['type']}): {e['description'][:80]}..."
                            for e in elements
                        ])
                    except:
                        culture_context = "- Matriarchal society with feminine authority structure"

                    # Try to get existing educational systems if available
                    try:
                        systems = await conn.fetch("""
                            SELECT name, system_type 
                            FROM EducationalSystems
                            LIMIT 2
                        """)
                        systems_context = ", ".join([
                            f"{s['name']} ({s['system_type']})"
                            for s in systems
                        ]) if systems else "No formal systems yet"
                    except:
                        systems_context = "No formal systems yet"

            prompt = f"""
            Generate {count} knowledge traditions for a matriarchal society.

            CULTURAL CONTEXT:
            {culture_context}

            EDUCATIONAL SYSTEMS:
            {systems_context}

            Create distinctive knowledge traditions that:
            1. Span different domains (art, craft, spiritual, practical, etc.)
            2. Use varied preservation methods
            3. Have appropriate matriarchal power dynamics
            4. Include gendered access considerations
            
            Return {count} KnowledgeTradition objects with all required fields.
            """
            
            # Create an agent with structured output
            tradition_agent = Agent(
                name="KnowledgeTraditionAgent",
                instructions="You create knowledge transmission traditions for matriarchal societies.",
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.9),
                output_type=List[KnowledgeTradition]
            )
            
            run_config = RunConfig(workflow_name="KnowledgeTraditionGeneration")
            result = await Runner.run(tradition_agent, prompt, context=run_ctx.context, run_config=run_config)

            traditions = result.final_output
            
            saved_traditions = []
            for tradition in traditions:
                try:
                    # Add the tradition to the database
                    tradition_id = await self._add_knowledge_tradition_impl(
                        run_ctx,
                        name=tradition.name,
                        tradition_type=tradition.tradition_type,
                        description=tradition.description,
                        knowledge_domain=tradition.knowledge_domain,
                        preservation_method=tradition.preservation_method,
                        access_requirements=tradition.access_requirements,
                        associated_group=tradition.associated_group,
                        examples=tradition.examples,
                        female_gatekeepers=tradition.female_gatekeepers,
                        gendered_access=tradition.gendered_access,
                        matriarchal_reinforcement=tradition.matriarchal_reinforcement
                    )
                    
                    # Add tradition to the result
                    tradition_dict = tradition.dict()
                    tradition_dict["id"] = tradition_id
                    saved_traditions.append(tradition_dict)
                    
                except Exception as e:
                    logger.error(f"Error saving knowledge tradition '{tradition.name}': {e}")

            return saved_traditions
