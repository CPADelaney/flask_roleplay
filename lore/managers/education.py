# lore/managers/education.py

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel, Field

# OpenAI Agents SDK imports
from agents import (
    Agent, function_tool, Runner, trace,
    GuardrailFunctionOutput, InputGuardrail, handoff,
    RunContextWrapper, RunConfig, ModelSettings
)

# Governance & others
from nyx.nyx_governance import AgentType, DirectivePriority, NyxUnifiedGovernor
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from nyx.directive_handler import DirectiveHandler

# Project imports
from embedding.vector_store import generate_embedding
from lore.managers.base_manager import BaseLoreManager
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# Import the matriarchal theming utilities if they exist
try:
    from lore.managers.utils import MatriarchalThemingUtils
except ImportError:
    # Create a simple placeholder if the utils don't exist
    class MatriarchalThemingUtils:
        @staticmethod
        def apply_matriarchal_theme(theme_type: str, text: str) -> str:
            return text

# ---------------------------------------------------------------------
# Context Type for SDK
# ---------------------------------------------------------------------
@dataclass
class EducationContext:
    """Context object for educational system operations"""
    user_id: int
    conversation_id: int
    manager: Optional['EducationalSystemManager'] = None
    governance_enabled: bool = True

# ---------------------------------------------------------------------
# Pydantic Models for Structured Outputs
# ---------------------------------------------------------------------
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
    
    class Config:
        extra = "forbid"

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
    
    class Config:
        extra = "forbid"

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
    
    class Config:
        extra = "forbid"

class KnowledgeExchangeResult(BaseModel):
    """Result of knowledge exchange between systems"""
    source_system: str
    target_system: str
    knowledge_domain: str
    transferable: List[Dict[str, Any]]
    requires_adaptation: List[Dict[str, Any]]
    restricted: List[Dict[str, Any]]
    recommendations: List[str]
    
    class Config:
        extra = "forbid"

class StreamingPhaseUpdate(BaseModel):
    """Update during streaming educational development"""
    phase: str
    status: str
    content: Optional[str] = None
    chunk: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "forbid"

# ---------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------
class CensorshipGuardrail(InputGuardrail):
    """Guardrail to detect and prevent inappropriate educational content."""
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
    
    async def __call__(self, ctx, agent, input_data: str) -> GuardrailFunctionOutput:
        """Check if educational content violates taboo subjects or includes inappropriate material."""
        forbidden_words = [
            "explicit", "graphic", "inappropriate", "harmful",
            "dangerous", "violent", "weapon"
        ]
        
        if self.strict_mode:
            forbidden_words.extend([
                "rebellion", "revolution", "overthrow", "resist", 
                "challenge authority", "disobey"
            ])
        
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

# ---------------------------------------------------------------------
# Educational System Manager Class
# ---------------------------------------------------------------------
class EducationalSystemManager(BaseLoreManager):
    """
    Enhanced manager for educational systems with streaming support,
    agent-to-agent knowledge exchange, structured outputs, and censorship guardrails.
    
    IMPORTANT: Governance initialization has been removed from ensure_initialized()
    to prevent circular dependencies. Governance is now handled externally by LoreSystem.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        
        # Guardrails
        self.censorship_guardrail = CensorshipGuardrail()
        self.strict_censorship_guardrail = CensorshipGuardrail(strict_mode=True)
        
        # Agents (will be initialized in initialize_agents)
        self.distribution_agent = None
        self.formal_education_agent = None
        self.apprenticeship_agent = None
        self.religious_education_agent = None
        self.education_agent = None
        self.knowledge_exchange_agent = None
    
    def _get_agent_type(self) -> AgentType:
        """Override to provide specific agent type."""
        return AgentType.NARRATIVE_CRAFTER
    
    def _get_agent_id(self) -> str:
        """Override to provide specific agent ID."""
        return "educational_system_manager"
    
    async def initialize_agents(self):
        """Initialize all specialized agents."""
        await super().initialize_agents()
        
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
        
        # Main education agent with handoffs
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
        
        logger.info(f"Educational agents initialized for user {self.user_id}")
    
    async def _initialize_tables(self):
        """Initialize database tables for educational systems."""
        table_definitions = {
            "EducationalSystems": """
                CREATE TABLE IF NOT EXISTS EducationalSystems (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
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
                    embedding VECTOR(1536),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_educationalsystems_user_conv 
                ON EducationalSystems(user_id, conversation_id);
                
                CREATE INDEX IF NOT EXISTS idx_educationalsystems_embedding 
                ON EducationalSystems USING ivfflat (embedding vector_cosine_ops);
            """,
            
            "KnowledgeTraditions": """
                CREATE TABLE IF NOT EXISTS KnowledgeTraditions (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
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
                    embedding VECTOR(1536),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_knowledgetraditions_user_conv 
                ON KnowledgeTraditions(user_id, conversation_id);
                
                CREATE INDEX IF NOT EXISTS idx_knowledgetraditions_embedding 
                ON KnowledgeTraditions USING ivfflat (embedding vector_cosine_ops);
            """,
            
            "TeachingContents": """
                CREATE TABLE IF NOT EXISTS TeachingContents (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (system_id) REFERENCES EducationalSystems(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_teachingcontents_system 
                ON TeachingContents(system_id);
                
                CREATE INDEX IF NOT EXISTS idx_teachingcontents_user_conv 
                ON TeachingContents(user_id, conversation_id);
                
                CREATE INDEX IF NOT EXISTS idx_teachingcontents_embedding 
                ON TeachingContents USING ivfflat (embedding vector_cosine_ops);
            """
        }
        
        logger.info(f"Initializing tables for {self.__class__.__name__}")
        
        # Add timeout protection to table creation
        try:
            await asyncio.wait_for(
                self.initialize_tables_for_class(table_definitions),
                timeout=10.0
            )
            logger.info(f"Table initialization complete for {self.__class__.__name__}")
        except asyncio.TimeoutError:
            logger.error(f"Table initialization timed out for {self.__class__.__name__}")
            raise RuntimeError("Table initialization timed out")
        except Exception as e:
            logger.error(f"Error during table initialization: {e}")
            raise
    
    # Internal helper methods
    async def _check_governance_permission(self, action_type: str, action_details: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check governance permission for an action."""
        if not self.governor:
            return True, None
        
        try:
            permission = await self.governor.check_action_permission(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="educational_system_manager",
                action_type=action_type,
                action_details=action_details
            )
            
            approved = permission.get("approved", True)
            reasoning = permission.get("reasoning", "Action not permitted by governance")
            return approved, reasoning
        except Exception as e:
            logger.error(f"Governance check failed: {e}")
            return True, None  # Default to allow if governance fails
    
    async def _save_educational_system_impl(self, system: EducationalSystem) -> int:
        """Internal implementation to save an educational system."""
        # Generate embedding
        embedding_text = f"{system.name} {system.system_type} {system.description} {' '.join(system.core_teachings)}"
        embedding = await generate_embedding(embedding_text)
        
        async with get_db_connection_context() as conn:
            system_id = await conn.fetchval("""
                INSERT INTO EducationalSystems (
                    user_id, conversation_id,
                    name, system_type, description, target_demographics,
                    controlled_by, core_teachings, teaching_methods, 
                    coming_of_age_rituals, knowledge_restrictions, embedding,
                    female_leadership_roles, male_roles, gender_specific_teachings,
                    taboo_subjects, censorship_level, censorship_enforcement
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                RETURNING id
            """,
            self.user_id, self.conversation_id,
            system.name, system.system_type, system.description, system.target_demographics,
            system.controlled_by, system.core_teachings, system.teaching_methods,
            system.coming_of_age_rituals, system.knowledge_restrictions, embedding,
            system.female_leadership_roles, system.male_roles, json.dumps(system.gender_specific_teachings),
            system.taboo_subjects, system.censorship_level, system.censorship_enforcement)
        
        # Cache the system
        cache_key = f"edu_system_{system_id}"
        self.set_cache(cache_key, system.dict())
        
        return system_id
    
    async def _save_knowledge_tradition_impl(self, tradition: KnowledgeTradition) -> int:
        """Internal implementation to save a knowledge tradition."""
        # Generate embedding
        embedding_text = f"{tradition.name} {tradition.tradition_type} {tradition.description} {tradition.knowledge_domain}"
        embedding = await generate_embedding(embedding_text)
        
        async with get_db_connection_context() as conn:
            tradition_id = await conn.fetchval("""
                INSERT INTO KnowledgeTraditions (
                    user_id, conversation_id,
                    name, tradition_type, description, knowledge_domain,
                    preservation_method, access_requirements,
                    associated_group, examples, embedding,
                    female_gatekeepers, gendered_access, matriarchal_reinforcement
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                RETURNING id
            """,
            self.user_id, self.conversation_id,
            tradition.name, tradition.tradition_type, tradition.description, tradition.knowledge_domain,
            tradition.preservation_method, tradition.access_requirements,
            tradition.associated_group, tradition.examples, embedding,
            tradition.female_gatekeepers, json.dumps(tradition.gendered_access), tradition.matriarchal_reinforcement)
        
        # Cache the tradition
        cache_key = f"knowledge_tradition_{tradition_id}"
        self.set_cache(cache_key, tradition.dict())
        
        return tradition_id
    
    async def _save_teaching_content_impl(self, system_id: int, content: TeachingContent) -> int:
        """Internal implementation to save teaching content."""
        # Check content with guardrail
        guardrail_result = await self.censorship_guardrail(None, None, content.description)
        if guardrail_result.tripwire_triggered:
            content.restricted = True
            content.restriction_reason = str(guardrail_result.output_info.get("forbidden_words", []))
        
        # Generate embedding
        embedding_text = f"{content.title} {content.description} {' '.join(content.key_points)}"
        embedding = await generate_embedding(embedding_text)
        
        async with get_db_connection_context() as conn:
            content_id = await conn.fetchval("""
                INSERT INTO TeachingContents (
                    user_id, conversation_id, system_id,
                    title, content_type, subject_area, description,
                    target_age_group, key_points, examples, exercises,
                    restricted, restriction_reason, embedding
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                RETURNING id
            """,
            self.user_id, self.conversation_id, system_id,
            content.title, content.content_type, content.subject_area, content.description,
            content.target_age_group, content.key_points, content.examples, 
            json.dumps(content.exercises), content.restricted, content.restriction_reason, embedding)
        
        return content_id
    
    async def _generate_educational_systems_impl(self, count: int) -> List[EducationalSystem]:
        """Internal implementation for generating educational systems."""
        # Fetch context from database
        async with get_db_connection_context() as conn:
            # Try to fetch faction info if available
            try:
                factions = await conn.fetch("""
                    SELECT name, type FROM Factions
                    WHERE user_id = $1 AND conversation_id = $2
                    LIMIT 5
                """, self.user_id, self.conversation_id)
                faction_names = [f"{f['name']} ({f['type']})" for f in factions]
            except:
                faction_names = ["Sisterhood of the Moon", "Matriarchal Council", "Forest Tribe"]
            
            # Try to fetch cultural elements if available  
            try:
                elements = await conn.fetch("""
                    SELECT name, type FROM CulturalElements
                    WHERE user_id = $1 AND conversation_id = $2
                    LIMIT 5
                """, self.user_id, self.conversation_id)
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
            context={"manager": self},
            run_config=run_config
        )
        
        return result.final_output
    
    def create_context(self) -> EducationContext:
        """Create a context object for agent execution."""
        return EducationContext(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            manager=self,
            governance_enabled=self.governor is not None
        )

    # Public methods for external use
    async def generate_educational_systems(self, _=None) -> List[Dict[str, Any]]:
        """
        Generate educational systems.
        This is a public method that can be called by LoreSystem.
        """
        # Run the generate_educational_systems tool function with proper context
        ctx = RunContextWrapper(context=self.create_context().__dict__)
        return await generate_educational_systems(ctx)
    
    async def generate_knowledge_traditions(self, _=None) -> List[Dict[str, Any]]:
        """
        Generate knowledge traditions.
        This is a public method that can be called by LoreSystem.
        """
        # Run the generate_knowledge_traditions tool function with proper context
        ctx = RunContextWrapper(context=self.create_context().__dict__)
        return await generate_knowledge_traditions(ctx)

# ---------------------------------------------------------------------
# Global Registry
# ---------------------------------------------------------------------
_education_managers = {}

async def get_education_manager(user_id: int, conversation_id: int) -> EducationalSystemManager:
    """Get or create an education manager instance."""
    key = f"{user_id}:{conversation_id}"
    if key not in _education_managers:
        manager = EducationalSystemManager(user_id, conversation_id)
        await manager.ensure_initialized()
        _education_managers[key] = manager
    return _education_managers[key]

# ---------------------------------------------------------------------
# Standalone Tool Functions
# ---------------------------------------------------------------------

@function_tool(strict_mode=False)
async def add_educational_system(
    ctx: RunContextWrapper[EducationContext],
    name: str,
    system_type: str,
    description: str,
    target_demographics: List[str],
    controlled_by: str,
    core_teachings: List[str],
    teaching_methods: List[str],
    coming_of_age_rituals: Optional[str] = None,
    knowledge_restrictions: Optional[str] = None,
    female_leadership_roles: Optional[List[str]] = None,
    male_roles: Optional[List[str]] = None,
    gender_specific_teachings: Optional[Dict[str, List[str]]] = None,
    taboo_subjects: Optional[List[str]] = None,
    censorship_level: int = 5,
    censorship_enforcement: Optional[str] = None
) -> Dict[str, Any]:
    """Add an educational system using canon system."""
    manager = await get_education_manager(ctx.context.user_id, ctx.context.conversation_id)
    
    # Check governance permission
    approved, reasoning = await manager._check_governance_permission(
        "add_educational_system",
        {"name": name, "type": system_type}
    )
    if not approved:
        return {"error": reasoning}
    
    with trace("AddEducationalSystem", metadata={"system_name": name}):
        # Apply matriarchal theming
        description = MatriarchalThemingUtils.apply_matriarchal_theme("education", description)
        
        # Set defaults for matriarchal elements
        female_leadership_roles = female_leadership_roles or ["Headmistress", "Teacher", "Mentor"]
        male_roles = male_roles or ["Assistant", "Aide", "Custodian"]
        gender_specific_teachings = gender_specific_teachings or {
            "female": ["Leadership", "Authority", "Decision-making"],
            "male": ["Service", "Support", "Compliance"]
        }
        taboo_subjects = taboo_subjects or ["Challenging feminine authority", "Male independence"]
        censorship_enforcement = censorship_enforcement or "Monitored by female leadership"
        
        # Prepare data package for canon
        system_data = {
            "name": name,
            "system_type": system_type,
            "description": description,
            "target_demographics": target_demographics,
            "controlled_by": controlled_by,
            "core_teachings": core_teachings,
            "teaching_methods": teaching_methods,
            "coming_of_age_rituals": coming_of_age_rituals,
            "knowledge_restrictions": knowledge_restrictions,
            "female_leadership_roles": female_leadership_roles,
            "male_roles": male_roles,
            "gender_specific_teachings": gender_specific_teachings,
            "taboo_subjects": taboo_subjects,
            "censorship_level": censorship_level,
            "censorship_enforcement": censorship_enforcement
        }
        
        # Use canon to create the system
        async with get_db_connection_context() as conn:
            from lore.core import canon
            
            # First establish controlling faction if specified
            if controlled_by:
                faction_id = await canon.find_or_create_faction(
                    ctx, conn, controlled_by, faction_type="educational_authority"
                )
            
            # Create the educational system
            system_id = await canon.find_or_create_educational_system(
                ctx, conn, **system_data
            )
        
        # Cache the system
        cache_key = f"edu_system_{system_id}"
        manager.set_cache(cache_key, system_data)
        
        return {"id": system_id, "status": "created"}

@function_tool(strict_mode=False)
async def add_knowledge_tradition(
    ctx: RunContextWrapper[EducationContext],
    name: str,
    tradition_type: str,
    description: str,
    knowledge_domain: str,
    preservation_method: Optional[str] = None,
    access_requirements: Optional[str] = None,
    associated_group: Optional[str] = None,
    examples: Optional[List[str]] = None,
    female_gatekeepers: bool = True,
    gendered_access: Optional[Dict[str, str]] = None,
    matriarchal_reinforcement: Optional[str] = None
) -> Dict[str, Any]:
    """Add a knowledge tradition using canon system."""
    manager = await get_education_manager(ctx.context.user_id, ctx.context.conversation_id)
    
    # Check governance permission
    approved, reasoning = await manager._check_governance_permission(
        "add_knowledge_tradition",
        {"name": name, "type": tradition_type}
    )
    if not approved:
        return {"error": reasoning}
    
    with trace("AddKnowledgeTradition", metadata={"tradition_name": name}):
        # Apply theming
        description = MatriarchalThemingUtils.apply_matriarchal_theme("tradition", description)
        
        # Set defaults
        examples = examples or []
        gendered_access = gendered_access or {
            "female": "Full access with advancement opportunities",
            "male": "Limited access under supervision"
        }
        matriarchal_reinforcement = matriarchal_reinforcement or "Emphasizes female wisdom and authority"
        
        # Prepare data for canon
        tradition_data = {
            "name": name,
            "tradition_type": tradition_type,
            "description": description,
            "knowledge_domain": knowledge_domain,
            "preservation_method": preservation_method,
            "access_requirements": access_requirements,
            "associated_group": associated_group,
            "examples": examples,
            "female_gatekeepers": female_gatekeepers,
            "gendered_access": gendered_access,
            "matriarchal_reinforcement": matriarchal_reinforcement
        }
        
        # Use canon to create
        async with get_db_connection_context() as conn:
            from lore.core import canon
            tradition_id = await canon.create_knowledge_tradition(ctx, conn, **tradition_data)
        
        # Cache
        cache_key = f"knowledge_tradition_{tradition_id}"
        manager.set_cache(cache_key, tradition_data)
        
        return {"id": tradition_id, "status": "created"}
        
@function_tool(strict_mode=False)
async def add_teaching_content(
    ctx: RunContextWrapper[EducationContext],
    system_id: int,
    title: str,
    content_type: str,
    subject_area: str,
    description: str,
    target_age_group: str,
    key_points: List[str],
    examples: Optional[List[str]] = None,
    exercises: Optional[List[Dict[str, Any]]] = None,
    restricted: bool = False,
    restriction_reason: Optional[str] = None
) -> Dict[str, Any]:
    """Add teaching content using canon system."""
    manager = await get_education_manager(ctx.context.user_id, ctx.context.conversation_id)
    
    # Check governance permission
    approved, reasoning = await manager._check_governance_permission(
        "add_teaching_content",
        {"system_id": system_id, "title": title}
    )
    if not approved:
        return {"error": reasoning}
    
    with trace("AddTeachingContent", metadata={"system_id": system_id, "title": title}):
        async with get_db_connection_context() as conn:
            # Verify system exists
            system = await conn.fetchrow("""
                SELECT * FROM EducationalSystems 
                WHERE id = $1 AND user_id = $2 AND conversation_id = $3
            """, system_id, ctx.context.user_id, ctx.context.conversation_id)
            
            if not system:
                return {"error": "Educational system not found"}
            
            # Check against taboo subjects
            if system['taboo_subjects']:
                for taboo in system['taboo_subjects']:
                    if taboo.lower() in description.lower() or any(taboo.lower() in kp.lower() for kp in key_points):
                        restricted = True
                        restriction_reason = f"Contains taboo subject: {taboo}"
                        
                        # Log this as canonical event
                        from lore.core import canon
                        await canon.log_canonical_event(
                            ctx, conn,
                            f"Teaching content '{title}' flagged for containing taboo subject in {system['name']}",
                            tags=["education", "censorship", "canon"],
                            significance=4
                        )
            
            # Check content with guardrail
            guardrail_result = await manager.censorship_guardrail(None, None, description)
            if guardrail_result.tripwire_triggered:
                restricted = True
                restriction_reason = str(guardrail_result.output_info.get("forbidden_words", []))
            
            # Apply theming
            description = MatriarchalThemingUtils.apply_matriarchal_theme("content", description)
            
            # Prepare data for canon
            content_data = {
                "title": title,
                "content_type": content_type,
                "subject_area": subject_area,
                "description": description,
                "target_age_group": target_age_group,
                "key_points": key_points,
                "examples": examples or [],
                "exercises": exercises or [],
                "restricted": restricted,
                "restriction_reason": restriction_reason
            }
            
            # Use canon to create
            from lore.core import canon
            content_id = await canon.create_teaching_content(ctx, conn, system_id, **content_data)
            
            # Update system if significant
            if len(key_points) > 5 or subject_area in ["leadership", "governance", "authority"]:
                from lore.core.lore_system import LoreSystem
                lore_system = await LoreSystem.get_instance(ctx.context.user_id, ctx.context.conversation_id)
                
                await lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="EducationalSystems",
                    entity_identifier={"id": system_id},
                    updates={"last_content_update": "CURRENT_TIMESTAMP"},
                    reason=f"Added significant teaching content: {title}"
                )
        
        return {"id": content_id, "status": "created", "restricted": restricted}
        
@function_tool
async def stream_educational_development(
    ctx: RunContextWrapper[EducationContext],
    system_name: str,
    system_type: str,
    matriarchy_level: int = 8
) -> AsyncGenerator[StreamingPhaseUpdate, None]:
    """
    Stream the development of a complete educational system with progressive updates.
    
    Yields:
        StreamingPhaseUpdate objects for each phase
    """
    manager = await get_education_manager(ctx.context.user_id, ctx.context.conversation_id)
    
    # Check governance permission
    approved, reasoning = await manager._check_governance_permission(
        "stream_educational_development",
        {"system_name": system_name, "system_type": system_type}
    )
    if not approved:
        yield StreamingPhaseUpdate(
            phase="error",
            status="governance_denied",
            content=reasoning
        )
        return
    
    with trace("StreamEducationalDevelopment", metadata={
        "system_name": system_name,
        "system_type": system_type,
        "matriarchy_level": matriarchy_level
    }):
        # Determine which specialized agent to use
        specialist_map = {
            "formal": manager.formal_education_agent,
            "school": manager.formal_education_agent,
            "academy": manager.formal_education_agent,
            "apprenticeship": manager.apprenticeship_agent,
            "mentorship": manager.apprenticeship_agent,
            "guild": manager.apprenticeship_agent,
            "religious": manager.religious_education_agent,
            "seminary": manager.religious_education_agent,
            "spiritual": manager.religious_education_agent
        }
        specialist = specialist_map.get(system_type.lower(), manager.education_agent)
        
        # Phase 1: Basic system outline
        yield StreamingPhaseUpdate(
            phase="outline",
            status="starting",
            content=f"Developing {system_name} - a {system_type} educational system..."
        )
        
        basic_prompt = f"""
        Create an outline for a {system_type} educational system named '{system_name}' with 
        a matriarchy level of {matriarchy_level}/10. Include:
        
        1. Core purpose
        2. Target demographics
        3. Main features
        4. Matriarchal authority structure
        """
        
        basic_result = await Runner.run(specialist, basic_prompt, context={"manager": manager})
        basic_outline = basic_result.final_output
        
        yield StreamingPhaseUpdate(
            phase="outline",
            status="complete",
            content=basic_outline
        )
        await asyncio.sleep(0.5)
        
        # Phase 2: Leadership structure with streaming
        yield StreamingPhaseUpdate(
            phase="leadership",
            status="starting",
            content="Developing leadership structure..."
        )
        
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
        
        # Use streaming for this phase
        stream_config = RunConfig(
            workflow_name="LeadershipStructureGeneration",
            trace_metadata={"component": "leadership_structure"}
        )
        
        streaming_result = Runner.run_streamed(
            specialist, 
            leadership_prompt, 
            context={"manager": manager},
            run_config=stream_config
        )
        
        # Process streaming events
        leadership_content = []
        async for event in streaming_result.stream_events():
            if event.type == "run_item_stream_event":
                if event.item.type == "message_output_item":
                    # Extract text from the message output
                    from agents.items import ItemHelpers
                    message_text = ItemHelpers.text_message_output(event.item)
                    leadership_content.append(message_text)
                    yield StreamingPhaseUpdate(
                        phase="leadership",
                        status="streaming",
                        chunk=message_text
                    )
        
        # Get the final result
        final_result = await streaming_result
        leadership_structure = "".join(leadership_content) if leadership_content else final_result.final_output
        
        yield StreamingPhaseUpdate(
            phase="leadership",
            status="complete",
            content=leadership_structure
        )
        await asyncio.sleep(0.5)
        
        # Phase 3: Curriculum and teaching methods
        yield StreamingPhaseUpdate(
            phase="curriculum",
            status="starting",
            content="Developing curriculum and teaching methods..."
        )
        
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
        
        curriculum_result = await Runner.run(specialist, curriculum_prompt, context={"manager": manager})
        curriculum = curriculum_result.final_output
        
        yield StreamingPhaseUpdate(
            phase="curriculum",
            status="complete",
            content=curriculum
        )
        await asyncio.sleep(0.5)
        
        # Phase 4: Knowledge restrictions and taboos
        yield StreamingPhaseUpdate(
            phase="restrictions",
            status="starting",
            content="Developing knowledge restrictions and taboo subjects..."
        )
        
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
        
        # Apply the censorship guardrail to this phase
        restrictions_agent = specialist.clone(
            input_guardrails=[manager.censorship_guardrail]
        )
        
        restrictions_result = await Runner.run(
            restrictions_agent, 
            restrictions_prompt, 
            context={"manager": manager}
        )
        restrictions = restrictions_result.final_output
        
        yield StreamingPhaseUpdate(
            phase="restrictions",
            status="complete",
            content=restrictions
        )
        await asyncio.sleep(0.5)
        
        # Phase 5: Compile and finalize
        yield StreamingPhaseUpdate(
            phase="finalizing",
            status="starting",
            content="Compiling complete educational system definition..."
        )
        
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
        system_definition_agent = specialist.clone(
            output_type=EducationalSystem
        )
        
        definition_result = await Runner.run(
            system_definition_agent, 
            complete_prompt, 
            context={"manager": manager}
        )
        system_definition = definition_result.final_output
        
        # Save the educational system
        try:
            system_id = await manager._save_educational_system_impl(system_definition)
            
            yield StreamingPhaseUpdate(
                phase="complete",
                status="saved",
                content=f"Educational system saved to database with ID: {system_id}",
                metadata={
                    "system_id": system_id,
                    "system": system_definition.dict()
                }
            )
        except Exception as e:
            yield StreamingPhaseUpdate(
                phase="error",
                status="save_failed",
                content=f"Error saving educational system: {str(e)}"
            )

@function_tool
async def exchange_knowledge_between_systems(
    ctx: RunContextWrapper[EducationContext],
    source_system_id: int,
    target_system_id: int,
    knowledge_domain: str
) -> Dict[str, Any]:
    """
    Facilitate knowledge exchange between two educational systems.
    
    Returns:
        Dictionary with exchange results
    """
    manager = await get_education_manager(ctx.context.user_id, ctx.context.conversation_id)
    
    # Check governance permission
    approved, reasoning = await manager._check_governance_permission(
        "exchange_knowledge_between_systems",
        {"source_id": source_system_id, "target_id": target_system_id}
    )
    if not approved:
        return {"error": reasoning}
    
    with trace("KnowledgeExchange", metadata={
        "source_id": source_system_id,
        "target_id": target_system_id,
        "domain": knowledge_domain
    }):
        # Fetch system data and teaching contents
        async with get_db_connection_context() as conn:
            source_system = await conn.fetchrow("""
                SELECT * FROM EducationalSystems 
                WHERE id = $1 AND user_id = $2 AND conversation_id = $3
            """, source_system_id, ctx.context.user_id, ctx.context.conversation_id)
            
            target_system = await conn.fetchrow("""
                SELECT * FROM EducationalSystems 
                WHERE id = $1 AND user_id = $2 AND conversation_id = $3
            """, target_system_id, ctx.context.user_id, ctx.context.conversation_id)
            
            if not source_system or not target_system:
                return {"error": "One or both systems not found"}
            
            # Get teaching contents from source system
            contents = await conn.fetch("""
                SELECT * FROM TeachingContents 
                WHERE system_id = $1 AND subject_area = $2 
                AND user_id = $3 AND conversation_id = $4
            """, source_system_id, knowledge_domain, ctx.context.user_id, ctx.context.conversation_id)
            
            source_contents = [dict(c) for c in contents]
        
        # Prepare data for analysis
        source_data = dict(source_system)
        target_data = dict(target_system)
        
        # Parse JSON fields
        for data in [source_data, target_data]:
            if data.get("gender_specific_teachings"):
                try:
                    data["gender_specific_teachings"] = json.loads(data["gender_specific_teachings"])
                except:
                    data["gender_specific_teachings"] = {}
        
        # Knowledge exchange prompt
        exchange_prompt = f"""
        Facilitate knowledge exchange between these two educational systems:
        
        SOURCE SYSTEM:
        {json.dumps(source_data, indent=2, default=str)}
        
        TARGET SYSTEM:
        {json.dumps(target_data, indent=2, default=str)}
        
        KNOWLEDGE DOMAIN:
        {knowledge_domain}
        
        AVAILABLE CONTENT:
        {json.dumps(source_contents, indent=2, default=str)}
        
        Analyze and return a KnowledgeExchangeResult with:
        1. transferable: Knowledge that can be transferred directly
        2. requires_adaptation: Knowledge needing modification
        3. restricted: Knowledge that cannot be shared
        4. recommendations: Specific recommendations for the exchange
        """
        
        # Use the knowledge exchange agent with structured output
        exchange_agent = manager.knowledge_exchange_agent.clone(
            output_type=KnowledgeExchangeResult
        )
        
        result = await Runner.run(exchange_agent, exchange_prompt, context={"manager": manager})
        exchange_results = result.final_output
        
        # Actually implement the knowledge transfer based on results
        transferred_count = 0
        adapted_count = 0
        restricted_count = 0
        
        async with get_db_connection_context() as conn:
            # Transfer content marked as transferable
            for item in exchange_results.transferable:
                if isinstance(item, dict) and "id" in item:
                    # Copy the teaching content to target system
                    original = await conn.fetchrow("""
                        SELECT * FROM TeachingContents WHERE id = $1
                    """, item["id"])
                    
                    if original:
                        new_content = TeachingContent(
                            title=f"{original['title']} (from {source_data['name']})",
                            content_type=original['content_type'],
                            subject_area=original['subject_area'],
                            description=original['description'],
                            target_age_group=original['target_age_group'],
                            key_points=original['key_points'],
                            examples=original['examples'],
                            exercises=json.loads(original['exercises']) if original['exercises'] else [],
                            restricted=False,
                            restriction_reason=None
                        )
                        
                        await manager._save_teaching_content_impl(target_system_id, new_content)
                        transferred_count += 1
            
            # Count adapted and restricted items
            adapted_count = len(exchange_results.requires_adaptation)
            restricted_count = len(exchange_results.restricted)
        
        return {
            "source_system": source_data["name"],
            "target_system": target_data["name"],
            "knowledge_domain": knowledge_domain,
            "exchange_results": exchange_results.dict(),
            "transferred_count": transferred_count,
            "adapted_count": adapted_count,
            "restricted_count": restricted_count
        }

@function_tool
async def generate_educational_systems(
    ctx: RunContextWrapper[EducationContext]
) -> List[Dict[str, Any]]:
    """Generate educational systems using canon system."""
    manager = await get_education_manager(ctx.context.user_id, ctx.context.conversation_id)
    
    # Check governance permission
    approved, reasoning = await manager._check_governance_permission(
        "generate_educational_systems",
        {}
    )
    if not approved:
        return [{"error": reasoning}]
    
    with trace("GenerateEducationalSystems", metadata={"user_id": ctx.context.user_id}):
        # Determine count
        dist_result = await Runner.run(
            manager.distribution_agent, 
            "How many educational systems should we generate? Return JSON with 'count'.",
            context={"manager": manager}
        )
        
        try:
            count = json.loads(dist_result.final_output).get("count", 3)
        except:
            count = 3
        
        # Generate systems
        systems = await manager._generate_educational_systems_impl(count)
        
        # Save all systems using canon
        saved_systems = []
        async with get_db_connection_context() as conn:
            from lore.core import canon
            
            for system in systems:
                try:
                    # First establish controlling faction
                    if system.controlled_by:
                        faction_id = await canon.find_or_create_faction(
                            ctx, conn, system.controlled_by, faction_type="educational_authority"
                        )
                    
                    # Apply theming
                    system.description = MatriarchalThemingUtils.apply_matriarchal_theme(
                        "education", system.description
                    )
                    
                    # Create via canon
                    system_id = await canon.find_or_create_educational_system(
                        ctx, conn, **system.dict()
                    )
                    
                    system_dict = system.dict()
                    system_dict["id"] = system_id
                    saved_systems.append(system_dict)
                    
                except Exception as e:
                    logger.error(f"Error saving educational system '{system.name}': {e}")
        
        return saved_systems

@function_tool
async def generate_knowledge_traditions(
    ctx: RunContextWrapper[EducationContext]
) -> List[Dict[str, Any]]:
    """Generate knowledge traditions using canon system."""
    manager = await get_education_manager(ctx.context.user_id, ctx.context.conversation_id)
    
    # Check governance permission
    approved, reasoning = await manager._check_governance_permission(
        "generate_knowledge_traditions",
        {}
    )
    if not approved:
        return [{"error": reasoning}]
    
    with trace("GenerateKnowledgeTraditions", metadata={"user_id": ctx.context.user_id}):
        # Let the LLM decide how many traditions to generate
        dist_prompt = (
            "We want to create a set of knowledge traditions for a matriarchal setting. "
            "How many should we make? Return JSON with a 'count' field. Example: {\"count\": 4}"
        )
        
        dist_config = RunConfig(workflow_name="KnowledgeTraditionCount")
        dist_result = await Runner.run(
            manager.distribution_agent, 
            dist_prompt, 
            context={"manager": manager}, 
            run_config=dist_config
        )
        
        try:
            dist_data = json.loads(dist_result.final_output)
            count = dist_data.get("count", 4)
        except json.JSONDecodeError:
            count = 4
        
        # Get context from database
        async with get_db_connection_context() as conn:
            # Try to get cultural elements
            try:
                elements = await conn.fetch("""
                    SELECT name, type, description 
                    FROM CulturalElements
                    WHERE user_id = $1 AND conversation_id = $2
                    LIMIT 3
                """, ctx.context.user_id, ctx.context.conversation_id)
                culture_context = "\n".join([
                    f"- {e['name']} ({e['type']}): {e['description'][:80]}..."
                    for e in elements
                ])
            except:
                culture_context = "- Matriarchal society with feminine authority structure"
            
            # Try to get existing educational systems
            try:
                systems = await conn.fetch("""
                    SELECT name, system_type 
                    FROM EducationalSystems
                    WHERE user_id = $1 AND conversation_id = $2
                    LIMIT 2
                """, ctx.context.user_id, ctx.context.conversation_id)
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
        result = await Runner.run(tradition_agent, prompt, context={"manager": manager})
        traditions = result.final_output
        
        # Save all traditions using canon
        saved_traditions = []
        async with get_db_connection_context() as conn:
            from lore.core import canon
            
            for tradition in traditions:
                try:
                    # Apply theming
                    tradition.description = MatriarchalThemingUtils.apply_matriarchal_theme(
                        "tradition", tradition.description
                    )
                    
                    # Create via canon
                    tradition_id = await canon.create_knowledge_tradition(
                        ctx, conn, **tradition.dict()
                    )
                    
                    tradition_dict = tradition.dict()
                    tradition_dict["id"] = tradition_id
                    saved_traditions.append(tradition_dict)
                    
                except Exception as e:
                    logger.error(f"Error saving knowledge tradition '{tradition.name}': {e}")
        
        return saved_traditions

@function_tool
async def search_educational_systems(
    ctx: RunContextWrapper[EducationContext],
    query: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Search educational systems by semantic similarity.
    
    Returns:
        List of matching systems with similarity scores
    """
    manager = await get_education_manager(ctx.context.user_id, ctx.context.conversation_id)
    
    # Check cache first
    cache_key = f"edu_search_{hash(query)}_{limit}"
    cached_results = manager.get_cache(cache_key)
    if cached_results:
        return cached_results
    
    # Generate embedding for query
    query_embedding = await generate_embedding(query)
    
    async with get_db_connection_context() as conn:
        results = await conn.fetch("""
            SELECT *, 1 - (embedding <=> $1) as similarity
            FROM EducationalSystems
            WHERE user_id = $2 AND conversation_id = $3 AND embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT $4
        """, query_embedding, ctx.context.user_id, ctx.context.conversation_id, limit)
        
        search_results = [dict(r) for r in results]
        
        # Cache results
        manager.set_cache(cache_key, search_results, ttl=300)
        
        return search_results

@function_tool
async def get_teaching_contents(
    ctx: RunContextWrapper[EducationContext],
    system_id: int,
    subject_area: Optional[str] = None,
    include_restricted: bool = False
) -> List[Dict[str, Any]]:
    """
    Get teaching contents for an educational system.
    
    Returns:
        List of teaching contents
    """
    manager = await get_education_manager(ctx.context.user_id, ctx.context.conversation_id)
    
    query = """
        SELECT * FROM TeachingContents
        WHERE system_id = $1 AND user_id = $2 AND conversation_id = $3
    """
    params = [system_id, ctx.context.user_id, ctx.context.conversation_id]
    
    if subject_area:
        query += " AND subject_area = $4"
        params.append(subject_area)
    
    if not include_restricted:
        query += " AND restricted = FALSE"
    
    query += " ORDER BY created_at DESC"
    
    async with get_db_connection_context() as conn:
        results = await conn.fetch(query, *params)
        
        return [dict(r) for r in results]

# ---------------------------------------------------------------------
# Agent Creation Functions
# ---------------------------------------------------------------------

def create_education_orchestrator() -> Agent[EducationContext]:
    """Create the main orchestration agent for education management."""
    return Agent[EducationContext](
        name="Education Orchestrator",
        instructions="""
        You orchestrate educational system management for a matriarchal RPG setting.
        You can:
        - Create and manage educational systems and knowledge traditions
        - Generate teaching content with appropriate restrictions
        - Facilitate knowledge exchange between systems
        - Search and analyze educational infrastructure
        
        Always consider matriarchal power dynamics and gender-based access in all operations.
        """,
        tools=[
            add_educational_system,
            add_knowledge_tradition,
            add_teaching_content,
            generate_educational_systems,
            generate_knowledge_traditions,
            exchange_knowledge_between_systems,
            search_educational_systems,
            get_teaching_contents
        ],
        model="gpt-4.1-nano",
        model_settings=ModelSettings(temperature=0.7)
    )

# ---------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------

async def cleanup_education_managers():
    """Close all education manager instances."""
    for manager in _education_managers.values():
        await manager.close()
    _education_managers.clear()
