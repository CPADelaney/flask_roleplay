# lore/managers/education.py

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
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
from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

# Project imports
from embedding.vector_store import generate_embedding
from lore.managers.base_manager import BaseLoreManager
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Context Type for SDK
# ---------------------------------------------------------------------
@dataclass
class EducationContext:
    """Context object for educational system operations"""
    user_id: int
    conversation_id: int
    manager: Optional['EducationalSystemManager'] = None

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
    transferable: List[str]
    requires_adaptation: List[str]
    restricted: List[str]
    transferred_count: int = 0
    adapted_count: int = 0
    restricted_count: int = 0
    
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
    Manager for educational systems in the matriarchal setting.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.censorship_guardrail = CensorshipGuardrail()
        self.strict_censorship_guardrail = CensorshipGuardrail(strict_mode=True)
        
        # Initialize agents
        self.distribution_agent = None
        self.formal_education_agent = None
        self.apprenticeship_agent = None
        self.religious_education_agent = None
        self.education_agent = None
        self.knowledge_exchange_agent = None
    
    async def initialize_agents(self):
        """Initialize all specialized agents."""
        await super().initialize_agents()
        
        # Distribution agent
        self.distribution_agent = Agent(
            name="EducationDistributionAgent",
            instructions=(
                "You decide how many educational systems or knowledge traditions to create. "
                "Return valid JSON with a 'count' field. Example: {\"count\": 4}"
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
                    tool_description_override="Design formal educational systems"
                ),
                handoff(
                    self.apprenticeship_agent,
                    tool_name_override="design_apprenticeship",
                    tool_description_override="Design apprenticeship systems"
                ),
                handoff(
                    self.religious_education_agent,
                    tool_name_override="design_religious_education",
                    tool_description_override="Design religious educational systems"
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
            model_settings=ModelSettings(temperature=0.7),
            output_type=KnowledgeExchangeResult
        )
    
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
                
                CREATE INDEX IF NOT EXISTS idx_teachingcontents_embedding 
                ON TeachingContents USING ivfflat (embedding vector_cosine_ops);
            """
        }
        
        await self.initialize_tables_from_definitions(table_definitions)
    
    # Internal methods
    async def _save_educational_system(self, system: EducationalSystem) -> int:
        """Save an educational system to the database."""
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
        
        return system_id
    
    async def _save_knowledge_tradition(self, tradition: KnowledgeTradition) -> int:
        """Save a knowledge tradition to the database."""
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
        
        return tradition_id
    
    def create_context(self) -> EducationContext:
        """Create a context object for agent execution."""
        return EducationContext(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            manager=self
        )

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
    system: Dict[str, Any]
) -> int:
    """
    Add an educational system to the database.
    
    Args:
        ctx: Context with user/conversation info
        system: Educational system data
        
    Returns:
        ID of created system
    """
    manager = await get_education_manager(ctx.context.user_id, ctx.context.conversation_id)
    
    # Convert dict to Pydantic model
    edu_system = EducationalSystem(**system)
    
    # Save to database
    return await manager._save_educational_system(edu_system)

@function_tool(strict_mode=False)
async def add_knowledge_tradition(
    ctx: RunContextWrapper[EducationContext],
    tradition: Dict[str, Any]
) -> int:
    """
    Add a knowledge tradition to the database.
    
    Args:
        ctx: Context with user/conversation info
        tradition: Knowledge tradition data
        
    Returns:
        ID of created tradition
    """
    manager = await get_education_manager(ctx.context.user_id, ctx.context.conversation_id)
    
    # Convert dict to Pydantic model
    knowledge_tradition = KnowledgeTradition(**tradition)
    
    # Save to database
    return await manager._save_knowledge_tradition(knowledge_tradition)

@function_tool
async def generate_educational_systems(
    ctx: RunContextWrapper[EducationContext],
    count: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Generate educational systems using AI agents.
    
    Args:
        ctx: Context with user/conversation info
        count: Number of systems to generate (None = AI decides)
        
    Returns:
        List of generated systems with IDs
    """
    manager = await get_education_manager(ctx.context.user_id, ctx.context.conversation_id)
    
    with trace("GenerateEducationalSystems", metadata={"user_id": ctx.context.user_id}):
        # Determine count if not provided
        if count is None:
            dist_result = await Runner.run(
                manager.distribution_agent,
                "How many educational systems should we create for a matriarchal society? (1-6)",
                context={"manager": manager}
            )
            try:
                count = json.loads(dist_result.final_output).get("count", 3)
            except:
                count = 3
        
        # Generate systems
        prompt = f"""
        Generate {count} educational systems for a matriarchal society.
        
        For each system:
        1. Choose type: formal, apprenticeship, or religious
        2. Create complete design with matriarchal themes
        3. Include gender dynamics and restrictions
        4. Define censorship and taboo subjects
        
        Return as List[EducationalSystem].
        """
        
        # Use education agent with structured output
        systems_agent = manager.education_agent.clone(
            output_type=List[EducationalSystem]
        )
        
        result = await Runner.run(systems_agent, prompt, context={"manager": manager})
        systems = result.final_output
        
        # Save all systems
        saved_systems = []
        for system in systems:
            try:
                system_id = await manager._save_educational_system(system)
                system_dict = system.dict()
                system_dict["id"] = system_id
                saved_systems.append(system_dict)
            except Exception as e:
                logger.error(f"Error saving system {system.name}: {e}")
        
        return saved_systems

@function_tool
async def generate_knowledge_traditions(
    ctx: RunContextWrapper[EducationContext],
    count: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Generate knowledge traditions using AI agents.
    
    Args:
        ctx: Context with user/conversation info
        count: Number of traditions to generate (None = AI decides)
        
    Returns:
        List of generated traditions with IDs
    """
    manager = await get_education_manager(ctx.context.user_id, ctx.context.conversation_id)
    
    with trace("GenerateKnowledgeTraditions", metadata={"user_id": ctx.context.user_id}):
        # Determine count if not provided
        if count is None:
            dist_result = await Runner.run(
                manager.distribution_agent,
                "How many knowledge traditions should we create? (1-6)",
                context={"manager": manager}
            )
            try:
                count = json.loads(dist_result.final_output).get("count", 4)
            except:
                count = 4
        
        # Generate traditions
        prompt = f"""
        Generate {count} knowledge traditions for a matriarchal society.
        
        Create diverse traditions spanning:
        - Different domains (craft, spiritual, practical)
        - Various preservation methods
        - Matriarchal gatekeeping
        - Gender-based access rules
        
        Return as List[KnowledgeTradition].
        """
        
        # Create agent with structured output
        traditions_agent = Agent(
            name="TraditionsGenerator",
            instructions="Generate knowledge traditions for matriarchal societies.",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9),
            output_type=List[KnowledgeTradition]
        )
        
        result = await Runner.run(traditions_agent, prompt, context={"manager": manager})
        traditions = result.final_output
        
        # Save all traditions
        saved_traditions = []
        for tradition in traditions:
            try:
                tradition_id = await manager._save_knowledge_tradition(tradition)
                tradition_dict = tradition.dict()
                tradition_dict["id"] = tradition_id
                saved_traditions.append(tradition_dict)
            except Exception as e:
                logger.error(f"Error saving tradition {tradition.name}: {e}")
        
        return saved_traditions

@function_tool
async def exchange_knowledge_between_systems(
    ctx: RunContextWrapper[EducationContext],
    source_system_id: int,
    target_system_id: int,
    knowledge_domain: str
) -> Dict[str, Any]:
    """
    Facilitate knowledge exchange between educational systems.
    
    Args:
        ctx: Context with user/conversation info
        source_system_id: ID of source system
        target_system_id: ID of target system
        knowledge_domain: Domain of knowledge to exchange
        
    Returns:
        Exchange results
    """
    manager = await get_education_manager(ctx.context.user_id, ctx.context.conversation_id)
    
    with trace("KnowledgeExchange", metadata={
        "source_id": source_system_id,
        "target_id": target_system_id,
        "domain": knowledge_domain
    }):
        # Fetch system data
        async with get_db_connection_context() as conn:
            source = await conn.fetchrow(
                "SELECT * FROM EducationalSystems WHERE id = $1 AND user_id = $2 AND conversation_id = $3",
                source_system_id, ctx.context.user_id, ctx.context.conversation_id
            )
            target = await conn.fetchrow(
                "SELECT * FROM EducationalSystems WHERE id = $1 AND user_id = $2 AND conversation_id = $3",
                target_system_id, ctx.context.user_id, ctx.context.conversation_id
            )
            
            if not source or not target:
                return {"error": "System(s) not found"}
        
        # Prepare prompt
        prompt = f"""
        Analyze knowledge exchange between:
        
        SOURCE: {source['name']} ({source['system_type']})
        - Controlled by: {source['controlled_by']}
        - Core teachings: {source['core_teachings']}
        - Restrictions: {source['knowledge_restrictions']}
        
        TARGET: {target['name']} ({target['system_type']})
        - Controlled by: {target['controlled_by']}
        - Core teachings: {target['core_teachings']}
        - Restrictions: {target['knowledge_restrictions']}
        
        DOMAIN: {knowledge_domain}
        
        Determine what can be transferred, adapted, or is restricted.
        """
        
        result = await Runner.run(
            manager.knowledge_exchange_agent,
            prompt,
            context={"manager": manager}
        )
        
        exchange_result = result.final_output
        return exchange_result.dict()

@function_tool
async def search_educational_systems(
    ctx: RunContextWrapper[EducationContext],
    query: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Search educational systems by semantic similarity.
    
    Args:
        ctx: Context with user/conversation info
        query: Search query
        limit: Maximum results
        
    Returns:
        List of matching systems
    """
    manager = await get_education_manager(ctx.context.user_id, ctx.context.conversation_id)
    
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
        You can create educational systems, knowledge traditions, and facilitate
        knowledge exchange between systems.
        """,
        tools=[
            add_educational_system,
            add_knowledge_tradition,
            generate_educational_systems,
            generate_knowledge_traditions,
            exchange_knowledge_between_systems,
            search_educational_systems
        ],
        model="gpt-4.1-nano"
    )

# ---------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------

async def cleanup_education_managers():
    """Close all education manager instances."""
    for manager in _education_managers.values():
        await manager.close()
    _education_managers.clear()
