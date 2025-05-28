# lore/managers/politics.py

import logging
import json
import random
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

# Agents SDK imports
from agents import Agent, function_tool, Runner, ModelSettings, RunContextWrapper, RunConfig, trace

from pydantic import BaseModel, Field

# Governance
from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

# Project imports
from utils.embedding_service import get_embedding

from lore.managers.base_manager import BaseLoreManager
from lore.utils.theming import MatriarchalThemingUtils

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Define Pydantic models for structured outputs
# -------------------------------------------------------------------------

class DiplomaticNegotiationResult(BaseModel):
    """Model for diplomatic negotiation results."""
    nation1_position: str
    nation2_position: str
    agreement_reached: bool
    agreement_terms: Optional[str] = None
    concessions_nation1: List[str] = []
    concessions_nation2: List[str] = []
    relationship_change: int = Field(0, description="Change from -5 to +5")
    notes: Optional[str] = None

class MediaCoverageItem(BaseModel):
    """Model for media coverage item."""
    media_name: str
    headline: str
    content: str
    bias: str
    emphasis: str
    bias_indicators: List[str] = []

# -------------------------------------------------------------------------
# (Optional) Agent for deciding distribution of conflicts
# -------------------------------------------------------------------------
distribution_agent = Agent(
    name="PoliticsDistributionAgent",
    instructions=(
        "You decide how many conflicts to generate or how to distribute them. "
        "Return JSON, e.g. {\"count\": 3}, or additional instructions.\n"
    ),
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.0)
)

# -------------------------------------------------------------------------
# Define the diplomatic negotiation class
# -------------------------------------------------------------------------
class DiplomaticNegotiation:
    """Class to handle diplomatic negotiations between nations."""
    
    def __init__(self, nation1_agent, nation2_agent, mediator_agent, nations_data, issue, max_rounds=5):
        self.nation1_agent = nation1_agent
        self.nation2_agent = nation2_agent
        self.mediator_agent = mediator_agent
        self.nations_data = nations_data
        self.issue = issue
        self.max_rounds = max_rounds
        self.conversation_history = []
        
    async def run(self, ctx) -> Dict[str, Any]:
        """Run the negotiation process."""
        with trace(
            "DiplomaticNegotiation",
            metadata={
                "nation1": self.nations_data['nation1']['name'],
                "nation2": self.nations_data['nation2']['name'],
                "issue": self.issue
            }
        ):
            # Initial positions
            nation1_position = await self._get_initial_position(ctx, self.nation1_agent, self.nations_data['nation1'])
            nation2_position = await self._get_initial_position(ctx, self.nation2_agent, self.nations_data['nation2'])
            
            self.conversation_history.append({
                "role": self.nations_data['nation1']['name'],
                "content": nation1_position
            })
            self.conversation_history.append({
                "role": self.nations_data['nation2']['name'],
                "content": nation2_position
            })
            
            # Negotiation rounds
            agreement = None
            round_num = 0
            while round_num < self.max_rounds and agreement is None:
                round_num += 1
                
                # Mediator summarizes and proposes
                mediator_proposal = await self._get_mediator_proposal(ctx, round_num)
                self.conversation_history.append({
                    "role": "Mediator",
                    "content": mediator_proposal
                })
                
                # Nations respond
                nation1_response = await self._get_nation_response(
                    ctx, self.nation1_agent, self.nations_data['nation1'], mediator_proposal, round_num)
                self.conversation_history.append({
                    "role": self.nations_data['nation1']['name'],
                    "content": nation1_response
                })
                
                nation2_response = await self._get_nation_response(
                    ctx, self.nation2_agent, self.nations_data['nation2'], mediator_proposal, round_num)
                self.conversation_history.append({
                    "role": self.nations_data['nation2']['name'],
                    "content": nation2_response
                })
                
                # Check for agreement
                agreement = await self._check_for_agreement(ctx, round_num)
            
            # Final result
            agreement_reached = agreement is not None
            result = {
                "nations": {
                    "nation1": self.nations_data['nation1']['name'],
                    "nation2": self.nations_data['nation2']['name']
                },
                "issue": self.issue,
                "rounds": round_num,
                "agreement_reached": agreement_reached,
                "conversation_history": self.conversation_history,
                "final_outcome": agreement if agreement_reached else "No agreement reached",
                "relationship_change": random.randint(-2, 3) if agreement_reached else random.randint(-5, 0)
            }
            
            return result
            
    async def _get_initial_position(self, ctx, agent, nation_data):
        """Get the initial position from a nation."""
        prompt = f"""
        You are representing {nation_data['name']} in diplomatic negotiations about: {self.issue}
        
        Express your nation's initial position and demands.
        
        NATION DATA:
        {json.dumps(nation_data, indent=2)}
        
        Return your nation's position in a clear, diplomatic manner.
        """
        
        result = await Runner.run(agent, prompt, context=ctx.context)
        return result.final_output
    
    async def _get_mediator_proposal(self, ctx, round_num):
        """Get a proposal from the mediator."""
        history_text = "\n\n".join([f"{item['role']}: {item['content']}" for item in self.conversation_history])
        
        prompt = f"""
        You are mediating a diplomatic negotiation about: {self.issue}
        
        This is round {round_num} of negotiations. Review the positions and provide a balanced proposal.
        
        CONVERSATION HISTORY:
        {history_text}
        
        Analyze both sides' positions and suggest a compromise that addresses key concerns.
        If this is the final round, push harder for resolution.
        """
        
        result = await Runner.run(self.mediator_agent, prompt, context=ctx.context)
        return result.final_output
    
    async def _get_nation_response(self, ctx, agent, nation_data, mediator_proposal, round_num):
        """Get a response from a nation to the mediator's proposal."""
        history_text = "\n\n".join([f"{item['role']}: {item['content']}" for item in self.conversation_history])
        
        prompt = f"""
        You are representing {nation_data['name']} in diplomatic negotiations about: {self.issue}
        
        This is round {round_num}. The mediator has proposed:
        
        MEDIATOR PROPOSAL:
        {mediator_proposal}
        
        PREVIOUS CONVERSATION:
        {history_text}
        
        Respond to this proposal from your nation's perspective. Consider:
        1. What aspects are acceptable?
        2. What remains problematic?
        3. What concessions might you offer if needed?
        
        {"This is the FINAL round. Consider accepting a reasonable compromise." if round_num >= self.max_rounds - 1 else ""}
        """
        
        result = await Runner.run(agent, prompt, context=ctx.context)
        return result.final_output
    
    async def _check_for_agreement(self, ctx, round_num):
        """Check if an agreement has been reached."""
        history_text = "\n\n".join([f"{item['role']}: {item['content']}" for item in self.conversation_history])
        
        prompt = f"""
        As the mediator, analyze if an agreement has been reached after round {round_num}.
        
        CONVERSATION HISTORY:
        {history_text}
        
        Return a JSON response with:
        - "agreement_reached": true or false
        - "agreement_terms": summary of the terms (if reached)
        - "key_concessions": what each side gave up
        - "notes": any additional observations
        
        If no agreement has been reached, return null for agreement_terms.
        """
        
        result = await Runner.run(self.mediator_agent, prompt, context=ctx.context)
        try:
            agreement_data = json.loads(result.final_output)
            if agreement_data.get("agreement_reached", False):
                return agreement_data
            return None
        except json.JSONDecodeError:
            return None


# -------------------------------------------------------------------------
#  FactionAgentProxy - representing a faction with its own goals
# -------------------------------------------------------------------------
class FactionAgentProxy:
    """
    Proxy class representing a faction in the world with competing goals and agency.
    """
    def __init__(self, faction_data: Dict[str, Any]):
        self.faction_data = faction_data
        self.agent = Agent(
            name=f"{faction_data['name']}Agent",
            instructions=self._build_instructions(),
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9)
        )
        self.actions_history = []
        self.goals = self._determine_faction_goals()

    def _build_instructions(self) -> str:
        """Build agent instructions based on faction data."""
        return f"""
        You represent {self.faction_data['name']}, a {self.faction_data.get('type', 'faction')} in a matriarchal world.
        Traits: {self.faction_data.get('cultural_traits', [])}
        Values: {self.faction_data.get('values', [])}
        
        Your purpose is to pursue your faction's goals while reacting realistically to world events.
        Always consider how your actions will affect your standing in the matriarchal power structure.
        """

    def _determine_faction_goals(self) -> List[str]:
        """Determine the faction's goals based on its data."""
        # This could be expanded to use an LLM to generate more nuanced goals:
        return self.faction_data.get('goals', ['Increase influence', 'Protect interests'])

    async def react_to_event(self, event: Dict[str, Any], ctx) -> Dict[str, Any]:
        """
        Have the faction react to a world event, producing a plan or statement.

        Returns:
          JSON-like dict with details on how the faction responds.
        """
        prompt = f"""
        Your faction ({self.faction_data['name']}) is reacting to this event:
        {json.dumps(event, indent=2)}
        
        Your faction's goals are: {self.goals}
        Your recent actions: {self.actions_history[-3:] if self.actions_history else 'None'}
        
        How do you react? Consider:
        1. Public statements
        2. Private actions
        3. Resource allocation
        4. Diplomatic initiatives
        5. Internal policy changes

        Return JSON with these fields:
        {{
            "public_statements": "...",
            "private_actions": "...",
            "resource_allocation": "...",
            "diplomatic_initiatives": "...",
            "internal_policy_changes": "..."
        }}
        """

        result = await Runner.run(self.agent, prompt, context=ctx.context)
        try:
            reaction = json.loads(result.final_output)
            self.actions_history.append({
                "event": event.get("name", "Unnamed event"),
                "reaction": reaction
            })
            return reaction
        except json.JSONDecodeError:
            return {"error": "Failed to parse faction reaction", "raw_output": result.final_output}


class WorldPoliticsManager(BaseLoreManager):
    """
    Consolidated manager for geopolitical landscape, conflicts, factions, and political evolutions.
    Handles nations, international relations, both international and domestic conflicts,
    and high-level political simulations.
    """

    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.cache_namespace = "world_politics"
        self.faction_proxies = {}  # Will store {faction_id: FactionAgentProxy} once initialized

    # ------------------------------------------------------------------------
    #                          DB Setup
    # ------------------------------------------------------------------------
    async def initialize_tables(self):
        """
        Initialize all required tables for geopolitics and conflicts.
        Expand or adjust as needed for new features (e.g., dynasties).
        """
        geo_tables = {
            "Nations": """
                CREATE TABLE IF NOT EXISTS Nations (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    government_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    relative_power INTEGER CHECK (relative_power BETWEEN 1 AND 10),
                    matriarchy_level INTEGER CHECK (matriarchy_level BETWEEN 1 AND 10),
                    population_scale TEXT,
                    major_resources TEXT[],
                    major_cities TEXT[],
                    cultural_traits TEXT[],
                    notable_features TEXT,
                    neighboring_nations TEXT[],
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_nations_embedding 
                ON Nations USING ivfflat (embedding vector_cosine_ops);
            """,
            "InternationalRelations": """
                CREATE TABLE IF NOT EXISTS InternationalRelations (
                    id SERIAL PRIMARY KEY,
                    nation1_id INTEGER NOT NULL,
                    nation2_id INTEGER NOT NULL,
                    relationship_type TEXT NOT NULL,
                    relationship_quality INTEGER CHECK (relationship_quality BETWEEN 1 AND 10),
                    description TEXT NOT NULL,
                    notable_conflicts TEXT[],
                    notable_alliances TEXT[],
                    trade_relations TEXT,
                    cultural_exchanges TEXT,
                    FOREIGN KEY (nation1_id) REFERENCES Nations(id) ON DELETE CASCADE,
                    FOREIGN KEY (nation2_id) REFERENCES Nations(id) ON DELETE CASCADE,
                    UNIQUE (nation1_id, nation2_id)
                );
            """
        }

        conflict_tables = {
            "NationalConflicts": """
                CREATE TABLE IF NOT EXISTS NationalConflicts (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    conflict_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity INTEGER CHECK (severity BETWEEN 1 AND 10),
                    status TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT,
                    involved_nations INTEGER[],
                    primary_aggressor INTEGER,
                    primary_defender INTEGER,
                    current_casualties TEXT,
                    economic_impact TEXT,
                    diplomatic_consequences TEXT,
                    public_opinion JSONB,
                    recent_developments TEXT[],
                    potential_resolution TEXT,
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_nationalconflicts_embedding 
                ON NationalConflicts USING ivfflat (embedding vector_cosine_ops);
            """,
            "ConflictNews": """
                CREATE TABLE IF NOT EXISTS ConflictNews (
                    id SERIAL PRIMARY KEY,
                    conflict_id INTEGER NOT NULL,
                    headline TEXT NOT NULL,
                    content TEXT NOT NULL,
                    publication_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source_nation INTEGER,
                    bias TEXT,
                    embedding VECTOR(1536),
                    FOREIGN KEY (conflict_id) REFERENCES NationalConflicts(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_conflictnews_embedding 
                ON ConflictNews USING ivfflat (embedding vector_cosine_ops);
            """,
            "DomesticIssues": """
                CREATE TABLE IF NOT EXISTS DomesticIssues (
                    id SERIAL PRIMARY KEY,
                    nation_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    issue_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity INTEGER CHECK (severity BETWEEN 1 AND 10),
                    status TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT,
                    supporting_factions TEXT[],
                    opposing_factions TEXT[],
                    neutral_factions TEXT[],
                    affected_demographics TEXT[],
                    public_opinion JSONB,
                    government_response TEXT,
                    recent_developments TEXT[],
                    political_impact TEXT,
                    social_impact TEXT,
                    economic_impact TEXT,
                    potential_resolution TEXT,
                    embedding VECTOR(1536),
                    FOREIGN KEY (nation_id) REFERENCES Nations(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_domesticissues_embedding 
                ON DomesticIssues USING ivfflat (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS idx_domesticissues_nation
                ON DomesticIssues(nation_id);
            """,
            "DomesticNews": """
                CREATE TABLE IF NOT EXISTS DomesticNews (
                    id SERIAL PRIMARY KEY,
                    issue_id INTEGER NOT NULL,
                    headline TEXT NOT NULL,
                    content TEXT NOT NULL,
                    publication_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source_faction TEXT,
                    bias TEXT,
                    embedding VECTOR(1536),
                    FOREIGN KEY (issue_id) REFERENCES DomesticIssues(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_domesticnews_embedding 
                ON DomesticNews USING ivfflat (embedding vector_cosine_ops);
            """
        }

        # Dynasty tracking table
        dynasty_tables = {
            "Dynasties": """
                CREATE TABLE IF NOT EXISTS Dynasties (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    founding_date TEXT NOT NULL,
                    ruling_nation INTEGER,
                    matriarch TEXT,
                    patriarch TEXT,
                    notable_members TEXT[],
                    family_traits TEXT[],
                    FOREIGN KEY (ruling_nation) REFERENCES Nations(id) ON DELETE SET NULL
                );
            """,
            "DynastyLineages": """
                CREATE TABLE IF NOT EXISTS DynastyLineages (
                    id SERIAL PRIMARY KEY,
                    dynasty_id INTEGER NOT NULL,
                    member_name TEXT NOT NULL,
                    birth_date TEXT,
                    death_date TEXT,
                    mother_name TEXT,
                    father_name TEXT,
                    inheritor_of TEXT,
                    is_ruler BOOLEAN DEFAULT FALSE,
                    notes TEXT,
                    FOREIGN KEY (dynasty_id) REFERENCES Dynasties(id) ON DELETE CASCADE
                );
            """
        }

        # Factions table
        faction_tables = {
            "Factions": """
                CREATE TABLE IF NOT EXISTS Factions (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT,
                    description TEXT,
                    values TEXT[],
                    goals TEXT[],
                    cultural_traits TEXT[],
                    territory TEXT[]  -- e.g. list of nation names or region IDs
                );
            """
        }

        all_tables = {
            **geo_tables, **conflict_tables, **dynasty_tables, **faction_tables
        }
        await self.initialize_tables_from_definitions(all_tables)

    async def ensure_initialized(self):
        """Ensure system is initialized with all tables."""
        if not self.initialized:
            await super().ensure_initialized()
            await self.initialize_tables()

    # ------------------------------------------------------------------------
    #   NATION & RELATIONS  (Add, Get, etc.)
    # ------------------------------------------------------------------------
    async def _add_nation_impl(
        self,
        ctx,
        name: str,
        government_type: str,
        description: str,
        relative_power: int,
        matriarchy_level: int,
        population_scale: str = None,
        major_resources: List[str] = None,
        major_cities: List[str] = None,
        cultural_traits: List[str] = None,
        notable_features: str = None,
        neighboring_nations: List[str] = None
    ) -> int:
        with trace(
            "AddNation",
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "nation_name": name}
        ):
            await self.ensure_initialized()
            major_resources = major_resources or []
            major_cities = major_cities or []
            cultural_traits = cultural_traits or []
            neighboring_nations = neighboring_nations or []
        
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    nation_id = await conn.fetchval("""
                        INSERT INTO Nations (
                            name, government_type, description, relative_power,
                            matriarchy_level, population_scale, major_resources,
                            major_cities, cultural_traits, notable_features,
                            neighboring_nations
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        RETURNING id
                    """,
                        name, government_type, description, relative_power,
                        matriarchy_level, population_scale, major_resources,
                        major_cities, cultural_traits, notable_features,
                        neighboring_nations
                    )
        
                    # Generate and store embedding
                    embedding_text = f"{name} {government_type} {description}"
                    await self.generate_and_store_embedding(embedding_text, conn, "Nations", "id", nation_id)
                    return nation_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_nation",
        action_description="Adding nation: {name}",
        id_from_context=lambda ctx: "world_politics_manager"
    )
    @function_tool
    async def add_nation(
        self,
        ctx,
        name: str,
        government_type: str,
        description: str,
        relative_power: int,
        matriarchy_level: int,
        population_scale: str = None,
        major_resources: List[str] = None,
        major_cities: List[str] = None,
        cultural_traits: List[str] = None,
        notable_features: str = None,
        neighboring_nations: List[str] = None
    ) -> int:
        """
        Add a nation to the database.
        
        Args:
            name: Name of the nation
            government_type: Type of government
            description: Description of the nation
            relative_power: Relative power level (1-10)
            matriarchy_level: Level of matriarchal control (1-10)
            population_scale: Description of population size
            major_resources: List of major resources
            major_cities: List of major cities
            cultural_traits: List of cultural traits
            notable_features: Notable features of the nation
            neighboring_nations: List of neighboring nation names
            
        Returns:
            ID of the created nation
        """
        return await self._add_nation_impl(
            ctx, name, government_type, description, relative_power, matriarchy_level,
            population_scale, major_resources, major_cities, cultural_traits, notable_features, neighboring_nations
        )

    async def _add_international_relation_impl(
        self,
        ctx,
        nation1_id: int,
        nation2_id: int,
        relationship_type: str,
        relationship_quality: int,
        description: str,
        notable_conflicts: List[str] = None,
        notable_alliances: List[str] = None,
        trade_relations: str = None,
        cultural_exchanges: str = None
    ) -> int:
        with trace(
            "AddInternationalRelation",
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "nation1_id": nation1_id,
                "nation2_id": nation2_id,
                "relationship_type": relationship_type
            }
        ):
            await self.ensure_initialized()
            notable_conflicts = notable_conflicts or []
            notable_alliances = notable_alliances or []
        
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    relation_id = await conn.fetchval("""
                        INSERT INTO InternationalRelations (
                            nation1_id, nation2_id, relationship_type,
                            relationship_quality, description, notable_conflicts,
                            notable_alliances, trade_relations, cultural_exchanges
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (nation1_id, nation2_id) DO UPDATE
                        SET relationship_type = EXCLUDED.relationship_type,
                            relationship_quality = EXCLUDED.relationship_quality,
                            description = EXCLUDED.description,
                            notable_conflicts = EXCLUDED.notable_conflicts,
                            notable_alliances = EXCLUDED.notable_alliances,
                            trade_relations = EXCLUDED.trade_relations,
                            cultural_exchanges = EXCLUDED.cultural_exchanges
                        RETURNING id
                    """,
                        nation1_id, nation2_id, relationship_type,
                        relationship_quality, description, notable_conflicts,
                        notable_alliances, trade_relations, cultural_exchanges
                    )
                    return relation_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_international_relation",
        action_description="Adding relation between nations",
        id_from_context=lambda ctx: "world_politics_manager"
    )
    @function_tool
    async def add_international_relation(
        self,
        ctx,
        nation1_id: int,
        nation2_id: int,
        relationship_type: str,
        relationship_quality: int,
        description: str,
        notable_conflicts: List[str] = None,
        notable_alliances: List[str] = None,
        trade_relations: str = None,
        cultural_exchanges: str = None
    ) -> int:
        """
        Add or update a relationship between two nations.
        
        Args:
            nation1_id: ID of the first nation
            nation2_id: ID of the second nation
            relationship_type: Type of relationship (allies, rivals, etc.)
            relationship_quality: Quality of the relationship (1-10)
            description: Description of the relationship
            notable_conflicts: List of notable conflicts
            notable_alliances: List of notable alliances
            trade_relations: Description of trade relations
            cultural_exchanges: Description of cultural exchanges
            
        Returns:
            ID of the created/updated relationship
        """
        return await self._add_international_relation_impl(
            ctx, nation1_id, nation2_id, relationship_type, relationship_quality,
            description, notable_conflicts, notable_alliances, trade_relations, cultural_exchanges
        )

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_all_nations",
        action_description="Getting all nations in the world",
        id_from_context=lambda ctx: "world_politics_manager"
    )
    @function_tool
    async def get_all_nations(self, ctx) -> List[Dict[str, Any]]:
        """
        Get all nations in the world.
        
        Returns:
            List of nation dictionaries
        """
        cache_key = f"all_nations_{self.user_id}_{self.conversation_id}"
        cached = self.get_cache(cache_key)
        if cached:
            return cached

        await self.ensure_initialized()
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                nations = await conn.fetch("""
                    SELECT id, name, government_type, description, relative_power,
                           matriarchy_level, population_scale, major_resources,
                           major_cities, cultural_traits, notable_features, 
                           neighboring_nations
                    FROM Nations
                    ORDER BY relative_power DESC
                """)
                result = [dict(n) for n in nations]
                self.set_cache(cache_key, result, ttl=3600)
                return result

    # ------------------------------------------------------------------------
    #   CONFLICT HANDLING
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_initial_conflicts",
        action_description="Generating initial national conflicts",
        id_from_context=lambda ctx: "world_politics_manager"
    )
    @function_tool
    async def generate_initial_conflicts(self, ctx, count: int = 3) -> List[Dict[str, Any]]:
        """
        Generate initial conflicts between nations with LLM-based approach.
        
        Args:
            count: Number of conflicts to generate
            
        Returns:
            List of generated conflict dictionaries
        """
        with trace(
            "GenerateInitialConflicts",
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "count": count}
        ):
            run_ctx = RunContextWrapper(context=ctx.context)
    
            # (Optional) Let an agent decide how many conflicts to generate
            distribution_prompt = (
                "We want to create some initial conflicts among nations. Currently set to {count}, but you can override.\n"
                "Return JSON with a 'count' field.\n"
            ).format(count=count)
            dist_config = RunConfig(workflow_name="ConflictDistribution")
            dist_result = await Runner.run(
                distribution_agent, 
                distribution_prompt,
                context=run_ctx.context,
                run_config=dist_config
            )
    
            try:
                dist_data = json.loads(dist_result.final_output)
                count = dist_data.get("count", count)
            except json.JSONDecodeError:
                pass  # fallback to the existing count
    
            # get nations
            nations = await self.get_all_nations(run_ctx)
            if len(nations) < 2:
                return []
    
            conflicts = []
            conflict_agent = Agent(
                name="NationalConflictAgent",
                instructions="You create realistic international conflicts for a fantasy world with matriarchal structures.",
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.9)
            )
    
            for _ in range(count):
                # pick random nations (ensuring at least 2)
                available_nations = [n for n in nations if not any(
                    n["id"] in c.get("involved_nations", []) for c in conflicts
                )]
                if len(available_nations) < 2:
                    available_nations = nations
    
                nation_pair = random.sample(available_nations, 2)
                matriarchy_diff = abs(
                    nation_pair[0].get("matriarchy_level", 5) -
                    nation_pair[1].get("matriarchy_level", 5)
                )
    
                if matriarchy_diff > 4:
                    conflict_types = ["ideological_dispute", "cultural_tension", "religious_conflict", "proxy_war"]
                elif matriarchy_diff > 2:
                    conflict_types = ["diplomatic_tension", "border_dispute", "trade_dispute", "resource_conflict"]
                else:
                    conflict_types = ["territorial_dispute", "trade_war", "succession_crisis", "alliance_dispute"]
    
                chosen_conflict_type = random.choice(conflict_types)
    
                prompt = f"""
                Generate a detailed international conflict between these two nations:
    
                NATION 1:
                {json.dumps(nation_pair[0], indent=2)}
    
                NATION 2:
                {json.dumps(nation_pair[1], indent=2)}
    
                Create a {chosen_conflict_type} that:
                1. Makes sense given the nations' characteristics
                2. Has appropriate severity and clear causes
                3. Includes realistic consequences
                4. Considers the matriarchal nature of the world
                5. Reflects the matriarchy level difference ({matriarchy_diff} points)
                
                Return JSON with fields:
                - name
                - conflict_type: "{chosen_conflict_type}"
                - description
                - severity (1-10)
                - status (active, escalating, etc.)
                - start_date
                - involved_nations (list of IDs)
                - primary_aggressor
                - primary_defender
                - current_casualties
                - economic_impact
                - diplomatic_consequences
                - public_opinion (object)
                - recent_developments (list)
                - potential_resolution
                """
    
                run_config = RunConfig(workflow_name="ConflictGeneration")
                result = await Runner.run(conflict_agent, prompt, context=run_ctx.context, run_config=run_config)
    
                try:
                    conflict_data = json.loads(result.final_output)
                    # Build embedding by calling your real embedding function:
                    embed_text = f"{conflict_data.get('name','Unnamed Conflict')} {conflict_data.get('description','')} {conflict_data.get('conflict_type','')}"
                    emb = await get_embedding(embed_text)
                    # If needed, ensure it's a list for Postgres vector usage:
                    if not isinstance(emb, list):
                        emb = emb.tolist()
    
                    # Insert DB
                    async with self.get_connection_pool() as pool:
                        async with pool.acquire() as conn:
                            conflict_id = await conn.fetchval("""
                                INSERT INTO NationalConflicts (
                                    name, conflict_type, description, severity, status,
                                    start_date, involved_nations, primary_aggressor, primary_defender,
                                    current_casualties, economic_impact, diplomatic_consequences,
                                    public_opinion, recent_developments, potential_resolution, embedding
                                )
                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                                RETURNING id
                            """,
                                conflict_data.get("name","Conflict X"),
                                conflict_data.get("conflict_type", chosen_conflict_type),
                                conflict_data.get("description",""),
                                conflict_data.get("severity",5),
                                conflict_data.get("status","active"),
                                conflict_data.get("start_date","Recently"),
                                conflict_data.get("involved_nations",[nation_pair[0]["id"],nation_pair[1]["id"]]),
                                conflict_data.get("primary_aggressor", nation_pair[0]["id"]),
                                conflict_data.get("primary_defender", nation_pair[1]["id"]),
                                conflict_data.get("current_casualties","Unknown"),
                                conflict_data.get("economic_impact","Unknown"),
                                conflict_data.get("diplomatic_consequences","Unknown"),
                                json.dumps(conflict_data.get("public_opinion",{})),
                                conflict_data.get("recent_developments",[]),
                                conflict_data.get("potential_resolution","TBD"),
                                emb
                            )
    
                            conflict_data["id"] = conflict_id
                            conflicts.append(conflict_data)
    
                            # Generate initial news
                            await self._generate_conflict_news(run_ctx, conflict_id, conflict_data, nation_pair)
    
                except Exception as e:
                    logger.error(f"Error generating conflict: {e}")
    
            return conflicts

    async def _generate_conflict_news(
        self,
        ctx,
        conflict_id: int,
        conflict_data: Dict[str, Any],
        nations: List[Dict[str, Any]]
    ) -> None:
        """
        Generate initial news articles about a newly created conflict.
        """
        with trace(
            "GenerateConflictNews",
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "conflict_id": conflict_id,
                "conflict_name": conflict_data.get("name", "Unnamed Conflict")
            }
        ):
            news_agent = Agent(
                name="ConflictNewsAgent",
                instructions="You create realistic news articles about new conflicts in a matriarchal fantasy world.",
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.8)
            )
    
            # We'll produce a single news article from each nation's perspective
            for nation in nations[:2]:
                bias = "pro_aggressor" if nation["id"] == conflict_data.get("primary_aggressor") else "pro_defender"
                prompt = f"""
                Generate a news article about this new conflict from {nation['name']}'s perspective:
    
                CONFLICT:
                {json.dumps(conflict_data, indent=2)}
    
                The coverage is {bias} biased. Must reflect matriarchal power structures, quoting female leaders, etc.
    
                Return JSON:
                - headline
                - content
                """
    
                run_config = RunConfig(workflow_name="ConflictNewsGeneration")
                result = await Runner.run(news_agent, prompt, context=ctx.context, run_config=run_config)
    
                try:
                    news_data = json.loads(result.final_output)
                    # Theming
                    if "content" in news_data:
                        news_data["content"] = MatriarchalThemingUtils.apply_matriarchal_theme("news", news_data["content"], emphasis_level=1)
    
                    embed_text = f"{news_data.get('headline','No Headline')} {news_data.get('content','')[:200]}"
                    emb = await get_embedding(embed_text)
                    if not isinstance(emb, list):
                        emb = emb.tolist()
    
                    async with self.get_connection_pool() as pool:
                        async with pool.acquire() as conn:
                            await conn.execute("""
                                INSERT INTO ConflictNews (
                                    conflict_id, headline, content, source_nation, bias, embedding
                                )
                                VALUES ($1, $2, $3, $4, $5, $6)
                            """,
                                conflict_id,
                                news_data.get("headline","No Headline"),
                                news_data.get("content",""),
                                nation["id"],
                                bias,
                                emb
                            )
    
                except Exception as e:
                    logger.error(f"Error generating conflict news: {e}")

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="stream_crisis_events",
        action_description="Streaming crisis events in real-time",
        id_from_context=lambda ctx: "world_politics_manager"
    )
    @function_tool
    async def stream_crisis_events(self, ctx, conflict_id: int) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream real-time updates about an evolving crisis, yielding events as they occur.
        
        Args:
            conflict_id: ID of the conflict to stream events for
            
        Yields:
            Event dictionaries with real-time updates
        """
        with trace(
            "StreamCrisisEvents",
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "conflict_id": conflict_id}
        ):
            conflict = await self._get_conflict_details(conflict_id)
            if not conflict:
                yield {"error": "Conflict not found"}
                return
    
            crisis_streaming_agent = Agent(
                name="CrisisStreamingAgent",
                instructions="Generate a stream of real-time developments in an ongoing crisis.",
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.9)
            )
    
            prompt = f"""
            Generate a series of real-time updates for this ongoing conflict:
            {json.dumps(conflict, indent=2)}
            
            Each update should include:
            - timestamp
            - location
            - event description
            - severity level (1-10)
            - parties involved
            - immediate consequences
            
            Provide updates as they might occur over a day of conflict.
            """
    
            run_ctx = RunContextWrapper(context=ctx.context)
            result = await Runner.run(crisis_streaming_agent, prompt, context=run_ctx.context)
    
            # Since this is not a streaming API, we'll parse the result and yield events one by one
            try:
                all_events = json.loads(result.final_output)
                if isinstance(all_events, list):
                    for event in all_events:
                        yield event
                else:
                    yield all_events
            except json.JSONDecodeError:
                # Try to parse line by line
                for line in result.final_output.split('\n'):
                    try:
                        event_data = json.loads(line.strip())
                        yield event_data
                    except json.JSONDecodeError:
                        pass

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="simulate_diplomatic_negotiation",
        action_description="Simulating diplomatic negotiations between nations",
        id_from_context=lambda ctx: "world_politics_manager"
    )
    @function_tool
    async def simulate_diplomatic_negotiation(self, ctx, nation1_id: int, nation2_id: int, issue: str) -> Dict[str, Any]:
        """
        Simulate diplomatic negotiations between two nations over a specific issue.
        
        Args:
            nation1_id: ID of the first nation
            nation2_id: ID of the second nation
            issue: Description of the issue being negotiated
            
        Returns:
            Dictionary with negotiation results
        """
        with trace(
            "SimulateDiplomaticNegotiation",
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata, 
                "nation1_id": nation1_id, 
                "nation2_id": nation2_id,
                "issue": issue
            }
        ):
            # Load the nation data
            nations = await self._load_negotiating_nations(nation1_id, nation2_id)
            if "error" in nations:
                return nations
    
            # Create agents for each nation with competing goals
            nation1_agent = Agent(
                name=f"{nations['nation1']['name']}Agent",
                instructions=(
                    f"You represent {nations['nation1']['name']}. Your goal is to maximize your "
                    f"nation's interests while finding a workable resolution. Traits: "
                    f"{nations['nation1'].get('cultural_traits', [])}"
                ),
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.9)
            )
            
            nation2_agent = Agent(
                name=f"{nations['nation2']['name']}Agent",
                instructions=(
                    f"You represent {nations['nation2']['name']}. Your goal is to maximize your "
                    f"nation's interests while finding a workable resolution. Traits: "
                    f"{nations['nation2'].get('cultural_traits', [])}"
                ),
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.9)
            )
    
            # Create a mediator agent
            mediator_agent = Agent(
                name="DiplomaticMediatorAgent",
                instructions=(
                    "You are a neutral diplomatic mediator. Your goal is to facilitate productive "
                    "negotiations and help reach a resolution that both parties can accept."
                ),
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.7)
            )
    
            # Set up the negotiation simulation
            negotiation = DiplomaticNegotiation(
                nation1_agent, 
                nation2_agent, 
                mediator_agent,
                nations,
                issue,
                max_rounds=5
            )
            
            # Run the simulation
            run_ctx = RunContextWrapper(context=ctx.context)
            results = await negotiation.run(run_ctx)
    
            # Update relations based on outcome
            await self._update_international_relations(nations, results)
            
            return results

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="simulate_media_coverage",
        action_description="Simulating media coverage of political events",
        id_from_context=lambda ctx: "world_politics_manager"
    )
    @function_tool
    async def simulate_media_coverage(self, ctx, event_id: int) -> Dict[str, Any]:
        """
        Simulate media coverage of a political event from different perspectives.
        
        Args:
            event_id: ID of the event to cover
            
        Returns:
            Dictionary with simulated media coverage from different outlets
        """
        with trace(
            "SimulateMediaCoverage",
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "event_id": event_id}
        ):
            run_ctx = RunContextWrapper(context=ctx.context)
            
            # Load event details
            event = await self._load_event_details(event_id)
            if "error" in event:
                return event
            
            # Create media outlet agents with different biases
            media_types = [
                {"name": "State Official", "bias": "pro_government", "reliability": 0.7},
                {"name": "Popular Tribune", "bias": "populist", "reliability": 0.6},
                {"name": "Noble Herald", "bias": "aristocratic", "reliability": 0.8},
                {"name": "Foreign Observer", "bias": "neutral", "reliability": 0.9}
            ]
            
            coverage = []
            for media in media_types:
                media_agent = Agent(
                    name=f"{media['name']}Agent",
                    instructions=(
                        f"You are {media['name']}, a news outlet with a {media['bias']} bias. "
                        f"Cover political events with your unique perspective. Maintain matriarchal themes."
                    ),
                    model="gpt-4.1-nano",
                    model_settings=ModelSettings(temperature=0.8)
                )
                
                prompt = f"""
                Write a news article covering this event:
                {json.dumps(event, indent=2)}
                
                Your outlet has a {media['bias']} bias.
                
                Return JSON with:
                - headline
                - content (article body)
                - emphasis (aspect of the event you highlight)
                - bias_indicators (how your bias shows)
                """
    
                result = await Runner.run(media_agent, prompt, context=run_ctx.context)
                try:
                    article = json.loads(result.final_output)
                    article["media_name"] = media["name"]
                    article["bias"] = media["bias"]
                    
                    # Add matriarchal theming to the content
                    if "content" in article:
                        article["content"] = MatriarchalThemingUtils.apply_matriarchal_theme("news", article["content"])
                    
                    coverage.append(article)
                    
                    # Store in database
                    await self._store_media_coverage(event_id, article)
                    
                except json.JSONDecodeError:
                    coverage.append({
                        "media_name": media["name"],
                        "error": "Failed to parse media coverage",
                        "raw_output": result.final_output
                    })
            
            return {"event": event, "coverage": coverage}

    # ------------------------------------------------------------------------
    #  DOMESTIC ISSUES
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_domestic_issues",
        action_description="Generating domestic issues for nation {nation_id}",
        id_from_context=lambda ctx: "world_politics_manager"
    )
    @function_tool
    async def generate_domestic_issues(self, ctx, nation_id: int, count: int = 2) -> List[Dict[str, Any]]:
        """
        Generate domestic issues for a specific nation with LLM-based logic.
        
        Args:
            nation_id: ID of the nation
            count: Number of issues to generate
            
        Returns:
            List of generated domestic issue dictionaries
        """
        with trace(
            "GenerateDomesticIssues",
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "nation_id": nation_id, "count": count}
        ):
            run_ctx = RunContextWrapper(context=ctx.context)
            await self.ensure_initialized()
    
            # Get nation details
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    nation = await conn.fetchrow("""
                        SELECT id, name, government_type, matriarchy_level, cultural_traits
                        FROM Nations
                        WHERE id = $1
                    """, nation_id)
                    if not nation:
                        return []
    
                    nation_data = dict(nation)
    
                    # Possibly get faction data, etc.
                    factions = await conn.fetch("""
                        SELECT id, name, type, description, values, goals, cultural_traits
                        FROM Factions
                        WHERE $1 = ANY(territory)  -- or some relevant check
                    """, nation_data.get("name"))
                    faction_data = [dict(f) for f in factions]
    
            issue_agent = Agent(
                name="DomesticIssueAgent",
                instructions="You create realistic domestic political and social issues in a matriarchal society.",
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.9)
            )
    
            issues = []
            for _ in range(count):
                prompt = f"""
                Generate a domestic issue for the following nation:
                
                NATION:
                {json.dumps(nation_data, indent=2)}
                
                FACTIONS:
                {json.dumps(faction_data, indent=2)}
                
                Include in the JSON:
                - name
                - issue_type
                - description
                - severity (1-10)
                - status (emerging, active, resolved, etc.)
                - start_date
                - supporting_factions
                - opposing_factions
                - neutral_factions
                - affected_demographics
                - public_opinion (JSON object)
                - government_response
                - recent_developments
                - political_impact
                - social_impact
                - economic_impact
                - potential_resolution
                """
                run_config = RunConfig(workflow_name="DomesticIssueGeneration")
                result = await Runner.run(issue_agent, prompt, context=run_ctx.context, run_config=run_config)
    
                try:
                    issue_data = json.loads(result.final_output)
                    if not all(k in issue_data for k in ["name","description","issue_type"]):
                        continue
    
                    # Insert DB (embedding using get_embedding)
                    embed_text = f"{issue_data['name']} {issue_data['description']} {issue_data['issue_type']}"
                    emb = await get_embedding(embed_text)
                    if not isinstance(emb, list):
                        emb = emb.tolist()
    
                    async with self.get_connection_pool() as pool:
                        async with pool.acquire() as conn:
                            issue_id = await conn.fetchval("""
                                INSERT INTO DomesticIssues (
                                    nation_id, name, issue_type, description, severity,
                                    status, start_date, supporting_factions, opposing_factions,
                                    neutral_factions, affected_demographics, public_opinion,
                                    government_response, recent_developments, political_impact,
                                    social_impact, economic_impact, potential_resolution, embedding
                                )
                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                                RETURNING id
                            """,
                                nation_id,
                                issue_data.get("name"),
                                issue_data.get("issue_type"),
                                issue_data.get("description"),
                                issue_data.get("severity",5),
                                issue_data.get("status","active"),
                                issue_data.get("start_date","Recently"),
                                issue_data.get("supporting_factions",[]),
                                issue_data.get("opposing_factions",[]),
                                issue_data.get("neutral_factions",[]),
                                issue_data.get("affected_demographics",[]),
                                json.dumps(issue_data.get("public_opinion",{})),
                                issue_data.get("government_response",""),
                                issue_data.get("recent_developments",[]),
                                issue_data.get("political_impact",""),
                                issue_data.get("social_impact",""),
                                issue_data.get("economic_impact",""),
                                issue_data.get("potential_resolution",""),
                                emb
                            )
    
                            issue_data["id"] = issue_id
                            issues.append(issue_data)
    
                            # Generate initial news
                            await self._generate_domestic_news(run_ctx, issue_id, issue_data, nation_data)
    
                except Exception as e:
                    logger.error(f"Error generating domestic issue: {e}")
    
            return issues

    async def _generate_domestic_news(
        self,
        ctx,
        issue_id: int,
        issue_data: Dict[str, Any],
        nation_data: Dict[str, Any]
    ) -> None:
        """
        Generate initial news articles about a newly created domestic issue.
        """
        with trace(
            "GenerateDomesticNews",
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "issue_id": issue_id,
                "issue_name": issue_data.get("name", "Unnamed Issue")
            }
        ):
            news_agent = Agent(
                name="DomesticNewsAgent",
                instructions="You create realistic news articles about domestic issues in a matriarchal society.",
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.8)
            )
    
            biases = ["supporting", "opposing", "neutral"]
            for bias in biases:
                prompt = f"""
                Generate a news article about this domestic issue from a {bias} perspective:
    
                ISSUE:
                {json.dumps(issue_data, indent=2)}
    
                NATION:
                {json.dumps(nation_data, indent=2)}
    
                Return JSON:
                - headline
                - content
                - source_faction (the faction or institution behind it)
                """
                run_config = RunConfig(workflow_name="DomesticNewsGeneration")
                result = await Runner.run(news_agent, prompt, context=ctx.context, run_config=run_config)
    
                try:
                    news_data = json.loads(result.final_output)
                    if not all(k in news_data for k in ["headline","content"]):
                        continue
    
                    # Theming
                    news_data["content"] = MatriarchalThemingUtils.apply_matriarchal_theme("news", news_data["content"], emphasis_level=1)
    
                    embed_text = f"{news_data.get('headline','No Headline')} {news_data.get('content','')[:200]}"
                    emb = await get_embedding(embed_text)
                    if not isinstance(emb, list):
                        emb = emb.tolist()
    
                    async with self.get_connection_pool() as pool:
                        async with pool.acquire() as conn:
                            await conn.fetchval("""
                                INSERT INTO DomesticNews (
                                    issue_id, headline, content, source_faction, bias, embedding
                                )
                                VALUES ($1, $2, $3, $4, $5, $6)
                                RETURNING id
                            """,
                                issue_id,
                                news_data.get("headline","No Headline"),
                                news_data.get("content",""),
                                news_data.get("source_faction","Unknown Source"),
                                bias,
                                emb
                            )
    
                except Exception as e:
                    logger.error(f"Error generating domestic news: {e}")

    # ------------------------------------------------------------------------
    #  GET ACTIVE CONFLICTS
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_active_conflicts",
        action_description="Getting active national conflicts",
        id_from_context=lambda ctx: "world_politics_manager"
    )
    @function_tool
    async def get_active_conflicts(self, ctx) -> List[Dict[str, Any]]:
        """
        Get all active conflicts from the DB.
        
        Returns:
            List of active conflict dictionaries
        """
        cache_key = "active_conflicts"
        cached = self.get_cache(cache_key)
        if cached:
            return cached

        await self.ensure_initialized()
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                conflicts = await conn.fetch("""
                    SELECT * FROM NationalConflicts
                    WHERE status != 'resolved'
                    ORDER BY severity DESC
                """)
                result = [dict(c) for c in conflicts]

                # parse JSON fields
                for c in result:
                    if "public_opinion" in c and c["public_opinion"]:
                        try:
                            c["public_opinion"] = json.loads(c["public_opinion"])
                        except:
                            pass

                self.set_cache(cache_key, result, ttl=3600)
                return result

    # ------------------------------------------------------------------------
    #  GET NATION POLITICS (comprehensive)
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_nation_politics",
        action_description="Getting complete political info for nation {nation_id}",
        id_from_context=lambda ctx: "world_politics_manager"
    )
    @function_tool
    async def get_nation_politics(self, ctx, nation_id: int) -> Dict[str, Any]:
        """
        Get comprehensive political information about a nation:
        details, international relations, conflicts, domestic issues, relevant news, etc.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            Dictionary with comprehensive nation political information
        """
        cache_key = f"nation_politics_{nation_id}"
        cached = self.get_cache(cache_key)
        if cached:
            return cached

        await self.ensure_initialized()
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                nation = await conn.fetchrow("""
                    SELECT * FROM Nations WHERE id = $1
                """, nation_id)
                if not nation:
                    return {"error": "Nation not found"}

                relations = await conn.fetch("""
                    SELECT r.*,
                           CASE WHEN r.nation1_id = $1 THEN r.nation2_id ELSE r.nation1_id END AS other_nation_id,
                           n.name AS other_nation_name, n.government_type AS other_government_type
                    FROM InternationalRelations r
                    JOIN Nations n ON (
                        CASE WHEN r.nation1_id = $1 THEN r.nation2_id ELSE r.nation1_id END = n.id
                    )
                    WHERE r.nation1_id = $1 OR r.nation2_id = $1
                """, nation_id)

                conflicts = await conn.fetch("""
                    SELECT c.*
                    FROM NationalConflicts c
                    WHERE $1 = ANY(c.involved_nations)
                    ORDER BY c.severity DESC
                """, nation_id)

                issues = await conn.fetch("""
                    SELECT *
                    FROM DomesticIssues
                    WHERE nation_id = $1
                    ORDER BY severity DESC
                """, nation_id)

                conflict_news = await conn.fetch("""
                    SELECT n.*
                    FROM ConflictNews n
                    JOIN NationalConflicts c ON n.conflict_id = c.id
                    WHERE $1 = ANY(c.involved_nations)
                      AND n.source_nation = $1
                    ORDER BY n.publication_date DESC
                    LIMIT 5
                """, nation_id)

                domestic_news = await conn.fetch("""
                    SELECT n.*
                    FROM DomesticNews n
                    JOIN DomesticIssues i ON n.issue_id = i.id
                    WHERE i.nation_id = $1
                    ORDER BY n.publication_date DESC
                    LIMIT 5
                """, nation_id)

                result = {
                    "nation": dict(nation),
                    "international_relations": [dict(r) for r in relations],
                    "conflicts": [dict(c) for c in conflicts],
                    "domestic_issues": [dict(i) for i in issues],
                    "news": {
                        "international": [dict(x) for x in conflict_news],
                        "domestic": [dict(x) for x in domestic_news]
                    }
                }

                # parse JSON fields
                for item in result["domestic_issues"]:
                    if "public_opinion" in item and item["public_opinion"]:
                        try:
                            item["public_opinion"] = json.loads(item["public_opinion"])
                        except:
                            pass
                for c in result["conflicts"]:
                    if "public_opinion" in c and c["public_opinion"]:
                        try:
                            c["public_opinion"] = json.loads(c["public_opinion"])
                        except:
                            pass

                self.set_cache(cache_key, result, ttl=3600)
                return result

    # ------------------------------------------------------------------------
    #  EVOLVE ALL CONFLICTS
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_all_conflicts",
        action_description="Evolving all conflicts by time passage",
        id_from_context=lambda ctx: "world_politics_manager"
    )
    @function_tool
    async def evolve_all_conflicts(self, ctx, days_passed: int = 30) -> Dict[str, Any]:
        """
        Evolve all active conflicts after a certain number of days,
        using an LLM to decide how each conflict changes.
        
        Args:
            days_passed: Number of days to simulate passing
            
        Returns:
            Dictionary with evolution results
        """
        with trace(
            "EvolveAllConflicts",
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "days_passed": days_passed}
        ):
            run_ctx = RunContextWrapper(context=ctx.context)
            active_conflicts = await self.get_active_conflicts(run_ctx)
            all_nations = await self.get_all_nations(run_ctx)
            nations_by_id = {n["id"]: n for n in all_nations}
    
            evolution_agent = Agent(
                name="ConflictEvolutionAgent",
                instructions="You evolve international conflicts over time in a matriarchal fantasy world.",
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.8)
            )
    
            evolution_results = {
                "days_passed": days_passed,
                "evolved_conflicts": [],
                "resolved_conflicts": [],
                "new_developments": [],
                "status_changes": []
            }
    
            for conflict in active_conflicts:
                conflict_id = conflict["id"]
                involved_nation_ids = conflict.get("involved_nations", [])
                involved_nations = [
                    nations_by_id.get(nid, {"id": nid, "name": "Unknown"})
                    for nid in involved_nation_ids
                ]
    
                prompt = f"""
                Evolve this conflict over {days_passed} days:
    
                CONFLICT:
                {json.dumps(conflict, indent=2)}
    
                INVOLVED NATIONS:
                {json.dumps(involved_nations, indent=2)}
    
                Consider:
                - current status: {conflict.get('status','active')}
                - severity: {conflict.get('severity',5)}/10
                - realistic progression and diplomacy
                - matriarchal power structure
                - possible resolution
    
                Return JSON:
                - conflict_id: {conflict_id}
                - new_status (active, escalating, resolved, etc.)
                - severity_change: int from -3 to +3
                - new_developments: array
                - casualties_update
                - economic_impact_update
                - diplomatic_consequences_update
                - resolution_details (if resolved)
                """
    
                run_config = RunConfig(workflow_name="ConflictEvolution")
                result = await Runner.run(evolution_agent, prompt, context=run_ctx.context, run_config=run_config)
    
                try:
                    evo_data = json.loads(result.final_output)
                    old_status = conflict.get("status","active")
                    new_status = evo_data.get("new_status", old_status)
                    severity_change = evo_data.get("severity_change", 0)
                    old_sev = conflict.get("severity",5)
                    new_severity = max(1, min(10, old_sev + severity_change))
    
                    was_resolved = new_status.lower() == "resolved"
    
                    if old_status != new_status:
                        evolution_results["status_changes"].append({
                            "conflict_id": conflict_id,
                            "conflict_name": conflict.get("name","Unnamed"),
                            "old_status": old_status,
                            "new_status": new_status
                        })
    
                    new_devs = evo_data.get("new_developments",[])
                    if new_devs:
                        evolution_results["new_developments"].append({
                            "conflict_id": conflict_id,
                            "conflict_name": conflict.get("name","Unnamed"),
                            "developments": new_devs
                        })
    
                    # Update DB
                    async with self.get_connection_pool() as pool:
                        async with pool.acquire() as conn:
                            await conn.execute("""
                                UPDATE NationalConflicts
                                SET status = $1,
                                    severity = $2,
                                    current_casualties = $3,
                                    economic_impact = $4,
                                    diplomatic_consequences = $5,
                                    recent_developments = recent_developments || $6,
                                    end_date = CASE WHEN $7 = TRUE THEN $8 ELSE end_date END
                                WHERE id = $9
                            """,
                                new_status,
                                new_severity,
                                evo_data.get("casualties_update", conflict.get("current_casualties")),
                                evo_data.get("economic_impact_update", conflict.get("economic_impact")),
                                evo_data.get("diplomatic_consequences_update", conflict.get("diplomatic_consequences")),
                                new_devs,
                                was_resolved,
                                "Recently",  # or an actual date string
                                conflict_id
                            )
    
                            # Possibly generate a news item from one involved nation's perspective
                            if new_devs and involved_nations:
                                await self._generate_conflict_update_news(
                                    run_ctx, conflict_id, conflict, evo_data, involved_nations[0]
                                )
    
                    updated_conflict = {**conflict,
                                        "status": new_status,
                                        "severity": new_severity,
                                        "new_developments": new_devs}
    
                    if was_resolved:
                        evolution_results["resolved_conflicts"].append({
                            "conflict_id": conflict_id,
                            "conflict_name": conflict.get("name","Unnamed"),
                            "resolution_details": evo_data.get("resolution_details","The conflict resolved.")
                        })
                    else:
                        evolution_results["evolved_conflicts"].append(updated_conflict)
    
                except Exception as e:
                    logger.error(f"Error evolving conflict {conflict_id}: {e}")
    
            self.invalidate_cache("active_conflicts")
            return evolution_results

    async def _generate_conflict_update_news(
        self,
        ctx,
        conflict_id: int,
        conflict: Dict[str, Any],
        evo_data: Dict[str, Any],
        nation: Dict[str, Any]
    ) -> None:
        """
        Generate a news update about recent conflict developments.
        """
        with trace(
            "GenerateConflictUpdateNews",
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "conflict_id": conflict_id,
                "conflict_name": conflict.get("name", "Unnamed Conflict")
            }
        ):
            news_agent = Agent(
                name="ConflictNewsUpdateAgent",
                instructions="You create news updates about evolving international conflicts in a matriarchal world.",
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.8)
            )
    
            devs = "\n".join([f"- {d}" for d in evo_data.get("new_developments",[])])
            prompt = f"""
            Generate a news article about these developments in an ongoing conflict:
    
            CONFLICT:
            {json.dumps(conflict, indent=2)}
    
            NEW DEVELOPMENTS:
            {devs}
    
            REPORTING NATION:
            {json.dumps(nation, indent=2)}
    
            Return JSON:
            - headline
            - content
            """
    
            run_config = RunConfig(workflow_name="ConflictUpdateNews")
            result = await Runner.run(news_agent, prompt, context=ctx.context, run_config=run_config)
    
            try:
                news_data = json.loads(result.final_output)
                if not all(k in news_data for k in ["headline","content"]):
                    return
    
                # Theming
                news_data["content"] = MatriarchalThemingUtils.apply_matriarchal_theme("news", news_data["content"], 1)
    
                embed_text = f"{news_data.get('headline','No Headline')} {news_data.get('content','')[:200]}"
                emb = await get_embedding(embed_text)
                if not isinstance(emb, list):
                    emb = emb.tolist()
    
                bias = "pro_aggressor" if nation["id"] == conflict.get("primary_aggressor") else "pro_defender"
    
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        await conn.execute("""
                            INSERT INTO ConflictNews (
                                conflict_id, headline, content, source_nation, bias, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                            conflict_id,
                            news_data.get("headline","No Headline"),
                            news_data.get("content",""),
                            nation["id"],
                            bias,
                            emb
                        )
    
            except Exception as e:
                logger.error(f"Error generating conflict update news: {e}")

    # ------------------------------------------------------------------------
    #  POLITICAL REFORM ENGINE
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="simulate_political_reforms",
        action_description="Model how governments evolve in response to internal and external pressures",
        id_from_context=lambda ctx: "world_politics_manager"
    )
    @function_tool
    async def simulate_political_reforms(self, ctx, nation_id: int) -> Dict[str, Any]:
        """
        Model how a nation's political system might evolve under pressure.
        Could shift government type, matriarchy level, or create new laws.
        
        Args:
            nation_id: ID of the nation to simulate reforms for
            
        Returns:
            Dictionary describing any changes applied
        """
        with trace(
            "SimulatePoliticalReforms",
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "nation_id": nation_id}
        ):
            run_ctx = RunContextWrapper(context=ctx.context)
            # Load existing nation info
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    nation = await conn.fetchrow("SELECT * FROM Nations WHERE id=$1", nation_id)
                    if not nation:
                        return {"error": "Nation not found"}
    
            nation_data = dict(nation)
            # Example prompt to an LLM:
            agent = Agent(
                name="PoliticalReformAgent",
                instructions="Consider internal and external pressures to propose feasible political reforms.",
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.8)
            )
    
            prompt = f"""
            The nation below might undergo political reforms due to recent pressures. 
            Suggest possible reforms (e.g. government structure changes, new laws, shifting matriarchy level, etc.):
    
            NATION DATA:
            {json.dumps(nation_data, indent=2)}
    
            Return JSON with:
            - new_government_type
            - matriarchy_level_change
            - major_reforms (list)
            - rationale
            """
    
            result = await Runner.run(agent, prompt, context=run_ctx.context)
            try:
                reforms = json.loads(result.final_output)
            except json.JSONDecodeError:
                return {"error": "Could not parse reforms", "raw_output": result.final_output}
    
            # Apply these changes to the DB
            new_govt_type = reforms.get("new_government_type") or nation_data["government_type"]
            new_matriarchy = nation_data["matriarchy_level"] + reforms.get("matriarchy_level_change", 0)
            new_matriarchy = max(1, min(10, new_matriarchy))  # clamp 1..10
    
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE Nations
                        SET government_type = $1,
                            matriarchy_level = $2
                        WHERE id = $3
                    """, new_govt_type, new_matriarchy, nation_id)
    
            reforms["updated_nation"] = {
                "id": nation_id,
                "government_type": new_govt_type,
                "matriarchy_level": new_matriarchy
            }
            return reforms

    # ------------------------------------------------------------------------
    #  DYNASTY TRACKING SYSTEM
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="track_dynasty_lineage",
        action_description="Follow a political family's lineage across generations",
        id_from_context=lambda ctx: "world_politics_manager"
    )
    @function_tool
    async def track_dynasty_lineage(
        self,
        ctx,
        dynasty_id: int,
        generations_to_advance: int = 1
    ) -> Dict[str, Any]:
        """
        Advance a dynasty by some generations,
        creating new members, possibly changing rulers, and storing it.
        
        Args:
            dynasty_id: ID of the dynasty to track
            generations_to_advance: Number of generations to advance the dynasty
            
        Returns:
            Dictionary with updated dynasty info
        """
        with trace(
            "TrackDynastyLineage",
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata, 
                "dynasty_id": dynasty_id,
                "generations": generations_to_advance
            }
        ):
            run_ctx = RunContextWrapper(context=ctx.context)
            # Load existing dynasty
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    dynasty = await conn.fetchrow("SELECT * FROM Dynasties WHERE id=$1", dynasty_id)
                    if not dynasty:
                        return {"error": "Dynasty not found"}
    
            dynasty_data = dict(dynasty)
            # We'll pass current dynasty to an agent to generate next generation details
            agent = Agent(
                name="DynastyAgent",
                instructions="You simulate how a dynasty evolves over multiple generations in a matriarchal fantasy world.",
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.7)
            )
    
            prompt = f"""
            The dynasty:
            {json.dumps(dynasty_data, indent=2)}
    
            Advance it by {generations_to_advance} generation(s). 
            Return JSON:
            {{
              "new_members": [...],  # each with name, birth_date, mother_name, father_name, is_ruler, notes
              "successions": [...],  # describe who inherits leadership, if any
              "update_nation_ruler": {{"nation_id": X, "new_ruler": "..."}}, # optional
              "summary": "..."
            }}
            """
    
            result = await Runner.run(agent, prompt, context=run_ctx.context)
            try:
                lineage_updates = json.loads(result.final_output)
            except json.JSONDecodeError:
                return {"error": "Could not parse lineage data", "raw_output": result.final_output}
    
            # Insert new members into DynastyLineages
            new_members = lineage_updates.get("new_members", [])
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    for m in new_members:
                        await conn.execute("""
                            INSERT INTO DynastyLineages (
                                dynasty_id, member_name, birth_date, death_date,
                                mother_name, father_name, inheritor_of, is_ruler, notes
                            )
                            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                        """,
                            dynasty_id,
                            m.get("name"),
                            m.get("birth_date"),
                            m.get("death_date"),
                            m.get("mother_name"),
                            m.get("father_name"),
                            m.get("inheritor_of"),
                            m.get("is_ruler", False),
                            m.get("notes","")
                        )
                    # Possibly also update the nation's current ruler if needed
                    if "update_nation_ruler" in lineage_updates:
                        info = lineage_updates["update_nation_ruler"]
                        new_ruler = info.get("new_ruler")
                        if new_ruler and info.get("nation_id"):
                            await conn.execute("""
                                UPDATE Nations
                                SET description = description || ' New ruler: ' || $2
                                WHERE id = $1
                            """, info["nation_id"], new_ruler)
    
            return {
                "dynasty_id": dynasty_id,
                "updates": lineage_updates
            }

    # ------------------------------------------------------------------------
    #  FACTION PROXIES INITIALIZATION
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="initialize_faction_proxies",
        action_description="Creating faction agent proxies with competing goals",
        id_from_context=lambda ctx: "world_politics_manager"
    )
    @function_tool
    async def initialize_faction_proxies(self, ctx) -> Dict[str, Any]:
        """
        Initialize agent proxies for all factions in the world.
        
        Returns:
            Dictionary with initialization status
        """
        with trace(
            "InitializeFactionProxies",
            group_id=self.trace_group_id,
            metadata=self.trace_metadata
        ):
            run_ctx = RunContextWrapper(context=ctx.context)
    
            await self.ensure_initialized()
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    factions = await conn.fetch("""
                        SELECT id, name, type, description, values, goals, cultural_traits, territory
                        FROM Factions
                    """)
    
            self.faction_proxies = {}
            for faction in factions:
                faction_data = dict(faction)
                proxy = FactionAgentProxy(faction_data)
                self.faction_proxies[faction_data['id']] = proxy
    
            return {"status": "success", "factions_initialized": len(self.faction_proxies)}

    # ------------------------------------------------------------------------
    #  HELPER METHODS (Loaders, etc.)
    # ------------------------------------------------------------------------
    async def _get_conflict_details(self, conflict_id: int) -> Optional[Dict[str, Any]]:
        """Load conflict details from DB."""
        await self.ensure_initialized()
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM NationalConflicts WHERE id=$1", conflict_id)
                if row:
                    conflict_data = dict(row)
                    if "public_opinion" in conflict_data and conflict_data["public_opinion"]:
                        try:
                            conflict_data["public_opinion"] = json.loads(conflict_data["public_opinion"])
                        except json.JSONDecodeError:
                            pass
                    return conflict_data
        return None

    async def _load_negotiating_nations(self, nation1_id: int, nation2_id: int) -> Dict[str, Any]:
        """Helper to load two nations for negotiation."""
        await self.ensure_initialized()
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                n1 = await conn.fetchrow("SELECT * FROM Nations WHERE id=$1", nation1_id)
                n2 = await conn.fetchrow("SELECT * FROM Nations WHERE id=$1", nation2_id)
                if not n1 or not n2:
                    return {"error": "One or both nations not found"}
                return {"nation1": dict(n1), "nation2": dict(n2)}

    async def _update_international_relations(self, nations: Dict[str, Any], negotiation_results: Dict[str, Any]):
        """Update international relations after negotiations."""
        if "relationship_change" not in negotiation_results:
            return
            
        relationship_change = negotiation_results.get("relationship_change", 0)
        if relationship_change == 0:
            return
            
        nation1_id = nations["nation1"]["id"]
        nation2_id = nations["nation2"]["id"]
        
        await self.ensure_initialized()
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if relation exists
                relation = await conn.fetchrow("""
                    SELECT * FROM InternationalRelations
                    WHERE (nation1_id = $1 AND nation2_id = $2)
                       OR (nation1_id = $2 AND nation2_id = $1)
                """, nation1_id, nation2_id)
                
                if relation:
                    # Update existing relation
                    new_quality = max(1, min(10, relation["relationship_quality"] + relationship_change))
                    
                    # Add negotiation outcome to description
                    agreement_reached = negotiation_results.get("agreement_reached", False)
                    agreement_desc = f"\nNegotiation on {negotiation_results.get('issue', 'an issue')} "
                    agreement_desc += "resulted in agreement. " if agreement_reached else "failed to reach agreement. "
                    
                    await conn.execute("""
                        UPDATE InternationalRelations
                        SET relationship_quality = $3,
                            description = description || $4
                        WHERE id = $5
                    """, new_quality, agreement_desc, relation["id"])
                else:
                    # Create new relation with default values plus negotiation results
                    relation_type = "friendly" if relationship_change > 0 else "neutral" if relationship_change == 0 else "tense"
                    quality = 5 + relationship_change  # Base of 5 plus change
                    quality = max(1, min(10, quality))  # Clamp to 1-10
                    
                    description = f"Relationship established following negotiations on {negotiation_results.get('issue', 'an issue')}. "
                    description += "Agreement was reached. " if negotiation_results.get("agreement_reached", False) else "No agreement was reached. "
                    
                    await self.add_international_relation(
                        RunContextWrapper(context=None),
                        nation1_id=nation1_id,
                        nation2_id=nation2_id,
                        relationship_type=relation_type,
                        relationship_quality=quality,
                        description=description
                    )

    async def _load_event_details(self, event_id: int) -> Dict[str, Any]:
        """
        Load event details from the DB. First checks for national conflicts,
        then domestic issues, falling back to a generic event schema.
        """
        await self.ensure_initialized()
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Try to load as a national conflict
                conflict = await conn.fetchrow("SELECT * FROM NationalConflicts WHERE id=$1", event_id)
                if conflict:
                    event_data = dict(conflict)
                    event_data["event_type"] = "national_conflict"
                    return event_data
                    
                # Try to load as a domestic issue
                issue = await conn.fetchrow("SELECT * FROM DomesticIssues WHERE id=$1", event_id)
                if issue:
                    event_data = dict(issue)
                    event_data["event_type"] = "domestic_issue"
                    return event_data
                    
        # Not found in either table
        return {"error": "Event not found", "event_id": event_id}

    async def _store_media_coverage(self, event_id: int, article: Dict[str, Any]) -> None:
        """Store coverage for a given event into DB."""
        await self.ensure_initialized()
        event = await self._load_event_details(event_id)
        
        if "error" in event:
            return
            
        event_type = event.get("event_type")
        
        # Prepare embedding
        embed_text = f"{article.get('headline','')} {article.get('content','')[:200]}"
        emb = await get_embedding(embed_text)
        if not isinstance(emb, list):
            emb = emb.tolist()
            
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                if event_type == "national_conflict":
                    # Store as conflict news
                    await conn.execute("""
                        INSERT INTO ConflictNews (
                            conflict_id, headline, content, bias, embedding
                        )
                        VALUES ($1, $2, $3, $4, $5)
                    """,
                        event_id,
                        article.get("headline", "No headline"),
                        article.get("content", "No content"),
                        article.get("bias", "neutral"),
                        emb
                    )
                elif event_type == "domestic_issue":
                    # Store as domestic news
                    await conn.execute("""
                        INSERT INTO DomesticNews (
                            issue_id, headline, content, source_faction, bias, embedding
                        )
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                        event_id,
                        article.get("headline", "No headline"),
                        article.get("content", "No content"),
                        article.get("media_name", "Unknown"),
                        article.get("bias", "neutral"),
                        emb
                    )

    async def generate_and_store_embedding(self, text: str, conn, table_name: str, key_name: str, key_value: int):
        """
        Helper to generate an embedding and store it in the specified table/column.
        Uses your real embedding service from utils.embedding_service.
        """
        emb = await get_embedding(text)
        if not isinstance(emb, list):
            emb = emb.tolist()
        await conn.execute(
            f"UPDATE {table_name} SET embedding=$1 WHERE {key_name}=$2",
            emb, key_value
        )

    # ------------------------------------------------------------------------
    #  REGISTRATION WITH GOVERNANCE
    # ------------------------------------------------------------------------
    async def register_with_governance(self):
        """
        Register with Nyx governance system.
        """
        await super().register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="world_politics_manager",
            directive_text="Manage nations, international relations, and conflicts in a matriarchal world.",
            scope="world_building",
            priority=DirectivePriority.MEDIUM
        )
