# lore/managers/politics.py

import logging
import json
import random
from typing import Dict, List, Any, Optional
from datetime import datetime

from agents import Agent, Runner
from agents.run_context import RunContextWrapper

from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

from embedding.vector_store import generate_embedding

from lore.core.base_manager import BaseLoreManager
from lore.utils.theming import MatriarchalThemingUtils

class WorldPoliticsManager(BaseLoreManager):
    """
    Consolidated manager for geopolitical landscape and conflicts.
    Handles nations, international relations, and both international and
    domestic conflicts.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.cache_namespace = "world_politics"
    
    async def _initialize_tables(self):
        """Initialize all required tables for geopolitics and conflicts"""
        # Nation and international relation tables
        geo_tables = {
            "Nations": """
                CREATE TABLE Nations (
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
                CREATE TABLE InternationalRelations (
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
        
        # Conflict tables
        conflict_tables = {
            "NationalConflicts": """
                CREATE TABLE NationalConflicts (
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
                CREATE TABLE ConflictNews (
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
                CREATE TABLE DomesticIssues (
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
                CREATE TABLE DomesticNews (
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
        
        # Initialize all tables
        all_tables = {**geo_tables, **conflict_tables}
        await self.initialize_tables_from_definitions(all_tables)
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_nation",
        action_description="Adding nation: {name}",
        id_from_context=lambda ctx: "world_politics_manager"
    )
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
        Add a nation to the database
        
        Args:
            name: Name of the nation
            government_type: Type of government
            description: Detailed description
            relative_power: Power level (1-10)
            matriarchy_level: How matriarchal (1-10)
            population_scale: Scale of population
            major_resources: Key resources
            major_cities: Key cities/settlements
            cultural_traits: Defining cultural traits
            notable_features: Other notable features
            neighboring_nations: Nations that border this one
            
        Returns:
            ID of the created nation
        """
        # Ensure tables exist
        await self.ensure_initialized()
        
        # Set defaults
        major_resources = major_resources or []
        major_cities = major_cities or []
        cultural_traits = cultural_traits or []
        neighboring_nations = neighboring_nations or []
        
        # Store in database
        async with await self.get_connection_pool() as pool:
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
                """, name, government_type, description, relative_power,
                     matriarchy_level, population_scale, major_resources,
                     major_cities, cultural_traits, notable_features,
                     neighboring_nations)
                
                # Generate and store embedding
                embedding_text = f"{name} {government_type} {description}"
                await self.generate_and_store_embedding(embedding_text, conn, "Nations", "id", nation_id)
                
                return nation_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_international_relation",
        action_description="Adding relation between nations",
        id_from_context=lambda ctx: "world_politics_manager"
    )
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
        Add a relation between two nations
        
        Args:
            nation1_id: ID of first nation
            nation2_id: ID of second nation
            relationship_type: Type of relationship (ally, rival, etc.)
            relationship_quality: Quality level (1-10)
            description: Description of relationship
            notable_conflicts: Notable conflicts
            notable_alliances: Notable alliances
            trade_relations: Description of trade
            cultural_exchanges: Description of cultural exchanges
            
        Returns:
            ID of the created relation
        """
        # Ensure tables exist
        await self.ensure_initialized()
        
        # Set defaults
        notable_conflicts = notable_conflicts or []
        notable_alliances = notable_alliances or []
        
        # Store in database
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                relation_id = await conn.fetchval("""
                    INSERT INTO InternationalRelations (
                        nation1_id, nation2_id, relationship_type,
                        relationship_quality, description, notable_conflicts,
                        notable_alliances, trade_relations, cultural_exchanges
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (nation1_id, nation2_id) DO UPDATE
                    SET relationship_type = $3,
                        relationship_quality = $4,
                        description = $5,
                        notable_conflicts = $6,
                        notable_alliances = $7,
                        trade_relations = $8,
                        cultural_exchanges = $9
                    RETURNING id
                """, nation1_id, nation2_id, relationship_type,
                     relationship_quality, description, notable_conflicts,
                     notable_alliances, trade_relations, cultural_exchanges)
                
                return relation_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_all_nations",
        action_description="Getting all nations in the world",
        id_from_context=lambda ctx: "world_politics_manager"
    )
    async def get_all_nations(self, ctx) -> List[Dict[str, Any]]:
        """
        Get all nations in the world
        
        Returns:
            List of all nations
        """
        # Check cache first
        cache_key = f"all_nations_{self.user_id}_{self.conversation_id}"
        cached = self.get_cache(cache_key)
        if cached:
            return cached
            
        # Ensure tables exist
        await self.ensure_initialized()
        
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                nations = await conn.fetch("""
                    SELECT id, name, government_type, description, relative_power,
                           matriarchy_level, population_scale, major_resources,
                           major_cities, cultural_traits, notable_features, 
                           neighboring_nations
                    FROM Nations
                    ORDER BY relative_power DESC
                """)
                
                result = [dict(nation) for nation in nations]
                
                # Cache the result
                self.set_cache(cache_key, result, ttl=3600)  # 1 hour TTL
                
                return result
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_initial_conflicts",
        action_description="Generating initial national conflicts",
        id_from_context=lambda ctx: "world_politics_manager"
    )
    async def generate_initial_conflicts(self, ctx, count: int = 3) -> List[Dict[str, Any]]:
        """
        Generate initial conflicts between nations with governance oversight.
        
        Args:
            count: Number of conflicts to generate
            
        Returns:
            List of generated conflicts
        """
        # Create the run context
        run_ctx = self.create_run_context(ctx)
        
        # Get nations for context
        nations = await self.get_all_nations(run_ctx)
        
        if len(nations) < 2:
            return []
        
        conflicts = []
        
        # Create agent for conflict generation
        conflict_agent = Agent(
            name="NationalConflictAgent",
            instructions="You create realistic international conflicts for a fantasy world with matriarchal power structures.",
            model="o3-mini"
        )
        
        for i in range(count):
            # Select random nations that aren't already in major conflicts
            available_nations = [n for n in nations if not any(
                n["id"] in c.get("involved_nations", []) for c in conflicts
            )]
            
            if len(available_nations) < 2:
                available_nations = nations  # Fallback if needed
            
            # Choose two random nations
            nation_pair = random.sample(available_nations, 2)
            
            # Determine conflict type based on nations' characteristics
            matriarchy_diff = abs(
                nation_pair[0].get("matriarchy_level", 5) - 
                nation_pair[1].get("matriarchy_level", 5)
            )
            
            # Higher difference makes ideological conflicts more likely
            if matriarchy_diff > 4:
                conflict_types = ["ideological_dispute", "cultural_tension", "religious_conflict", "proxy_war"]
            elif matriarchy_diff > 2:
                conflict_types = ["diplomatic_tension", "border_dispute", "trade_dispute", "resource_conflict"]
            else:
                conflict_types = ["territorial_dispute", "trade_war", "succession_crisis", "alliance_dispute"]
                
            # Randomly select conflict type
            conflict_type = random.choice(conflict_types)
            
            # Create prompt for the agent
            prompt = f"""
            Generate a detailed international conflict between these two nations:
            
            NATION 1:
            {json.dumps(nation_pair[0], indent=2)}
            
            NATION 2:
            {json.dumps(nation_pair[1], indent=2)}
            
            Create a {conflict_type} that:
            1. Makes sense given the nations' characteristics
            2. Has appropriate severity and clear causes
            3. Includes realistic consequences and casualties
            4. Considers the matriarchal nature of the world
            5. Reflects how the differing matriarchy levels ({matriarchy_diff} point difference) might cause tension
            
            Return a JSON object with:
            - name: Name of the conflict
            - conflict_type: "{conflict_type}"
            - description: Detailed description
            - severity: Severity level (1-10)
            - status: Current status (active, escalating, etc.)
            - start_date: When it started (narrative date)
            - involved_nations: IDs of involved nations
            - primary_aggressor: ID of the primary aggressor
            - primary_defender: ID of the primary defender
            - current_casualties: Description of casualties so far
            - economic_impact: Description of economic impact
            - diplomatic_consequences: Description of diplomatic fallout
            - public_opinion: Object with nation IDs as keys and opinion descriptions as values
            - recent_developments: Array of recent events in the conflict
            - potential_resolution: Potential ways it might end
            """
            
            # Get response from agent
            result = await Runner.run(conflict_agent, prompt, context=run_ctx.context)
            
            try:
                # Parse response
                conflict_data = json.loads(result.final_output)
                
                # Ensure required fields exist
                if not all(k in conflict_data for k in ["name", "description", "conflict_type", "severity", "status"]):
                    continue
                
                # Generate embedding
                embedding_text = f"{conflict_data['name']} {conflict_data['description']} {conflict_data['conflict_type']}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with await self.get_connection_pool() as pool:
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
                        conflict_data.get("name"), 
                        conflict_data.get("conflict_type"),
                        conflict_data.get("description"),
                        conflict_data.get("severity", 5),
                        conflict_data.get("status", "active"),
                        conflict_data.get("start_date", "Recently"),
                        conflict_data.get("involved_nations", [nation_pair[0]["id"], nation_pair[1]["id"]]),
                        conflict_data.get("primary_aggressor", nation_pair[0]["id"]),
                        conflict_data.get("primary_defender", nation_pair[1]["id"]),
                        conflict_data.get("current_casualties", "Unknown"),
                        conflict_data.get("economic_impact", "Unknown"),
                        conflict_data.get("diplomatic_consequences", "Unknown"),
                        json.dumps(conflict_data.get("public_opinion", {})),
                        conflict_data.get("recent_developments", []),
                        conflict_data.get("potential_resolution", "Unknown"),
                        embedding)
                        
                        # Generate initial news about this conflict
                        await self._generate_conflict_news(run_ctx, conflict_id, conflict_data, nation_pair)
                        
                        # Add to result
                        conflict_data["id"] = conflict_id
                        conflicts.append(conflict_data)
                        
            except Exception as e:
                logging.error(f"Error generating conflict: {e}")
        
        return conflicts
    
    async def _generate_conflict_news(
        self, 
        ctx, 
        conflict_id: int, 
        conflict_data: Dict[str, Any],
        nations: List[Dict[str, Any]]
    ) -> None:
        """Generate initial news articles about a conflict"""
        # Create agent for news generation
        news_agent = Agent(
            name="ConflictNewsAgent",
            instructions="You create realistic news articles about international conflicts in a matriarchal world.",
            model="o3-mini"
        )
        
        # Generate one news article from each nation's perspective
        for i, nation in enumerate(nations[:2]):
            bias = "pro_defender" if nation["id"] == conflict_data.get("primary_defender") else "pro_aggressor"
            
            # Create prompt for the agent
            prompt = f"""
            Generate a news article about this conflict from the perspective of {nation["name"]}:
            
            CONFLICT:
            {json.dumps(conflict_data, indent=2)}
            
            REPORTING NATION:
            {json.dumps(nation, indent=2)}
            
            Create a news article that:
            1. Has a clear {bias} bias
            2. Includes quotes from officials (primarily women in positions of power)
            3. Covers the key facts but with the nation's spin
            4. Has a catchy headline
            5. Reflects matriarchal power structures in its language and reporting style
            
            Return a JSON object with:
            - headline: The article headline
            - content: The full article content (300-500 words)
            """
            
            # Get response from agent
            result = await Runner.run(news_agent, prompt, context=ctx.context)
            
            try:
                # Parse response
                news_data = json.loads(result.final_output)
                
                # Ensure required fields exist
                if not all(k in news_data for k in ["headline", "content"]):
                    continue
                
                # Apply matriarchal theming to content
                news_data["content"] = MatriarchalThemingUtils._replace_gendered_words(news_data["content"])
                
                # Generate embedding
                embedding_text = f"{news_data['headline']} {news_data['content'][:200]}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with await self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        await conn.execute("""
                            INSERT INTO ConflictNews (
                                conflict_id, headline, content, source_nation, bias, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """, 
                        conflict_id,
                        news_data.get("headline"), 
                        news_data.get("content"),
                        nation["id"],
                        bias,
                        embedding)
                        
            except Exception as e:
                logging.error(f"Error generating conflict news: {e}")
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_domestic_issues",
        action_description="Generating domestic issues for nation {nation_id}",
        id_from_context=lambda ctx: "world_politics_manager"
    )
    async def generate_domestic_issues(self, ctx, nation_id: int, count: int = 2) -> List[Dict[str, Any]]:
        """
        Generate domestic issues for a specific nation with governance oversight.
        
        Args:
            nation_id: ID of the nation
            count: Number of issues to generate
            
        Returns:
            List of generated domestic issues
        """
        # Create the run context
        run_ctx = self.create_run_context(ctx)
        
        # Get nation details
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                nation = await conn.fetchrow("""
                    SELECT id, name, government_type, matriarchy_level, cultural_traits
                    FROM Nations
                    WHERE id = $1
                """, nation_id)
                
                if not nation:
                    return []
                
                nation_data = dict(nation)
                
                # Get factions in this nation
                factions = await conn.fetch("""
                    SELECT id, name, type, description, values
                    FROM Factions
                    WHERE $1 = ANY(territory)
                """, nation_data.get("name"))
                
                faction_data = [dict(faction) for faction in factions]
        
        # Create agent for domestic issue generation
        issue_agent = Agent(
            name="DomesticIssueAgent",
            instructions="You create realistic domestic political and social issues for fantasy nations with matriarchal power structures.",
            model="o3-mini"
        )
        
        # Determine issue types based on nation characteristics
        issue_types = []
        
        # Higher matriarchy has different issues than lower
        matriarchy_level = nation_data.get("matriarchy_level", 5)
        
        if matriarchy_level >= 8:
            # High matriarchy issues
            issue_types.extend([
                "male_rights_movement", "traditionalist_opposition", "matriarchy_reform", 
                "male_separatism", "gender_hierarchy_legislation"
            ])
        elif matriarchy_level <= 3:
            # Low matriarchy issues
            issue_types.extend([
                "feminist_movement", "equality_legislation", "patriarchal_opposition",
                "female_leadership_controversy", "gender_role_debates"
            ])
        else:
            # Balanced matriarchy issues
            issue_types.extend([
                "gender_balance_debate", "power_sharing_reform", "traditionalist_vs_progressive"
            ])
        
        # Universal issue types
        universal_issues = [
            "economic_crisis", "environmental_disaster", "disease_outbreak",
            "succession_dispute", "religious_controversy", "tax_reform",
            "military_service_debate", "trade_regulation", "education_policy",
            "infrastructure_development", "foreign_policy_shift", "corruption_scandal",
            "resource_scarcity", "technological_change", "constitutional_crisis",
            "land_rights_dispute", "criminal_justice_reform", "public_safety_concerns",
            "media_censorship", "social_services_funding"
        ]
        
        issue_types.extend(universal_issues)
        
        # Generate issues
        issues = []
        selected_types = random.sample(issue_types, min(count, len(issue_types)))
        
        for issue_type in selected_types:
            # Create prompt for the agent
            prompt = f"""
            Generate a domestic political or social issue for this nation:
            
            NATION:
            {json.dumps(nation_data, indent=2)}
            
            FACTIONS:
            {json.dumps(faction_data, indent=2)}
            
            Create a {issue_type} issue that:
            1. Makes sense given the nation's characteristics
            2. Creates realistic societal tension and debate
            3. Involves multiple factions or groups
            4. Considers the matriarchal level of the society ({matriarchy_level}/10)
                        
                        Return a JSON object with:
                        - name: Name of the issue/controversy
                        - issue_type: "{issue_type}"
                        - description: Detailed description
                        - severity: Severity level (1-10)
                        - status: Current status (emerging, active, waning, resolved)
                        - start_date: When it started (narrative date)
                        - supporting_factions: Groups supporting one side
                        - opposing_factions: Groups opposing
                        - neutral_factions: Groups remaining neutral
                        - affected_demographics: Demographics most affected
                        - public_opinion: Object describing opinion distribution
                        - government_response: How the government is responding
                        - recent_developments: Array of recent events in this issue
                        - political_impact: Impact on political landscape
                        - social_impact: Impact on society
                        - economic_impact: Economic consequences
                        - potential_resolution: Potential ways it might resolve
                        """
                        
                        # Get response from agent
                        result = await Runner.run(issue_agent, prompt, context=run_ctx.context)
                        
                        try:
                            # Parse response
                            issue_data = json.loads(result.final_output)
                            
                            # Ensure required fields exist
                            if not all(k in issue_data for k in ["name", "description", "issue_type"]):
                                continue
                            
                            # Add nation_id
                            issue_data["nation_id"] = nation_id
                            
                            # Generate embedding
                            embedding_text = f"{issue_data['name']} {issue_data['description']} {issue_data['issue_type']}"
                            embedding = await generate_embedding(embedding_text)
                            
                            # Store in database
                            async with await self.get_connection_pool() as pool:
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
                                    issue_data.get("severity", 5),
                                    issue_data.get("status", "active"),
                                    issue_data.get("start_date", "Recently"),
                                    issue_data.get("supporting_factions", []),
                                    issue_data.get("opposing_factions", []),
                                    issue_data.get("neutral_factions", []),
                                    issue_data.get("affected_demographics", []),
                                    json.dumps(issue_data.get("public_opinion", {})),
                                    issue_data.get("government_response", ""),
                                    issue_data.get("recent_developments", []),
                                    issue_data.get("political_impact", ""),
                                    issue_data.get("social_impact", ""),
                                    issue_data.get("economic_impact", ""),
                                    issue_data.get("potential_resolution", ""),
                                    embedding)
                                    
                                    # Generate initial news about this issue
                                    await self._generate_domestic_news(run_ctx, issue_id, issue_data, nation_data)
                                    
                                    # Add to result
                                    issue_data["id"] = issue_id
                                    issues.append(issue_data)
                                    
                        except Exception as e:
                            logging.error(f"Error generating domestic issue: {e}")
                    
                    return issues
                    
                async def _generate_domestic_news(
                    self, 
                    ctx, 
                    issue_id: int, 
                    issue_data: Dict[str, Any],
                    nation_data: Dict[str, Any]
                ) -> None:
                    """Generate initial news articles about a domestic issue"""
                    # Create agent for news generation
                    news_agent = Agent(
                        name="DomesticNewsAgent",
                        instructions="You create realistic news articles about domestic political issues in a matriarchal society.",
                        model="o3-mini"
                    )
                    
                    # Generate news articles from different perspectives
                    biases = ["supporting", "opposing", "neutral"]
                    
                    for bias in biases:
                        # Create prompt for the agent
                        prompt = f"""
                        Generate a news article about this domestic issue from a {bias} perspective:
                        
                        ISSUE:
                        {json.dumps(issue_data, indent=2)}
                        
                        NATION:
                        {json.dumps(nation_data, indent=2)}
                        
                        Create a news article that:
                        1. Has a clear {bias} bias toward the issue
                        2. Includes quotes from relevant figures
                        3. Covers the key facts but with the appropriate spin
                        4. Has a catchy headline
                        5. Reflects the matriarchal power structures of society
                        
                        Return a JSON object with:
                        - headline: The article headline
                        - content: The full article content (300-500 words)
                        - source_faction: The faction or institution publishing this
                        """
                        
                        # Get response from agent
                        result = await Runner.run(news_agent, prompt, context=ctx.context)
                        
                        try:
                            # Parse response
                            news_data = json.loads(result.final_output)
                            
                            # Ensure required fields exist
                            if not all(k in news_data for k in ["headline", "content"]):
                                continue
                            
                            # Apply matriarchal theming
                            news_data["content"] = MatriarchalThemingUtils.apply_matriarchal_theme("news", news_data["content"], 1)
                            
                            # Store in database
                            async with await self.get_connection_pool() as pool:
                                async with pool.acquire() as conn:
                                    news_id = await conn.fetchval("""
                                        INSERT INTO DomesticNews (
                                            issue_id, headline, content, source_faction, bias
                                        )
                                        VALUES ($1, $2, $3, $4, $5)
                                        RETURNING id
                                    """, 
                                    issue_id,
                                    news_data.get("headline"), 
                                    news_data.get("content"),
                                    news_data.get("source_faction", "Unknown Source"),
                                    bias)
                                    
                                    # Generate and store embedding
                                    embedding_text = f"{news_data['headline']} {news_data['content'][:200]}"
                                    await self.generate_and_store_embedding(embedding_text, conn, "DomesticNews", "id", news_id)
                                    
                        except Exception as e:
                            logging.error(f"Error generating domestic news: {e}")
                
                @with_governance(
                    agent_type=AgentType.NARRATIVE_CRAFTER,
                    action_type="get_active_conflicts",
                    action_description="Getting active national conflicts",
                    id_from_context=lambda ctx: "world_politics_manager"
                )
                async def get_active_conflicts(self, ctx) -> List[Dict[str, Any]]:
                    """
                    Get all active conflicts with governance oversight.
                    
                    Returns:
                        List of active conflicts
                    """
                    # Check cache first
                    cache_key = "active_conflicts"
                    cached = self.get_cache(cache_key)
                    if cached:
                        return cached
                    
                    # Query database for active conflicts
                    async with await self.get_connection_pool() as pool:
                        async with pool.acquire() as conn:
                            conflicts = await conn.fetch("""
                                SELECT * FROM NationalConflicts
                                WHERE status != 'resolved'
                                ORDER BY severity DESC
                            """)
                            
                            # Convert to list of dicts
                            result = [dict(conflict) for conflict in conflicts]
                            
                            # Parse JSON fields
                            for conflict in result:
                                if "public_opinion" in conflict and conflict["public_opinion"]:
                                    try:
                                        conflict["public_opinion"] = json.loads(conflict["public_opinion"])
                                    except:
                                        pass
                            
                            # Cache result
                            self.set_cache(cache_key, result, ttl=3600)  # 1 hour TTL
                            
                            return result
                            
                @with_governance(
                    agent_type=AgentType.NARRATIVE_CRAFTER,
                    action_type="get_nation_politics",
                    action_description="Getting complete political information for nation {nation_id}",
                    id_from_context=lambda ctx: "world_politics_manager"
                )
                async def get_nation_politics(self, ctx, nation_id: int) -> Dict[str, Any]:
                    """
                    Get comprehensive political information about a nation.
                    
                    Args:
                        nation_id: ID of the nation
                        
                    Returns:
                        Dictionary with complete political information
                    """
                    # Check cache first
                    cache_key = f"nation_politics_{nation_id}"
                    cached = self.get_cache(cache_key)
                    if cached:
                        return cached
                        
                    # Ensure tables exist
                    await self.ensure_initialized()
                    
                    # Query for all needed information
                    async with await self.get_connection_pool() as pool:
                        async with pool.acquire() as conn:
                            # 1. Get nation details
                            nation = await conn.fetchrow("""
                                SELECT * FROM Nations WHERE id = $1
                            """, nation_id)
                            
                            if not nation:
                                return {"error": "Nation not found"}
                            
                            # 2. Get international relations
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
                            
                            # 3. Get conflicts involving this nation
                            conflicts = await conn.fetch("""
                                SELECT c.*
                                FROM NationalConflicts c
                                WHERE $1 = ANY(c.involved_nations)
                                ORDER BY c.severity DESC
                            """, nation_id)
                            
                            # 4. Get domestic issues
                            issues = await conn.fetch("""
                                SELECT * FROM DomesticIssues
                                WHERE nation_id = $1
                                ORDER BY severity DESC
                            """, nation_id)
                            
                            # 5. Get related news (conflict + domestic)
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
                            
                            # Process the results
                            result = {
                                "nation": dict(nation),
                                "international_relations": [dict(rel) for rel in relations],
                                "conflicts": [dict(conflict) for conflict in conflicts],
                                "domestic_issues": [dict(issue) for issue in issues],
                                "news": {
                                    "international": [dict(news) for news in conflict_news],
                                    "domestic": [dict(news) for news in domestic_news]
                                }
                            }
                            
                            # Parse JSON fields
                            for item in result["domestic_issues"]:
                                if "public_opinion" in item and item["public_opinion"]:
                                    try:
                                        item["public_opinion"] = json.loads(item["public_opinion"])
                                    except:
                                        pass
                            
                            for item in result["conflicts"]:
                                if "public_opinion" in item and item["public_opinion"]:
                                    try:
                                        item["public_opinion"] = json.loads(item["public_opinion"])
                                    except:
                                        pass
                            
                            # Cache the result
                            self.set_cache(cache_key, result, ttl=3600)  # 1 hour TTL
                            
                            return result
                
                @with_governance(
                    agent_type=AgentType.NARRATIVE_CRAFTER,
                    action_type="evolve_all_conflicts",
                    action_description="Evolving all conflicts by time passage",
                    id_from_context=lambda ctx: "world_politics_manager"
                )
                async def evolve_all_conflicts(self, ctx, days_passed: int = 30) -> Dict[str, Any]:
                    """
                    Evolve all active conflicts based on time passing.
                    
                    Args:
                        days_passed: Number of days to simulate
                        
                    Returns:
                        Dictionary with evolution results
                    """
                    # Create run context
                    run_ctx = self.create_run_context(ctx)
                    
                    # Get active conflicts
                    active_conflicts = await self.get_active_conflicts(run_ctx)
                    
                    # Get nations for context
                    all_nations = await self.get_all_nations(run_ctx)
                    nations_by_id = {n["id"]: n for n in all_nations}
                    
                    # Create agent for conflict evolution
                    evolution_agent = Agent(
                        name="ConflictEvolutionAgent",
                        instructions="You evolve international conflicts over time in a matriarchal fantasy world.",
                        model="o3-mini"
                    )
                    
                    # Track evolution results
                    evolution_results = {
                        "days_passed": days_passed,
                        "evolved_conflicts": [],
                        "resolved_conflicts": [],
                        "new_developments": [],
                        "status_changes": []
                    }
                    
                    # Process each active conflict
                    for conflict in active_conflicts:
                        conflict_id = conflict["id"]
                        
                        # Get involved nations
                        involved_nation_ids = conflict.get("involved_nations", [])
                        involved_nations = [nations_by_id.get(nid, {"id": nid, "name": "Unknown Nation"}) 
                                           for nid in involved_nation_ids if nid in nations_by_id]
                        
                        # Create prompt for evolution
                        prompt = f"""
                        Evolve this conflict over a period of {days_passed} days:
                        
                        CONFLICT:
                        {json.dumps(conflict, indent=2)}
                        
                        INVOLVED NATIONS:
                        {json.dumps(involved_nations, indent=2)}
                        
                        Consider how this conflict would evolve over {days_passed} days, factoring in:
                        1. The current status ({conflict.get('status', 'active')})
                        2. The severity level ({conflict.get('severity', 5)}/10)
                        3. Realistic conflict progression and diplomacy
                        4. The matriarchal power dynamics of this world
                        
                        Return a JSON object with:
                        - conflict_id: {conflict_id}
                        - new_status: Updated status (active, escalating, de-escalating, resolved, etc.)
                        - severity_change: Change in severity (-3 to +3)
                        - new_developments: Array of new events that occurred
                        - casualties_update: Updated casualty information
                        - economic_impact_update: Updated economic impact
                        - diplomatic_consequences_update: Updated diplomatic consequences
                        - resolution_details: Details on resolution (if resolved)
                        """
                        
                        # Get response from agent
                        result = await Runner.run(evolution_agent, prompt, context=run_ctx.context)
                        
                        try:
                            # Parse response
                            evolution_data = json.loads(result.final_output)
                            
                            # Calculate new severity
                            old_severity = conflict.get("severity", 5)
                            severity_change = evolution_data.get("severity_change", 0)
                            new_severity = max(1, min(10, old_severity + severity_change))
                            
                            # Track status change
                            old_status = conflict.get("status", "active")
                            new_status = evolution_data.get("new_status", old_status)
                            
                            if old_status != new_status:
                                evolution_results["status_changes"].append({
                                    "conflict_id": conflict_id,
                                    "conflict_name": conflict.get("name", "Unnamed Conflict"),
                                    "old_status": old_status,
                                    "new_status": new_status
                                })
                            
                            # Track if resolved
                            was_resolved = new_status.lower() == "resolved"
                            
                            # New developments
                            new_developments = evolution_data.get("new_developments", [])
                            if new_developments:
                                evolution_results["new_developments"].append({
                                    "conflict_id": conflict_id,
                                    "conflict_name": conflict.get("name", "Unnamed Conflict"),
                                    "developments": new_developments
                                })
                            
                            # Update the conflict in the database
                            async with await self.get_connection_pool() as pool:
                                async with pool.acquire() as conn:
                                    await conn.execute("""
                                        UPDATE NationalConflicts
                                        SET status = $1,
                                            severity = $2,
                                            current_casualties = $3,
                                            economic_impact = $4,
                                            diplomatic_consequences = $5,
                                            recent_developments = recent_developments || $6,
                                            end_date = $7
                                        WHERE id = $8
                                    """,
                                    new_status,
                                    new_severity,
                                    evolution_data.get("casualties_update", conflict.get("current_casualties")),
                                    evolution_data.get("economic_impact_update", conflict.get("economic_impact")),
                                    evolution_data.get("diplomatic_consequences_update", conflict.get("diplomatic_consequences")),
                                    new_developments,
                                    "Recently" if was_resolved else None,  # End date if resolved
                                    conflict_id)
                                    
                                    # Add conflict news for significant developments
                                    if new_developments and involved_nations:
                                        # Generate news from perspective of one involved nation
                                        nation = involved_nations[0]
                                        await self._generate_conflict_update_news(run_ctx, conflict_id, 
                                                                               conflict, evolution_data, nation)
                            
                            # Add to evolution results
                            updated_conflict = {**conflict, 
                                               "status": new_status,
                                               "severity": new_severity,
                                               "new_developments": new_developments}
                                               
                            if was_resolved:
                                evolution_results["resolved_conflicts"].append({
                                    "conflict_id": conflict_id,
                                    "conflict_name": conflict.get("name", "Unnamed Conflict"),
                                    "resolution_details": evolution_data.get("resolution_details", "The conflict has concluded.")
                                })
                            else:
                                evolution_results["evolved_conflicts"].append(updated_conflict)
                            
                        except Exception as e:
                            logging.error(f"Error evolving conflict {conflict_id}: {e}")
                    
                    # Invalidate relevant caches
                    self.invalidate_cache("active_conflicts")
                    
                    return evolution_results
                
                async def _generate_conflict_update_news(
                    self,
                    ctx,
                    conflict_id: int,
                    conflict: Dict[str, Any],
                    evolution_data: Dict[str, Any],
                    nation: Dict[str, Any]
                ) -> None:
                    """Generate news about conflict developments"""
                    # Create agent for news generation
                    news_agent = Agent(
                        name="ConflictNewsUpdateAgent",
                        instructions="You create news updates about evolving international conflicts in a matriarchal world.",
                        model="o3-mini"
                    )
                    
                    # Build news prompt
                    developments_text = "\n".join([f"- {dev}" for dev in evolution_data.get("new_developments", [])])
                    
                    prompt = f"""
                    Generate a news article about these new developments in an ongoing conflict:
                    
                    CONFLICT:
                    {json.dumps(conflict, indent=2)}
                    
                    NEW DEVELOPMENTS:
                    {developments_text}
                    
                    REPORTING NATION:
                    {json.dumps(nation, indent=2)}
                    
                    Create a news article that:
                    1. Reports on the latest developments in the conflict
                    2. Has a perspective aligned with {nation["name"]}'s interests
                    3. Includes quotes from officials (primarily women in positions of power)
                    4. Has a catchy headline
                    5. Reflects matriarchal power structures in its language and reporting style
                    
                    Return a JSON object with:
                    - headline: The article headline
                    - content: The full article content (300-500 words)
                    """
                    
                    # Get response from agent
                    result = await Runner.run(news_agent, prompt, context=ctx.context)
                    
                    try:
                        # Parse response
                        news_data = json.loads(result.final_output)
                        
                        # Ensure required fields exist
                        if not all(k in news_data for k in ["headline", "content"]):
                            return
                        
                        # Apply matriarchal theming to content
                        news_data["content"] = MatriarchalThemingUtils.apply_matriarchal_theme("news", news_data["content"], 1)
                        
                        # Generate embedding
                        embedding_text = f"{news_data['headline']} {news_data['content'][:200]}"
                        embedding = await generate_embedding(embedding_text)
                        
                        # Store in database
                        async with await self.get_connection_pool() as pool:
                            async with pool.acquire() as conn:
                                await conn.execute("""
                                    INSERT INTO ConflictNews (
                                        conflict_id, headline, content, source_nation, bias, embedding
                                    )
                                    VALUES ($1, $2, $3, $4, $5, $6)
                                """, 
                                conflict_id,
                                news_data.get("headline"), 
                                news_data.get("content"),
                                nation["id"],
                                "pro_aggressor" if nation["id"] == conflict.get("primary_aggressor") else "pro_defender",
                                embedding)
                                
                    except Exception as e:
                        logging.error(f"Error generating conflict update news: {e}")
                
                async def register_with_governance(self):
                    """Register with Nyx governance system."""
                    await super().register_with_governance(
                        agent_type=AgentType.NARRATIVE_CRAFTER,
                        agent_id="world_politics_manager",
                        directive_text="Manage nations, international relations, and conflicts in a matriarchal world.",
                        scope="world_building",
                        priority=DirectivePriority.MEDIUM
                    )
