# lore/national_conflict_system.py

import logging
import random
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# Agents SDK imports
from agents import Agent, ModelSettings, function_tool, Runner
from agents.run_context import RunContextWrapper

# Nyx governance integration
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType
from nyx.governance_helpers import with_governance, with_governance_permission

# Database functionality
from db.connection import get_db_connection
from utils.caching import LoreCache

# Import existing modules
from lore.lore_manager import LoreManager
from lore.enhanced_lore import GeopoliticalSystemManager, EmergentLoreSystem

# Initialize cache for conflicts
CONFLICT_CACHE = LoreCache(max_size=100, ttl=3600)  # 1 hour TTL

class NationalConflictSystem:
    """
    System for managing, generating, and evolving national and international
    conflicts that serve as background elements in the world.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_manager = LoreManager(user_id, conversation_id)
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
        self.governor = None
        
    async def initialize_governance(self):
        """Initialize Nyx governance connection"""
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
        return self.governor
        
    async def initialize_tables(self):
        """Ensure conflict system tables exist"""
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if Conflicts table exists
                conflicts_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'nationalconflicts'
                    );
                """)
                
                if not conflicts_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE NationalConflicts (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            conflict_type TEXT NOT NULL, -- war, trade_dispute, diplomatic_tension, etc.
                            description TEXT NOT NULL,
                            severity INTEGER CHECK (severity BETWEEN 1 AND 10),
                            status TEXT NOT NULL, -- active, resolved, escalating, de-escalating
                            start_date TEXT NOT NULL,
                            end_date TEXT, -- NULL if ongoing
                            involved_nations INTEGER[], -- IDs of nations involved
                            primary_aggressor INTEGER, -- Nation ID of aggressor
                            primary_defender INTEGER, -- Nation ID of defender
                            current_casualties TEXT, -- Description of casualties so far
                            economic_impact TEXT, -- Description of economic impact
                            diplomatic_consequences TEXT, -- Description of diplomatic fallout
                            public_opinion JSONB, -- Public opinion in different nations
                            recent_developments TEXT[], -- Recent events in the conflict
                            potential_resolution TEXT, -- Potential ways it might end
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_nationalconflicts_embedding 
                        ON NationalConflicts USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("NationalConflicts table created")
                
                # Check if ConflictNews table exists
                news_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'conflictnews'
                    );
                """)
                
                if not news_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE ConflictNews (
                            id SERIAL PRIMARY KEY,
                            conflict_id INTEGER NOT NULL,
                            headline TEXT NOT NULL,
                            content TEXT NOT NULL,
                            publication_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            source_nation INTEGER, -- Nation ID where this news originated
                            bias TEXT, -- pro_aggressor, pro_defender, neutral
                            embedding VECTOR(1536),
                            FOREIGN KEY (conflict_id) REFERENCES NationalConflicts(id) ON DELETE CASCADE
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_conflictnews_embedding 
                        ON ConflictNews USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("ConflictNews table created")

                # Check if DomesticIssues table exists
                domestic_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'domesticissues'
                    );
                """)
                
                if not domestic_exists:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE DomesticIssues (
                            id SERIAL PRIMARY KEY,
                            nation_id INTEGER NOT NULL,
                            name TEXT NOT NULL,
                            issue_type TEXT NOT NULL, -- civil_rights, political_controversy, economic_crisis, etc.
                            description TEXT NOT NULL,
                            severity INTEGER CHECK (severity BETWEEN 1 AND 10),
                            status TEXT NOT NULL, -- emerging, active, waning, resolved
                            start_date TEXT NOT NULL,
                            end_date TEXT, -- NULL if ongoing
                            supporting_factions TEXT[], -- Groups supporting one side
                            opposing_factions TEXT[], -- Groups opposing
                            neutral_factions TEXT[], -- Groups remaining neutral
                            affected_demographics TEXT[], -- Demographics most affected
                            public_opinion JSONB, -- Opinion distribution
                            government_response TEXT, -- How the government is responding
                            recent_developments TEXT[], -- Recent events in this issue
                            political_impact TEXT, -- Impact on political landscape
                            social_impact TEXT, -- Impact on society
                            economic_impact TEXT, -- Economic consequences
                            potential_resolution TEXT, -- Potential ways it might resolve
                            embedding VECTOR(1536),
                            FOREIGN KEY (nation_id) REFERENCES Nations(id) ON DELETE CASCADE
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_domesticissues_embedding 
                        ON DomesticIssues USING ivfflat (embedding vector_cosine_ops);
                        
                        CREATE INDEX IF NOT EXISTS idx_domesticissues_nation
                        ON DomesticIssues(nation_id);
                    """)
                    
                    logging.info("DomesticIssues table created")
                
                    # Check if DomesticNews table exists
                    domestic_news_exist = await conn.fetchval("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'domesticnews'
                        );
                    """)
                    
                    if not domestic_news_exist:
                        # Create the table
                        await conn.execute("""
                            CREATE TABLE DomesticNews (
                                id SERIAL PRIMARY KEY,
                                issue_id INTEGER NOT NULL,
                                headline TEXT NOT NULL,
                                content TEXT NOT NULL,
                                publication_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                source_faction TEXT, -- Faction perspective
                                bias TEXT, -- supporting, opposing, neutral
                                embedding VECTOR(1536),
                                FOREIGN KEY (issue_id) REFERENCES DomesticIssues(id) ON DELETE CASCADE
                            );
                        """)
                        
                        # Create index
                        await conn.execute("""
                            CREATE INDEX IF NOT EXISTS idx_domesticnews_embedding 
                            ON DomesticNews USING ivfflat (embedding vector_cosine_ops);
                        """)
                        
                        logging.info("DomesticNews table created")
        
        @with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            action_type="generate_domestic_issues",
            action_description="Generating domestic issues for nation {nation_id}",
            id_from_context=lambda ctx: "national_conflict_system"
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
            run_ctx = RunContextWrapper(context=ctx.context)
            
            # Get nation details
            async with self.lore_manager.get_connection_pool() as pool:
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
                instructions="You create realistic domestic political and social issues for fantasy nations.",
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
                    async with self.lore_manager.get_connection_pool() as pool:
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
                instructions="You create realistic news articles about domestic political issues.",
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
                    
                    # Generate embedding
                    embedding_text = f"{news_data['headline']} {news_data['content'][:200]}"
                    embedding = await generate_embedding(embedding_text)
                    
                    # Store in database
                    async with self.lore_manager.get_connection_pool() as pool:
                        async with pool.acquire() as conn:
                            await conn.execute("""
                                INSERT INTO DomesticNews (
                                    issue_id, headline, content, source_faction, bias, embedding
                                )
                                VALUES ($1, $2, $3, $4, $5, $6)
                            """, 
                            issue_id,
                            news_data.get("headline"), 
                            news_data.get("content"),
                            news_data.get("source_faction", "Unknown Source"),
                            bias,
                            embedding)
                            
                except Exception as e:
                    logging.error(f"Error generating domestic news: {e}")
        
        @with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            action_type="evolve_domestic_issues",
            action_description="Evolving domestic issues for nation {nation_id}",
            id_from_context=lambda ctx: "national_conflict_system"
        )
        async def evolve_domestic_issues(self, ctx, nation_id: int, days_passed: int = 7) -> List[Dict[str, Any]]:
            """
            Evolve domestic issues over time with governance oversight.
            
            Args:
                nation_id: ID of the nation
                days_passed: Number of days to simulate passing
                
            Returns:
                List of updated domestic issues
            """
            # Create the run context
            run_ctx = RunContextWrapper(context=ctx.context)
            
            # Get active domestic issues
            async with self.lore_manager.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    issues = await conn.fetch("""
                        SELECT * FROM DomesticIssues
                        WHERE nation_id = $1 AND status != 'resolved'
                    """, nation_id)
                    
                    issue_data = [dict(issue) for issue in issues]
                    
                    # Get nation details
                    nation = await conn.fetchrow("""
                        SELECT id, name, government_type, matriarchy_level
                        FROM Nations
                        WHERE id = $1
                    """, nation_id)
                    
                    nation_data = dict(nation) if nation else {}
            
            if not issue_data or not nation_data:
                return []
            
            # Create agent for issue evolution
            evolution_agent = Agent(
                name="DomesticIssueEvolutionAgent",
                instructions="You evolve domestic political issues realistically over time.",
                model="o3-mini"
            )
            
            # Update each issue
            updated_issues = []
            for issue in issue_data:
                # Calculate probability of change based on issue type and severity
                change_probability = min(0.8, (issue.get("severity", 5) / 10) + (days_passed / 30))
                
                if random.random() > change_probability:
                    continue  # No change for this issue
                
                # Create prompt for the agent
                prompt = f"""
                Evolve this domestic issue after {days_passed} days have passed:
                
                CURRENT ISSUE STATE:
                {json.dumps(issue, indent=2)}
                
                NATION:
                {json.dumps(nation_data, indent=2)}
                
                Progress this issue realistically by:
                1. Determining if it escalates, de-escalates, or remains similar
                2. Generating 1-2 new developments that have occurred
                3. Updating public opinion, government response, and impacts
                4. Considering if it might be resolved or reach a new phase
                
                Return a JSON object with:
                - id: The original issue ID
                - status: Updated status (emerging, active, waning, resolved)
                - severity: Updated severity (1-10)
                - new_developments: Array of 1-2 new developments to add
                - public_opinion: Updated public opinion object
                - government_response: Updated government response
                - political_impact: Updated political impact
                - social_impact: Updated social impact
                - economic_impact: Updated economic impact
                - end_date: End date if resolved, otherwise null
                - resolution_description: Description of resolution if resolved
                """
                
                # Get response from agent
                result = await Runner.run(evolution_agent, prompt, context=run_ctx.context)
                
                try:
                    # Parse response
                    update_data = json.loads(result.final_output)
                    
                    # Ensure required fields exist
                    if "id" not in update_data or update_data["id"] != issue["id"]:
                        continue
                    
                    # Get new developments to add
                    new_developments = update_data.get("new_developments", [])
                    
                    # Combine with existing developments but limit to most recent 10
                    all_developments = issue.get("recent_developments", []) + new_developments
                    if len(all_developments) > 10:
                        all_developments = all_developments[-10:]
                    
                    # Update issue in database
                    async with self.lore_manager.get_connection_pool() as pool:
                        async with pool.acquire() as conn:
                            await conn.execute("""
                                UPDATE DomesticIssues
                                SET status = $1,
                                    severity = $2,
                                    recent_developments = $3,
                                    public_opinion = $4,
                                    government_response = $5,
                                    political_impact = $6,
                                    social_impact = $7,
                                    economic_impact = $8,
                                    end_date = $9
                                WHERE id = $10
                            """, 
                            update_data.get("status", issue.get("status")),
                            update_data.get("severity", issue.get("severity")),
                            all_developments,
                            json.dumps(update_data.get("public_opinion", {})),
                            update_data.get("government_response", issue.get("government_response")),
                            update_data.get("political_impact", issue.get("political_impact")),
                            update_data.get("social_impact", issue.get("social_impact")),
                            update_data.get("economic_impact", issue.get("economic_impact")),
                            update_data.get("end_date") if update_data.get("status") == "resolved" else None,
                            issue["id"])
                    
                    # Generate news about the development
                    if new_developments:
                        # Choose a random bias perspective for this news update
                        bias = random.choice(["supporting", "opposing", "neutral"])
                        await self._generate_domestic_update_news(
                            run_ctx, 
                            issue["id"], 
                            {**issue, **update_data, "recent_developments": new_developments}, 
                            nation_data,
                            bias
                        )
                    
                    # Add to result
                    updated = {**issue, **update_data, "recent_developments": all_developments}
                    updated_issues.append(updated)
                    
                    # Clear cache
                    CONFLICT_CACHE.invalidate_pattern(f"domestic_issue_{issue['id']}")
                    
                except Exception as e:
                    logging.error(f"Error evolving domestic issue {issue['id']}: {e}")
            
            return updated_issues
        
        async def _generate_domestic_update_news(
            self, 
            ctx, 
            issue_id: int, 
            issue_data: Dict[str, Any],
            nation_data: Dict[str, Any],
            bias: str = "neutral"
        ) -> None:
            """Generate news update about issue developments"""
            # Create agent for news generation
            news_agent = Agent(
                name="DomesticNewsUpdateAgent",
                instructions="You create realistic news updates about domestic political issues.",
                model="o3-mini"
            )
            
            # Focus on most recent development
            recent_development = issue_data.get("recent_developments", ["No new developments"])[0]
            
            # Create prompt for the agent
            prompt = f"""
            Generate a news update about this domestic issue development from a {bias} perspective:
            
            ISSUE:
            {json.dumps(issue_data, indent=2)}
            
            RECENT DEVELOPMENT:
            {recent_development}
            
            NATION:
            {json.dumps(nation_data, indent=2)}
            
            Create a news article that:
            1. Focuses on the recent development
            2. Has a clear {bias} bias
            3. Includes quotes from officials or involved parties
            4. Has a catchy headline
            
            Return a JSON object with:
            - headline: The article headline
            - content: The full article content (200-300 words)
            - source_faction: Which group or outlet is publishing this
            """
            
            # Get response from agent
            result = await Runner.run(news_agent, prompt, context=ctx.context)
            
            try:
                # Parse response
                news_data = json.loads(result.final_output)
                
                # Ensure required fields exist
                if not all(k in news_data for k in ["headline", "content"]):
                    return
                
                # Generate embedding
                embedding_text = f"{news_data['headline']} {news_data['content'][:200]}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with self.lore_manager.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        await conn.execute("""
                            INSERT INTO DomesticNews (
                                issue_id, headline, content, source_faction, bias, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """, 
                        issue_id,
                        news_data.get("headline"), 
                        news_data.get("content"),
                        news_data.get("source_faction", "Unknown Source"),
                        bias,
                        embedding)
                        
            except Exception as e:
                logging.error(f"Error generating domestic news update: {e}")
        
        @with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            action_type="get_nation_domestic_issues",
            action_description="Getting domestic issues for nation {nation_id}",
            id_from_context=lambda ctx: "national_conflict_system"
        )
        async def get_nation_domestic_issues(self, ctx, nation_id: int) -> List[Dict[str, Any]]:
            """
            Get all domestic issues for a nation with governance oversight.
            
            Args:
                nation_id: ID of the nation
                
            Returns:
                List of domestic issues
            """
            # Check cache first
            cache_key = f"nation_domestic_issues_{nation_id}_{self.user_id}_{self.conversation_id}"
            cached = CONFLICT_CACHE.get(cache_key)
            if cached:
                return cached
            
            # Query database for domestic issues
            async with self.lore_manager.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    issues = await conn.fetch("""
                        SELECT * FROM DomesticIssues
                        WHERE nation_id = $1
                        ORDER BY severity DESC
                    """, nation_id)
                    
                    # Convert to list of dicts
                    result = [dict(issue) for issue in issues]
                    
                    # Parse JSON fields
                    for issue in result:
                        if "public_opinion" in issue and issue["public_opinion"]:
                            try:
                                issue["public_opinion"] = json.loads(issue["public_opinion"])
                            except:
                                pass
                    
                    # Cache result
                    CONFLICT_CACHE.set(cache_key, result)
                    
                    return result
        
        @with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            action_type="get_domestic_issue_news",
            action_description="Getting news for domestic issue {issue_id}",
            id_from_context=lambda ctx: "national_conflict_system"
        )
        async def get_domestic_issue_news(self, ctx, issue_id: int) -> List[Dict[str, Any]]:
            """
            Get news articles about a specific domestic issue with governance oversight.
            
            Args:
                issue_id: ID of the issue
                
            Returns:
                List of news articles
            """
            # Check cache first
            cache_key = f"domestic_news_{issue_id}_{self.user_id}_{self.conversation_id}"
            cached = CONFLICT_CACHE.get(cache_key)
            if cached:
                return cached
            
            # Query database for news about this issue
            async with self.lore_manager.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    news = await conn.fetch("""
                        SELECT n.*
                        FROM DomesticNews n
                        WHERE n.issue_id = $1
                        ORDER BY n.publication_date DESC
                    """, issue_id)
                    
                    # Convert to list of dicts
                    result = [dict(article) for article in news]
                    
                    # Cache result
                    CONFLICT_CACHE.set(cache_key, result)
                    
                    return result
        
        @with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            action_type="get_all_recent_news",
            action_description="Getting all recent news",
            id_from_context=lambda ctx: "national_conflict_system"
        )
        async def get_all_recent_news(self, ctx, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
            """
            Get all recent news across international conflicts and domestic issues with governance oversight.
            
            Args:
                limit: Maximum number of news articles to return per category
                
            Returns:
                Dictionary with international and domestic news
            """
            # Check cache first
            cache_key = f"all_recent_news_{limit}_{self.user_id}_{self.conversation_id}"
            cached = CONFLICT_CACHE.get(cache_key)
            if cached:
                return cached
            
            # Query database for recent international news
            async with self.lore_manager.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    int_news = await conn.fetch("""
                        SELECT n.*, nations.name as source_nation_name, c.name as conflict_name
                        FROM ConflictNews n
                        LEFT JOIN Nations ON n.source_nation = Nations.id
                        LEFT JOIN NationalConflicts c ON n.conflict_id = c.id
                        ORDER BY n.publication_date DESC
                        LIMIT $1
                    """, limit)
                    
                    # Get domestic news
                    dom_news = await conn.fetch("""
                        SELECT n.*, i.name as issue_name, i.nation_id
                        FROM DomesticNews n
                        JOIN DomesticIssues i ON n.issue_id = i.id
                        ORDER BY n.publication_date DESC
                        LIMIT $1
                    """, limit)
                    
                    # Join with nation names
                    nations_map = {}
                    for row in dom_news:
                        nation_id = row["nation_id"]
                        if nation_id not in nations_map:
                            nation = await conn.fetchrow("""
                                SELECT name FROM Nations WHERE id = $1
                            """, nation_id)
                            nations_map[nation_id] = nation["name"] if nation else "Unknown Nation"
                    
                    # Convert to list of dicts
                    int_result = [dict(article) for article in int_news]
                    dom_result = []
                    
                    for article in dom_news:
                        article_dict = dict(article)
                        article_dict["nation_name"] = nations_map.get(article["nation_id"], "Unknown Nation")
                        dom_result.append(article_dict)
                    
                    # Compile result
                    result = {
                        "international_news": int_result,
                        "domestic_news": dom_result
                    }
                    
                    # Cache result
                    CONFLICT_CACHE.set(cache_key, result)
                    
                    return result
    
        # Initialize the world with both international conflicts AND domestic issues
        async def initialize_world_conflicts(self, ctx):
            """Initialize the world with conflicts and domestic issues"""
            # First generate international conflicts
            conflicts = await self.generate_initial_conflicts(ctx, count=3)
            
            # Then generate domestic issues for each nation
            nations = await self.geopolitical_manager.get_all_nations(ctx)
            
            domestic_issues = []
            for nation in nations:
                nation_issues = await self.generate_domestic_issues(ctx, nation["id"], count=2)
                domestic_issues.extend(nation_issues)
            
            return {
                "international_conflicts": conflicts,
                "domestic_issues": domestic_issues
            }
            
        # Evolve both international conflicts AND domestic issues over time
        async def evolve_all_conflicts(self, ctx, days_passed: int = 7):
            """Evolve all conflicts and domestic issues over time"""
            # First evolve international conflicts
            evolved_conflicts = await self.evolve_conflicts(ctx, days_passed)
            
            # Then evolve domestic issues for each nation
            evolved_issues = []
            
            # Get all nations
            nations = await self.geopolitical_manager.get_all_nations(ctx)
            
            for nation in nations:
                nation_issues = await self.evolve_domestic_issues(ctx, nation["id"], days_passed)
                evolved_issues.extend(nation_issues)
            
            return {
                "evolved_international_conflicts": evolved_conflicts,
                "evolved_domestic_issues": evolved_issues
            }
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_initial_conflicts",
        action_description="Generating initial national conflicts",
        id_from_context=lambda ctx: "national_conflict_system"
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
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get nations for context
        nations = await self.geopolitical_manager.get_all_nations(run_ctx)
        
        if len(nations) < 2:
            return []
        
        conflicts = []
        
        # Create agent for conflict generation
        conflict_agent = Agent(
            name="NationalConflictAgent",
            instructions="You create realistic international conflicts for a fantasy world.",
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
                
                # Add to database
                from embedding.vector_store import generate_embedding
                
                # Generate embedding
                embedding_text = f"{conflict_data['name']} {conflict_data['description']} {conflict_data['conflict_type']}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with self.lore_manager.get_connection_pool() as pool:
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
            instructions="You create realistic news articles about international conflicts.",
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
            2. Includes quotes from officials
            3. Covers the key facts but with the nation's spin
            4. Has a catchy headline
            
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
                
                # Generate embedding
                from embedding.vector_store import generate_embedding
                embedding_text = f"{news_data['headline']} {news_data['content'][:200]}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with self.lore_manager.get_connection_pool() as pool:
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
        action_type="evolve_conflicts",
        action_description="Evolving national conflicts over time",
        id_from_context=lambda ctx: "national_conflict_system"
    )
    async def evolve_conflicts(self, ctx, days_passed: int = 7) -> List[Dict[str, Any]]:
        """
        Evolve existing conflicts over time with governance oversight.
        
        Args:
            days_passed: Number of days to simulate passing
            
        Returns:
            List of updated conflicts
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get active conflicts
        active_conflicts = await self.get_active_conflicts(run_ctx)
        
        if not active_conflicts:
            return []
        
        # Create agent for conflict evolution
        conflict_agent = Agent(
            name="ConflictEvolutionAgent",
            instructions="You evolve international conflicts realistically over time.",
            model="o3-mini"
        )
        
        # Update each conflict
        updated_conflicts = []
        for conflict in active_conflicts:
            # Calculate probability of change based on conflict type and severity
            change_probability = min(0.8, (conflict.get("severity", 5) / 10) + (days_passed / 30))
            
            if random.random() > change_probability:
                continue  # No change for this conflict
            
            # Determine what kind of change
            status = conflict.get("status", "active")
            
            if status == "resolved":
                continue  # Skip resolved conflicts
            
            # Get nations involved
            nation_ids = conflict.get("involved_nations", [])
            nations = []
            
            for nation_id in nation_ids:
                async with self.lore_manager.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        nation = await conn.fetchrow("""
                            SELECT id, name, government_type, matriarchy_level
                            FROM Nations
                            WHERE id = $1
                        """, nation_id)
                        
                        if nation:
                            nations.append(dict(nation))
            
            # Create prompt for the agent
            prompt = f"""
            Evolve this international conflict after {days_passed} days have passed:
            
            CURRENT CONFLICT STATE:
            {json.dumps(conflict, indent=2)}
            
            INVOLVED NATIONS:
            {json.dumps(nations, indent=2)}
            
            Progress this conflict realistically by:
            1. Determining if it escalates, de-escalates, or remains similar
            2. Generating 1-2 new developments that have occurred
            3. Updating casualties, economic impact, and diplomatic consequences
            4. Considering if it might be resolved or reach a new phase
            
            Return a JSON object with:
            - id: The original conflict ID
            - status: Updated status (active, escalating, de-escalating, resolved)
            - severity: Updated severity (1-10)
            - new_developments: Array of 1-2 new developments to add
            - current_casualties: Updated casualties description
            - economic_impact: Updated economic impact description
            - diplomatic_consequences: Updated diplomatic consequences
            - end_date: End date if resolved, otherwise null
            - resolution_description: Description of resolution if resolved
            """
            
            # Get response from agent
            result = await Runner.run(conflict_agent, prompt, context=run_ctx.context)
            
            try:
                # Parse response
                update_data = json.loads(result.final_output)
                
                # Ensure required fields exist
                if "id" not in update_data or update_data["id"] != conflict["id"]:
                    continue
                
                # Get new developments to add
                new_developments = update_data.get("new_developments", [])
                
                # Combine with existing developments but limit to most recent 10
                all_developments = conflict.get("recent_developments", []) + new_developments
                if len(all_developments) > 10:
                    all_developments = all_developments[-10:]
                
                # Update conflict in database
                async with self.lore_manager.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        await conn.execute("""
                            UPDATE NationalConflicts
                            SET status = $1,
                                severity = $2,
                                recent_developments = $3,
                                current_casualties = $4,
                                economic_impact = $5,
                                diplomatic_consequences = $6,
                                end_date = $7
                            WHERE id = $8
                        """, 
                        update_data.get("status", conflict.get("status")),
                        update_data.get("severity", conflict.get("severity")),
                        all_developments,
                        update_data.get("current_casualties", conflict.get("current_casualties")),
                        update_data.get("economic_impact", conflict.get("economic_impact")),
                        update_data.get("diplomatic_consequences", conflict.get("diplomatic_consequences")),
                        update_data.get("end_date") if update_data.get("status") == "resolved" else None,
                        conflict["id"])
                
                # Generate news about the development
                if new_developments:
                    for nation in nations[:2]:  # Generate news from perspectives of the two main nations
                        await self._generate_conflict_update_news(
                            run_ctx, 
                            conflict["id"], 
                            {**conflict, **update_data, "recent_developments": new_developments}, 
                            nation
                        )
                
                # Add to result
                updated = {**conflict, **update_data, "recent_developments": all_developments}
                updated_conflicts.append(updated)
                
                # Clear cache
                CONFLICT_CACHE.invalidate_pattern(f"conflict_{conflict['id']}")
                
            except Exception as e:
                logging.error(f"Error evolving conflict {conflict['id']}: {e}")
        
        return updated_conflicts
    
    async def _generate_conflict_update_news(
        self, 
        ctx, 
        conflict_id: int, 
        conflict_data: Dict[str, Any],
        nation: Dict[str, Any]
    ) -> None:
        """Generate news update about conflict developments"""
        # Create agent for news generation
        news_agent = Agent(
            name="ConflictNewsUpdateAgent",
            instructions="You create realistic news updates about international conflicts.",
            model="o3-mini"
        )
        
        # Determine bias
        bias = "pro_defender" if nation["id"] == conflict_data.get("primary_defender") else "pro_aggressor"
        
        # Focus on most recent development
        recent_development = conflict_data.get("recent_developments", ["No new developments"])[0]
        
        # Create prompt for the agent
        prompt = f"""
        Generate a news update about this conflict development from the perspective of {nation["name"]}:
        
        CONFLICT:
        {json.dumps(conflict_data, indent=2)}
        
        RECENT DEVELOPMENT:
        {recent_development}
        
        REPORTING NATION:
        {json.dumps(nation, indent=2)}
        
        Create a news article that:
        1. Focuses on the recent development
        2. Has a clear {bias} bias
        3. Includes quotes from officials
        4. Has a catchy headline
        
        Return a JSON object with:
        - headline: The article headline
        - content: The full article content (200-300 words)
        """
        
        # Get response from agent
        result = await Runner.run(news_agent, prompt, context=ctx.context)
        
        try:
            # Parse response
            news_data = json.loads(result.final_output)
            
            # Ensure required fields exist
            if not all(k in news_data for k in ["headline", "content"]):
                return
            
            # Generate embedding
            from embedding.vector_store import generate_embedding
            embedding_text = f"{news_data['headline']} {news_data['content'][:200]}"
            embedding = await generate_embedding(embedding_text)
            
            # Store in database
            async with self.lore_manager.get_connection_pool() as pool:
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
            logging.error(f"Error generating conflict news update: {e}")
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_active_conflicts",
        action_description="Getting active national conflicts",
        id_from_context=lambda ctx: "national_conflict_system"
    )
    async def get_active_conflicts(self, ctx) -> List[Dict[str, Any]]:
        """
        Get all active conflicts with governance oversight.
        
        Returns:
            List of active conflicts
        """
        # Check cache first
        cache_key = f"active_conflicts_{self.user_id}_{self.conversation_id}"
        cached = CONFLICT_CACHE.get(cache_key)
        if cached:
            return cached
        
        # Query database for active conflicts
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                conflicts = await conn.fetch("""
                    SELECT * FROM NationalConflicts
                    WHERE status != 'resolved'
                    ORDER BY severity DESC
                """)
                
                # Convert to list of dicts
                result = [dict(conflict) for conflict in conflicts]
                
                # Cache result
                CONFLICT_CACHE.set(cache_key, result)
                
                return result
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_conflict_news",
        action_description="Getting news for conflict {conflict_id}",
        id_from_context=lambda ctx: "national_conflict_system"
    )
    async def get_conflict_news(self, ctx, conflict_id: int) -> List[Dict[str, Any]]:
        """
        Get news articles about a specific conflict with governance oversight.
        
        Args:
            conflict_id: ID of the conflict
            
        Returns:
            List of news articles
        """
        # Check cache first
        cache_key = f"conflict_news_{conflict_id}_{self.user_id}_{self.conversation_id}"
        cached = CONFLICT_CACHE.get(cache_key)
        if cached:
            return cached
        
        # Query database for news about this conflict
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                news = await conn.fetch("""
                    SELECT n.*, nations.name as source_nation_name
                    FROM ConflictNews n
                    LEFT JOIN Nations ON n.source_nation = Nations.id
                    WHERE n.conflict_id = $1
                    ORDER BY n.publication_date DESC
                """, conflict_id)
                
                # Convert to list of dicts
                result = [dict(article) for article in news]
                
                # Cache result
                CONFLICT_CACHE.set(cache_key, result)
                
                return result
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_recent_conflict_news",
        action_description="Getting recent conflict news for the world",
        id_from_context=lambda ctx: "national_conflict_system"
    )
    async def get_recent_conflict_news(self, ctx, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent news across all conflicts with governance oversight.
        
        Args:
            limit: Maximum number of news articles to return
            
        Returns:
            List of recent news articles
        """
        # Check cache first
        cache_key = f"recent_news_{limit}_{self.user_id}_{self.conversation_id}"
        cached = CONFLICT_CACHE.get(cache_key)
        if cached:
            return cached
        
        # Query database for recent news
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                news = await conn.fetch("""
                    SELECT n.*, nations.name as source_nation_name, c.name as conflict_name
                    FROM ConflictNews n
                    LEFT JOIN Nations ON n.source_nation = Nations.id
                    LEFT JOIN NationalConflicts c ON n.conflict_id = c.id
                    ORDER BY n.publication_date DESC
                    LIMIT $1
                """, limit)
                
                # Convert to list of dicts
                result = [dict(article) for article in news]
                
                # Cache result
                CONFLICT_CACHE.set(cache_key, result)
                
                return result
    
    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await self.initialize_governance()
        
        # Register this system with governance
        await self.governor.register_agent(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="national_conflict_system",
            agent_instance=self
        )
        
        # Issue a directive for conflict system
        await self.governor.issue_directive(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="national_conflict_system",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Generate and manage realistic national conflicts as background elements.",
                "scope": "world_building"
            },
            priority=5,  # Medium priority
            duration_minutes=24*60  # 24 hours
        )
        
        logging.info(f"NationalConflictSystem registered with Nyx governance for user {self.user_id}, conversation {self.conversation_id}")
