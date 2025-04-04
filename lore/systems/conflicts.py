# lore/systems/conflicts.py

import logging
import json
import random
from datetime import datetime
from typing import Dict, List, Any, Optional

from agents import Agent, Runner
from agents.run_context import RunContextWrapper

from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

from embedding.vector_store import generate_embedding

from lore.core.base_manager import BaseLoreManager
from lore.managers.geopolitical import GeopoliticalSystemManager
from lore.utils.theming import MatriarchalThemingUtils

class NationalConflictSystem(BaseLoreManager):
    """
    System for managing, generating, and evolving national and international
    conflicts that serve as background elements in the world.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
        self.cache_namespace = "national_conflicts"
    
    async def ensure_initialized(self):
        """Ensure system is initialized"""
        if not self.initialized:
            await super().ensure_initialized()
            await self.initialize_tables()
            await self.register_with_governance()
            self.initialized = True
    
    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await super().register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="national_conflict_system",
            directive_text="Create and manage national conflicts and domestic issues with matriarchal power dynamics.",
            scope="world_building",
            priority=DirectivePriority.MEDIUM
        )
        
        logging.info(f"NationalConflictSystem registered with governance for user {self.user_id}, conversation {self.conversation_id}")
    
    async def initialize_tables(self):
        """Ensure conflict system tables exist"""
        table_definitions = {
            "NationalConflicts": """
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
                    source_nation INTEGER, -- Nation ID where this news originated
                    bias TEXT, -- pro_aggressor, pro_defender, neutral
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
                    source_faction TEXT, -- Faction perspective
                    bias TEXT, -- supporting, opposing, neutral
                    embedding VECTOR(1536),
                    FOREIGN KEY (issue_id) REFERENCES DomesticIssues(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_domesticnews_embedding 
                ON DomesticNews USING ivfflat (embedding vector_cosine_ops);
            """
        }
        
        await self.initialize_tables_for_class(table_definitions)

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
        run_ctx = self.create_run_context(ctx)
        
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
                
                # Store in database
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        issue_id = await conn.fetchval("""
                            INSERT INTO DomesticIssues (
                                nation_id, name, issue_type, description, severity,
                                status, start_date, supporting_factions, opposing_factions,
                                neutral_factions, affected_demographics, public_opinion,
                                government_response, recent_developments, political_impact,
                                social_impact, economic_impact, potential_resolution
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
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
                        issue_data.get("potential_resolution", ""))
                        
                        # Generate and store embedding
                        embedding_text = f"{issue_data['name']} {issue_data['description']} {issue_data['issue_type']}"
                        await self.generate_and_store_embedding(embedding_text, conn, "DomesticIssues", "id", issue_id)
                        
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
                
                # Apply matriarchal theming to content
                news_data["content"] = MatriarchalThemingUtils.apply_matriarchal_theme("news", news_data["content"], emphasis_level=1)
                
                # Store in database
                async with self.get_connection_pool() as pool:
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
        run_ctx = self.create_run_context(ctx)
        
        # Get nations for context
        nations = await self.geopolitical_manager.get_all_nations(run_ctx)
        
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
                
                # Apply matriarchal theming to description
                conflict_data["description"] = MatriarchalThemingUtils.apply_matriarchal_theme("conflict", conflict_data["description"], emphasis_level=1)
                
                # Store in database
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        conflict_id = await conn.fetchval("""
                            INSERT INTO NationalConflicts (
                                name, conflict_type, description, severity, status,
                                start_date, involved_nations, primary_aggressor, primary_defender,
                                current_casualties, economic_impact, diplomatic_consequences,
                                public_opinion, recent_developments, potential_resolution
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
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
                        conflict_data.get("potential_resolution", "Unknown"))
                        
                        # Generate and store embedding
                        embedding_text = f"{conflict_data['name']} {conflict_data['description']} {conflict_data['conflict_type']}"
                        await self.generate_and_store_embedding(embedding_text, conn, "NationalConflicts", "id", conflict_id)
                        
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
                news_data["content"] = MatriarchalThemingUtils.apply_matriarchal_theme("news", news_data["content"], emphasis_level=1)
                
                # Store in database
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        news_id = await conn.fetchval("""
                            INSERT INTO ConflictNews (
                                conflict_id, headline, content, source_nation, bias
                            )
                            VALUES ($1, $2, $3, $4, $5)
                            RETURNING id
                        """, 
                        conflict_id,
                        news_data.get("headline"), 
                        news_data.get("content"),
                        nation["id"],
                        bias)
                        
                        # Generate and store embedding
                        embedding_text = f"{news_data['headline']} {news_data['content'][:200]}"
                        await self.generate_and_store_embedding(embedding_text, conn, "ConflictNews", "id", news_id)
                        
            except Exception as e:
                logging.error(f"Error generating conflict news: {e}")

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
        cache_key = "active_conflicts"
        cached = self.get_cache(cache_key)
        if cached:
            return cached
        
        # Query database for active conflicts
        async with self.get_connection_pool() as pool:
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
                self.set_cache(cache_key, result)
                
                return result

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_nation_issues",
        action_description="Getting domestic issues for nation {nation_id}",
        id_from_context=lambda ctx: "national_conflict_system"
    )
    async def get_nation_issues(self, ctx, nation_id: int) -> List[Dict[str, Any]]:
        """
        Get all domestic issues for a nation with governance oversight.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            List of domestic issues
        """
        # Check cache first
        cache_key = f"nation_domestic_issues_{nation_id}"
        cached = self.get_cache(cache_key)
        if cached:
            return cached
        
        # Query database for domestic issues
        async with self.get_connection_pool() as pool:
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
                self.set_cache(cache_key, result)
                
                return result
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_all_conflicts",
        action_description="Evolving all conflicts by days_passed",
        id_from_context=lambda ctx: "national_conflict_system"
    )
    async def evolve_all_conflicts(self, ctx, days_passed: int = 7) -> Dict[str, Any]:
        """
        Evolve all active conflicts based on time passed
        
        Args:
            days_passed: Number of days that have passed
            
        Returns:
            Dictionary with evolution results
        """
        # Implementation would go here
        # This would update conflicts' statuses, add developments, etc.
        return {"message": "Conflicts evolved", "days_passed": days_passed}
