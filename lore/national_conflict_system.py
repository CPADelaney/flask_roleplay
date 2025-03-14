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
