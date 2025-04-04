# lore/systems/religion.py

import logging
import json
import random
from typing import Dict, List, Any, Optional

from agents import Agent, Runner
from agents.run_context import RunContextWrapper

from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

from embedding.vector_store import generate_embedding

from lore.core.base_manager import BaseLoreManager
from lore.managers.geopolitical import GeopoliticalSystemManager
from lore.utils.theming import MatriarchalThemingUtils
from lore.managers.faith import FaithSystem

class ReligiousDistributionSystem(BaseLoreManager):
    """
    Extends the FaithSystem to handle religious distribution across regions,
    including religious diversity, state religions, and religious laws.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
        self.faith_system = FaithSystem(user_id, conversation_id)
        self.cache_namespace = "religious_distribution"
    
    async def ensure_initialized(self):
        """Ensure system is initialized"""
        if not self.initialized:
            await super().ensure_initialized()
            # First initialize base faith system tables
            await self.faith_system.initialize_tables()
            await self.initialize_tables()
            await self.register_with_governance()
            self.initialized = True
    
    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await super().register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="religious_distribution_system",
            directive_text="Distribute religions across nations with matriarchal power structures.",
            scope="world_building",
            priority=DirectivePriority.MEDIUM
        )
        
        logging.info(f"ReligiousDistributionSystem registered with governance for user {self.user_id}, conversation {self.conversation_id}")
        
    async def initialize_tables(self):
        """Ensure religious distribution tables exist"""
        table_definitions = {
            "NationReligion": """
                CREATE TABLE NationReligion (
                    id SERIAL PRIMARY KEY,
                    nation_id INTEGER NOT NULL,
                    state_religion BOOLEAN DEFAULT FALSE,
                    primary_pantheon_id INTEGER, -- Main pantheon if any
                    pantheon_distribution JSONB, -- Distribution of pantheons by percentage
                    religiosity_level INTEGER CHECK (religiosity_level BETWEEN 1 AND 10),
                    religious_tolerance INTEGER CHECK (religious_tolerance BETWEEN 1 AND 10),
                    religious_leadership TEXT, -- Who leads religion nationally
                    religious_laws JSONB, -- Religious laws in effect
                    religious_holidays TEXT[], -- Major religious holidays
                    religious_conflicts TEXT[], -- Current religious tensions
                    religious_minorities TEXT[], -- Description of minority faiths
                    embedding VECTOR(1536),
                    FOREIGN KEY (nation_id) REFERENCES Nations(id) ON DELETE CASCADE,
                    FOREIGN KEY (primary_pantheon_id) REFERENCES Pantheons(id) ON DELETE SET NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_nationreligion_embedding 
                ON NationReligion USING ivfflat (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS idx_nationreligion_nation
                ON NationReligion(nation_id);
            """,
            
            "RegionalReligiousPractice": """
                CREATE TABLE RegionalReligiousPractice (
                    id SERIAL PRIMARY KEY,
                    nation_id INTEGER NOT NULL,
                    practice_id INTEGER NOT NULL, -- Reference to ReligiousPractices
                    regional_variation TEXT, -- How practice differs in this region
                    importance INTEGER CHECK (importance BETWEEN 1 AND 10),
                    frequency TEXT, -- How often practiced locally
                    local_additions TEXT, -- Any local additions to the practice
                    gender_differences TEXT, -- Any local gender differences
                    embedding VECTOR(1536),
                    FOREIGN KEY (nation_id) REFERENCES Nations(id) ON DELETE CASCADE,
                    FOREIGN KEY (practice_id) REFERENCES ReligiousPractices(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_regionalreligiouspractice_embedding 
                ON RegionalReligiousPractice USING ivfflat (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS idx_regionalreligiouspractice_nation
                ON RegionalReligiousPractice(nation_id);
            """
        }
        
        await self.initialize_tables_for_class(table_definitions)
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="distribute_religions",
        action_description="Distributing religions across nations",
        id_from_context=lambda ctx: "religious_distribution_system"
    )
    async def distribute_religions(self, ctx) -> List[Dict[str, Any]]:
        """
        Distribute religions across nations with governance oversight.
        
        Returns:
            List of national religion distributions
        """
        # Create the run context
        run_ctx = self.create_run_context(ctx)
        
        # Get nations and pantheons
        nations = await self.geopolitical_manager.get_all_nations(run_ctx)
        
        # Get pantheons through the faith system
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheons = await conn.fetch("""
                    SELECT id, name, description, matriarchal_elements
                    FROM Pantheons
                """)
                
                # Convert to list of dicts
                pantheon_data = [dict(pantheon) for pantheon in pantheons]
        
        if not nations or not pantheon_data:
            return []
        
        # Create agent for religious distribution
        distribution_agent = Agent(
            name="ReligiousDistributionAgent",
            instructions="You distribute religious pantheons across fantasy nations in a matriarchal world.",
            model="o3-mini"
        )
        
        distributions = []
        for nation in nations:
            # Create prompt for distribution
            prompt = f"""
            Determine religious distribution for this nation:
            
            NATION:
            {json.dumps(nation, indent=2)}
            
            AVAILABLE PANTHEONS:
            {json.dumps(pantheon_data, indent=2)}
            
            Create a realistic religious distribution that:
            1. Considers the nation's matriarchy level ({nation.get("matriarchy_level", 5)}/10)
            2. Determines whether it has a state religion
            3. Distributes pantheons in percentages
            4. Establishes religious laws and practices
            5. Emphasizes matriarchal and feminine aspects of religion
            
            Return a JSON object with:
            - nation_id: The nation ID
            - state_religion: Boolean indicating if there's a state religion
            - primary_pantheon_id: ID of main pantheon (or null if none)
            - pantheon_distribution: Object mapping pantheon IDs to percentage of population
            - religiosity_level: Overall religiosity (1-10)
            - religious_tolerance: Tolerance level (1-10)
            - religious_leadership: Who leads religion nationally (favor matriarchal leadership)
            - religious_laws: Object describing religious laws in effect
            - religious_holidays: Array of major religious holidays
            - religious_conflicts: Array of current religious tensions
            - religious_minorities: Array of minority faith descriptions
            """
            
            # Get response from agent
            result = await Runner.run(distribution_agent, prompt, context=run_ctx.context)
            
            try:
                # Parse JSON response
                distribution_data = json.loads(result.final_output)
                
                # Ensure required fields
                if not all(k in distribution_data for k in ["nation_id", "religiosity_level"]):
                    continue
                
                # Store in database
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        distribution_id = await conn.fetchval("""
                            INSERT INTO NationReligion (
                                nation_id, state_religion, primary_pantheon_id, pantheon_distribution,
                                religiosity_level, religious_tolerance, religious_leadership,
                                religious_laws, religious_holidays, religious_conflicts,
                                religious_minorities
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                            RETURNING id
                        """,
                        distribution_data.get("nation_id"),
                        distribution_data.get("state_religion", False),
                        distribution_data.get("primary_pantheon_id"),
                        json.dumps(distribution_data.get("pantheon_distribution", {})),
                        distribution_data.get("religiosity_level", 5),
                        distribution_data.get("religious_tolerance", 5),
                        distribution_data.get("religious_leadership"),
                        json.dumps(distribution_data.get("religious_laws", {})),
                        distribution_data.get("religious_holidays", []),
                        distribution_data.get("religious_conflicts", []),
                        distribution_data.get("religious_minorities", []))
                        
                        # Generate and store embedding
                        embedding_text = f"religion {nation['name']} {distribution_data.get('religious_leadership', '')} {distribution_data.get('religious_tolerance', 5)}"
                        await self.generate_and_store_embedding(embedding_text, conn, "NationReligion", "id", distribution_id)
                        
                        distribution_data["id"] = distribution_id
                        distributions.append(distribution_data)
                        
                        # Now generate regional religious practices
                        if distribution_data.get("primary_pantheon_id"):
                            await self._generate_regional_practices(run_ctx, distribution_data)
            
            except Exception as e:
                logging.error(f"Error distributing religion for nation {nation['id']}: {e}")
        
        return distributions
    
    async def _generate_regional_practices(self, ctx, distribution_data: Dict[str, Any]) -> None:
        """Generate regional variations of religious practices"""
        # Get pantheons and practices
        nation_id = distribution_data.get("nation_id")
        primary_pantheon_id = distribution_data.get("primary_pantheon_id")
        
        if not primary_pantheon_id:
            return
        
        # Get religious practices for this pantheon
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                practices = await conn.fetch("""
                    SELECT id, name, practice_type, description, purpose
                    FROM ReligiousPractices
                    WHERE pantheon_id = $1
                """, primary_pantheon_id)
                
                # Convert to list of dicts
                practice_data = [dict(practice) for practice in practices]
                
                # Get nation details
                nation = await conn.fetchrow("""
                    SELECT name, government_type, matriarchy_level, cultural_traits
                    FROM Nations
                    WHERE id = $1
                """, nation_id)
                
                nation_data = dict(nation) if nation else {}
        
        if not practice_data or not nation_data:
            return
        
        # Create agent for regional practice generation
        practice_agent = Agent(
            name="RegionalPracticeAgent",
            instructions="You create regional variations of religious practices for a matriarchal society.",
            model="o3-mini"
        )
        
        for practice in practice_data:
            # Create prompt for practice variation
            prompt = f"""
            Create a regional variation of this religious practice for this nation:
            
            NATION:
            {json.dumps(nation_data, indent=2)}
            
            RELIGIOUS PRACTICE:
            {json.dumps(practice, indent=2)}
            
            RELIGIOUS CONTEXT:
            Religiosity level: {distribution_data.get("religiosity_level", 5)}
            Religious tolerance: {distribution_data.get("religious_tolerance", 5)}
            
            Create a regional variation that:
            1. Adapts the practice to local culture
            2. Considers the nation's matriarchy level ({nation_data.get("matriarchy_level", 5)}/10)
            3. Emphasizes feminine power and authority
            4. Feels authentic to both the practice and the nation
            
            Return a JSON object with:
            - practice_id: ID of the original practice
            - regional_variation: How the practice is modified regionally
            - importance: Importance in this region (1-10)
            - frequency: How often practiced locally
            - local_additions: Any local additions to the practice
            - gender_differences: Any local gender differences
            """
            
            # Get response from agent
            result = await Runner.run(practice_agent, prompt, context=ctx.context)
            
            try:
                # Parse JSON response
                variation_data = json.loads(result.final_output)
                
                # Ensure required fields
                if not all(k in variation_data for k in ["practice_id", "regional_variation"]):
                    continue
                
                # Apply matriarchal theming
                variation_data["regional_variation"] = MatriarchalThemingUtils.apply_matriarchal_theme("religion", variation_data["regional_variation"], emphasis_level=1)
                
                # Store in database
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        practice_id = await conn.fetchval("""
                            INSERT INTO RegionalReligiousPractice (
                                nation_id, practice_id, regional_variation,
                                importance, frequency, local_additions,
                                gender_differences
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            RETURNING id
                        """,
                        nation_id,
                        variation_data.get("practice_id"),
                        variation_data.get("regional_variation"),
                        variation_data.get("importance", 5),
                        variation_data.get("frequency"),
                        variation_data.get("local_additions"),
                        variation_data.get("gender_differences"))
                        
                        # Generate and store embedding
                        embedding_text = f"practice {practice['name']} {variation_data['regional_variation']}"
                        await self.generate_and_store_embedding(embedding_text, conn, "RegionalReligiousPractice", "id", practice_id)
            
            except Exception as e:
                logging.error(f"Error generating regional practice variation: {e}")

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_nation_religion",
        action_description="Getting religious information for nation {nation_id}",
        id_from_context=lambda ctx: "religious_distribution_system"
    )
    async def get_nation_religion(self, ctx, nation_id: int) -> Dict[str, Any]:
        """
        Get comprehensive religious information about a nation with governance oversight.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            Dictionary with nation's religious information
        """
        # Check cache first
        cache_key = f"nation_religion_{nation_id}"
        cached = self.get_cache(cache_key)
        if cached:
            return cached
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get nation details
                nation = await conn.fetchrow("""
                    SELECT id, name, government_type, matriarchy_level
                    FROM Nations
                    WHERE id = $1
                """, nation_id)
                
                if not nation:
                    return {"error": "Nation not found"}
                
                # Get religious distribution
                religion = await conn.fetchrow("""
                    SELECT * FROM NationReligion
                    WHERE nation_id = $1
                """, nation_id)
                
                if not religion:
                    return {"error": "No religious data for this nation"}
                
                # Get primary pantheon
                primary_pantheon_id = religion["primary_pantheon_id"]
                primary_pantheon = None
                if primary_pantheon_id:
                    pantheon = await conn.fetchrow("""
                        SELECT id, name, description, matriarchal_elements
                        FROM Pantheons
                        WHERE id = $1
                    """, primary_pantheon_id)
                    
                    if pantheon:
                        primary_pantheon = dict(pantheon)
                
                # Get regional practices
                practices = await conn.fetch("""
                    SELECT r.*, p.name as practice_name, p.practice_type, p.purpose
                    FROM RegionalReligiousPractice r
                    JOIN ReligiousPractices p ON r.practice_id = p.id
                    WHERE r.nation_id = $1
                """, nation_id)
                
                # Get holy sites in this nation
                holy_sites = await conn.fetch("""
                    SELECT h.* 
                    FROM HolySites h
                    JOIN Locations l ON h.location_id = l.id
                    JOIN LoreConnections lc ON l.id = lc.target_id
                    JOIN Nations n ON lc.source_id = n.id
                    WHERE n.id = $1 AND lc.source_type = 'Nations' AND lc.target_type = 'Locations'
                """, nation_id)
                
                # Compile result
                result = {
                    "nation": dict(nation),
                    "religion": dict(religion),
                    "primary_pantheon": primary_pantheon,
                    "regional_practices": [dict(practice) for practice in practices],
                    "holy_sites": [dict(site) for site in holy_sites]
                }
                
                # Parse JSON fields
                if "pantheon_distribution" in result["religion"] and result["religion"]["pantheon_distribution"]:
                    try:
                        result["religion"]["pantheon_distribution"] = json.loads(result["religion"]["pantheon_distribution"])
                    except:
                        pass
                
                if "religious_laws" in result["religion"] and result["religion"]["religious_laws"]:
                    try:
                        result["religion"]["religious_laws"] = json.loads(result["religion"]["religious_laws"])
                    except:
                        pass
                
                # Cache the result
                self.set_cache(cache_key, result)
                
                return result
