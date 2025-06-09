# logic/conflict_system/generation.py
"""
Conflict generation with world state analysis
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from agents import Agent, Runner, ModelSettings
from db.connection import get_db_connection_context
from embedding.vector_store import generate_embedding

from .core import ConflictCore, CONFLICT_ARCHETYPES, ConflictScale
from .world_analyzer import WorldStateAnalyzer

logger = logging.getLogger(__name__)

# Conflict Generation Agent
conflict_generation_agent = Agent(
    name="Conflict Generation Agent",
    model_settings=ModelSettings(model="gpt-4o", temperature=0.8),
    instructions="""
    You create organic conflicts based on world state analysis.
    
    Generate conflicts that:
    1. Emerge naturally from existing tensions
    2. Involve stakeholders with genuine motivations
    3. Offer multiple resolution approaches
    4. Create interesting moral dilemmas
    5. Include femdom power dynamics where appropriate
    
    Consider:
    - Historical context and grievances
    - Economic and resource factors
    - Faction dynamics and rivalries
    - Character relationships
    - Narrative timing
    
    Output complete conflict structure as JSON.
    """
)

class ConflictGenerator:
    """Advanced conflict generation system"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.core = ConflictCore(user_id, conversation_id)
        self.analyzer = WorldStateAnalyzer(user_id, conversation_id)
    
    async def generate_organic_conflict(self, 
                                      preferred_scale: Optional[ConflictScale] = None,
                                      force_archetype: Optional[str] = None) -> Dict[str, Any]:
        """Generate a conflict that feels organic to the world state"""
        # Analyze world state
        world_state = await self.analyzer.analyze_world_state()
        
        # Select appropriate archetype
        if force_archetype and force_archetype in CONFLICT_ARCHETYPES:
            archetype = CONFLICT_ARCHETYPES[force_archetype]
        else:
            archetype = await self._select_archetype(world_state, preferred_scale)
        
        # Get relevant NPCs
        npcs = await self._get_relevant_npcs(archetype, world_state)
        
        # Generate conflict details
        conflict_data = await self._generate_conflict_details(
            archetype, world_state, npcs
        )
        
        # Create conflict and related data
        conflict_id = await self._create_full_conflict(conflict_data)
        
        return await self.core.get_conflict_details(conflict_id)
    
    async def _select_archetype(self, world_state: Dict[str, Any],
                              preferred_scale: Optional[ConflictScale]) -> Any:
        """Select appropriate conflict archetype based on world pressure"""
        pressure = world_state['total_pressure']
        
        # Filter by preferred scale
        candidates = []
        for archetype in CONFLICT_ARCHETYPES.values():
            if preferred_scale is None or archetype.scale == preferred_scale:
                candidates.append(archetype)
        
        # Weight by pressure
        weights = []
        for archetype in candidates:
            if pressure < 50 and archetype.scale == ConflictScale.PERSONAL:
                weights.append(3.0)
            elif pressure < 150 and archetype.scale == ConflictScale.LOCAL:
                weights.append(2.0)
            elif pressure < 300 and archetype.scale == ConflictScale.REGIONAL:
                weights.append(2.0)
            elif pressure >= 300 and archetype.scale == ConflictScale.WORLD:
                weights.append(3.0)
            else:
                weights.append(1.0)
        
        # Weighted random selection
        return random.choices(candidates, weights=weights)[0]
    
    async def _get_relevant_npcs(self, archetype: Any,
                                world_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get NPCs relevant to the conflict archetype"""
        async with get_db_connection_context() as conn:
            # Base query
            query_parts = ["""
                SELECT n.*, 
                       COUNT(DISTINCT sl.link_id) as relationship_count,
                       AVG(sl.link_level) as avg_relationship
                FROM NPCStats n
                LEFT JOIN SocialLinks sl ON 
                    (sl.entity1_id = n.npc_id AND sl.entity1_type = 'npc')
                    OR (sl.entity2_id = n.npc_id AND sl.entity2_type = 'npc')
                WHERE n.user_id = $1 AND n.conversation_id = $2
                    AND n.introduced = TRUE
            """]
            
            # Add filters based on archetype
            if archetype.scale == ConflictScale.PERSONAL:
                query_parts.append("AND (n.intensity > 70 OR n.closeness > 70)")
            elif archetype.scale in [ConflictScale.LOCAL, ConflictScale.REGIONAL]:
                query_parts.append("AND n.dominance > 60")
            
            query_parts.extend([
                "GROUP BY n.npc_id",
                "ORDER BY n.dominance DESC, relationship_count DESC",
                f"LIMIT {archetype.max_stakeholders * 2}"
            ])
            
            npcs = await conn.fetch(" ".join(query_parts), 
                                   self.user_id, self.conversation_id)
            
            return [dict(n) for n in npcs]
    
    async def _generate_conflict_details(self, archetype: Any,
                                       world_state: Dict[str, Any],
                                       npcs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate specific conflict details using AI"""
        # Select stakeholder NPCs
        stakeholder_count = random.randint(
            archetype.min_stakeholders,
            min(archetype.max_stakeholders, len(npcs))
        )
        stakeholder_npcs = random.sample(npcs, stakeholder_count)
        
        # Create generation prompt
        prompt = f"""
        Generate a {archetype.name} conflict based on this world state:
        
        World Pressure: {world_state['total_pressure']}
        Primary Driver: {world_state['primary_driver']}
        Scale: {archetype.scale.value}
        
        Stakeholder NPCs:
        {json.dumps([{
            'name': npc['npc_name'],
            'dominance': npc['dominance'],
            'intensity': npc['intensity'],
            'relationships': npc.get('relationship_count', 0)
        } for npc in stakeholder_npcs], indent=2)}
        
        Recent Tensions:
        {json.dumps(world_state.get('recent_tensions', [])[:3], indent=2)}
        
        Create a conflict with:
        - Compelling name and description
        - Clear root cause from world state
        - Stakeholder motivations (public and private)
        - 3-5 resolution paths
        - Femdom manipulation opportunities
        """
        
        result = await Runner.run(
            conflict_generation_agent,
            prompt,
            self.core.ctx
        )
        
        conflict_data = json.loads(result.final_output)
        conflict_data['archetype'] = archetype.name
        conflict_data['estimated_duration'] = archetype.base_duration
        conflict_data['stakeholder_npcs'] = stakeholder_npcs
        
        return conflict_data
    
    async def _create_full_conflict(self, conflict_data: Dict[str, Any]) -> int:
        """Create conflict with all related data"""
        async with get_db_connection_context() as conn:
            async with conn.transaction():
                # Create base conflict
                conflict_id = await self.core.create_conflict(conflict_data)
                
                # Create stakeholders
                for stakeholder in conflict_data.get('stakeholders', []):
                    await self._create_stakeholder(conn, conflict_id, stakeholder)
                
                # Create resolution paths
                for path in conflict_data.get('resolution_paths', []):
                    await self._create_resolution_path(conn, conflict_id, path)
                
                # Create initial manipulation attempts
                await self._create_initial_manipulations(
                    conn, conflict_id, 
                    conflict_data.get('manipulation_opportunities', [])
                )
                
                return conflict_id
    
    async def _create_stakeholder(self, conn, conflict_id: int,
                                stakeholder_data: Dict[str, Any]):
        """Create a conflict stakeholder"""
        await conn.execute("""
            INSERT INTO ConflictStakeholders (
                conflict_id, npc_id, faction_id, faction_name,
                public_motivation, private_motivation, desired_outcome,
                involvement_level, leadership_ambition, faction_standing
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """,
        conflict_id, stakeholder_data['npc_id'],
        stakeholder_data.get('faction_id'),
        stakeholder_data.get('faction_name'),
        stakeholder_data['public_motivation'],
        stakeholder_data['private_motivation'],
        stakeholder_data['desired_outcome'],
        stakeholder_data.get('involvement_level', 5),
        stakeholder_data.get('leadership_ambition', 50),
        stakeholder_data.get('faction_standing', 50)
        )
        
        # Create secrets
        for secret in stakeholder_data.get('secrets', []):
            await conn.execute("""
                INSERT INTO StakeholderSecrets (
                    conflict_id, npc_id, secret_id, secret_type,
                    content, target_npc_id
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """,
            conflict_id, stakeholder_data['npc_id'],
            f"secret_{conflict_id}_{stakeholder_data['npc_id']}_{secret['type']}",
            secret['type'], secret['content'], secret.get('target_npc_id')
            )
    
    async def _create_resolution_path(self, conn, conflict_id: int,
                                    path_data: Dict[str, Any]):
        """Create a resolution path"""
        await conn.execute("""
            INSERT INTO ResolutionPaths (
                conflict_id, path_id, name, description,
                approach_type, difficulty, requirements,
                stakeholders_involved, key_challenges
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
        conflict_id, path_data['path_id'],
        path_data['name'], path_data['description'],
        path_data['approach_type'], path_data.get('difficulty', 5),
        json.dumps(path_data.get('requirements', {})),
        json.dumps(path_data.get('stakeholders_involved', [])),
        json.dumps(path_data.get('key_challenges', []))
        )
    
    async def _create_initial_manipulations(self, conn, conflict_id: int,
                                          opportunities: List[Dict[str, Any]]):
        """Create initial manipulation attempts"""
        for opp in opportunities[:2]:  # Limit initial attempts
            await conn.execute("""
                INSERT INTO PlayerManipulationAttempts (
                    conflict_id, user_id, conversation_id,
                    npc_id, manipulation_type, content, goal,
                    leverage_used, intimacy_level, success
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NULL)
            """,
            conflict_id, self.user_id, self.conversation_id,
            opp['npc_id'], opp['type'], opp['content'],
            json.dumps(opp.get('goal', {})),
            json.dumps(opp.get('leverage', {})),
            opp.get('intimacy_level', 0)
            )
