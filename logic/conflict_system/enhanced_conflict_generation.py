# logic/conflict_system/enhanced_conflict_generation.py
"""
Enhanced Conflict Generation System that deeply integrates with canon and world state
"""

import logging
import json
import random
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from agents import Agent, function_tool, ModelSettings, RunContextWrapper, Runner
from db.connection import get_db_connection_context
from lore.core import canon
from logic.conflict_system.conflict_agents import (
    ConflictContext, 
    initialize_conflict_assistants,
    ask_assistant
)
from embedding.vector_store import generate_embedding
from logic.relationship_integration import RelationshipIntegration

logger = logging.getLogger(__name__)

class ConflictArchetype:
    """Defines different types of conflicts and their characteristics"""
    
    ARCHETYPES = {
        "personal_dispute": {
            "scale": "personal",
            "min_stakeholders": 2,
            "max_stakeholders": 4,
            "facets": {"personal": 0.7, "political": 0.2, "crisis": 0.1},
            "base_duration": 3,
            "examples": ["romantic rivalry", "friendship betrayal", "family feud"]
        },
        "faction_rivalry": {
            "scale": "local",
            "min_stakeholders": 4,
            "max_stakeholders": 8,
            "facets": {"political": 0.6, "personal": 0.3, "crisis": 0.1},
            "base_duration": 7,
            "examples": ["territory dispute", "resource competition", "ideological clash"]
        },
        "succession_crisis": {
            "scale": "regional",
            "min_stakeholders": 5,
            "max_stakeholders": 10,
            "facets": {"political": 0.7, "personal": 0.2, "mystery": 0.1},
            "base_duration": 14,
            "examples": ["leadership vacuum", "contested inheritance", "coup attempt"]
        },
        "economic_collapse": {
            "scale": "regional",
            "min_stakeholders": 6,
            "max_stakeholders": 12,
            "facets": {"crisis": 0.7, "political": 0.3},
            "base_duration": 21,
            "examples": ["trade war", "resource depletion", "currency crisis"]
        },
        "religious_schism": {
            "scale": "world",
            "min_stakeholders": 8,
            "max_stakeholders": 15,
            "facets": {"political": 0.4, "personal": 0.3, "mystery": 0.3},
            "base_duration": 30,
            "examples": ["doctrine dispute", "prophet emergence", "sacred site conflict"]
        },
        "apocalyptic_threat": {
            "scale": "world",
            "min_stakeholders": 10,
            "max_stakeholders": 20,
            "facets": {"crisis": 0.8, "political": 0.2},
            "base_duration": 60,
            "examples": ["plague outbreak", "environmental disaster", "ancient evil awakening"]
        }
    }

class WorldStateAnalyzer:
    """Analyzes the current world state to identify conflict potential"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.ctx = RunContextWrapper({
            "user_id": user_id,
            "conversation_id": conversation_id
        })
        self._assistants = None  # Lazy initialization
    
    async def _get_assistants(self):
        """Get or initialize assistants"""
        if self._assistants is None:
            self._assistants = await initialize_conflict_assistants()
        return self._assistants
    
    async def analyze_conflict_potential(self) -> Dict[str, Any]:
        """Analyze world state for conflict opportunities"""
        async with get_db_connection_context() as conn:
            # Get recent events and changes
            recent_events = await self._get_recent_canonical_events(conn)
            
            # Analyze different conflict sources
            relationship_tensions = await self._analyze_relationship_tensions(conn)
            faction_dynamics = await self._analyze_faction_dynamics(conn)
            economic_stress = await self._analyze_economic_factors(conn)
            historical_grievances = await self._analyze_historical_grievances(conn)
            regional_tensions = await self._analyze_regional_tensions(conn)
            
            # Get current narrative context
            narrative_context = await self._get_narrative_context(conn)
            
            return {
                "recent_events": recent_events,
                "relationship_tensions": relationship_tensions,
                "faction_dynamics": faction_dynamics,
                "economic_stress": economic_stress,
                "historical_grievances": historical_grievances,
                "regional_tensions": regional_tensions,
                "narrative_context": narrative_context,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _get_recent_canonical_events(self, conn) -> List[Dict[str, Any]]:
        """Get recent canonical events that might spark conflicts"""
        events = await conn.fetch("""
            SELECT ce.event_text, ce.tags, ce.significance, ce.timestamp,
                   um.entity_type, um.entity_id
            FROM CanonicalEvents ce
            LEFT JOIN unified_memories um ON um.memory_text = ce.event_text
            WHERE ce.user_id = $1 AND ce.conversation_id = $2
                AND ce.timestamp > NOW() - INTERVAL '7 days'
                AND ce.significance >= 6
            ORDER BY ce.timestamp DESC
            LIMIT 20
        """, self.user_id, self.conversation_id)
        
        return [dict(e) for e in events]
    
    async def _analyze_relationship_tensions(self, conn) -> Dict[str, Any]:
        """Analyze NPC relationships for conflict potential"""
        # Get NPCs with high negative emotions
        tensions = await conn.fetch("""
            SELECT n1.npc_id as npc1_id, n1.npc_name as npc1_name,
                   n2.npc_id as npc2_id, n2.npc_name as npc2_name,
                   sl.link_type, sl.link_level, sl.dynamics
            FROM SocialLinks sl
            JOIN NPCStats n1 ON sl.entity1_id = n1.npc_id AND sl.entity1_type = 'npc'
            JOIN NPCStats n2 ON sl.entity2_id = n2.npc_id AND sl.entity2_type = 'npc'
            WHERE sl.user_id = $1 AND sl.conversation_id = $2
                AND (sl.link_level < -50 OR sl.dynamics->>'hostility' > '50')
            ORDER BY sl.link_level ASC
            LIMIT 10
        """, self.user_id, self.conversation_id)
        
        # Get NPCs with unresolved grudges
        grudges = await conn.fetch("""
            SELECT ch.affected_npc_id, ch.grudge_level, ch.narrative_impact,
                   c.conflict_name, n.npc_name
            FROM ConflictHistory ch
            JOIN Conflicts c ON ch.conflict_id = c.conflict_id
            JOIN NPCStats n ON ch.affected_npc_id = n.npc_id
            WHERE ch.user_id = $1 AND ch.conversation_id = $2
                AND ch.grudge_level > 50
                AND ch.has_triggered_consequence = FALSE
            ORDER BY ch.grudge_level DESC
            LIMIT 10
        """, self.user_id, self.conversation_id)
        
        return {
            "high_tensions": [dict(t) for t in tensions],
            "unresolved_grudges": [dict(g) for g in grudges],
            "tension_score": len(tensions) * 10 + sum(g['grudge_level'] for g in grudges) / 10
        }
    
    async def _analyze_faction_dynamics(self, conn) -> Dict[str, Any]:
        """Analyze faction relationships and power dynamics"""
        # Get recent power shifts
        power_shifts = await conn.fetch("""
            SELECT faction_name, SUM(change_amount) as total_change,
                   COUNT(*) as shift_count,
                   array_agg(cause ORDER BY created_at DESC) as causes
            FROM FactionPowerShifts
            WHERE user_id = $1 AND conversation_id = $2
                AND created_at > NOW() - INTERVAL '14 days'
            GROUP BY faction_name
            HAVING ABS(SUM(change_amount)) > 3
            ORDER BY ABS(SUM(change_amount)) DESC
        """, self.user_id, self.conversation_id)
        
        # Get faction rivalries - FIX: Check if rivals field exists and handle properly
        rivalries = await conn.fetch("""
            SELECT f1.id as faction1_id, f1.name as faction1_name,
                   f2.id as faction2_id, f2.name as faction2_name,
                   f1.power_level as f1_power, f2.power_level as f2_power
            FROM Factions f1
            CROSS JOIN Factions f2
            WHERE f1.user_id = $1 AND f1.conversation_id = $2
                AND f2.user_id = $1 AND f2.conversation_id = $2
                AND f1.id < f2.id
                AND (
                    -- Check if rivals column exists and contains the other faction
                    (f1.rivals IS NOT NULL AND f2.id = ANY(f1.rivals::int[])) 
                    OR 
                    (f2.rivals IS NOT NULL AND f1.id = ANY(f2.rivals::int[]))
                )
        """, self.user_id, self.conversation_id)
        
        # Alternative simpler query if the above doesn't work:
        # Just get all faction pairs and check rivalry elsewhere
        if not rivalries:
            rivalries = await conn.fetch("""
                SELECT f1.id as faction1_id, f1.name as faction1_name,
                       f2.id as faction2_id, f2.name as faction2_name,
                       f1.power_level as f1_power, f2.power_level as f2_power
                FROM Factions f1
                CROSS JOIN Factions f2
                WHERE f1.user_id = $1 AND f1.conversation_id = $2
                    AND f2.user_id = $1 AND f2.conversation_id = $2
                    AND f1.id < f2.id
                    AND ABS(f1.power_level - f2.power_level) < 3  -- Similar power levels suggest rivalry
                LIMIT 5
            """, self.user_id, self.conversation_id)
        
        return {
            "power_shifts": [dict(p) for p in power_shifts],
            "active_rivalries": [dict(r) for r in rivalries],
            "instability_score": sum(abs(p['total_change']) for p in power_shifts) if power_shifts else 0
        }
    
    async def _analyze_economic_factors(self, conn) -> Dict[str, Any]:
        """Analyze economic stress factors"""
        # Get resource scarcity
        resource_changes = await conn.fetch("""
            SELECT resource_type, AVG(amount_changed) as avg_change,
                   COUNT(*) as transaction_count
            FROM ResourceHistoryLog
            WHERE user_id = $1 AND conversation_id = $2
                AND timestamp > NOW() - INTERVAL '7 days'
                AND amount_changed < 0
            GROUP BY resource_type
            HAVING AVG(amount_changed) < -5
        """, self.user_id, self.conversation_id)
        
        # Get player resource status
        player_resources = await conn.fetchrow("""
            SELECT money, supplies, influence
            FROM PlayerResources
            WHERE user_id = $1 AND conversation_id = $2
        """, self.user_id, self.conversation_id)
        
        return {
            "resource_scarcity": [dict(r) for r in resource_changes],
            "player_resources": dict(player_resources) if player_resources else {},
            "economic_stress": len(resource_changes) * 20
        }
    
    async def _analyze_historical_grievances(self, conn) -> Dict[str, Any]:
        """Analyze historical events that could resurface"""
        # Get significant past events
        historical_events = await conn.fetch("""
            SELECT he.name, he.description, he.significance,
                   he.involved_entities, he.consequences,
                   he.disputed_facts, he.date_description
            FROM HistoricalEvents he
            WHERE he.user_id = $1 AND he.conversation_id = $2
                AND he.significance >= 7
                AND he.disputed_facts IS NOT NULL
                AND array_length(he.disputed_facts, 1) > 0
            ORDER BY he.significance DESC
            LIMIT 5
        """, self.user_id, self.conversation_id)
        
        # Get commemorations that might spark conflict
        commemorations = await conn.fetch("""
            SELECT he.name, he.commemorations, he.cultural_impact
            FROM HistoricalEvents he
            WHERE he.user_id = $1 AND he.conversation_id = $2
                AND he.commemorations IS NOT NULL
                AND array_length(he.commemorations, 1) > 0
                AND he.date_description LIKE '%anniversary%'
            LIMIT 5
        """, self.user_id, self.conversation_id)
        
        return {
            "disputed_history": [dict(h) for h in historical_events],
            "upcoming_commemorations": [dict(c) for c in commemorations],
            "grievance_score": sum(h['significance'] for h in historical_events)
        }
    
    async def _analyze_regional_tensions(self, conn) -> Dict[str, Any]:
        """Analyze regional and location-based tensions"""
        # Get border disputes
        border_disputes = await conn.fetch("""
            SELECT bd.*, r1.name as region1_name, r2.name as region2_name
            FROM BorderDisputes bd
            JOIN GeographicRegions r1 ON bd.region1_id = r1.id
            JOIN GeographicRegions r2 ON bd.region2_id = r2.id
            WHERE bd.status IN ('active', 'escalating')
            ORDER BY bd.severity DESC
            LIMIT 5
        """, self.user_id, self.conversation_id)
        
        # Get contested locations
        contested_locations = await conn.fetch("""
            SELECT l.location_name, l.strategic_value,
                   l.cultural_significance, l.access_restrictions
            FROM Locations l
            WHERE l.user_id = $1 AND l.conversation_id = $2
                AND l.strategic_value >= 7
                AND array_length(l.access_restrictions, 1) > 0
            LIMIT 5
        """, self.user_id, self.conversation_id)
        
        return {
            "border_disputes": [dict(b) for b in border_disputes],
            "contested_locations": [dict(c) for c in contested_locations],
            "territorial_tension": len(border_disputes) * 30 + len(contested_locations) * 20
        }
    
    async def _get_narrative_context(self, conn) -> Dict[str, Any]:
        """Get current narrative context"""
        # Get current narrative stage
        narrative_info = await conn.fetchrow("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id = $1 AND conversation_id = $2
                AND key IN ('CurrentNarrativeId', 'CurrentDay', 'CurrentLocation')
        """, self.user_id, self.conversation_id)
        
        return dict(narrative_info) if narrative_info else {}

class OrganicConflictGenerator:
    """Generates conflicts that feel organic and connected to the world state"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.analyzer = WorldStateAnalyzer(user_id, conversation_id)
        self.context = ConflictContext(user_id, conversation_id)
        self._assistants = None  # Lazy initialization
        
    async def _get_assistants(self):
        """Get or initialize assistants"""
        if self._assistants is None:
            self._assistants = await initialize_conflict_assistants()
        return self._assistants
        
    async def generate_contextual_conflict(self, 
                                         preferred_scale: Optional[str] = None,
                                         force_archetype: Optional[str] = None) -> Dict[str, Any]:
        """Generate a conflict based on current world state"""
        
        # Analyze world state
        world_state = await self.analyzer.analyze_conflict_potential()
        
        # Determine conflict pressure
        total_pressure = (
            world_state["relationship_tensions"]["tension_score"] +
            world_state["faction_dynamics"]["instability_score"] +
            world_state["economic_stress"]["economic_stress"] +
            world_state["historical_grievances"]["grievance_score"] +
            world_state["regional_tensions"]["territorial_tension"]
        )
        
        # Use interpreter agent to identify best conflict opportunities
        interpretation_prompt = f"""
        Analyze this world state data and identify the most compelling conflict opportunity:
        
        {json.dumps(world_state, indent=2)}
        
        Total pressure score: {total_pressure}
        Preferred scale: {preferred_scale or 'any'}
        
        Consider all factors and suggest the most dramatically appropriate conflict.
        """
        
        # Initialize assistants if not already done
        assistants = await self._get_assistants()
        
        interpretation_result = await ask_assistant(
            assistants["world_state_interpreter"],
            interpretation_prompt,
            self.context
        )
        # Extract the actual interpretation text
        interpretation = interpretation_result if isinstance(interpretation_result, str) else str(interpretation_result)
        
        # Determine appropriate archetype
        if force_archetype:
            archetype = force_archetype
        else:
            archetype = await self._select_archetype_by_pressure(total_pressure, world_state)
        
        # Generate the actual conflict
        conflict_data = await self._generate_conflict_from_archetype(
            archetype, 
            world_state,
            interpretation
        )
        
        # Enrich with canonical connections
        enriched_conflict = await self._enrich_with_canon(conflict_data)
        
        return enriched_conflict
    
    async def _select_archetype_by_pressure(self, pressure: float, 
                                          world_state: Dict[str, Any]) -> str:
        """Select appropriate conflict archetype based on pressure and context"""
        
        # Pressure thresholds
        if pressure < 50:
            archetypes = ["personal_dispute"]
        elif pressure < 100:
            archetypes = ["personal_dispute", "faction_rivalry"]
        elif pressure < 200:
            archetypes = ["faction_rivalry", "succession_crisis"]
        elif pressure < 300:
            archetypes = ["succession_crisis", "economic_collapse"]
        elif pressure < 400:
            archetypes = ["economic_collapse", "religious_schism"]
        else:
            archetypes = ["religious_schism", "apocalyptic_threat"]
        
        # Weight by specific factors
        weights = {}
        for archetype in archetypes:
            weight = 1.0
            
            if archetype == "personal_dispute" and world_state["relationship_tensions"]["high_tensions"]:
                weight *= 2.0
            elif archetype == "faction_rivalry" and world_state["faction_dynamics"]["active_rivalries"]:
                weight *= 2.0
            elif archetype == "economic_collapse" and world_state["economic_stress"]["resource_scarcity"]:
                weight *= 2.0
            elif archetype == "succession_crisis" and any("coup" in str(e) for e in world_state["recent_events"]):
                weight *= 2.0
                
            weights[archetype] = weight
        
        # Weighted random selection
        total_weight = sum(weights.values())
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for archetype, weight in weights.items():
            cumulative += weight
            if r <= cumulative:
                return archetype
                
        return archetypes[-1]  # Fallback
    
    async def _generate_conflict_from_archetype(self, archetype: str,
                                               world_state: Dict[str, Any],
                                               interpretation: str) -> Dict[str, Any]:
        """Generate specific conflict details from archetype and world state"""
        
        archetype_data = ConflictArchetype.ARCHETYPES[archetype]
        
        # Get relevant NPCs based on archetype
        npcs = await self._get_relevant_npcs_for_archetype(archetype, world_state)
        
        # Get relevant locations
        locations = await self._get_relevant_locations(archetype, world_state)
        
        # Use seed agent to generate conflict
        generation_prompt = f"""
        Create a {archetype} conflict based on this world state:
        
        World Analysis: {interpretation}
        
        Archetype: {archetype}
        Scale: {archetype_data['scale']}
        Example types: {', '.join(archetype_data['examples'])}
        
        Available NPCs: {json.dumps(npcs, indent=2)}
        Relevant Locations: {json.dumps(locations, indent=2)}
        Recent Events: {json.dumps(world_state['recent_events'][:5], indent=2)}
        
        The conflict should:
        - Emerge naturally from the described tensions
        - Involve {archetype_data['min_stakeholders']}-{archetype_data['max_stakeholders']} stakeholders
        - Have clear resolution paths
        - Include femdom power dynamics where appropriate
        
        Generate a complete conflict structure with JSON format including:
        - conflict_name
        - description
        - stakeholders (array with npc_name, role, faction_name, public_motivation, private_motivation, desired_outcome)
        - resolution_paths (array with path_id, name, description, approach_type, difficulty, requirements, stakeholders_involved, key_challenges)
        """
        
        # Initialize assistants if not already done
        assistants = await self._get_assistants()
            
        result = await ask_assistant(
            assistants["conflict_seed"],
            generation_prompt,
            self.context
        )
        
        # Parse and structure the result
        conflict_data = result if isinstance(result, dict) else json.loads(result)
        
        # Add archetype metadata
        conflict_data.update({
            "archetype": archetype,
            "scale": archetype_data["scale"],
            "facets": archetype_data["facets"],
            "estimated_duration": archetype_data["base_duration"],
            "generation_context": {
                "world_pressure": sum([
                    world_state["relationship_tensions"]["tension_score"],
                    world_state["faction_dynamics"]["instability_score"],
                    world_state["economic_stress"]["economic_stress"]
                ]),
                "primary_driver": self._identify_primary_driver(world_state)
            }
        })
        
        return conflict_data
    
    async def _get_relevant_npcs_for_archetype(self, archetype: str,
                                              world_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get NPCs relevant to the conflict archetype"""
        async with get_db_connection_context() as conn:
            # Note: We're reading existing NPCs, not creating, so this is fine
            # But if we need to create NPCs based on the query, we should use canon
            
            if archetype == "personal_dispute":
                npcs = await conn.fetch("""
                    SELECT DISTINCT n.*, 
                           array_agg(f.name) as faction_affiliations
                    FROM NPCStats n
                    LEFT JOIN Factions f ON n.affiliations @> ARRAY[f.id]
                    WHERE n.user_id = $1 AND n.conversation_id = $2
                        AND n.introduced = TRUE
                        AND (n.intensity > 70 OR n.dominance > 70)
                    GROUP BY n.npc_id
                    ORDER BY n.intensity DESC
                    LIMIT 10
                """, self.user_id, self.conversation_id)
                
            elif archetype in ["faction_rivalry", "succession_crisis"]:
                # Get faction leaders and high-ranking members
                npcs = await conn.fetch("""
                    SELECT DISTINCT n.*, 
                           f.name as primary_faction,
                           f.power_level as faction_power
                    FROM NPCStats n
                    JOIN Factions f ON n.affiliations @> ARRAY[f.id]
                    WHERE n.user_id = $1 AND n.conversation_id = $2
                        AND n.introduced = TRUE
                        AND (n.dominance > 60 OR f.power_level > 5)
                    ORDER BY n.dominance DESC, f.power_level DESC
                    LIMIT 15
                """, self.user_id, self.conversation_id)
                
            else:
                # Get all prominent NPCs
                npcs = await conn.fetch("""
                    SELECT n.*, 
                           array_agg(DISTINCT f.name) as faction_affiliations,
                           COUNT(DISTINCT sl.link_id) as relationship_count
                    FROM NPCStats n
                    LEFT JOIN Factions f ON n.affiliations @> ARRAY[f.id]
                    LEFT JOIN SocialLinks sl ON 
                        (sl.entity1_id = n.npc_id AND sl.entity1_type = 'npc')
                        OR (sl.entity2_id = n.npc_id AND sl.entity2_type = 'npc')
                    WHERE n.user_id = $1 AND n.conversation_id = $2
                        AND n.introduced = TRUE
                    GROUP BY n.npc_id
                    HAVING COUNT(DISTINCT sl.link_id) > 2
                    ORDER BY n.dominance DESC, COUNT(DISTINCT sl.link_id) DESC
                    LIMIT 20
                """, self.user_id, self.conversation_id)
            
            return [dict(n) for n in npcs]
    
    async def _get_relevant_locations(self, archetype: str,
                                    world_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get locations relevant to the conflict"""
        async with get_db_connection_context() as conn:
            if archetype in ["personal_dispute"]:
                # Get intimate/personal locations
                locations = await conn.fetch("""
                    SELECT location_name, description, cultural_significance
                    FROM Locations
                    WHERE user_id = $1 AND conversation_id = $2
                        AND location_type IN ('residence', 'private', 'social')
                    LIMIT 5
                """, self.user_id, self.conversation_id)
                
            elif archetype in ["faction_rivalry", "succession_crisis"]:
                # Get strategic locations
                locations = await conn.fetch("""
                    SELECT location_name, description, strategic_value,
                           cultural_significance, economic_importance
                    FROM Locations
                    WHERE user_id = $1 AND conversation_id = $2
                        AND (strategic_value >= 7 OR economic_importance = 'high')
                    ORDER BY strategic_value DESC
                    LIMIT 5
                """, self.user_id, self.conversation_id)
                
            else:
                # Get major locations
                locations = await conn.fetch("""
                    SELECT location_name, description, 
                           strategic_value, population_density
                    FROM Locations
                    WHERE user_id = $1 AND conversation_id = $2
                        AND population_density IN ('high', 'very high')
                    ORDER BY strategic_value DESC
                    LIMIT 5
                """, self.user_id, self.conversation_id)
            
            return [dict(l) for l in locations]
    
    async def _enrich_with_canon(self, conflict_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich conflict with canonical references and connections"""
        async with get_db_connection_context() as conn:
            ctx = RunContextWrapper({
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
            # Add historical context
            if "root_cause" in conflict_data:
                # Search for related historical events
                search_vector = await generate_embedding(conflict_data["root_cause"])
                
                historical_context = await conn.fetch("""
                    SELECT name, description, significance,
                           1 - (embedding <=> $1) AS relevance
                    FROM HistoricalEvents
                    WHERE user_id = $2 AND conversation_id = $3
                        AND embedding IS NOT NULL
                    ORDER BY embedding <=> $1
                    LIMIT 3
                """, search_vector, self.user_id, self.conversation_id)
                
                conflict_data["historical_context"] = [dict(h) for h in historical_context]
                
                # Create canonical links to historical events
                for event in historical_context[:1]:  # Link to most relevant
                    if event['relevance'] > 0.7:
                        await canon.log_canonical_event(
                            ctx, conn,
                            f"Conflict '{conflict_data['conflict_name']}' echoes historical event '{event['name']}'",
                            tags=["conflict", "history", "connection"],
                            significance=6
                        )
            
            # Add cultural context
            if conflict_data.get("scale") in ["regional", "world"]:
                cultural_elements = await conn.fetch("""
                    SELECT name, description, significance
                    FROM CulturalElements
                    WHERE significance >= 7
                    ORDER BY significance DESC
                    LIMIT 3
                """)
                
                conflict_data["cultural_stakes"] = [dict(c) for c in cultural_elements]
            
            # Add myth/legend connections for larger conflicts
            if conflict_data.get("archetype") == "apocalyptic_threat":
                myths = await conn.fetch("""
                    SELECT name, description, believability
                    FROM UrbanMyths
                    WHERE believability >= 7
                    ORDER BY spread_rate DESC
                    LIMIT 2
                """)
                
                conflict_data["prophetic_elements"] = [dict(m) for m in myths]
                
                # Create canonical connection to myths
                for myth in myths[:1]:
                    await canon.log_canonical_event(
                        ctx, conn,
                        f"Conflict '{conflict_data['conflict_name']}' fulfills prophecy from myth '{myth['name']}'",
                        tags=["conflict", "prophecy", "myth"],
                        significance=9
                    )
            
            return conflict_data
    
    def _identify_primary_driver(self, world_state: Dict[str, Any]) -> str:
        """Identify the primary driver of conflict"""
        drivers = {
            "relationships": world_state["relationship_tensions"]["tension_score"],
            "factions": world_state["faction_dynamics"]["instability_score"],
            "economics": world_state["economic_stress"]["economic_stress"],
            "history": world_state["historical_grievances"]["grievance_score"],
            "territory": world_state["regional_tensions"]["territorial_tension"]
        }
        
        return max(drivers.items(), key=lambda x: x[1])[0]


# ========== IMPLEMENTATION FUNCTIONS (for direct Python calls) ==========

async def generate_organic_conflict_impl(
    ctx: RunContextWrapper,
    preferred_scale: Optional[str] = None,
    force_archetype: Optional[str] = None
) -> Dict[str, Any]:
    """
    Implementation: Generate a conflict that feels organic based on world state.
    """
    # Fix: Access context as a dict
    context = ctx.context if hasattr(ctx, 'context') else ctx
    user_id = context.get('user_id') if isinstance(context, dict) else context.user_id
    conversation_id = context.get('conversation_id') if isinstance(context, dict) else context.conversation_id
    
    generator = OrganicConflictGenerator(user_id, conversation_id)
    
    try:
        conflict_data = await generator.generate_contextual_conflict(
            preferred_scale=preferred_scale,
            force_archetype=force_archetype
        )
        
        async with get_db_connection_context() as conn:
            # Get current day using canon helper
            current_day = await conn.fetchval("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentDay'
            """, user_id, conversation_id)
            current_day = int(current_day) if current_day else 1
            
            # Create conflict record
            conflict_id = await conn.fetchval("""
                INSERT INTO Conflicts (
                    user_id, conversation_id, conflict_name, conflict_type,
                    description, progress, phase, start_day, estimated_duration,
                    success_rate, is_active
                ) VALUES ($1, $2, $3, $4, $5, 0, 'brewing', $6, $7, $8, TRUE)
                RETURNING conflict_id
            """, 
            user_id, conversation_id,
            conflict_data["conflict_name"], conflict_data["archetype"],
            conflict_data["description"], current_day,
            conflict_data["estimated_duration"], 0.5
            )
            
            # Create stakeholders using canon
            for stakeholder in conflict_data.get("stakeholders", []):
                # Ensure NPC exists canonically
                npc_id = await canon.find_or_create_npc(
                    ctx, conn,
                    npc_name=stakeholder["npc_name"],
                    role=stakeholder.get("role"),
                    affiliations=stakeholder.get("faction_affiliations", [])
                )
                
                # Ensure faction exists if specified
                faction_id = None
                if stakeholder.get("faction_name"):
                    faction_id = await canon.find_or_create_faction(
                        ctx, conn,
                        faction_name=stakeholder["faction_name"],
                        type=stakeholder.get("faction_type", "organization")
                    )
                
                await conn.execute("""
                    INSERT INTO ConflictStakeholders (
                        conflict_id, npc_id, faction_id, faction_name,
                        public_motivation, private_motivation, desired_outcome,
                        involvement_level
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                conflict_id, npc_id, faction_id, stakeholder.get("faction_name"),
                stakeholder["public_motivation"], stakeholder["private_motivation"],
                stakeholder["desired_outcome"], stakeholder.get("involvement_level", 5)
                )
            
            # Create resolution paths
            for path in conflict_data.get("resolution_paths", []):
                await conn.execute("""
                    INSERT INTO ResolutionPaths (
                        conflict_id, path_id, name, description,
                        approach_type, difficulty, requirements,
                        stakeholders_involved, key_challenges
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                conflict_id, path["path_id"], path["name"], path["description"],
                path["approach_type"], path["difficulty"],
                json.dumps(path.get("requirements", {})),
                json.dumps(path["stakeholders_involved"]),
                json.dumps(path["key_challenges"])
                )
            
            # Log canonical event
            await canon.log_canonical_event(
                ctx, conn,
                f"Conflict emerged: {conflict_data['conflict_name']} - {conflict_data['description'][:100]}...",
                tags=["conflict", conflict_data["archetype"], "emergence"],
                significance=8
            )
            
            conflict_data["conflict_id"] = conflict_id
            
        return conflict_data
        
    except Exception as e:
        logger.error(f"Error generating organic conflict: {e}", exc_info=True)
        return {"error": str(e)}

async def analyze_conflict_pressure_impl(ctx: RunContextWrapper) -> Dict[str, Any]:
    """
    Implementation: Analyze current world state for conflict pressure.
    
    Returns:
        Analysis of conflict potential
    """
    # Fix: Access context as a dict
    context = ctx.context if hasattr(ctx, 'context') else ctx
    user_id = context.get('user_id') if isinstance(context, dict) else context.user_id
    conversation_id = context.get('conversation_id') if isinstance(context, dict) else context.conversation_id
    
    analyzer = WorldStateAnalyzer(user_id, conversation_id)
    
    try:
        analysis = await analyzer.analyze_conflict_potential()
        
        # Calculate total pressure
        total_pressure = (
            analysis["relationship_tensions"]["tension_score"] +
            analysis["faction_dynamics"]["instability_score"] +
            analysis["economic_stress"]["economic_stress"] +
            analysis["historical_grievances"]["grievance_score"] +
            analysis["regional_tensions"]["territorial_tension"]
        )
        
        analysis["total_pressure"] = total_pressure
        analysis["recommended_action"] = _get_pressure_recommendation(total_pressure)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing conflict pressure: {e}", exc_info=True)
        return {"error": str(e)}


# ========== TOOL FUNCTIONS (for agent tool system) ==========

@function_tool
async def generate_organic_conflict(
    ctx: RunContextWrapper,
    preferred_scale: Optional[str] = None,
    force_archetype: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a conflict that feels organic based on world state."""
    return await generate_organic_conflict_impl(ctx, preferred_scale, force_archetype)

@function_tool
async def analyze_conflict_pressure(ctx: RunContextWrapper) -> Dict[str, Any]:
    """
    Analyze current world state for conflict pressure.
    
    Returns:
        Analysis of conflict potential
    """
    return await analyze_conflict_pressure_impl(ctx)


# ========== HELPER FUNCTIONS ==========

def _get_pressure_recommendation(pressure: float) -> str:
    """Get recommendation based on pressure level"""
    if pressure < 50:
        return "Low pressure - personal conflicts only"
    elif pressure < 100:
        return "Moderate pressure - faction rivalries emerging"
    elif pressure < 200:
        return "High pressure - succession crises possible"
    elif pressure < 300:
        return "Critical pressure - economic collapse imminent"
    elif pressure < 400:
        return "Extreme pressure - religious/ideological schisms forming"
    else:
        return "Apocalyptic pressure - world-ending threats manifesting"
