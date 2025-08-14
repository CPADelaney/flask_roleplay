# logic/conflict_system/slice_of_life_conflicts.py
"""
Slice-of-life conflict system with LLM-generated dynamic content.
Replaces hardcoded patterns with intelligent, contextual generation.
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from agents import Agent, ModelSettings, function_tool, RunContextWrapper
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ===============================================================================
# ENUMS (Preserved from original)
# ===============================================================================

class SliceOfLifeConflictType(Enum):
    """Types of subtle daily conflicts"""
    PERMISSION_PATTERNS = "permission_patterns"
    ROUTINE_DOMINANCE = "routine_dominance"
    BOUNDARY_EROSION = "boundary_erosion"
    SOCIAL_PECKING_ORDER = "social_pecking_order"
    FRIENDSHIP_BOUNDARIES = "friendship_boundaries"
    ROLE_EXPECTATIONS = "role_expectations"
    FINANCIAL_CONTROL = "financial_control"
    SUBTLE_RIVALRY = "subtle_rivalry"
    PREFERENCE_SUBMISSION = "preference_submission"
    CARE_DEPENDENCY = "care_dependency"
    INDEPENDENCE_STRUGGLE = "independence_struggle"
    PASSIVE_AGGRESSION = "passive_aggression"
    EMOTIONAL_LABOR = "emotional_labor"
    DECISION_FATIGUE = "decision_fatigue"
    MASK_SLIPPAGE = "mask_slippage"
    DOMESTIC_HIERARCHY = "domestic_hierarchy"
    GROOMING_PATTERNS = "grooming_patterns"
    GASLIGHTING_GENTLE = "gaslighting_gentle"
    SOCIAL_ISOLATION = "social_isolation"
    CONDITIONING_RESISTANCE = "conditioning_resistance"

class ConflictIntensity(Enum):
    """How overtly the conflict manifests"""
    SUBTEXT = "subtext"
    TENSION = "tension"
    PASSIVE = "passive"
    DIRECT = "direct"
    CONFRONTATION = "confrontation"

class ResolutionApproach(Enum):
    """How conflicts resolve in slice-of-life"""
    GRADUAL_ACCEPTANCE = "gradual_acceptance"
    SUBTLE_RESISTANCE = "subtle_resistance"
    NEGOTIATED_COMPROMISE = "negotiated_compromise"
    ESTABLISHED_PATTERN = "established_pattern"
    THIRD_PARTY_INFLUENCE = "third_party_influence"
    TIME_EROSION = "time_erosion"

@dataclass
class SliceOfLifeStake:
    """What's at stake in mundane conflicts"""
    stake_type: str
    description: str
    daily_impact: str
    relationship_impact: str
    accumulation_factor: float

@dataclass
class DailyConflictEvent:
    """A conflict moment embedded in daily routine"""
    activity_type: str
    conflict_manifestation: str
    choice_presented: bool
    accumulation_impact: float
    npc_reactions: Dict[int, str]

# ===============================================================================
# EMERGENT CONFLICT DETECTOR WITH LLM
# ===============================================================================

class EmergentConflictDetector:
    """Detects conflicts emerging from daily interactions using LLM analysis"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._pattern_analyzer = None
        
    @property
    def pattern_analyzer(self) -> Agent:
        """Lazy-load pattern analysis agent"""
        if self._pattern_analyzer is None:
            self._pattern_analyzer = Agent(
                name="Pattern Analyzer",
                instructions="""
                Analyze memory patterns and relationship dynamics for emerging slice-of-life conflicts.
                
                Look for:
                - Recurring permission-seeking behaviors
                - Subtle boundary violations
                - Power imbalances in daily routines
                - Emotional labor disparities
                - Control patterns disguised as care
                - Gradual isolation tactics
                - Decision fatigue patterns
                
                Generate nuanced, contextual conflict descriptions that feel organic to daily life.
                Focus on subtle tensions rather than dramatic confrontations.
                """,
                model="gpt-5-nano",
            )
        return self._pattern_analyzer
    
    async def detect_brewing_tensions(self) -> List[Dict[str, Any]]:
        """Analyze recent interactions for emerging conflicts with LLM"""
        
        async with get_db_connection_context() as conn:
            # Get recent memory patterns
            memory_patterns = await conn.fetch("""
                SELECT 
                    entity_id,
                    entity_type,
                    memory_text,
                    emotional_valence,
                    tags
                FROM enhanced_memories
                WHERE user_id = $1 
                AND conversation_id = $2
                AND created_at > NOW() - INTERVAL '3 days'
                ORDER BY created_at DESC
                LIMIT 100
            """, self.user_id, self.conversation_id)
            
            # Get relationship dynamics
            relationships = await conn.fetch("""
                SELECT 
                    entity1_id,
                    entity2_id,
                    dimension,
                    current_value,
                    recent_delta
                FROM relationship_dimensions
                WHERE user_id = $1 
                AND conversation_id = $2
                AND dimension IN ('dominance', 'control', 'dependency', 'resistance')
            """, self.user_id, self.conversation_id)
        
        if not memory_patterns and not relationships:
            return []
        
        # Use LLM to analyze patterns
        conflicts = await self._analyze_patterns_with_llm(memory_patterns, relationships)
        
        return conflicts
    
    async def _analyze_patterns_with_llm(
        self, 
        memories: List, 
        relationships: List
    ) -> List[Dict]:
        """Use LLM to detect emerging tensions from patterns"""
        
        # Prepare context for LLM
        memory_summary = self._summarize_memories(memories[:20])
        relationship_summary = self._summarize_relationships(relationships)
        
        prompt = f"""
        Analyze these recent patterns for emerging slice-of-life conflicts:
        
        Recent Memories:
        {memory_summary}
        
        Relationship Dynamics:
        {relationship_summary}
        
        Identify 1-3 brewing tensions that could become subtle daily conflicts.
        For each, provide:
        1. Conflict type (permission_patterns, boundary_erosion, etc.)
        2. Intensity level (subtext, tension, passive)
        3. A specific, contextual description
        4. Evidence from the patterns
        
        Format as JSON array.
        """
        
        response = await self.pattern_analyzer.run(prompt)
        
        try:
            tensions = json.loads(response.content)
            
            # Map to internal format
            conflicts = []
            for tension in tensions:
                conflicts.append({
                    'type': SliceOfLifeConflictType[tension.get('conflict_type', 'SUBTLE_RIVALRY').upper()],
                    'intensity': ConflictIntensity[tension.get('intensity', 'TENSION').upper()],
                    'description': tension.get('description', 'A subtle tension emerges'),
                    'evidence': tension.get('evidence', []),
                    'tension_level': random.uniform(0.3, 0.7)
                })
            
            return conflicts
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return []
    
    def _summarize_memories(self, memories: List) -> str:
        """Create a summary of memories for LLM context"""
        summary = []
        for m in memories[:10]:
            summary.append(f"- {m['memory_text']} (emotion: {m.get('emotional_valence', 'neutral')})")
        return "\n".join(summary) if summary else "No recent significant memories"
    
    def _summarize_relationships(self, relationships: List) -> str:
        """Create a summary of relationship dynamics"""
        summary = []
        for r in relationships[:5]:
            summary.append(
                f"- {r['dimension']}: {r['current_value']:.2f} "
                f"(recent change: {r.get('recent_delta', 0):.2f})"
            )
        return "\n".join(summary) if summary else "No significant relationship dynamics"

# ===============================================================================
# SLICE OF LIFE CONFLICT MANAGER WITH LLM
# ===============================================================================

class SliceOfLifeConflictManager:
    """Manages conflicts through daily activities with LLM generation"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._conflict_agent = None
        
    @property
    def conflict_agent(self) -> Agent:
        """Lazy-load conflict generation agent"""
        if self._conflict_agent is None:
            self._conflict_agent = Agent(
                name="Slice of Life Conflict Director",
                instructions="""
                Generate subtle conflicts embedded in daily routines.
                
                Focus on:
                - Mundane moments where power dynamics emerge
                - Small choices that reveal larger patterns
                - Passive-aggressive behaviors disguised as normal interaction
                - The accumulation of tiny surrenders
                - Conflicts that feel realistic and relatable
                
                Avoid dramatic confrontations. Keep everything subtle and slice-of-life.
                Make NPCs feel like real people with complex motivations.
                """,
                model="gpt-5-nano",
            )
        return self._conflict_agent
    
    async def embed_conflict_in_activity(
        self,
        conflict_id: int,
        activity_type: str,
        participating_npcs: List[int]
    ) -> DailyConflictEvent:
        """Generate how a conflict manifests in a daily activity"""
        
        # Get conflict details
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow("""
                SELECT * FROM Conflicts WHERE conflict_id = $1
            """, conflict_id)
            
            # Get NPC details for context
            npc_details = []
            for npc_id in participating_npcs[:3]:
                npc = await conn.fetchrow("""
                    SELECT name, personality_traits FROM NPCs WHERE npc_id = $1
                """, npc_id)
                if npc:
                    npc_details.append(f"{npc['name']} ({npc.get('personality_traits', 'unknown')})")
        
        # Generate manifestation with LLM
        prompt = f"""
        Generate how this conflict manifests during {activity_type}:
        
        Conflict Type: {conflict['conflict_type']}
        Intensity: {conflict.get('intensity', 'tension')}
        Phase: {conflict.get('phase', 'active')}
        NPCs Present: {', '.join(npc_details)}
        
        Create:
        1. A specific manifestation (1-2 sentences)
        2. Whether player gets an explicit choice (true/false)
        3. Impact on conflict progression (0.0-1.0)
        4. Brief NPC reactions (one line each)
        
        Keep it subtle and realistic to daily life.
        Format as JSON.
        """
        
        response = await self.conflict_agent.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            # Build NPC reactions dict
            npc_reactions = {}
            if 'npc_reactions' in result:
                for i, npc_id in enumerate(participating_npcs[:len(result['npc_reactions'])]):
                    npc_reactions[npc_id] = result['npc_reactions'][i]
            
            return DailyConflictEvent(
                activity_type=activity_type,
                conflict_manifestation=result.get('manifestation', 'A subtle tension colors the moment'),
                choice_presented=result.get('choice_presented', False),
                accumulation_impact=float(result.get('impact', 0.1)),
                npc_reactions=npc_reactions
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse conflict event: {e}")
            # Return fallback
            return DailyConflictEvent(
                activity_type=activity_type,
                conflict_manifestation="The underlying tension affects the interaction",
                choice_presented=False,
                accumulation_impact=0.1,
                npc_reactions={}
            )

# ===============================================================================
# PATTERN-BASED RESOLUTION WITH LLM
# ===============================================================================

class PatternBasedResolution:
    """Resolves conflicts based on accumulated patterns using LLM"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._resolution_agent = None
    
    @property
    def resolution_agent(self) -> Agent:
        """Lazy-load resolution agent"""
        if self._resolution_agent is None:
            self._resolution_agent = Agent(
                name="Conflict Resolution Analyst",
                instructions="""
                Analyze conflict patterns to determine natural resolutions.
                
                Consider:
                - How small daily choices accumulate
                - Whether patterns are becoming entrenched
                - If resistance is increasing or decreasing
                - Natural resolution through routine establishment
                - The role of exhaustion and decision fatigue
                
                Generate resolutions that feel organic and earned, not dramatic.
                Focus on patterns solidifying or slowly dissolving.
                """,
                model="gpt-5-nano",
            )
        return self._resolution_agent
    
    async def check_resolution_by_pattern(self, conflict_id: int) -> Optional[Dict[str, Any]]:
        """Check if conflict should resolve based on patterns"""
        
        async with get_db_connection_context() as conn:
            # Get conflict history
            conflict = await conn.fetchrow("""
                SELECT * FROM Conflicts WHERE conflict_id = $1
            """, conflict_id)
            
            # Get related memories
            memories = await conn.fetch("""
                SELECT memory_text, emotional_valence, created_at
                FROM enhanced_memories
                WHERE user_id = $1 AND conversation_id = $2
                AND tags @> ARRAY[$3::text]
                ORDER BY created_at DESC
                LIMIT 20
            """, self.user_id, self.conversation_id, f"conflict_{conflict_id}")
        
        if not memories or conflict['progress'] < 50:
            return None
        
        # Use LLM to analyze resolution potential
        prompt = f"""
        Analyze if this conflict should naturally resolve:
        
        Conflict: {conflict['conflict_type']} (Progress: {conflict['progress']}%)
        Phase: {conflict['phase']}
        Recent Pattern:
        {self._format_memory_pattern(memories[:10])}
        
        Determine:
        1. Should it resolve? (yes/no)
        2. Resolution type (gradual_acceptance, subtle_resistance, etc.)
        3. Brief description of how it resolves
        4. New established patterns or routines
        
        Format as JSON. Only resolve if patterns clearly indicate it.
        """
        
        response = await self.resolution_agent.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            if result.get('should_resolve', False):
                return {
                    'resolution_type': ResolutionApproach[result.get('type', 'TIME_EROSION').upper()],
                    'description': result.get('description', 'The conflict fades into routine'),
                    'new_patterns': result.get('new_patterns', []),
                    'final_state': 'resolved'
                }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse resolution: {e}")
        
        return None
    
    def _format_memory_pattern(self, memories: List) -> str:
        """Format memories for LLM analysis"""
        pattern = []
        for m in memories:
            pattern.append(f"- {m['memory_text']}")
        return "\n".join(pattern)

# ===============================================================================
# CONFLICT DAILY INTEGRATION WITH LLM
# ===============================================================================

class ConflictDailyIntegration:
    """Integrates conflicts with daily routines using LLM"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._integration_agent = None
    
    @property
    def integration_agent(self) -> Agent:
        """Lazy-load integration agent"""
        if self._integration_agent is None:
            self._integration_agent = Agent(
                name="Daily Conflict Integrator",
                instructions="""
                Weave conflicts naturally into daily activities.
                
                Make conflicts feel like:
                - Natural extensions of routine interactions
                - Subtle undercurrents rather than focal points
                - Realistic relationship dynamics
                - Accumulations of small moments
                
                Generate specific details that make scenes feel lived-in and real.
                """,
                model="gpt-5-nano",
            )
        return self._integration_agent
    
    async def get_conflicts_for_time_of_day(self, time_of_day: str) -> List[Dict]:
        """Get conflicts appropriate for current time"""
        
        async with get_db_connection_context() as conn:
            conflicts = await conn.fetch("""
                SELECT c.*, array_agg(cs.npc_id) as stakeholder_npcs
                FROM Conflicts c
                LEFT JOIN conflict_stakeholders cs ON c.conflict_id = cs.conflict_id
                WHERE c.user_id = $1 AND c.conversation_id = $2
                AND c.is_active = true
                GROUP BY c.conflict_id
            """, self.user_id, self.conversation_id)
        
        # Filter conflicts appropriate for time
        appropriate = []
        for conflict in conflicts:
            if await self._is_appropriate_for_time(conflict, time_of_day):
                appropriate.append(dict(conflict))
        
        return appropriate
    
    async def _is_appropriate_for_time(self, conflict: Dict, time_of_day: str) -> bool:
        """Determine if conflict fits the time of day"""
        
        # Use LLM for intelligent filtering
        prompt = f"""
        Is this conflict appropriate for {time_of_day}?
        
        Conflict Type: {conflict['conflict_type']}
        Intensity: {conflict.get('intensity', 'tension')}
        
        Consider typical daily patterns and when such tensions would naturally surface.
        Answer: yes or no
        """
        
        response = await self.integration_agent.run(prompt)
        return 'yes' in response.content.lower()
    
    async def weave_conflict_into_routine(
        self,
        activity_type: str,
        conflict_ids: List[int]
    ) -> Dict[str, Any]:
        """Weave multiple conflicts into routine activity"""
        
        if not conflict_ids:
            return {'conflicts_woven': False}
        
        # Get conflict details
        async with get_db_connection_context() as conn:
            conflicts = await conn.fetch("""
                SELECT * FROM Conflicts 
                WHERE conflict_id = ANY($1::int[])
            """, conflict_ids)
        
        # Generate integrated narrative
        prompt = f"""
        Weave these conflicts into {activity_type}:
        
        Active Conflicts:
        {self._format_conflicts_for_prompt(conflicts)}
        
        Create:
        1. How conflicts subtly affect the activity
        2. Any micro-aggressions or subtle power plays
        3. Player agency level (high/medium/low)
        4. Suggested NPC behaviors
        
        Keep everything subtle and slice-of-life.
        Format as JSON.
        """
        
        response = await self.integration_agent.run(prompt)
        
        try:
            result = json.loads(response.content)
            return {
                'conflicts_woven': True,
                'activity_modified': result.get('activity_effect', ''),
                'micro_aggressions': result.get('micro_aggressions', []),
                'player_agency': result.get('player_agency', 'medium'),
                'npc_behaviors': result.get('npc_behaviors', {})
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to weave conflicts: {e}")
            return {'conflicts_woven': False}
    
    def _format_conflicts_for_prompt(self, conflicts: List) -> str:
        """Format conflicts for LLM prompt"""
        formatted = []
        for c in conflicts[:3]:  # Limit to avoid token overflow
            formatted.append(f"- {c['conflict_type']} ({c.get('intensity', 'tension')})")
        return "\n".join(formatted)

# ===============================================================================
# INTEGRATION HOOKS (Preserved with LLM enhancement)
# ===============================================================================

@function_tool
async def generate_slice_of_life_conflict(
    ctx: RunContextWrapper,
    current_activity: str,
    present_npcs: List[int],
    recent_patterns: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """Generate a conflict appropriate for current slice-of-life context"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    detector = EmergentConflictDetector(user_id, conversation_id)
    tensions = await detector.detect_brewing_tensions()
    
    if not tensions:
        return None
    
    selected_tension = tensions[0]
    
    async with get_db_connection_context() as conn:
        conflict_id = await conn.fetchval("""
            INSERT INTO Conflicts
            (user_id, conversation_id, conflict_type, conflict_name, 
             description, intensity, phase, is_active, progress)
            VALUES ($1, $2, $3, $4, $5, $6, 'emerging', true, 0)
            RETURNING conflict_id
        """, user_id, conversation_id, 
        selected_tension['type'].value,
        f"{current_activity} tension",
        selected_tension['description'],
        selected_tension['intensity'].value)
        
        for npc_id in present_npcs:
            await conn.execute("""
                INSERT INTO conflict_stakeholders
                (conflict_id, npc_id, faction, involvement_level)
                VALUES ($1, $2, 'neutral', 'participant')
            """, conflict_id, npc_id)
    
    return {
        'conflict_id': conflict_id,
        'type': selected_tension['type'].value,
        'intensity': selected_tension['intensity'].value,
        'embedded_in_activity': current_activity,
        'participating_npcs': present_npcs
    }

@function_tool
async def process_daily_conflict_progression(
    ctx: RunContextWrapper,
    conflict_id: int,
    activity_type: str,
    player_choice: Optional[str] = None
) -> Dict[str, Any]:
    """Progress a conflict through daily activity"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = SliceOfLifeConflictManager(user_id, conversation_id)
    
    async with get_db_connection_context() as conn:
        stakeholders = await conn.fetch("""
            SELECT npc_id FROM conflict_stakeholders
            WHERE conflict_id = $1
        """, conflict_id)
    
    npc_ids = [s['npc_id'] for s in stakeholders]
    
    event = await manager.embed_conflict_in_activity(conflict_id, activity_type, npc_ids)
    
    resolver = PatternBasedResolution(user_id, conversation_id)
    resolution = await resolver.check_resolution_by_pattern(conflict_id)
    
    return {
        'event': {
            'manifestation': event.conflict_manifestation,
            'choice_available': event.choice_presented,
            'npc_reactions': event.npc_reactions
        },
        'progress_impact': event.accumulation_impact,
        'resolution': resolution
    }

@function_tool  
async def analyze_conflict_subtext(
    ctx: RunContextWrapper,
    scene_description: str,
    participating_npcs: List[int]
) -> Dict[str, Any]:
    """Analyze a scene for hidden conflict dynamics"""
    
    agent = Agent(
        name="Subtext Analyzer",
        instructions="""
        Analyze scenes for hidden conflicts and power dynamics.
        Look for:
        - Unspoken tensions
        - Power plays disguised as care
        - Boundary testing
        - Submission patterns
        - Control through routine
        
        Return insights about what's really happening beneath the surface.
        """,
        model="gpt-5-nano",
    )
    
    prompt = f"""
    Scene: {scene_description}
    NPCs present: {participating_npcs}
    
    Analyze the subtext and hidden dynamics.
    Return as JSON with: hidden_tensions, control_mechanisms, 
    escalation_potential (0-1), and primary_dynamic.
    """
    
    response = await agent.run(prompt)
    
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {
            'hidden_tensions': [],
            'control_mechanisms': [],
            'escalation_potential': 0.5,
            'primary_dynamic': 'unclear'
        }
