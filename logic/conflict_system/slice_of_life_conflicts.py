# logic/conflict_system/slice_of_life_conflicts.py
"""
Enhanced Conflict System for Open-World Slice-of-Life RPG
Focuses on subtle tensions emerging from daily routines and power dynamics
"""

import logging
import json
import random
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

from agents import Agent, function_tool, ModelSettings, RunContextWrapper, Runner
from db.connection import get_db_connection_context
from lore.core import canon

logger = logging.getLogger(__name__)

# ===============================================================================
# SLICE-OF-LIFE CONFLICT TYPES
# ===============================================================================

class SliceOfLifeConflictType(Enum):
    """Conflicts that emerge from daily life rather than dramatic events"""
    
    # Domestic Control Conflicts
    ROUTINE_DOMINANCE = "routine_dominance"  # Who controls daily schedules
    DOMESTIC_HIERARCHY = "domestic_hierarchy"  # Who makes household decisions
    CARE_DEPENDENCY = "care_dependency"  # Control through caretaking
    
    # Social Dynamic Conflicts
    SOCIAL_PECKING_ORDER = "social_pecking_order"  # Workplace/social hierarchies
    FRIENDSHIP_BOUNDARIES = "friendship_boundaries"  # Testing relationship limits
    SUBTLE_RIVALRY = "subtle_rivalry"  # Unspoken competition
    
    # Resource & Permission Conflicts  
    FINANCIAL_CONTROL = "financial_control"  # Who manages money
    PERMISSION_PATTERNS = "permission_patterns"  # Asking vs deciding
    PREFERENCE_SUBMISSION = "preference_submission"  # Whose tastes matter
    
    # Identity & Role Conflicts
    MASK_SLIPPAGE = "mask_slippage"  # NPCs revealing true selves
    ROLE_EXPECTATIONS = "role_expectations"  # Pressure to conform
    INDEPENDENCE_STRUGGLE = "independence_struggle"  # Maintaining autonomy
    
    # Emergent Pattern Conflicts
    HABIT_ENFORCEMENT = "habit_enforcement"  # Establishing routines
    BOUNDARY_EROSION = "boundary_erosion"  # Gradual limit pushing
    CONDITIONING_RESISTANCE = "conditioning_resistance"  # Fighting patterns


class ConflictIntensity(Enum):
    """How overtly the conflict manifests"""
    SUBTEXT = "subtext"  # Completely hidden in normal interaction
    TENSION = "tension"  # Noticeable but unspoken
    PASSIVE = "passive"  # Passive-aggressive behaviors
    DIRECT = "direct"  # Open but civil disagreement
    CONFRONTATION = "confrontation"  # Actual argument (rare)


class ResolutionApproach(Enum):
    """How conflicts resolve in slice-of-life"""
    GRADUAL_ACCEPTANCE = "gradual_acceptance"  # Slowly giving in
    SUBTLE_RESISTANCE = "subtle_resistance"  # Quiet defiance
    NEGOTIATED_COMPROMISE = "negotiated_compromise"  # Finding middle ground
    ESTABLISHED_PATTERN = "established_pattern"  # Falling into routine
    THIRD_PARTY_INFLUENCE = "third_party_influence"  # Others affect outcome
    TIME_EROSION = "time_erosion"  # Conflict fades naturally


@dataclass
class SliceOfLifeStake:
    """What's at stake in mundane conflicts"""
    stake_type: str  # "autonomy", "preference", "schedule", "social_standing", etc.
    description: str
    daily_impact: str  # How it affects routine
    relationship_impact: str  # How it affects dynamics
    accumulation_factor: float  # How much small choices matter (0.0-1.0)


# ===============================================================================
# CONFLICT GENERATION FROM DAILY PATTERNS
# ===============================================================================

class EmergentConflictDetector:
    """Detects conflicts emerging from daily interactions and patterns"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
    async def detect_brewing_tensions(self) -> List[Dict[str, Any]]:
        """Analyze recent interactions for emerging conflicts"""
        
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
            
            # Get NPC schedules and routines
            npc_routines = await conn.fetch("""
                SELECT 
                    npc_id,
                    schedule_type,
                    schedule_data
                FROM npc_schedules
                WHERE user_id = $1 
                AND conversation_id = $2
                AND is_active = true
            """, self.user_id, self.conversation_id)
        
        # Analyze for tension patterns
        brewing_tensions = []
        
        # Check for routine dominance patterns
        routine_conflicts = self._detect_routine_conflicts(memory_patterns, npc_routines)
        brewing_tensions.extend(routine_conflicts)
        
        # Check for permission patterns
        permission_conflicts = self._detect_permission_patterns(memory_patterns)
        brewing_tensions.extend(permission_conflicts)
        
        # Check for relationship boundary testing
        boundary_conflicts = self._detect_boundary_testing(relationships, memory_patterns)
        brewing_tensions.extend(boundary_conflicts)
        
        return brewing_tensions
    
    def _detect_routine_conflicts(self, memories: List, routines: List) -> List[Dict]:
        """Detect conflicts over daily routines and schedules"""
        conflicts = []
        
        # Look for patterns of schedule conflicts
        schedule_mentions = [m for m in memories if any(
            word in m['memory_text'].lower() 
            for word in ['schedule', 'routine', 'time', 'late', 'early', 'when']
        )]
        
        if len(schedule_mentions) >= 3:
            conflicts.append({
                'type': SliceOfLifeConflictType.ROUTINE_DOMINANCE,
                'intensity': ConflictIntensity.SUBTEXT,
                'description': 'Unspoken tension over daily schedules',
                'evidence': schedule_mentions[:3]
            })
        
        return conflicts
    
    def _detect_permission_patterns(self, memories: List) -> List[Dict]:
        """Detect patterns of asking permission vs making decisions"""
        conflicts = []
        
        permission_words = ['can i', 'may i', 'should i', 'is it okay', 'do you mind']
        permission_memories = [
            m for m in memories 
            if any(phrase in m['memory_text'].lower() for phrase in permission_words)
        ]
        
        if len(permission_memories) >= 5:
            conflicts.append({
                'type': SliceOfLifeConflictType.PERMISSION_PATTERNS,
                'intensity': ConflictIntensity.TENSION,
                'description': 'Growing pattern of seeking approval',
                'evidence': permission_memories[:5]
            })
        
        return conflicts
    
    def _detect_boundary_testing(self, relationships: List, memories: List) -> List[Dict]:
        """Detect NPCs testing relationship boundaries"""
        conflicts = []
        
        for rel in relationships:
            if rel['dimension'] == 'resistance' and rel['recent_delta'] < -0.1:
                conflicts.append({
                    'type': SliceOfLifeConflictType.BOUNDARY_EROSION,
                    'intensity': ConflictIntensity.SUBTEXT,
                    'description': 'Gradual pushing of comfort boundaries',
                    'npc_id': rel['entity1_id'] if rel['entity1_id'] != self.user_id else rel['entity2_id']
                })
        
        return conflicts


# ===============================================================================
# CONFLICT PROGRESSION THROUGH DAILY LIFE
# ===============================================================================

@dataclass
class DailyConflictEvent:
    """A conflict moment embedded in daily routine"""
    activity_type: str  # "breakfast", "commute", "work", "shopping", etc.
    conflict_manifestation: str  # How the conflict shows up
    choice_presented: bool  # Whether player gets explicit choice
    accumulation_impact: float  # How much this affects the overall conflict
    npc_reactions: Dict[int, str]  # How NPCs respond


class SliceOfLifeConflictManager:
    """Manages conflicts through daily activities rather than dramatic confrontations"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.conflict_agent = self._create_conflict_agent()
        
    def _create_conflict_agent(self) -> Agent:
        """Create agent for generating slice-of-life conflict content"""
        return Agent(
            name="Slice of Life Conflict Director",
            instructions="""
            Generate subtle conflicts embedded in daily routines for a slice-of-life game.
            
            Focus on:
            - Conflicts that manifest through mundane activities
            - Power dynamics hidden in everyday choices
            - Slow accumulation of control through routine
            - Tension expressed through subtext not confrontation
            
            Avoid:
            - Dramatic confrontations
            - Explicit power struggles
            - Sudden relationship changes
            - Binary win/lose outcomes
            
            Conflicts should feel like natural friction in daily life.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.7)
        )
    
    async def embed_conflict_in_activity(
        self, 
        conflict_id: int,
        activity_type: str,
        participating_npcs: List[int]
    ) -> DailyConflictEvent:
        """Embed a conflict moment in a daily activity"""
        
        async with get_db_connection_context() as conn:
            # Get conflict details
            conflict = await conn.fetchrow("""
                SELECT * FROM Conflicts
                WHERE conflict_id = $1
            """, conflict_id)
            
            # Get NPC personalities and current moods
            npcs = await conn.fetch("""
                SELECT 
                    npc_id,
                    npc_name,
                    dominance,
                    personality_traits,
                    current_mood
                FROM NPCStats
                WHERE npc_id = ANY($1)
            """, participating_npcs)
        
        # Generate conflict manifestation
        prompt = f"""
        Activity: {activity_type}
        Conflict: {conflict['conflict_type']} - {conflict['description']}
        NPCs present: {json.dumps([dict(n) for n in npcs])}
        Intensity: {conflict['intensity']}
        
        Generate a subtle moment where this conflict manifests during {activity_type}.
        Return JSON with:
        - manifestation: How the conflict shows (1-2 sentences)
        - player_choice: Optional choice for player (null if none)
        - npc_behaviors: How each NPC acts (subtle, not dramatic)
        - subtext: What's really happening beneath the surface
        """
        
        response = await Runner.run(self.conflict_agent, prompt)
        result = json.loads(response.output)
        
        # Create the event
        event = DailyConflictEvent(
            activity_type=activity_type,
            conflict_manifestation=result['manifestation'],
            choice_presented=result['player_choice'] is not None,
            accumulation_impact=0.1,  # Small impacts that add up
            npc_reactions={
                npc['npc_id']: result['npc_behaviors'].get(str(npc['npc_id']), '')
                for npc in npcs
            }
        )
        
        # Record the event
        await self._record_conflict_event(conflict_id, event)
        
        return event
    
    async def _record_conflict_event(self, conflict_id: int, event: DailyConflictEvent):
        """Record a conflict event in the database"""
        
        async with get_db_connection_context() as conn:
            # Update conflict progress (small increments)
            await conn.execute("""
                UPDATE Conflicts
                SET 
                    progress = LEAST(progress + $1, 100),
                    last_event_at = CURRENT_TIMESTAMP
                WHERE conflict_id = $2
            """, event.accumulation_impact * 10, conflict_id)
            
            # Create memory of the event
            await conn.execute("""
                INSERT INTO conflict_memories
                (conflict_id, memory_text, significance, memory_type)
                VALUES ($1, $2, $3, 'daily_moment')
            """, conflict_id, event.conflict_manifestation, 3)


# ===============================================================================
# RESOLUTION THROUGH PATTERNS
# ===============================================================================

class PatternBasedResolution:
    """Resolves conflicts through accumulated patterns rather than decisive moments"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
    async def check_resolution_by_pattern(self, conflict_id: int) -> Optional[Dict[str, Any]]:
        """Check if a conflict has resolved through accumulated patterns"""
        
        async with get_db_connection_context() as conn:
            # Get conflict and its events
            conflict = await conn.fetchrow("""
                SELECT * FROM Conflicts
                WHERE conflict_id = $1
            """, conflict_id)
            
            events = await conn.fetch("""
                SELECT * FROM conflict_memories
                WHERE conflict_id = $1
                ORDER BY created_at DESC
                LIMIT 20
            """, conflict_id)
            
            # Count pattern types
            acceptance_count = sum(1 for e in events if 'accept' in e['memory_text'].lower())
            resistance_count = sum(1 for e in events if 'resist' in e['memory_text'].lower())
            compromise_count = sum(1 for e in events if 'compromise' in e['memory_text'].lower())
            
            # Determine if pattern is strong enough for resolution
            total_events = len(events)
            if total_events >= 10:
                if acceptance_count / total_events > 0.6:
                    return await self._resolve_by_acceptance(conflict_id, conflict)
                elif resistance_count / total_events > 0.6:
                    return await self._resolve_by_resistance(conflict_id, conflict)
                elif compromise_count / total_events > 0.4:
                    return await self._resolve_by_compromise(conflict_id, conflict)
                elif conflict['progress'] >= 80:
                    return await self._resolve_by_time(conflict_id, conflict)
        
        return None
    
    async def _resolve_by_acceptance(self, conflict_id: int, conflict: Dict) -> Dict[str, Any]:
        """Resolve conflict through gradual acceptance"""
        
        resolution = {
            'resolution_type': ResolutionApproach.GRADUAL_ACCEPTANCE,
            'description': 'The new pattern has become accepted routine',
            'relationship_changes': {
                'submission': 0.1,
                'resistance': -0.1,
                'dependency': 0.05
            },
            'new_routines_established': True
        }
        
        await self._finalize_resolution(conflict_id, resolution)
        return resolution
    
    async def _resolve_by_resistance(self, conflict_id: int, conflict: Dict) -> Dict[str, Any]:
        """Resolve conflict through sustained resistance"""
        
        resolution = {
            'resolution_type': ResolutionApproach.SUBTLE_RESISTANCE,
            'description': 'Boundaries have been quietly but firmly maintained',
            'relationship_changes': {
                'autonomy': 0.1,
                'resistance': 0.05,
                'respect': 0.1
            },
            'patterns_rejected': True
        }
        
        await self._finalize_resolution(conflict_id, resolution)
        return resolution
    
    async def _resolve_by_compromise(self, conflict_id: int, conflict: Dict) -> Dict[str, Any]:
        """Resolve conflict through finding middle ground"""
        
        resolution = {
            'resolution_type': ResolutionApproach.NEGOTIATED_COMPROMISE,
            'description': 'A comfortable routine has been negotiated',
            'relationship_changes': {
                'trust': 0.1,
                'understanding': 0.1
            },
            'shared_control_established': True
        }
        
        await self._finalize_resolution(conflict_id, resolution)
        return resolution
    
    async def _resolve_by_time(self, conflict_id: int, conflict: Dict) -> Dict[str, Any]:
        """Resolve conflict through time passing"""
        
        resolution = {
            'resolution_type': ResolutionApproach.TIME_EROSION,
            'description': 'The issue has faded into background noise',
            'relationship_changes': {},
            'conflict_dormant': True
        }
        
        await self._finalize_resolution(conflict_id, resolution)
        return resolution
    
    async def _finalize_resolution(self, conflict_id: int, resolution: Dict):
        """Finalize conflict resolution in database"""
        
        async with get_db_connection_context() as conn:
            await conn.execute("""
                UPDATE Conflicts
                SET 
                    is_active = FALSE,
                    progress = 100,
                    phase = 'resolved',
                    outcome = $1,
                    resolution_description = $2,
                    resolved_at = CURRENT_TIMESTAMP
                WHERE conflict_id = $3
            """, resolution['resolution_type'].value, resolution['description'], conflict_id)


# ===============================================================================
# INTEGRATION HOOKS
# ===============================================================================

@function_tool
async def generate_slice_of_life_conflict(
    ctx: RunContextWrapper,
    current_activity: str,
    present_npcs: List[int],
    recent_patterns: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate a conflict appropriate for current slice-of-life context
    
    Args:
        ctx: Context wrapper
        current_activity: What the player is doing
        present_npcs: NPCs currently present
        recent_patterns: Recent behavioral patterns observed
        
    Returns:
        Generated conflict or None if no appropriate tension
    """
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    # Detect emerging tensions
    detector = EmergentConflictDetector(user_id, conversation_id)
    tensions = await detector.detect_brewing_tensions()
    
    if not tensions:
        return None
    
    # Select most appropriate tension for current context
    selected_tension = tensions[0]  # Could be more sophisticated
    
    # Generate the conflict
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
        
        # Add stakeholders
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
    """
    Progress a conflict through daily activity
    
    Args:
        ctx: Context wrapper
        conflict_id: ID of the conflict
        activity_type: Type of daily activity
        player_choice: Optional player choice made
        
    Returns:
        Progression result including any resolution
    """
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = SliceOfLifeConflictManager(user_id, conversation_id)
    
    # Get participating NPCs
    async with get_db_connection_context() as conn:
        stakeholders = await conn.fetch("""
            SELECT npc_id FROM conflict_stakeholders
            WHERE conflict_id = $1
        """, conflict_id)
    
    npc_ids = [s['npc_id'] for s in stakeholders]
    
    # Generate conflict event for this activity
    event = await manager.embed_conflict_in_activity(conflict_id, activity_type, npc_ids)
    
    # Check for pattern-based resolution
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
    """
    Analyze a scene for hidden conflict dynamics
    
    Args:
        ctx: Context wrapper
        scene_description: Description of current scene
        participating_npcs: NPCs in the scene
        
    Returns:
        Analysis of hidden tensions and power dynamics
    """
    
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
        
        Return insights about what's really happening.
        """,
        model="gpt-5-nano",
        model_settings=ModelSettings(temperature=0.6)
    )
    
    prompt = f"""
    Scene: {scene_description}
    NPCs present: {participating_npcs}
    
    Analyze the subtext and hidden dynamics.
    Return JSON with:
    - primary_dynamic: Main power dynamic at play
    - hidden_tensions: List of unspoken conflicts
    - control_mechanisms: How control is being exercised
    - resistance_points: Where pushback might occur
    - escalation_potential: How this might develop
    """
    
    response = await Runner.run(agent, prompt)
    return json.loads(response.output)


# ===============================================================================
# DAILY INTEGRATION
# ===============================================================================

class ConflictDailyIntegration:
    """Integrates conflicts seamlessly into daily life simulation"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
    async def get_conflicts_for_time_of_day(self, time_phase: str) -> List[Dict]:
        """Get conflicts appropriate for current time of day"""
        
        async with get_db_connection_context() as conn:
            conflicts = await conn.fetch("""
                SELECT c.*, 
                       array_agg(cs.npc_id) as stakeholder_npcs
                FROM Conflicts c
                LEFT JOIN conflict_stakeholders cs ON c.conflict_id = cs.conflict_id
                WHERE c.user_id = $1 
                AND c.conversation_id = $2
                AND c.is_active = true
                AND c.intensity != 'confrontation'
                GROUP BY c.conflict_id
            """, self.user_id, self.conversation_id)
        
        # Filter by appropriateness for time
        appropriate = []
        for conflict in conflicts:
            if self._is_appropriate_for_time(conflict, time_phase):
                appropriate.append(dict(conflict))
        
        return appropriate
    
    def _is_appropriate_for_time(self, conflict: Dict, time_phase: str) -> bool:
        """Determine if conflict fits the time of day"""
        
        conflict_type = conflict['conflict_type']
        
        morning_conflicts = [
            'ROUTINE_DOMINANCE', 'DOMESTIC_HIERARCHY', 'PERMISSION_PATTERNS'
        ]
        
        work_conflicts = [
            'SOCIAL_PECKING_ORDER', 'SUBTLE_RIVALRY', 'ROLE_EXPECTATIONS'
        ]
        
        evening_conflicts = [
            'CARE_DEPENDENCY', 'PREFERENCE_SUBMISSION', 'BOUNDARY_EROSION'
        ]
        
        if time_phase in ['morning', 'wake_up'] and conflict_type in morning_conflicts:
            return True
        elif time_phase in ['work_hours', 'afternoon'] and conflict_type in work_conflicts:
            return True
        elif time_phase in ['evening', 'night'] and conflict_type in evening_conflicts:
            return True
        
        # Low-intensity conflicts can happen anytime
        return conflict['intensity'] in ['subtext', 'tension']
    
    async def weave_conflict_into_routine(
        self,
        routine_activity: str,
        active_conflicts: List[int]
    ) -> Dict[str, Any]:
        """Weave active conflicts into routine activities"""
        
        if not active_conflicts:
            return {'conflicts_present': False}
        
        # Select primary conflict for this moment
        primary_conflict = active_conflicts[0]
        
        manager = SliceOfLifeConflictManager(self.user_id, self.conversation_id)
        
        # Get NPCs involved
        async with get_db_connection_context() as conn:
            npcs = await conn.fetch("""
                SELECT DISTINCT cs.npc_id
                FROM conflict_stakeholders cs
                JOIN NPCStats ns ON cs.npc_id = ns.npc_id
                WHERE cs.conflict_id = $1
                AND ns.current_location = (
                    SELECT current_location 
                    FROM player_state 
                    WHERE user_id = $2 
                    AND conversation_id = $3
                )
            """, primary_conflict, self.user_id, self.conversation_id)
        
        if not npcs:
            return {'conflicts_present': False}
        
        npc_ids = [n['npc_id'] for n in npcs]
        
        # Generate conflict moment
        event = await manager.embed_conflict_in_activity(
            primary_conflict, routine_activity, npc_ids
        )
        
        return {
            'conflicts_present': True,
            'primary_conflict': primary_conflict,
            'manifestation': event.conflict_manifestation,
            'player_agency': event.choice_presented,
            'npc_behaviors': event.npc_reactions,
            'impact': event.accumulation_impact
        }
