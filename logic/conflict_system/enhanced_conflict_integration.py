# logic/conflict_system/enhanced_conflict_integration.py
"""
Integration module connecting the slice-of-life conflict system with all game systems
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from agents import function_tool, RunContextWrapper
from db.connection import get_db_connection_context

# Import core conflict systems
from logic.conflict_system.slice_of_life_conflicts import (
    SliceOfLifeConflictType,
    ConflictIntensity,
    EmergentConflictDetector,
    SliceOfLifeConflictManager,
    PatternBasedResolution,
    ConflictDailyIntegration
)

# Import other game systems
from logic.time_cycle import get_current_time_model, ActivityType
from logic.dynamic_relationships import OptimizedRelationshipManager
from npcs.npc_handler import NPCHandler
from story_agent.world_director_agent import WorldDirector, WorldMood
from lore.matriarchal_lore_system import MatriarchalLoreSystem
from logic.narrative_events import check_for_narrative_moments
from context.memory_manager import get_memory_manager

logger = logging.getLogger(__name__)

# ===============================================================================
# MASTER INTEGRATION CLASS
# ===============================================================================

class EnhancedConflictSystemIntegration:
    """
    Master integration class that connects conflicts with all game systems
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Core conflict components
        self.detector = EmergentConflictDetector(user_id, conversation_id)
        self.manager = SliceOfLifeConflictManager(user_id, conversation_id)
        self.resolver = PatternBasedResolution(user_id, conversation_id)
        self.daily_integration = ConflictDailyIntegration(user_id, conversation_id)
        
        # Other system connections (lazy loaded)
        self._relationship_manager = None
        self._npc_handler = None
        self._world_director = None
        self._lore_system = None
        self._memory_manager = None
        
    # ========== System Connections ==========
    
    @property
    def relationship_manager(self):
        if self._relationship_manager is None:
            self._relationship_manager = OptimizedRelationshipManager(
                self.user_id, self.conversation_id
            )
        return self._relationship_manager
    
    @property
    def npc_handler(self):
        if self._npc_handler is None:
            self._npc_handler = NPCHandler(self.user_id, self.conversation_id)
        return self._npc_handler
    
    @property
    def world_director(self):
        if self._world_director is None:
            self._world_director = WorldDirector(self.user_id, self.conversation_id)
        return self._world_director
    
    @property
    def lore_system(self):
        if self._lore_system is None:
            self._lore_system = MatriarchalLoreSystem(self.user_id, self.conversation_id)
        return self._lore_system
    
    @property
    def memory_manager(self):
        if self._memory_manager is None:
            self._memory_manager = get_memory_manager(self.user_id, self.conversation_id)
        return self._memory_manager
    
    # ========== Conflict Generation from World State ==========
    
    async def generate_contextual_conflict(self, ctx: RunContextWrapper) -> Optional[Dict[str, Any]]:
        """
        Generate a conflict based on current world state and context
        """
        
        # Get current world state
        world_state = await self.world_director.get_world_state()
        current_time = await get_current_time_model(ctx)
        
        # Get active NPCs at current location
        player_location = await self._get_player_location()
        present_npcs = await self.npc_handler.get_npcs_at_location(player_location)
        
        # Check relationship dynamics for tension sources
        relationship_tensions = await self._analyze_relationship_tensions(present_npcs)
        
        # Check NPC narrative stages for potential conflicts
        npc_tensions = await self._analyze_npc_progression_tensions(present_npcs)
        
        # Check lore-based tensions (matriarchal society dynamics)
        lore_tensions = await self._analyze_lore_tensions(present_npcs)
        
        # Combine and prioritize tension sources
        all_tensions = {
            'relationship': relationship_tensions,
            'npc_progression': npc_tensions,
            'lore': lore_tensions,
            'emergent': await self.detector.detect_brewing_tensions()
        }
        
        # Select most appropriate tension for current context
        selected_tension = self._select_tension_for_context(
            all_tensions, 
            world_state,
            current_time
        )
        
        if not selected_tension:
            return None
        
        # Generate the conflict
        return await self._create_conflict_from_tension(
            selected_tension,
            present_npcs,
            world_state
        )
    
    async def _analyze_relationship_tensions(self, npcs: List[int]) -> List[Dict]:
        """Analyze relationship dynamics for potential conflicts"""
        
        tensions = []
        
        for npc_id in npcs:
            # Get relationship with player
            relationship = await self.relationship_manager.get_relationship(
                'npc', npc_id, 'player', self.user_id
            )
            
            if not relationship:
                continue
            
            dimensions = relationship.get('dimensions', {})
            
            # Check for tension patterns
            if dimensions.get('dominance', 0) > 0.6 and dimensions.get('resistance', 0) > 0.3:
                tensions.append({
                    'type': SliceOfLifeConflictType.BOUNDARY_EROSION,
                    'source': 'relationship',
                    'npc_id': npc_id,
                    'tension_level': dimensions['dominance'] * dimensions['resistance']
                })
            
            if dimensions.get('dependency', 0) > 0.5 and dimensions.get('autonomy', 0) > 0.4:
                tensions.append({
                    'type': SliceOfLifeConflictType.INDEPENDENCE_STRUGGLE,
                    'source': 'relationship',
                    'npc_id': npc_id,
                    'tension_level': dimensions['dependency'] * dimensions['autonomy']
                })
        
        return tensions
    
    async def _analyze_npc_progression_tensions(self, npcs: List[int]) -> List[Dict]:
        """Analyze NPC narrative progression for conflicts"""
        
        tensions = []
        
        async with get_db_connection_context() as conn:
            for npc_id in npcs:
                progression = await conn.fetchrow("""
                    SELECT * FROM NPCNarrativeProgression
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                """, npc_id, self.user_id, self.conversation_id)
                
                if not progression:
                    continue
                
                # Check for mask slippage opportunities
                if progression['narrative_stage'] in ['Growing_Intimacy', 'Mask_Slips']:
                    npc_stats = await conn.fetchrow("""
                        SELECT mask_integrity FROM NPCStats WHERE npc_id = $1
                    """, npc_id)
                    
                    if npc_stats and npc_stats['mask_integrity'] < 70:
                        tensions.append({
                            'type': SliceOfLifeConflictType.MASK_SLIPPAGE,
                            'source': 'npc_progression',
                            'npc_id': npc_id,
                            'tension_level': (100 - npc_stats['mask_integrity']) / 100
                        })
        
        return tensions
    
    async def _analyze_lore_tensions(self, npcs: List[int]) -> List[Dict]:
        """Analyze lore-based tensions (matriarchal society dynamics)"""
        
        tensions = []
        
        # Get current lore state
        lore_context = await self.lore_system.get_current_cultural_tensions()
        
        # Check for social hierarchy conflicts
        if lore_context.get('social_mobility_tension', 0) > 0.5:
            tensions.append({
                'type': SliceOfLifeConflictType.SOCIAL_PECKING_ORDER,
                'source': 'lore',
                'context': 'matriarchal_hierarchy',
                'tension_level': lore_context['social_mobility_tension']
            })
        
        # Check for role expectation conflicts
        if lore_context.get('gender_role_tension', 0) > 0.4:
            tensions.append({
                'type': SliceOfLifeConflictType.ROLE_EXPECTATIONS,
                'source': 'lore',
                'context': 'societal_expectations',
                'tension_level': lore_context['gender_role_tension']
            })
        
        return tensions
    
    def _select_tension_for_context(
        self, 
        all_tensions: Dict[str, List],
        world_state: Any,
        current_time: Any
    ) -> Optional[Dict]:
        """Select the most appropriate tension for current context"""
        
        # Flatten all tensions
        flat_tensions = []
        for source, tensions in all_tensions.items():
            for tension in tensions:
                tension['source_category'] = source
                flat_tensions.append(tension)
        
        if not flat_tensions:
            return None
        
        # Score tensions based on context
        for tension in flat_tensions:
            score = tension.get('tension_level', 0.5)
            
            # Boost score based on world mood
            if hasattr(world_state, 'world_mood'):
                mood = world_state.world_mood
                if mood == WorldMood.TENSE and tension['type'] in [
                    SliceOfLifeConflictType.BOUNDARY_EROSION,
                    SliceOfLifeConflictType.PERMISSION_PATTERNS
                ]:
                    score *= 1.5
                elif mood == WorldMood.INTIMATE and tension['type'] in [
                    SliceOfLifeConflictType.CARE_DEPENDENCY,
                    SliceOfLifeConflictType.PREFERENCE_SUBMISSION
                ]:
                    score *= 1.5
            
            # Boost score based on time of day
            if current_time.time_of_day == "Morning" and tension['type'] in [
                SliceOfLifeConflictType.ROUTINE_DOMINANCE,
                SliceOfLifeConflictType.DOMESTIC_HIERARCHY
            ]:
                score *= 1.3
            
            tension['context_score'] = score
        
        # Sort by score and return highest
        flat_tensions.sort(key=lambda x: x.get('context_score', 0), reverse=True)
        return flat_tensions[0]
    
    async def _create_conflict_from_tension(
        self,
        tension: Dict,
        present_npcs: List[int],
        world_state: Any
    ) -> Dict[str, Any]:
        """Create a conflict from selected tension"""
        
        async with get_db_connection_context() as conn:
            # Determine intensity based on world state
            intensity = ConflictIntensity.SUBTEXT
            if hasattr(world_state, 'tension_level'):
                if world_state.tension_level > 0.7:
                    intensity = ConflictIntensity.TENSION
                elif world_state.tension_level > 0.9:
                    intensity = ConflictIntensity.PASSIVE
            
            # Create conflict record
            conflict_id = await conn.fetchval("""
                INSERT INTO Conflicts
                (user_id, conversation_id, conflict_type, conflict_name,
                 description, intensity, phase, is_active, progress)
                VALUES ($1, $2, $3, $4, $5, $6, 'emerging', true, 0)
                RETURNING conflict_id
            """, self.user_id, self.conversation_id,
            tension['type'].value,
            f"{tension['type'].value.replace('_', ' ').title()}",
            f"Tension arising from {tension['source_category']}",
            intensity.value)
            
            # Add stakeholders
            stakeholder_npcs = []
            if 'npc_id' in tension:
                stakeholder_npcs = [tension['npc_id']]
            else:
                stakeholder_npcs = present_npcs[:2]  # Limit to 2 primary NPCs
            
            for npc_id in stakeholder_npcs:
                await conn.execute("""
                    INSERT INTO conflict_stakeholders
                    (conflict_id, npc_id, faction, involvement_level)
                    VALUES ($1, $2, 'neutral', 'primary')
                """, conflict_id, npc_id)
            
            # Create initial memory
            await self.memory_manager.create_memory(
                entity_type='conflict',
                entity_id=conflict_id,
                memory_text=f"A subtle tension emerges: {tension['type'].value}",
                importance='medium',
                tags=['conflict', 'emerging', tension['source_category']]
            )
        
        return {
            'conflict_id': conflict_id,
            'type': tension['type'].value,
            'intensity': intensity.value,
            'source': tension['source_category'],
            'stakeholders': stakeholder_npcs,
            'initial_tension_level': tension.get('tension_level', 0.5)
        }
    
    # ========== Daily Activity Integration ==========
    
    async def integrate_conflicts_with_daily_routine(
        self,
        ctx: RunContextWrapper,
        activity_type: str,
        activity_description: str
    ) -> Dict[str, Any]:
        """
        Integrate active conflicts into daily activities
        """
        
        # Get current time and location
        current_time = await get_current_time_model(ctx)
        player_location = await self._get_player_location()
        
        # Get conflicts appropriate for this time/activity
        appropriate_conflicts = await self.daily_integration.get_conflicts_for_time_of_day(
            current_time.time_of_day
        )
        
        if not appropriate_conflicts:
            return {
                'conflicts_active': False,
                'activity_proceeds_normally': True
            }
        
        # Get NPCs present for this activity
        present_npcs = await self.npc_handler.get_npcs_at_location(player_location)
        
        # Filter conflicts to those with stakeholders present
        active_conflicts = []
        for conflict in appropriate_conflicts:
            stakeholder_npcs = conflict.get('stakeholder_npcs', [])
            if any(npc in present_npcs for npc in stakeholder_npcs):
                active_conflicts.append(conflict)
        
        if not active_conflicts:
            return {
                'conflicts_active': False,
                'activity_proceeds_normally': True
            }
        
        # Weave conflicts into the activity
        conflict_integration = await self.daily_integration.weave_conflict_into_routine(
            activity_type,
            [c['conflict_id'] for c in active_conflicts]
        )
        
        # Process NPC reactions based on their personalities and states
        npc_reactions = await self._generate_npc_reactions(
            conflict_integration.get('npc_behaviors', {}),
            activity_type
        )
        
        # Check for narrative moments triggered by conflict
        narrative_moments = await check_for_narrative_moments(
            self.user_id, self.conversation_id
        )
        
        # Update relationship dynamics based on conflict progression
        if conflict_integration.get('conflicts_present'):
            await self._update_relationship_dynamics(
                conflict_integration['primary_conflict'],
                conflict_integration.get('impact', 0)
            )
        
        return {
            'conflicts_active': True,
            'primary_conflict': conflict_integration.get('primary_conflict'),
            'manifestation': conflict_integration.get('manifestation'),
            'player_choices': self._generate_player_choices(conflict_integration),
            'npc_reactions': npc_reactions,
            'narrative_moments': narrative_moments,
            'relationship_impact': conflict_integration.get('impact', 0)
        }
    
    async def _generate_npc_reactions(
        self, 
        npc_behaviors: Dict[int, str],
        activity_type: str
    ) -> Dict[int, Dict]:
        """Generate detailed NPC reactions based on personalities"""
        
        reactions = {}
        
        for npc_id, behavior in npc_behaviors.items():
            npc_data = await self.npc_handler.get_npc_details(npc_id)
            
            if not npc_data:
                continue
            
            # Consider NPC personality and current state
            personality = npc_data.get('personality_traits', {})
            dominance = npc_data.get('dominance', 50)
            current_mood = npc_data.get('current_mood', 'neutral')
            
            reaction = {
                'behavior': behavior,
                'subtle_cues': self._generate_subtle_cues(
                    dominance, personality, current_mood
                ),
                'affects_player_agency': dominance > 60 and activity_type in [
                    'domestic', 'routine', 'care'
                ]
            }
            
            reactions[npc_id] = reaction
        
        return reactions
    
    def _generate_subtle_cues(
        self,
        dominance: int,
        personality: Dict,
        mood: str
    ) -> List[str]:
        """Generate subtle behavioral cues based on NPC state"""
        
        cues = []
        
        if dominance > 70:
            cues.append("Their presence fills the space naturally")
        elif dominance > 50:
            cues.append("A quiet expectation in their manner")
        
        if personality.get('manipulative', 0) > 0.5:
            cues.append("Something calculating behind the warmth")
        
        if mood == 'playful':
            cues.append("An amused glint suggesting hidden thoughts")
        elif mood == 'stern':
            cues.append("The weight of unspoken expectations")
        
        return cues
    
    def _generate_player_choices(self, conflict_integration: Dict) -> List[Dict]:
        """Generate contextual player choices"""
        
        choices = []
        
        if conflict_integration.get('player_agency'):
            choices.extend([
                {
                    'id': 'comply',
                    'text': 'Go along with the established pattern',
                    'subtext': 'Reinforces existing dynamics',
                    'impact': {'submission': 0.05, 'resistance': -0.05}
                },
                {
                    'id': 'negotiate',
                    'text': 'Suggest a small modification',
                    'subtext': 'Tests boundaries gently',
                    'impact': {'autonomy': 0.03, 'trust': 0.02}
                },
                {
                    'id': 'resist',
                    'text': 'Quietly maintain your preference',
                    'subtext': 'Subtle assertion of independence',
                    'impact': {'resistance': 0.05, 'tension': 0.03}
                }
            ])
        
        return choices
    
    async def _update_relationship_dynamics(
        self,
        conflict_id: int,
        impact: float
    ):
        """Update relationship dimensions based on conflict progression"""
        
        async with get_db_connection_context() as conn:
            # Get conflict stakeholders
            stakeholders = await conn.fetch("""
                SELECT npc_id FROM conflict_stakeholders
                WHERE conflict_id = $1
            """, conflict_id)
            
            for stakeholder in stakeholders:
                npc_id = stakeholder['npc_id']
                
                # Small incremental changes
                dimension_changes = {
                    'tension': impact * 0.1,
                    'pattern_strength': impact * 0.05
                }
                
                await self.relationship_manager.update_dimensions(
                    'npc', npc_id,
                    'player', self.user_id,
                    dimension_changes
                )
    
    # ========== Pattern Detection and Resolution ==========
    
    async def check_for_pattern_resolutions(self) -> List[Dict[str, Any]]:
        """Check all active conflicts for pattern-based resolution"""
        
        resolutions = []
        
        async with get_db_connection_context() as conn:
            active_conflicts = await conn.fetch("""
                SELECT conflict_id FROM Conflicts
                WHERE user_id = $1 
                AND conversation_id = $2
                AND is_active = true
                AND progress >= 50
            """, self.user_id, self.conversation_id)
        
        for conflict in active_conflicts:
            resolution = await self.resolver.check_resolution_by_pattern(
                conflict['conflict_id']
            )
            
            if resolution:
                resolutions.append({
                    'conflict_id': conflict['conflict_id'],
                    'resolution': resolution
                })
                
                # Create narrative moment for resolution
                await self._create_resolution_narrative(
                    conflict['conflict_id'],
                    resolution
                )
        
        return resolutions
    
    async def _create_resolution_narrative(
        self,
        conflict_id: int,
        resolution: Dict
    ):
        """Create a narrative moment for conflict resolution"""
        
        narrative_text = f"A pattern has settled: {resolution['description']}"
        
        # Add to memories
        await self.memory_manager.create_memory(
            entity_type='narrative',
            entity_id=conflict_id,
            memory_text=narrative_text,
            importance='high',
            tags=['resolution', 'pattern', 'conflict_end']
        )
        
        # Update world state if significant
        if resolution.get('new_routines_established'):
            await self.world_director.register_routine_change(
                f"Conflict {conflict_id} established new patterns"
            )
    
    # ========== Helper Methods ==========
    
    async def _get_player_location(self) -> str:
        """Get current player location"""
        
        async with get_db_connection_context() as conn:
            location = await conn.fetchval("""
                SELECT current_location FROM player_state
                WHERE user_id = $1 AND conversation_id = $2
            """, self.user_id, self.conversation_id)
        
        return location or "Home"


# ===============================================================================
# PUBLIC API FUNCTIONS
# ===============================================================================

@function_tool
async def process_conflict_in_scene(
    ctx: RunContextWrapper,
    scene_type: str,
    activity: str,
    present_npcs: List[int]
) -> Dict[str, Any]:
    """
    Main function to process conflicts within a scene
    
    This is the primary integration point for the game loop
    """
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    integration = EnhancedConflictSystemIntegration(user_id, conversation_id)
    
    # Check for new conflict generation
    if random.random() < 0.15:  # 15% chance of new conflict
        new_conflict = await integration.generate_contextual_conflict(ctx)
        if new_conflict:
            logger.info(f"Generated new conflict: {new_conflict['type']}")
    
    # Integrate existing conflicts with current activity
    conflict_integration = await integration.integrate_conflicts_with_daily_routine(
        ctx, activity, f"Player is {activity}"
    )
    
    # Check for resolutions
    resolutions = await integration.check_for_pattern_resolutions()
    
    return {
        'conflicts_active': conflict_integration.get('conflicts_active', False),
        'manifestation': conflict_integration.get('manifestation'),
        'player_choices': conflict_integration.get('player_choices', []),
        'npc_reactions': conflict_integration.get('npc_reactions', {}),
        'resolutions': resolutions,
        'narrative_moments': conflict_integration.get('narrative_moments', [])
    }


@function_tool
async def analyze_scene_for_conflict_potential(
    ctx: RunContextWrapper,
    scene_description: str,
    npcs_present: List[int],
    recent_events: List[str]
) -> Dict[str, Any]:
    """
    Analyze a scene for potential conflict generation
    """
    
    from logic.conflict_system.slice_of_life_conflicts import analyze_conflict_subtext
    
    # Get subtext analysis
    subtext = await analyze_conflict_subtext(ctx, scene_description, npcs_present)
    
    # Determine if conditions are right for conflict
    tension_score = 0.0
    
    if subtext.get('hidden_tensions'):
        tension_score += len(subtext['hidden_tensions']) * 0.1
    
    if subtext.get('control_mechanisms'):
        tension_score += len(subtext['control_mechanisms']) * 0.15
    
    if subtext.get('escalation_potential', 0) > 0.5:
        tension_score += 0.2
    
    # Check recent events for patterns
    pattern_keywords = ['permission', 'decide', 'allow', 'should', 'must']
    pattern_count = sum(
        1 for event in recent_events 
        if any(kw in event.lower() for kw in pattern_keywords)
    )
    tension_score += pattern_count * 0.05
    
    should_generate = tension_score > 0.3
    
    return {
        'tension_score': tension_score,
        'should_generate_conflict': should_generate,
        'primary_dynamic': subtext.get('primary_dynamic'),
        'potential_conflict_types': _suggest_conflict_types(subtext),
        'subtext_analysis': subtext
    }


def _suggest_conflict_types(subtext: Dict) -> List[str]:
    """Suggest appropriate conflict types based on subtext"""
    
    suggestions = []
    
    if 'permission' in str(subtext.get('control_mechanisms', [])).lower():
        suggestions.append(SliceOfLifeConflictType.PERMISSION_PATTERNS.value)
    
    if 'routine' in str(subtext.get('primary_dynamic', '')).lower():
        suggestions.append(SliceOfLifeConflictType.ROUTINE_DOMINANCE.value)
    
    if 'boundary' in str(subtext.get('hidden_tensions', [])).lower():
        suggestions.append(SliceOfLifeConflictType.BOUNDARY_EROSION.value)
    
    return suggestions


# ===============================================================================
# MIGRATION HELPERS
# ===============================================================================

async def migrate_existing_conflicts(user_id: int, conversation_id: int):
    """
    Helper to migrate existing dramatic conflicts to slice-of-life system
    """
    
    async with get_db_connection_context() as conn:
        # Get existing conflicts
        old_conflicts = await conn.fetch("""
            SELECT * FROM Conflicts
            WHERE user_id = $1 AND conversation_id = $2
            AND is_active = true
        """, user_id, conversation_id)
        
        for conflict in old_conflicts:
            # Map old types to new slice-of-life types
            mapping = {
                'faction_rivalry': SliceOfLifeConflictType.SOCIAL_PECKING_ORDER,
                'personal_dispute': SliceOfLifeConflictType.FRIENDSHIP_BOUNDARIES,
                'succession_crisis': SliceOfLifeConflictType.ROLE_EXPECTATIONS,
                'economic_collapse': SliceOfLifeConflictType.FINANCIAL_CONTROL
            }
            
            new_type = mapping.get(
                conflict['conflict_type'],
                SliceOfLifeConflictType.SUBTLE_RIVALRY
            )
            
            # Update to new system
            await conn.execute("""
                UPDATE Conflicts
                SET 
                    conflict_type = $1,
                    intensity = $2,
                    phase = 'integrated'
                WHERE conflict_id = $3
            """, new_type.value, ConflictIntensity.TENSION.value, conflict['conflict_id'])
    
    logger.info(f"Migrated {len(old_conflicts)} conflicts to slice-of-life system")
