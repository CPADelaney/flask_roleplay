# logic/conflict_system/conflict_synthesizer.py
"""
Conflict Output Synthesizer Module

This module acts as the coordination layer between all conflict subsystems,
synthesizing their outputs into cohesive, unified information for the game.
It doesn't replace existing modules but orchestrates them intelligently.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

from agents import function_tool, RunContextWrapper
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ===============================================================================
# SYNTHESIS DATA STRUCTURES
# ===============================================================================

@dataclass
class ConflictSnapshot:
    """Unified snapshot of all conflict states at a moment"""
    timestamp: datetime
    active_conflicts: List[Dict]
    power_dynamics: Dict[str, Any]
    tension_map: Dict[str, float]
    resolution_queue: List[Dict]
    narrative_hooks: List[str]
    player_agency_options: List[Dict]
    npc_involvement: Dict[int, List[str]]  # NPC ID -> conflict roles
    world_impact: Dict[str, Any]
    metadata: Dict = field(default_factory=dict)

@dataclass
class SynthesizedOutput:
    """Final synthesized output for consumption by other systems"""
    primary_tension: Optional[str]
    active_dynamics: List[Dict]
    scene_modifiers: Dict[str, Any]
    dialogue_hints: List[str]
    player_choices: List[Dict]
    npc_behaviors: Dict[int, Dict]
    environmental_cues: List[str]
    narrative_text: Optional[str]
    ui_elements: Dict[str, Any]
    priority_score: float

class ConflictPriority(Enum):
    """Priority levels for conflict presentation"""
    CRITICAL = 5  # Must be addressed immediately
    HIGH = 4      # Should influence current scene
    MEDIUM = 3    # Background tension
    LOW = 2       # Subtle undercurrent
    DORMANT = 1   # Not currently relevant

# ===============================================================================
# CONFLICT OUTPUT SYNTHESIZER
# ===============================================================================

class ConflictOutputSynthesizer:
    """
    Orchestrates and synthesizes outputs from all conflict subsystems.
    This is the coordination layer that ensures coherent conflict presentation.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Track which systems have provided input
        self._system_inputs = {}
        self._synthesis_cache = None
        self._last_synthesis_time = None
        
    async def synthesize_conflict_state(
        self,
        scene_context: Dict[str, Any],
        requested_outputs: Optional[List[str]] = None
    ) -> SynthesizedOutput:
        """
        Main synthesis function that coordinates all conflict systems.
        
        Args:
            scene_context: Current scene information
            requested_outputs: Specific outputs needed (e.g., ['dialogue', 'choices'])
            
        Returns:
            Synthesized, cohesive conflict output
        """
        # Gather inputs from all conflict subsystems
        inputs = await self._gather_system_inputs(scene_context)
        
        # Detect and resolve contradictions
        harmonized = await self._harmonize_inputs(inputs)
        
        # Prioritize conflicts for current context
        prioritized = await self._prioritize_conflicts(harmonized, scene_context)
        
        # Generate unified narrative elements
        narrative = await self._generate_narrative_synthesis(prioritized)
        
        # Create player interaction options
        player_options = await self._synthesize_player_options(prioritized)
        
        # Determine NPC behaviors based on all conflicts
        npc_behaviors = await self._synthesize_npc_behaviors(prioritized)
        
        # Generate environmental/scene modifications
        scene_mods = await self._synthesize_scene_modifiers(prioritized)
        
        # Package into final output
        output = SynthesizedOutput(
            primary_tension=self._extract_primary_tension(prioritized),
            active_dynamics=prioritized['active_dynamics'],
            scene_modifiers=scene_mods,
            dialogue_hints=narrative['dialogue_hints'],
            player_choices=player_options,
            npc_behaviors=npc_behaviors,
            environmental_cues=narrative['environmental_cues'],
            narrative_text=narrative.get('text'),
            ui_elements=await self._generate_ui_elements(prioritized),
            priority_score=self._calculate_overall_priority(prioritized)
        )
        
        # Cache the synthesis
        self._synthesis_cache = output
        self._last_synthesis_time = datetime.now()
        
        # Log synthesis event
        await self._log_synthesis(output)
        
        return output
    
    # ========== INPUT GATHERING ==========
    
    async def _gather_system_inputs(self, context: Dict) -> Dict[str, Any]:
        """Gather inputs from all conflict subsystems"""
        inputs = {}
        
        # 1. SLICE-OF-LIFE CONFLICTS
        from logic.conflict_system.slice_of_life_conflicts import (
            SliceOfLifeConflictManager, EmergentConflictDetector, PatternBasedResolution
        )
        
        slice_manager = SliceOfLifeConflictManager(self.user_id, self.conversation_id)
        detector = EmergentConflictDetector(self.user_id, self.conversation_id)
        pattern_resolver = PatternBasedResolution(self.user_id, self.conversation_id)
        
        inputs['slice_of_life'] = {
            'active': await slice_manager.get_active_conflicts(),
            'tensions': await detector.detect_brewing_tensions(),
            'pending_resolutions': []  # Check for pattern-based resolutions
        }
        
        # Check each slice conflict for pattern resolution
        for conflict in inputs['slice_of_life']['active']:
            resolution = await pattern_resolver.check_resolution_by_pattern(conflict.get('conflict_id'))
            if resolution:
                inputs['slice_of_life']['pending_resolutions'].append(resolution)
        
        # 2. ENHANCED CONFLICT INTEGRATION
        from logic.conflict_system.enhanced_conflict_integration import (
            EnhancedConflictSystemIntegration
        )
        
        integration = EnhancedConflictSystemIntegration(self.user_id, self.conversation_id)
        inputs['enhanced_integration'] = {
            'daily_dynamics': await integration.daily_integration.get_current_dynamics(),
            'pattern_resolutions': await integration.check_for_pattern_resolutions()
        }
        
        # 3. MAIN CONFLICT SYSTEM INTEGRATION  
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        
        main_system = ConflictSystemIntegration(self.user_id, self.conversation_id)
        inputs['main_conflicts'] = {
            'active': await main_system.get_active_conflicts(),
            'vitals': await main_system.update_player_vitals(context.get('activity_type', 'idle')),
            'daily_update': await main_system.run_daily_update() if context.get('is_new_day') else None
        }
        
        # Get detailed conflict info for each active conflict
        inputs['main_conflicts']['details'] = {}
        for conflict in inputs['main_conflicts']['active']:
            conflict_id = conflict.get('conflict_id')
            if conflict_id:
                inputs['main_conflicts']['details'][conflict_id] = await main_system.get_conflict_details(conflict_id)
        
        # 4. CONFLICT RESOLUTION SYSTEM
        try:
            from logic.conflict_system.conflict_resolution import ConflictResolutionSystem
            resolution_system = ConflictResolutionSystem(self.user_id, self.conversation_id)
            inputs['resolution_system'] = {
                'pending_resolutions': await resolution_system.get_pending_resolutions(),
                'resolution_history': await resolution_system.get_recent_resolutions()
            }
        except ImportError:
            inputs['resolution_system'] = {'available': False}
        
        # 5. POWER DYNAMICS FROM AGENT INTERACTION
        from story_agent.agent_interaction import (
            check_for_power_exchange, update_relationship_from_exchange
        )
        
        # Check for active power exchanges with present NPCs
        inputs['power_exchanges'] = []
        for npc_id in context.get('npcs_present', []):
            exchange = await check_for_power_exchange(
                self.user_id, 
                self.conversation_id,
                context.get('scene_focus', 'routine'),
                npc_id
            )
            if exchange:
                inputs['power_exchanges'].append(exchange)
        
        # 6. GOVERNANCE-LEVEL CONFLICTS
        try:
            from nyx.governance.conflict import ConflictAnalyzer
            governance = ConflictAnalyzer()
            inputs['governance_conflicts'] = await governance.analyze_conflicts({
                'user_id': self.user_id,
                'conversation_id': self.conversation_id
            })
        except ImportError:
            inputs['governance_conflicts'] = {'available': False}
        
        # 7. POLITICAL/GEOPOLITICAL CONFLICTS
        try:
            from lore.managers.politics import PoliticalManager
            from lore.managers.geopolitical import GeopoliticalManager
            
            political_mgr = PoliticalManager()
            geo_mgr = GeopoliticalManager()
            
            inputs['political_conflicts'] = {
                'active_political': await political_mgr.get_active_conflicts(),
                'geopolitical': await geo_mgr.get_regional_conflicts()
            }
        except ImportError:
            inputs['political_conflicts'] = {'available': False}
        
        # 8. TIME-BASED CONTEXT
        from logic.time_cycle import (
            get_current_time_model, 
            check_for_conflict_events,
            process_conflict_time_advancement
        )
        
        time_model = await get_current_time_model(self.user_id, self.conversation_id)
        inputs['temporal_context'] = {
            'time_of_day': time_model.time_of_day,
            'activity': time_model.current_activity,
            'conflict_events': await check_for_conflict_events(self.user_id, self.conversation_id),
            'time_advancement': await process_conflict_time_advancement(
                self.user_id, 
                self.conversation_id,
                context.get('activity_type', 'idle')
            )
        }
        
        # 9. NPC-SPECIFIC CONFLICT STATES
        npcs_present = context.get('npcs_present', [])
        inputs['npc_conflicts'] = {}
        
        for npc_id in npcs_present:
            inputs['npc_conflicts'][npc_id] = await self._get_npc_conflict_state(npc_id)
        
        # 10. WORLD STATE IMPACTS
        from story_agent.world_director_agent import WorldDirector
        
        world_director = WorldDirector(self.user_id, self.conversation_id)
        world_state = await world_director.get_world_state()
        
        inputs['world_state'] = {
            'tensions': world_state.tension_factors if hasattr(world_state, 'tension_factors') else {},
            'mood': world_state.world_mood if hasattr(world_state, 'world_mood') else 'neutral',
            'power_dynamics': world_state.recent_power_exchanges if hasattr(world_state, 'recent_power_exchanges') else []
        }
        
        # 11. MEMORY-BASED CONFLICT TRACKING
        from context.memory_manager import get_memory_manager
        
        memory_mgr = get_memory_manager(self.user_id, self.conversation_id)
        inputs['conflict_memories'] = {
            'recent_resolutions': await memory_mgr.get_memories_by_type('conflict_resolution', limit=5),
            'conflict_patterns': await memory_mgr.get_memories_by_type('conflict_pattern', limit=10)
        }
        
        # 12. EVENT SYSTEM CONFLICTS
        from logic.event_system import EventSystem
        
        event_system = EventSystem()
        await event_system.initialize(self.user_id, self.conversation_id)
        inputs['event_conflicts'] = await event_system.get_conflict_events()
        
        # 13. NARRATIVE EVENT CONFLICTS
        from logic.narrative_events import check_for_narrative_moments
        
        inputs['narrative_conflicts'] = await check_for_narrative_moments(
            self.user_id, 
            self.conversation_id,
            context
        )
        
        return inputs
    
    # ========== HARMONIZATION ==========
    
    async def _harmonize_inputs(self, inputs: Dict) -> Dict[str, Any]:
        """
        Detect and resolve contradictions between different conflict systems.
        Ensures all conflicts work together coherently.
        """
        harmonized = {
            'conflicts': [],
            'tensions': defaultdict(float),
            'active_dynamics': [],
            'power_exchanges': [],
            'resolutions_pending': [],
            'contradictions_resolved': [],
            'vitals_impact': {},
            'governance_directives': []
        }
        
        all_conflicts = []
        
        # 1. Process slice-of-life conflicts
        for conflict in inputs.get('slice_of_life', {}).get('active', []):
            conflict['source_system'] = 'slice_of_life'
            conflict['weight'] = 0.7  # Subtle conflicts have lower weight
            conflict['category'] = 'interpersonal'
            all_conflicts.append(conflict)
        
        # Add pending slice-of-life resolutions
        harmonized['resolutions_pending'].extend(
            inputs.get('slice_of_life', {}).get('pending_resolutions', [])
        )
        
        # 2. Process main system conflicts
        for conflict in inputs.get('main_conflicts', {}).get('active', []):
            conflict['source_system'] = 'main'
            conflict['weight'] = 1.0
            conflict['category'] = 'major'
            # Add detailed info if available
            conflict_id = conflict.get('conflict_id')
            if conflict_id and conflict_id in inputs.get('main_conflicts', {}).get('details', {}):
                conflict['details'] = inputs['main_conflicts']['details'][conflict_id]
            all_conflicts.append(conflict)
        
        # 3. Process enhanced integration conflicts
        if 'enhanced_integration' in inputs:
            # Add pattern resolutions
            harmonized['resolutions_pending'].extend(
                inputs['enhanced_integration'].get('pattern_resolutions', [])
            )
            # Add daily dynamics as active dynamics
            harmonized['active_dynamics'].extend(
                inputs['enhanced_integration'].get('daily_dynamics', [])
            )
        
        # 4. Process power exchanges
        harmonized['power_exchanges'] = inputs.get('power_exchanges', [])
        
        # 5. Process political/geopolitical conflicts
        if inputs.get('political_conflicts', {}).get('available', True):
            for conflict in inputs.get('political_conflicts', {}).get('active_political', []):
                conflict['source_system'] = 'political'
                conflict['weight'] = 0.5  # Background political tensions
                conflict['category'] = 'political'
                all_conflicts.append(conflict)
        
        # 6. Process governance conflicts
        if inputs.get('governance_conflicts', {}).get('available', True):
            governance_data = inputs.get('governance_conflicts', {})
            if governance_data:
                harmonized['governance_directives'] = governance_data.get('directives', [])
                # Add high-priority governance conflicts
                for conflict in governance_data.get('conflicts', []):
                    conflict['source_system'] = 'governance'
                    conflict['weight'] = 1.5  # Governance has high priority
                    conflict['category'] = 'systemic'
                    all_conflicts.append(conflict)
        
        # 7. Process temporal conflict events
        for event in inputs.get('temporal_context', {}).get('conflict_events', []):
            # Convert events to temporary conflicts
            temp_conflict = {
                'conflict_id': f"event_{event.get('conflict_id')}",
                'source_system': 'event',
                'weight': 0.8,
                'category': 'event',
                'description': event.get('description'),
                'temporary': True
            }
            all_conflicts.append(temp_conflict)
        
        # 8. Process vitals impact
        harmonized['vitals_impact'] = inputs.get('main_conflicts', {}).get('vitals', {})
        
        # 9. Check for contradictions between ALL conflicts
        for i, c1 in enumerate(all_conflicts):
            for c2 in all_conflicts[i+1:]:
                contradiction = await self._detect_contradiction(c1, c2)
                if contradiction:
                    resolved = await self._resolve_contradiction(c1, c2, contradiction)
                    harmonized['contradictions_resolved'].append(resolved)
        
        # 10. Aggregate tension levels from all sources
        tension_sources = [
            ('slice_of_life', 1.0),
            ('world_state', 1.2),
            ('temporal_context', 0.8),
            ('narrative_conflicts', 0.9)
        ]
        
        for source_key, weight in tension_sources:
            if source_key in inputs and isinstance(inputs[source_key], dict):
                tensions = inputs[source_key].get('tensions', {})
                for tension_type, level in tensions.items():
                    harmonized['tensions'][tension_type] += level * weight
        
        # Add power exchange tensions
        for exchange in harmonized['power_exchanges']:
            intensity = exchange.get('intensity', 0.5)
            harmonized['tensions']['power'] += intensity * 0.3
            harmonized['tensions']['sexual'] += intensity * 0.1
        
        # Normalize tension levels
        max_tension = max(harmonized['tensions'].values()) if harmonized['tensions'] else 1.0
        if max_tension > 0:
            for key in harmonized['tensions']:
                harmonized['tensions'][key] = min(1.0, harmonized['tensions'][key] / max_tension)
        
        # 11. Extract memory patterns
        if 'conflict_memories' in inputs:
            for memory in inputs['conflict_memories'].get('conflict_patterns', []):
                # Use memory patterns to adjust conflict weights
                pattern_type = memory.get('pattern_type')
                for conflict in all_conflicts:
                    if pattern_type in str(conflict.get('conflict_type', '')).lower():
                        conflict['weight'] *= 1.2  # Boost conflicts matching memory patterns
        
        harmonized['conflicts'] = all_conflicts
        
        return harmonized
    
    # ========== PRIORITIZATION ==========
    
    async def _prioritize_conflicts(
        self,
        harmonized: Dict,
        context: Dict
    ) -> Dict[str, Any]:
        """Prioritize conflicts based on current scene context"""
        
        prioritized = {
            'primary': None,
            'secondary': [],
            'background': [],
            'active_dynamics': harmonized.get('active_dynamics', []),
            'tension_map': harmonized.get('tensions', {})
        }
        
        # Score each conflict for current relevance
        scored_conflicts = []
        
        for conflict in harmonized.get('conflicts', []):
            score = await self._calculate_conflict_relevance(conflict, context)
            scored_conflicts.append((score, conflict))
        
        # Sort by score
        scored_conflicts.sort(key=lambda x: x[0], reverse=True)
        
        # Assign to priority tiers
        if scored_conflicts:
            prioritized['primary'] = scored_conflicts[0][1]
            
            if len(scored_conflicts) > 1:
                prioritized['secondary'] = [c[1] for c in scored_conflicts[1:3]]
            
            if len(scored_conflicts) > 3:
                prioritized['background'] = [c[1] for c in scored_conflicts[3:6]]
        
        return prioritized
    
    # ========== NARRATIVE SYNTHESIS ==========
    
    async def _generate_narrative_synthesis(self, prioritized: Dict) -> Dict[str, Any]:
        """Generate unified narrative elements from prioritized conflicts"""
        
        narrative = {
            'text': None,
            'dialogue_hints': [],
            'environmental_cues': [],
            'subtext_layers': []
        }
        
        # Generate primary narrative if there's a main conflict
        if prioritized.get('primary'):
            primary = prioritized['primary']
            
            # Create narrative text based on conflict type and intensity
            narrative['text'] = await self._create_narrative_text(primary)
            
            # Generate dialogue hints
            narrative['dialogue_hints'] = await self._generate_dialogue_hints(
                primary,
                prioritized.get('secondary', [])
            )
        
        # Add environmental cues from all active conflicts
        all_active = [prioritized.get('primary')] + prioritized.get('secondary', [])
        all_active = [c for c in all_active if c]  # Filter None
        
        for conflict in all_active:
            cues = await self._extract_environmental_cues(conflict)
            narrative['environmental_cues'].extend(cues)
        
        # Layer in subtext from background conflicts
        for bg_conflict in prioritized.get('background', []):
            subtext = await self._generate_subtext(bg_conflict)
            if subtext:
                narrative['subtext_layers'].append(subtext)
        
        return narrative
    
    # ========== PLAYER OPTIONS SYNTHESIS ==========
    
    async def _synthesize_player_options(self, prioritized: Dict) -> List[Dict]:
        """Synthesize coherent player choices from all active conflicts"""
        
        options = []
        used_archetypes = set()  # Avoid duplicate option types
        
        # Generate options for primary conflict
        if prioritized.get('primary'):
            primary_options = await self._generate_conflict_options(
                prioritized['primary'],
                is_primary=True
            )
            options.extend(primary_options)
            for opt in primary_options:
                used_archetypes.add(opt.get('archetype'))
        
        # Add nuanced options from secondary conflicts
        for conflict in prioritized.get('secondary', []):
            secondary_options = await self._generate_conflict_options(
                conflict,
                is_primary=False,
                exclude_archetypes=used_archetypes
            )
            
            # Merge compatible options
            for sec_opt in secondary_options:
                merged = False
                for pri_opt in options:
                    if self._can_merge_options(pri_opt, sec_opt):
                        pri_opt['subtext'] += f" (also affects {conflict.get('conflict_name', 'other tension')})"
                        pri_opt['multi_impact'] = True
                        merged = True
                        break
                
                if not merged and len(options) < 5:  # Max 5 options
                    options.append(sec_opt)
        
        # Ensure there's always a neutral option
        if not any(opt.get('archetype') == 'neutral' for opt in options):
            options.append({
                'id': 'observe',
                'text': 'Simply observe for now',
                'archetype': 'neutral',
                'subtext': 'Maintain current dynamics',
                'impact': {}
            })
        
        return options
    
    # ========== NPC BEHAVIOR SYNTHESIS ==========
    
    async def _synthesize_npc_behaviors(self, prioritized: Dict) -> Dict[int, Dict]:
        """Determine NPC behaviors based on all active conflicts"""
        
        behaviors = {}
        
        # Get all NPCs involved in any conflict
        involved_npcs = set()
        
        if prioritized.get('primary'):
            involved_npcs.update(prioritized['primary'].get('stakeholders', []))
        
        for conflict in prioritized.get('secondary', []):
            involved_npcs.update(conflict.get('stakeholders', []))
        
        # Generate behavior for each NPC
        for npc_id in involved_npcs:
            behavior = await self._determine_npc_behavior(
                npc_id,
                prioritized,
                self._get_npc_conflict_involvement(npc_id, prioritized)
            )
            behaviors[npc_id] = behavior
        
        return behaviors
    
    # ========== SCENE MODIFICATION ==========
    
    async def _synthesize_scene_modifiers(self, prioritized: Dict) -> Dict[str, Any]:
        """Generate scene modifications based on active conflicts"""
        
        modifiers = {
            'atmosphere': 'neutral',
            'tension_level': 0.0,
            'hidden_elements': [],
            'power_dynamics_visible': False,
            'special_interactions': []
        }
        
        # Set atmosphere based on primary conflict
        if prioritized.get('primary'):
            primary = prioritized['primary']
            modifiers['atmosphere'] = self._determine_atmosphere(primary)
            modifiers['tension_level'] = primary.get('intensity', 0.5)
        
        # Add tension from secondary conflicts
        for conflict in prioritized.get('secondary', []):
            modifiers['tension_level'] += conflict.get('intensity', 0) * 0.3
        
        # Cap tension level
        modifiers['tension_level'] = min(1.0, modifiers['tension_level'])
        
        # Determine if power dynamics should be visible
        if any('power' in str(c.get('conflict_type', '')).lower() 
               for c in [prioritized.get('primary')] + prioritized.get('secondary', [])
               if c):
            modifiers['power_dynamics_visible'] = True
        
        # Add special interactions based on conflict states
        modifiers['special_interactions'] = await self._get_special_interactions(prioritized)
        
        return modifiers
    
    # ========== HELPER METHODS ==========
    
    async def _detect_contradiction(self, c1: Dict, c2: Dict) -> Optional[str]:
        """Detect if two conflicts contradict each other"""
        
        # Check if same NPCs have conflicting roles
        c1_npcs = set(c1.get('stakeholders', []))
        c2_npcs = set(c2.get('stakeholders', []))
        
        shared_npcs = c1_npcs & c2_npcs
        
        if shared_npcs:
            # Check for role conflicts
            if (c1.get('conflict_type') == 'dominance_challenge' and 
                c2.get('conflict_type') == 'submission_pattern'):
                return 'role_conflict'
        
        # Check for mutually exclusive states
        if (c1.get('phase') == 'resolution' and 
            c2.get('phase') == 'emerging' and
            c1.get('conflict_type') == c2.get('conflict_type')):
            return 'phase_conflict'
        
        return None
    
    async def _resolve_contradiction(
        self,
        c1: Dict,
        c2: Dict,
        contradiction_type: str
    ) -> Dict:
        """Resolve a detected contradiction"""
        
        resolution = {
            'type': contradiction_type,
            'conflicts': [c1.get('conflict_id'), c2.get('conflict_id')],
            'resolution': 'merged'
        }
        
        if contradiction_type == 'role_conflict':
            # Prioritize the more intense conflict
            if c1.get('intensity', 0) > c2.get('intensity', 0):
                c2['suppressed'] = True
                resolution['kept'] = c1.get('conflict_id')
            else:
                c1['suppressed'] = True
                resolution['kept'] = c2.get('conflict_id')
        
        elif contradiction_type == 'phase_conflict':
            # Keep the more advanced conflict
            phase_order = ['emerging', 'developing', 'active', 'climax', 'resolution']
            c1_phase_idx = phase_order.index(c1.get('phase', 'emerging'))
            c2_phase_idx = phase_order.index(c2.get('phase', 'emerging'))
            
            if c1_phase_idx > c2_phase_idx:
                c2['delayed'] = True
                resolution['kept'] = c1.get('conflict_id')
            else:
                c1['delayed'] = True
                resolution['kept'] = c2.get('conflict_id')
        
        return resolution
    
    async def _calculate_conflict_relevance(
        self,
        conflict: Dict,
        context: Dict
    ) -> float:
        """Calculate how relevant a conflict is to current scene"""
        
        score = 0.0
        
        # Base score from conflict weight
        score += conflict.get('weight', 0.5)
        
        # Boost if NPCs involved are present
        present_npcs = set(context.get('npcs_present', []))
        conflict_npcs = set(conflict.get('stakeholders', []))
        
        if conflict_npcs & present_npcs:
            score += 0.3 * len(conflict_npcs & present_npcs)
        
        # Boost based on conflict phase
        phase_scores = {
            'emerging': 0.1,
            'developing': 0.2,
            'active': 0.4,
            'climax': 0.6,
            'resolution': 0.3
        }
        score += phase_scores.get(conflict.get('phase'), 0)
        
        # Boost if location is relevant
        if conflict.get('location') == context.get('location'):
            score += 0.2
        
        # Adjust for time of day appropriateness
        time_appropriate = await self._is_time_appropriate(conflict, context)
        if time_appropriate:
            score += 0.1
        else:
            score *= 0.7
        
        return min(1.0, score)
    
    async def _is_time_appropriate(self, conflict: Dict, context: Dict) -> bool:
        """Check if conflict is appropriate for current time"""
        
        time_of_day = context.get('time_of_day', 'Day')
        conflict_type = conflict.get('conflict_type', '')
        
        # Domestic conflicts more appropriate during routine times
        if 'domestic' in conflict_type.lower() or 'routine' in conflict_type.lower():
            return time_of_day in ['Morning', 'Evening']
        
        # Social conflicts during day
        if 'social' in conflict_type.lower():
            return time_of_day in ['Day', 'Afternoon']
        
        # Intimate/power conflicts in private moments
        if 'intimate' in conflict_type.lower() or 'power' in conflict_type.lower():
            return time_of_day in ['Evening', 'Night']
        
        return True  # Default to appropriate
    
    async def _get_npc_conflict_state(self, npc_id: int) -> Dict:
        """Get conflict state for specific NPC"""
        
        async with get_db_connection_context() as conn:
            conflicts = await conn.fetch("""
                SELECT c.* FROM Conflicts c
                JOIN conflict_stakeholders cs ON c.conflict_id = cs.conflict_id
                WHERE cs.npc_id = $1 
                AND c.user_id = $2
                AND c.conversation_id = $3
                AND c.is_active = true
            """, npc_id, self.user_id, self.conversation_id)
        
        return {
            'active_count': len(conflicts),
            'conflicts': [dict(c) for c in conflicts]
        }
    
    def _get_npc_conflict_involvement(
        self,
        npc_id: int,
        prioritized: Dict
    ) -> List[str]:
        """Get list of conflict types NPC is involved in"""
        
        involvement = []
        
        if prioritized.get('primary'):
            if npc_id in prioritized['primary'].get('stakeholders', []):
                involvement.append('primary')
        
        for conflict in prioritized.get('secondary', []):
            if npc_id in conflict.get('stakeholders', []):
                involvement.append('secondary')
        
        return involvement
    
    async def _log_synthesis(self, output: SynthesizedOutput):
        """Log synthesis event for debugging and analysis"""
        
        logger.debug(f"Synthesized conflict output: priority={output.priority_score}, "
                    f"choices={len(output.player_choices)}, "
                    f"npcs={len(output.npc_behaviors)}")

# ===============================================================================
# PUBLIC API FUNCTIONS
# ===============================================================================

@function_tool
async def get_unified_conflict_state(
    ctx: RunContextWrapper,
    scene_context: Dict[str, Any]
) -> str:
    """
    Get the unified, synthesized conflict state for the current scene.
    
    This is the main entry point for other systems to get conflict information.
    """
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    synthesizer = ConflictOutputSynthesizer(user_id, conversation_id)
    output = await synthesizer.synthesize_conflict_state(scene_context)
    
    return json.dumps({
        'primary_tension': output.primary_tension,
        'active_dynamics': output.active_dynamics,
        'scene_modifiers': output.scene_modifiers,
        'dialogue_hints': output.dialogue_hints,
        'player_choices': output.player_choices,
        'npc_behaviors': output.npc_behaviors,
        'environmental_cues': output.environmental_cues,
        'narrative_text': output.narrative_text,
        'ui_elements': output.ui_elements,
        'priority_score': output.priority_score
    })

@function_tool
async def update_conflict_synthesis_cache(
    ctx: RunContextWrapper,
    system_name: str,
    update_data: Dict[str, Any]
) -> str:
    """
    Allow individual conflict systems to push updates to the synthesizer.
    
    This enables real-time coordination without constant re-querying.
    """
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    synthesizer = ConflictOutputSynthesizer(user_id, conversation_id)
    
    # Store the update
    synthesizer._system_inputs[system_name] = {
        'data': update_data,
        'timestamp': datetime.now()
    }
    
    # Mark cache as stale
    synthesizer._synthesis_cache = None
    
    return json.dumps({
        'success': True,
        'system': system_name,
        'update_received': True
    })

@function_tool
async def get_conflict_priority_queue(
    ctx: RunContextWrapper
) -> str:
    """
    Get a prioritized queue of conflicts that need resolution or attention.
    
    Useful for AI decision-making about what to focus on.
    """
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    synthesizer = ConflictOutputSynthesizer(user_id, conversation_id)
    
    # Get all conflicts and prioritize them
    context = {'npcs_present': [], 'location': 'current'}
    inputs = await synthesizer._gather_system_inputs(context)
    harmonized = await synthesizer._harmonize_inputs(inputs)
    prioritized = await synthesizer._prioritize_conflicts(harmonized, context)
    
    # Create priority queue
    queue = []
    
    if prioritized.get('primary'):
        queue.append({
            'priority': ConflictPriority.CRITICAL.value,
            'conflict': prioritized['primary'],
            'action_required': True
        })
    
    for conflict in prioritized.get('secondary', []):
        queue.append({
            'priority': ConflictPriority.HIGH.value,
            'conflict': conflict,
            'action_required': False
        })
    
    for conflict in prioritized.get('background', []):
        queue.append({
            'priority': ConflictPriority.MEDIUM.value,
            'conflict': conflict,
            'action_required': False
        })
    
    return json.dumps({
        'queue': queue,
        'total_active': len(queue),
        'immediate_action_needed': any(q['action_required'] for q in queue)
    })
