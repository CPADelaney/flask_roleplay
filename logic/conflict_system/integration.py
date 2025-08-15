# logic/conflict_system/integration.py
"""
Master Integration Module - Interface to ConflictSynthesizer
Provides high-level API for game systems to interact with the conflict system
"""

import logging
import json
import random
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from agents import Agent, function_tool, ModelSettings, RunContextWrapper, Runner
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ===============================================================================
# INTEGRATION MODES
# ===============================================================================

class IntegrationMode(Enum):
    """Modes of conflict system integration"""
    FULL_IMMERSION = "full_immersion"  # All systems active
    STORY_FOCUS = "story_focus"  # Prioritize narrative
    SOCIAL_DYNAMICS = "social_dynamics"  # Focus on relationships
    BACKGROUND_AWARE = "background_aware"  # Emphasize world events
    PLAYER_CENTRIC = "player_centric"  # Focus on player agency
    EMERGENT = "emergent"  # Let patterns emerge naturally
    GUIDED = "guided"  # Actively guide experience


@dataclass
class IntegrationState:
    """Current state of integrated conflict system"""
    mode: IntegrationMode
    active_modules: Set[str]
    conflict_count: int
    complexity_score: float
    narrative_coherence: float
    player_engagement: float
    system_health: float


# ===============================================================================
# CONFLICT SYSTEM INTERFACE
# ===============================================================================

class ConflictSystemInterface:
    """
    High-level interface to the conflict system.
    Works through ConflictSynthesizer without duplicating orchestration.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Current integration mode
        self.current_mode = IntegrationMode.EMERGENT
        
        # Monitoring agents (lazy loaded)
        self._experience_monitor = None
        self._mode_optimizer = None
        
        # Cache for synthesizer reference
        self._synthesizer_cache = None
    
    # ========== Agent Properties ==========
    
    @property
    def experience_monitor(self) -> Agent:
        """Monitor overall player experience"""
        if self._experience_monitor is None:
            self._experience_monitor = Agent(
                name="Experience Monitor",
                instructions="""
                Monitor the overall conflict experience for the player.
                
                Track:
                - Engagement levels
                - Narrative coherence
                - Pacing and rhythm
                - Emotional journey
                - Satisfaction indicators
                
                Recommend adjustments to optimize experience.
                """,
                model="gpt-5-nano",
            )
        return self._experience_monitor
    
    @property
    def mode_optimizer(self) -> Agent:
        """Optimize integration mode based on context"""
        if self._mode_optimizer is None:
            self._mode_optimizer = Agent(
                name="Mode Optimizer",
                instructions="""
                Determine optimal integration mode for current context.
                
                Consider:
                - Player engagement patterns
                - Story progression
                - System load
                - Narrative needs
                - Player preferences
                
                Recommend mode changes to enhance experience.
                """,
                model="gpt-5-nano",
            )
        return self._mode_optimizer
    
    # ========== Synthesizer Access ==========
    
    async def _get_synthesizer(self):
        """Get or cache synthesizer instance"""
        if not self._synthesizer_cache:
            from logic.conflict_system.conflict_synthesizer import get_synthesizer
            self._synthesizer_cache = await get_synthesizer(self.user_id, self.conversation_id)
        return self._synthesizer_cache
    
    # ========== High-Level Operations ==========
    
    async def initialize_system(self, mode: IntegrationMode = IntegrationMode.EMERGENT) -> Dict[str, Any]:
        """Initialize the conflict system with specified mode"""
        
        self.current_mode = mode
        
        # Get synthesizer (this initializes all subsystems)
        synthesizer = await self._get_synthesizer()
        
        # Store mode preference
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO integration_settings
                (user_id, conversation_id, mode, updated_at)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, conversation_id)
                DO UPDATE SET mode = $3, updated_at = CURRENT_TIMESTAMP
            """, self.user_id, self.conversation_id, mode.value)
        
        # Get initial state
        state = await synthesizer.get_system_state()
        
        return {
            'initialized': True,
            'mode': mode.value,
            'subsystems_loaded': len(state.get('subsystems', {})),
            'system_health': state['metrics']['system_health']
        }
    
    async def process_game_scene(
        self,
        scene_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a game scene through the conflict system"""
        
        # Add mode context
        scene_data['integration_mode'] = self.current_mode.value
        
        # Process through synthesizer
        synthesizer = await self._get_synthesizer()
        result = await synthesizer.process_scene(scene_data)
        
        # Monitor experience
        experience_quality = await self._assess_experience_quality(result)
        
        # Add experience metadata
        result['experience_quality'] = experience_quality
        
        # Check if mode should change
        if experience_quality < 0.5:
            new_mode = await self._recommend_mode_change(scene_data, experience_quality)
            if new_mode != self.current_mode:
                result['recommended_mode_change'] = new_mode.value
        
        return result
    
    async def create_conflict(
        self,
        conflict_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new conflict"""
        
        # Add mode-specific adjustments
        context['integration_mode'] = self.current_mode.value
        
        # Create through synthesizer
        synthesizer = await self._get_synthesizer()
        result = await synthesizer.create_conflict(conflict_type, context)
        
        return result
    
    async def resolve_conflict(
        self,
        conflict_id: int,
        resolution_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve a conflict"""
        
        # Resolve through synthesizer
        synthesizer = await self._get_synthesizer()
        result = await synthesizer.resolve_conflict(
            conflict_id,
            resolution_type,
            context
        )
        
        return result
    
    async def get_system_status(self) -> IntegrationState:
        """Get comprehensive system status"""
        
        synthesizer = await self._get_synthesizer()
        state = await synthesizer.get_system_state()
        
        # Build integration state
        return IntegrationState(
            mode=self.current_mode,
            active_modules=set(state.get('subsystems', {}).keys()),
            conflict_count=state['metrics']['active_conflicts'],
            complexity_score=state['metrics']['complexity_score'],
            narrative_coherence=state['metrics'].get('narrative_coherence', 1.0),
            player_engagement=state['metrics'].get('player_engagement', 0.5),
            system_health=state['metrics']['system_health']
        )
    
    async def optimize_experience(self) -> Dict[str, Any]:
        """Optimize the conflict experience"""
        
        # Get current state
        state = await self.get_system_status()
        
        optimizations = []
        
        # Check various optimization opportunities
        if state.complexity_score > 0.8:
            optimizations.append(await self._reduce_complexity())
        
        if state.narrative_coherence < 0.5:
            optimizations.append(await self._improve_coherence())
        
        if state.player_engagement < 0.3:
            optimizations.append(await self._boost_engagement())
        
        if state.system_health < 0.5:
            optimizations.append(await self._heal_system())
        
        return {
            'optimizations_performed': optimizations,
            'new_state': await self.get_system_status()
        }
    
    # ========== Mode Management ==========
    
    async def set_mode(self, mode: IntegrationMode) -> Dict[str, Any]:
        """Change the integration mode"""
        
        old_mode = self.current_mode
        self.current_mode = mode
        
        # Calculate what changes
        changes = await self._calculate_mode_changes(old_mode, mode)
        
        # Store preference
        async with get_db_connection_context() as conn:
            await conn.execute("""
                UPDATE integration_settings
                SET mode = $1, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = $2 AND conversation_id = $3
            """, mode.value, self.user_id, self.conversation_id)
        
        return {
            'mode_changed': True,
            'previous_mode': old_mode.value,
            'new_mode': mode.value,
            'changes': changes
        }
    
    async def _calculate_mode_changes(
        self,
        old_mode: IntegrationMode,
        new_mode: IntegrationMode
    ) -> Dict[str, Any]:
        """Calculate what changes between modes"""
        
        changes = {
            'subsystem_priorities': {},
            'complexity_adjustment': 0,
            'focus_shift': ""
        }
        
        # Define mode characteristics
        mode_profiles = {
            IntegrationMode.FULL_IMMERSION: {
                'complexity': 1.0,
                'all_systems': True,
                'focus': 'complete experience'
            },
            IntegrationMode.STORY_FOCUS: {
                'complexity': 0.7,
                'priority': ['canon', 'flow', 'victory'],
                'focus': 'narrative progression'
            },
            IntegrationMode.SOCIAL_DYNAMICS: {
                'complexity': 0.6,
                'priority': ['social', 'multiparty', 'leverage'],
                'focus': 'relationships'
            },
            IntegrationMode.BACKGROUND_AWARE: {
                'complexity': 0.5,
                'priority': ['background'],
                'focus': 'world events'
            },
            IntegrationMode.PLAYER_CENTRIC: {
                'complexity': 0.4,
                'priority': ['victory', 'template'],
                'focus': 'player agency'
            },
            IntegrationMode.EMERGENT: {
                'complexity': 0.5,
                'all_systems': True,
                'focus': 'natural emergence'
            },
            IntegrationMode.GUIDED: {
                'complexity': 0.6,
                'priority': ['flow', 'template'],
                'focus': 'curated experience'
            }
        }
        
        old_profile = mode_profiles[old_mode]
        new_profile = mode_profiles[new_mode]
        
        changes['complexity_adjustment'] = new_profile['complexity'] - old_profile['complexity']
        changes['focus_shift'] = f"{old_profile['focus']} -> {new_profile['focus']}"
        
        if 'priority' in new_profile:
            for system in new_profile['priority']:
                changes['subsystem_priorities'][system] = 'high'
        
        return changes
    
    # ========== Experience Optimization ==========
    
    async def _assess_experience_quality(self, scene_result: Dict[str, Any]) -> float:
        """Assess the quality of the experience"""
        
        quality_score = 0.5  # Base score
        
        # Check for various quality indicators
        if scene_result.get('choices'):
            quality_score += 0.1  # Player has choices
        
        if scene_result.get('manifestations'):
            quality_score += 0.1  # Conflicts are manifesting
        
        if scene_result.get('narrative_summary'):
            quality_score += 0.1  # Good narrative flow
        
        if scene_result.get('victories'):
            quality_score += 0.2  # Progress/resolution
        
        # Check for problems
        if scene_result.get('edge_cases'):
            quality_score -= 0.2  # System issues
        
        return max(0.0, min(1.0, quality_score))
    
    async def _recommend_mode_change(
        self,
        context: Dict[str, Any],
        current_quality: float
    ) -> IntegrationMode:
        """Recommend a mode change based on context"""
        
        prompt = f"""
        Recommend integration mode based on context:
        
        Current Mode: {self.current_mode.value}
        Experience Quality: {current_quality:.1%}
        Context: {json.dumps(context)}
        
        Available modes:
        - full_immersion: All systems active
        - story_focus: Prioritize narrative
        - social_dynamics: Focus on relationships
        - background_aware: Emphasize world events
        - player_centric: Maximum player agency
        - emergent: Natural pattern emergence
        - guided: Curated experience
        
        Which mode would improve the experience?
        Return just the mode name.
        """
        
        response = await Runner.run(self.mode_optimizer, prompt)
        
        try:
            mode_name = response.output.strip().lower().replace(' ', '_')
            return IntegrationMode[mode_name.upper()]
        except (KeyError, AttributeError):
            return self.current_mode  # Keep current if recommendation fails
    
    async def _reduce_complexity(self) -> str:
        """Reduce system complexity"""
        synthesizer = await self._get_synthesizer()
        
        # Get lowest priority conflicts
        async with get_db_connection_context() as conn:
            low_priority = await conn.fetch("""
                SELECT conflict_id FROM Conflicts
                WHERE user_id = $1 AND conversation_id = $2
                AND is_active = true
                AND intensity IN ('subtle', 'tension')
                ORDER BY progress ASC
                LIMIT 2
            """, self.user_id, self.conversation_id)
        
        # Resolve them
        for conflict in low_priority:
            await synthesizer.resolve_conflict(
                conflict['conflict_id'],
                'natural_resolution',
                {'reason': 'complexity_reduction'}
            )
        
        return f"Resolved {len(low_priority)} low-priority conflicts"
    
    async def _improve_coherence(self) -> str:
        """Improve narrative coherence"""
        synthesizer = await self._get_synthesizer()
        
        # Trigger edge case detection and recovery
        from logic.conflict_system.conflict_synthesizer import SystemEvent, EventType, SubsystemType
        
        event = SystemEvent(
            event_id=f"coherence_{datetime.now().timestamp()}",
            event_type=EventType.HEALTH_CHECK,
            source_subsystem=SubsystemType.CANON,
            payload={'check_coherence': True},
            target_subsystems={SubsystemType.EDGE_HANDLER}
        )
        
        await synthesizer.emit_event(event)
        
        return "Triggered coherence check and recovery"
    
    async def _boost_engagement(self) -> str:
        """Boost player engagement"""
        synthesizer = await self._get_synthesizer()
        
        # Generate an interesting conflict from template
        from logic.conflict_system.conflict_synthesizer import SubsystemType
        
        template_subsystem = synthesizer._subsystems.get(SubsystemType.TEMPLATE)
        if template_subsystem:
            # Create engaging conflict
            await synthesizer.create_conflict(
                'template_personal_boundaries',
                {'boost_engagement': True}
            )
            return "Generated engaging conflict"
        
        return "Engagement boost attempted"
    
    async def _heal_system(self) -> str:
        """Heal system issues"""
        synthesizer = await self._get_synthesizer()
        
        # Trigger comprehensive health check
        from logic.conflict_system.conflict_synthesizer import SystemEvent, EventType
        
        event = SystemEvent(
            event_id=f"heal_{datetime.now().timestamp()}",
            event_type=EventType.HEALTH_CHECK,
            source_subsystem=None,
            payload={'comprehensive': True}
        )
        
        await synthesizer.emit_event(event)
        
        return "System healing initiated"


# ===============================================================================
# PUBLIC API FUNCTIONS
# ===============================================================================

@function_tool
async def initialize_conflict_system(
    ctx: RunContextWrapper,
    mode: str = "emergent"
) -> Dict[str, Any]:
    """Initialize the master conflict system"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    interface = ConflictSystemInterface(user_id, conversation_id)
    
    try:
        integration_mode = IntegrationMode[mode.upper()]
    except KeyError:
        integration_mode = IntegrationMode.EMERGENT
    
    result = await interface.initialize_system(integration_mode)
    
    return result


@function_tool
async def process_conflict_scene(
    ctx: RunContextWrapper,
    activity: str,
    location: str,
    present_npcs: List[int],
    recent_events: List[str]
) -> Dict[str, Any]:
    """Process a scene through the conflict system"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    interface = ConflictSystemInterface(user_id, conversation_id)
    
    scene_data = {
        'activity': activity,
        'location': location,
        'present_npcs': present_npcs,
        'recent_events': recent_events,
        'timestamp': datetime.now().isoformat()
    }
    
    result = await interface.process_game_scene(scene_data)
    
    return result


@function_tool
async def adjust_conflict_mode(
    ctx: RunContextWrapper,
    new_mode: str
) -> Dict[str, Any]:
    """Adjust the conflict system integration mode"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    interface = ConflictSystemInterface(user_id, conversation_id)
    
    try:
        mode = IntegrationMode[new_mode.upper()]
        result = await interface.set_mode(mode)
    except KeyError:
        result = {'error': f'Invalid mode: {new_mode}'}
    
    return result


@function_tool
async def get_conflict_system_status(
    ctx: RunContextWrapper
) -> Dict[str, Any]:
    """Get comprehensive status of conflict system"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    interface = ConflictSystemInterface(user_id, conversation_id)
    
    state = await interface.get_system_status()
    
    return {
        'mode': state.mode.value,
        'active_modules': list(state.active_modules),
        'metrics': {
            'conflict_count': state.conflict_count,
            'complexity': state.complexity_score,
            'coherence': state.narrative_coherence,
            'engagement': state.player_engagement,
            'health': state.system_health
        },
        'recommendation': 'healthy' if state.system_health > 0.7 else 'needs_attention'
    }


@function_tool
async def optimize_conflict_experience(
    ctx: RunContextWrapper
) -> Dict[str, Any]:
    """Optimize the conflict experience based on current state"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    interface = ConflictSystemInterface(user_id, conversation_id)
    
    result = await interface.optimize_experience()
    
    return result


@function_tool
async def create_contextual_conflict(
    ctx: RunContextWrapper,
    conflict_type: str,
    participants: List[int],
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a new conflict with context"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    interface = ConflictSystemInterface(user_id, conversation_id)
    
    full_context = {
        'participants': participants,
        **context
    }
    
    result = await interface.create_conflict(conflict_type, full_context)
    
    return result


@function_tool
async def resolve_active_conflict(
    ctx: RunContextWrapper,
    conflict_id: int,
    resolution_type: str = "natural",
    resolution_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Resolve an active conflict"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    interface = ConflictSystemInterface(user_id, conversation_id)
    
    result = await interface.resolve_conflict(
        conflict_id,
        resolution_type,
        resolution_context or {}
    )
    
    return result
