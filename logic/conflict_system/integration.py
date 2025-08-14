# logic/conflict_system/integration.py
"""
Master Integration Module with LLM-powered orchestration
Coordinates all conflict system modules into a unified experience
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

# Import all conflict system modules
from logic.conflict_system.slice_of_life_conflicts import (
    SliceOfLifeConflictManager, EmergentConflictDetector
)
from logic.conflict_system.enhanced_conflict_integration import (
    EnhancedConflictSystemIntegration
)
from logic.conflict_system.tension import TensionSystem
from logic.conflict_system.social_circle import SocialCircleConflictManager
from logic.conflict_system.leverage import LeverageSystem
from logic.conflict_system.multi_party_dynamics import MultiPartyConflictOrchestrator
from logic.conflict_system.conflict_flow import ConflictFlowManager
from logic.conflict_system.autonomous_stakeholder_actions import AutonomousStakeholderSystem
from logic.conflict_system.background_grand_conflicts import BackgroundConflictManager
from logic.conflict_system.conflict_victory import ConflictVictoryManager
from logic.conflict_system.conflict_canon import ConflictCanonManager
from logic.conflict_system.conflict_synthesizer import ConflictSynthesizer
from logic.conflict_system.dynamic_conflict_template import DynamicConflictTemplateSystem
from logic.conflict_system.edge_cases import ConflictEdgeCaseHandler

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
# MASTER CONFLICT INTEGRATION
# ===============================================================================

class MasterConflictIntegration:
    """Master orchestrator for all conflict system modules"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize all subsystems
        self.slice_of_life = SliceOfLifeConflictManager(user_id, conversation_id)
        self.emergence_detector = EmergentConflictDetector(user_id, conversation_id)
        self.enhanced_integration = EnhancedConflictSystemIntegration(user_id, conversation_id)
        self.tension_system = TensionSystem(user_id, conversation_id)
        self.social_circle = SocialCircleConflictManager(user_id, conversation_id)
        self.leverage_system = LeverageSystem(user_id, conversation_id)
        self.multi_party = MultiPartyConflictOrchestrator(user_id, conversation_id)
        self.flow_manager = ConflictFlowManager(user_id, conversation_id)
        self.stakeholder_system = AutonomousStakeholderSystem(user_id, conversation_id)
        self.background_manager = BackgroundConflictManager(user_id, conversation_id)
        self.victory_manager = ConflictVictoryManager(user_id, conversation_id)
        self.canon_manager = ConflictCanonManager(user_id, conversation_id)
        self.synthesizer = ConflictSynthesizer(user_id, conversation_id)
        self.template_system = DynamicConflictTemplateSystem(user_id, conversation_id)
        self.edge_handler = ConflictEdgeCaseHandler(user_id, conversation_id)
        
        # Integration mode
        self.current_mode = IntegrationMode.EMERGENT
        
        # Lazy-loaded agents
        self._master_orchestrator = None
        self._experience_designer = None
        self._coherence_keeper = None
        self._engagement_monitor = None
        self._system_balancer = None
    
    # ========== Agent Properties ==========
    
    @property
    def master_orchestrator(self) -> Agent:
        if self._master_orchestrator is None:
            self._master_orchestrator = Agent(
                name="Master Conflict Orchestrator",
                instructions="""
                Orchestrate all conflict systems into a unified experience.
                
                Coordinate:
                - Which systems to activate when
                - How systems interact
                - Overall narrative flow
                - Complexity management
                - Player experience optimization
                
                Create seamless, engaging conflict experiences that feel natural.
                Balance depth with accessibility, drama with playability.
                """,
                model="gpt-5-nano",
            )
        return self._master_orchestrator
    
    @property
    def experience_designer(self) -> Agent:
        if self._experience_designer is None:
            self._experience_designer = Agent(
                name="Conflict Experience Designer",
                instructions="""
                Design the overall conflict experience for players.
                
                Focus on:
                - Emotional journey
                - Dramatic pacing
                - Meaningful choices
                - Satisfying resolutions
                - Memorable moments
                
                Turn system interactions into compelling experiences.
                """,
                model="gpt-5-nano",
            )
        return self._experience_designer
    
    @property
    def coherence_keeper(self) -> Agent:
        if self._coherence_keeper is None:
            self._coherence_keeper = Agent(
                name="Narrative Coherence Keeper",
                instructions="""
                Maintain narrative coherence across all conflict systems.
                
                Ensure:
                - Consistent characterization
                - Logical progression
                - Timeline integrity
                - Thematic unity
                - World consistency
                
                Keep the story making sense no matter how complex it gets.
                """,
                model="gpt-5-nano",
            )
        return self._coherence_keeper
    
    @property
    def engagement_monitor(self) -> Agent:
        if self._engagement_monitor is None:
            self._engagement_monitor = Agent(
                name="Player Engagement Monitor",
                instructions="""
                Monitor and optimize player engagement with conflicts.
                
                Track:
                - Player interest signals
                - Choice patterns
                - Engagement drops
                - Confusion indicators
                - Satisfaction markers
                
                Adjust the experience to keep players invested.
                """,
                model="gpt-5-nano",
            )
        return self._engagement_monitor
    
    @property
    def system_balancer(self) -> Agent:
        if self._system_balancer is None:
            self._system_balancer = Agent(
                name="System Load Balancer",
                instructions="""
                Balance the load across conflict systems.
                
                Manage:
                - System activation
                - Resource allocation
                - Complexity distribution
                - Performance optimization
                - Graceful degradation
                
                Keep everything running smoothly without overwhelming anything.
                """,
                model="gpt-5-nano",
            )
        return self._system_balancer
    
    # ========== Master Orchestration ==========
    
    async def process_scene(
        self,
        scene_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a scene through all relevant conflict systems"""
        
        # Determine which systems should be active
        active_systems = await self._determine_active_systems(scene_context)
        
        # Check system health first
        health_check = await self.edge_handler.scan_for_edge_cases()
        if health_check:
            await self._handle_edge_cases(health_check)
        
        # Process through active systems
        results = {}
        
        if 'emergence' in active_systems:
            tensions = await self.emergence_detector.detect_brewing_tensions()
            results['emerging_tensions'] = tensions
        
        if 'tension' in active_systems:
            tension_state = await self.tension_system.get_current_tension_state()
            results['tension'] = tension_state
        
        if 'social' in active_systems:
            social_dynamics = await self.social_circle.process_social_dynamics(
                scene_context.get('present_npcs', [])
            )
            results['social'] = social_dynamics
        
        if 'background' in active_systems:
            background = await self.background_manager.get_conversation_topics()
            results['background_flavor'] = background
        
        if 'flow' in active_systems:
            flow_state = await self.flow_manager.get_current_flow_state()
            results['flow'] = flow_state
        
        # Check for victories
        victory_checks = await self._check_all_victories()
        if victory_checks:
            results['victories'] = victory_checks
        
        # Synthesize if needed
        synthesis = await self._check_for_synthesis()
        if synthesis:
            results['synthesis'] = synthesis
        
        # Generate integrated response
        integrated = await self._integrate_results(results, scene_context)
        
        return integrated
    
    async def _determine_active_systems(
        self,
        context: Dict[str, Any]
    ) -> Set[str]:
        """Determine which systems should be active"""
        
        prompt = f"""
        Determine which conflict systems to activate:
        
        Context: {json.dumps(context)}
        Current Mode: {self.current_mode.value}
        Active Conflicts: {await self._get_conflict_count()}
        
        Available systems:
        - emergence: Detect new tensions
        - tension: Manage tension levels
        - social: Social dynamics
        - leverage: Power dynamics
        - multiparty: Complex stakeholder interactions
        - flow: Pacing and rhythm
        - background: World events
        - victory: Check victory conditions
        - synthesis: Combine conflicts
        
        Return JSON:
        {{
            "active_systems": ["system names to activate"],
            "reasoning": "Why these systems",
            "priority_system": "Most important system",
            "integration_strategy": "How to combine them"
        }}
        """
        
        response = await Runner.run(self.master_orchestrator, prompt)
        data = json.loads(response.output)
        return set(data['active_systems'])
    
    async def _integrate_results(
        self,
        results: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Integrate results from all systems"""
        
        prompt = f"""
        Integrate conflict system results into unified experience:
        
        Results: {json.dumps(results)}
        Context: {json.dumps(context)}
        Mode: {self.current_mode.value}
        
        Create integrated response that:
        - Combines all active elements
        - Maintains narrative coherence
        - Provides clear player choices
        - Sets appropriate mood
        - Advances the story
        
        Return JSON:
        {{
            "narrative_summary": "What's happening",
            "active_conflicts": ["current conflicts"],
            "player_choices": [
                {{
                    "id": "choice_id",
                    "text": "Choice text",
                    "systems_affected": ["which systems this affects"],
                    "potential_outcomes": ["possible results"]
                }}
            ],
            "atmospheric_elements": ["mood and tone elements"],
            "npc_behaviors": {{
                "npc_id": "behavior influenced by conflicts"
            }},
            "world_state_changes": ["how the world changed"],
            "next_beat": "What comes next"
        }}
        """
        
        response = await Runner.run(self.experience_designer, prompt)
        return json.loads(response.output)
    
    # ========== Mode Management ==========
    
    async def set_integration_mode(
        self,
        mode: IntegrationMode,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Change the integration mode"""
        
        self.current_mode = mode
        
        # Adjust systems based on mode
        adjustments = await self._calculate_mode_adjustments(mode, context)
        
        # Apply adjustments
        await self._apply_mode_adjustments(adjustments)
        
        return {
            'mode_changed': True,
            'new_mode': mode.value,
            'adjustments': adjustments
        }
    
    async def _calculate_mode_adjustments(
        self,
        mode: IntegrationMode,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate adjustments for mode change"""
        
        prompt = f"""
        Calculate system adjustments for mode change:
        
        New Mode: {mode.value}
        Context: {json.dumps(context)}
        
        Modes:
        - full_immersion: All systems active, maximum complexity
        - story_focus: Prioritize narrative coherence
        - social_dynamics: Emphasize relationships
        - background_aware: World events prominent
        - player_centric: Maximum player agency
        - emergent: Natural pattern emergence
        - guided: Active narrative guidance
        
        Return JSON:
        {{
            "system_weights": {{
                "system_name": 0.0 to 1.0 priority
            }},
            "complexity_target": 0.0 to 1.0,
            "pacing_preference": "slow/medium/fast",
            "narrative_style": "descriptive style",
            "player_agency_level": "low/medium/high",
            "background_prominence": 0.0 to 1.0
        }}
        """
        
        response = await Runner.run(self.system_balancer, prompt)
        return json.loads(response.output)
    
    async def _apply_mode_adjustments(
        self,
        adjustments: Dict[str, Any]
    ):
        """Apply mode adjustments to systems"""
        
        # Store adjustments in database
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO integration_settings
                (user_id, conversation_id, mode, adjustments, updated_at)
                VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, conversation_id)
                DO UPDATE SET mode = $3, adjustments = $4, updated_at = CURRENT_TIMESTAMP
            """, self.user_id, self.conversation_id,
            self.current_mode.value, json.dumps(adjustments))
    
    # ========== Health Monitoring ==========
    
    async def get_system_health(self) -> IntegrationState:
        """Get current health of integrated system"""
        
        # Get metrics
        conflict_count = await self._get_conflict_count()
        complexity = await self._calculate_complexity()
        coherence = await self._assess_coherence()
        engagement = await self._measure_engagement()
        health = await self._calculate_system_health()
        
        # Determine active modules
        active_modules = await self._get_active_modules()
        
        return IntegrationState(
            mode=self.current_mode,
            active_modules=active_modules,
            conflict_count=conflict_count,
            complexity_score=complexity,
            narrative_coherence=coherence,
            player_engagement=engagement,
            system_health=health
        )
    
    async def _get_conflict_count(self) -> int:
        """Get count of active conflicts"""
        
        async with get_db_connection_context() as conn:
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM Conflicts
                WHERE user_id = $1 AND conversation_id = $2
                AND is_active = true
            """, self.user_id, self.conversation_id)
        
        return count
    
    async def _calculate_complexity(self) -> float:
        """Calculate overall system complexity"""
        
        conflict_count = await self._get_conflict_count()
        
        # Get synthesis count
        async with get_db_connection_context() as conn:
            synthesis_count = await conn.fetchval("""
                SELECT COUNT(*) FROM conflict_synthesis
                WHERE primary_conflict_id IN (
                    SELECT conflict_id FROM Conflicts
                    WHERE user_id = $1 AND conversation_id = $2
                    AND is_active = true
                )
            """, self.user_id, self.conversation_id)
        
        # Calculate complexity (0-1)
        base_complexity = min(1.0, conflict_count / 10)
        synthesis_modifier = synthesis_count * 0.1
        
        return min(1.0, base_complexity + synthesis_modifier)
    
    async def _assess_coherence(self) -> float:
        """Assess narrative coherence"""
        
        # Check for contradictions
        edge_cases = await self.edge_handler.scan_for_edge_cases()
        contradiction_count = sum(
            1 for case in edge_cases
            if case.case_type.value == 'contradiction'
        )
        
        # Calculate coherence (1.0 = perfect, 0.0 = broken)
        coherence = max(0.0, 1.0 - (contradiction_count * 0.2))
        
        return coherence
    
    async def _measure_engagement(self) -> float:
        """Measure player engagement"""
        
        # Get recent player choices
        async with get_db_connection_context() as conn:
            recent_choices = await conn.fetchval("""
                SELECT COUNT(*) FROM player_choices
                WHERE user_id = $1 AND conversation_id = $2
                AND created_at > CURRENT_TIMESTAMP - INTERVAL '1 hour'
            """, self.user_id, self.conversation_id)
        
        # Simple engagement metric
        engagement = min(1.0, recent_choices / 10)
        
        return engagement
    
    async def _calculate_system_health(self) -> float:
        """Calculate overall system health"""
        
        complexity = await self._calculate_complexity()
        coherence = await self._assess_coherence()
        engagement = await self._measure_engagement()
        
        # Weight factors
        health = (
            coherence * 0.4 +  # Most important
            engagement * 0.3 +
            (1.0 - complexity) * 0.3  # Lower complexity is healthier
        )
        
        return health
    
    async def _get_active_modules(self) -> Set[str]:
        """Get currently active modules"""
        
        active = set()
        
        # Check which modules have recent activity
        async with get_db_connection_context() as conn:
            # Check for recent tensions
            tensions = await conn.fetchval("""
                SELECT COUNT(*) FROM tension_events
                WHERE user_id = $1 AND conversation_id = $2
                AND created_at > CURRENT_TIMESTAMP - INTERVAL '1 hour'
            """, self.user_id, self.conversation_id)
            if tensions > 0:
                active.add('tension')
            
            # Check for social dynamics
            social = await conn.fetchval("""
                SELECT COUNT(*) FROM social_dynamics
                WHERE user_id = $1 AND conversation_id = $2
                AND updated_at > CURRENT_TIMESTAMP - INTERVAL '1 hour'
            """, self.user_id, self.conversation_id)
            if social > 0:
                active.add('social')
            
            # Always include edge handler
            active.add('edge_handler')
        
        return active
    
    # ========== Utility Methods ==========
    
    async def _check_all_victories(self) -> Optional[List[Dict[str, Any]]]:
        """Check all conflicts for victory conditions"""
        
        victories = []
        
        async with get_db_connection_context() as conn:
            active_conflicts = await conn.fetch("""
                SELECT conflict_id FROM Conflicts
                WHERE user_id = $1 AND conversation_id = $2
                AND is_active = true
            """, self.user_id, self.conversation_id)
        
        for conflict in active_conflicts:
            # Get current state
            state = {'conflict_id': conflict['conflict_id']}  # Simplified
            achievements = await self.victory_manager.check_victory_conditions(
                conflict['conflict_id'],
                state
            )
            
            if achievements:
                victories.extend(achievements)
        
        return victories if victories else None
    
    async def _check_for_synthesis(self) -> Optional[Dict[str, Any]]:
        """Check if conflicts should be synthesized"""
        
        conflict_count = await self._get_conflict_count()
        
        if conflict_count >= 3:
            # Get conflicts
            async with get_db_connection_context() as conn:
                conflicts = await conn.fetch("""
                    SELECT conflict_id FROM Conflicts
                    WHERE user_id = $1 AND conversation_id = $2
                    AND is_active = true
                    LIMIT 5
                """, self.user_id, self.conversation_id)
            
            conflict_ids = [c['conflict_id'] for c in conflicts]
            
            # Analyze for synthesis
            analysis = await self.synthesizer.analyze_conflicts_for_synthesis(
                conflict_ids
            )
            
            if analysis['should_synthesize']:
                return analysis
        
        return None
    
    async def _handle_edge_cases(
        self,
        edge_cases: List[Any]
    ):
        """Handle detected edge cases"""
        
        for case in edge_cases:
            if case.severity > 0.7:  # High severity
                # Auto-recover
                await self.edge_handler.execute_recovery(case, 0)


# ===============================================================================
# MASTER INTEGRATION FUNCTIONS
# ===============================================================================

@function_tool
async def initialize_conflict_system(
    ctx: RunContextWrapper,
    mode: str = "emergent"
) -> Dict[str, Any]:
    """Initialize the master conflict system"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    master = MasterConflictIntegration(user_id, conversation_id)
    
    # Set initial mode
    await master.set_integration_mode(
        IntegrationMode(mode),
        {'initialization': True}
    )
    
    # Initialize background world
    await master.background_manager.orchestrator.generate_background_conflict()
    
    # Get initial state
    state = await master.get_system_health()
    
    return {
        'initialized': True,
        'mode': state.mode.value,
        'active_modules': list(state.active_modules),
        'system_health': state.system_health
    }


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
    
    master = MasterConflictIntegration(user_id, conversation_id)
    
    scene_context = {
        'activity': activity,
        'location': location,
        'present_npcs': present_npcs,
        'recent_events': recent_events,
        'timestamp': datetime.now().isoformat()
    }
    
    result = await master.process_scene(scene_context)
    
    return result


@function_tool
async def adjust_conflict_mode(
    ctx: RunContextWrapper,
    new_mode: str
) -> Dict[str, Any]:
    """Adjust the conflict system integration mode"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    master = MasterConflictIntegration(user_id, conversation_id)
    
    result = await master.set_integration_mode(
        IntegrationMode(new_mode),
        {'mode_change': True}
    )
    
    return result


@function_tool
async def get_conflict_system_status(
    ctx: RunContextWrapper
) -> Dict[str, Any]:
    """Get comprehensive status of conflict system"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    master = MasterConflictIntegration(user_id, conversation_id)
    
    # Get system health
    state = await master.get_system_health()
    
    # Check for issues
    edge_cases = await master.edge_handler.scan_for_edge_cases()
    
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
        'issues': [
            {
                'type': case.case_type.value,
                'severity': case.severity
            }
            for case in edge_cases
        ],
        'recommendation': 'healthy' if state.system_health > 0.7 else 'needs_attention'
    }


@function_tool
async def optimize_conflict_experience(
    ctx: RunContextWrapper
) -> Dict[str, Any]:
    """Optimize the conflict experience based on current state"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    master = MasterConflictIntegration(user_id, conversation_id)
    
    # Get current state
    state = await master.get_system_health()
    
    # Determine optimizations needed
    optimizations = []
    
    if state.complexity_score > 0.8:
        # Too complex - simplify
        optimizations.append('reduce_complexity')
        # Auto-resolve some conflicts
        
    if state.narrative_coherence < 0.5:
        # Poor coherence - fix contradictions
        optimizations.append('improve_coherence')
        edge_cases = await master.edge_handler.scan_for_edge_cases()
        for case in edge_cases:
            await master.edge_handler.execute_recovery(case, 0)
    
    if state.player_engagement < 0.3:
        # Low engagement - add hooks
        optimizations.append('boost_engagement')
        # Generate new interesting conflict
        await master.template_system.generate_conflict_from_template(
            1,  # Use first template
            {'boost_engagement': True}
        )
    
    return {
        'optimizations_performed': optimizations,
        'new_health': (await master.get_system_health()).system_health,
        'recommendations': [
            'Continue monitoring' if state.system_health > 0.7
            else 'Consider mode change'
        ]
    }
