# logic/conflict_system/conflict_synthesizer.py
"""
Conflict Synthesizer - Central Orchestration System

This module acts as the central orchestrator for all conflict-related operations,
ensuring all conflict subsystems work together in harmony. It doesn't merge conflicts
narratively, but rather synthesizes the various conflict modules into a cohesive system.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

from agents import Agent, function_tool, ModelSettings, RunContextWrapper, Runner
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ===============================================================================
# ORCHESTRATION TYPES
# ===============================================================================

class OrchestrationMode(Enum):
    """How conflict modules are orchestrated"""
    SEQUENTIAL = "sequential"  # Process through modules one by one
    PARALLEL = "parallel"  # Process through modules simultaneously
    SELECTIVE = "selective"  # Only use specific modules
    ADAPTIVE = "adaptive"  # Dynamically choose based on context
    PRIORITY = "priority"  # Process by priority order


class ConflictPhase(Enum):
    """Unified conflict phases across all modules"""
    BREWING = "brewing"
    ACTIVE = "active"
    CLIMAX = "climax"
    RESOLUTION = "resolution"
    AFTERMATH = "aftermath"


@dataclass
class ConflictState:
    """Unified state representation across all modules"""
    conflict_id: int
    phase: ConflictPhase
    intensity: float
    complexity: float
    stakeholder_states: Dict[int, Any]
    resolution_paths: List[Dict[str, Any]]
    active_events: List[Dict[str, Any]]
    module_states: Dict[str, Any]


@dataclass
class OrchestrationResult:
    """Result of orchestrating conflict operations"""
    success: bool
    mode: OrchestrationMode
    modules_engaged: List[str]
    primary_result: Dict[str, Any]
    module_results: Dict[str, Any]
    cascade_effects: List[Dict[str, Any]]
    state_changes: Dict[str, Any]
    narrative_impact: float


# ===============================================================================
# CONFLICT SYNTHESIZER - CENTRAL ORCHESTRATOR
# ===============================================================================

class ConflictSynthesizer:
    """
    Central orchestrator for all conflict subsystems.
    Ensures all conflict modules work together seamlessly.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Conflict subsystems (lazy loaded)
        self._subsystems = {}
        self._subsystem_loaders = {
            'resolution': self._load_resolution_system,
            'generation': self._load_generation_system,
            'stakeholders': self._load_stakeholder_system,
            'integration': self._load_integration_system,
            'analysis': self._load_analysis_system,
            'rewards': self._load_rewards_system
        }
        
        # Orchestration state
        self.active_conflicts = {}
        self.orchestration_history = []
        self.module_performance = defaultdict(lambda: {'calls': 0, 'successes': 0, 'failures': 0})
        
        # Coordination agents (lazy loaded)
        self._orchestration_director = None
        self._state_coordinator = None
        self._complexity_monitor = None
        self._cascade_analyzer = None
        
        # Cache for performance
        self._state_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
    # ========== Subsystem Loading ==========
    
    async def _load_resolution_system(self):
        """Load conflict resolution system"""
        from logic.conflict_system.conflict_resolution import ConflictResolutionSystem
        system = ConflictResolutionSystem(self.user_id, self.conversation_id)
        await system.initialize()
        return system
    
    async def _load_generation_system(self):
        """Load conflict generation system"""
        from logic.conflict_system.enhanced_conflict_generation import EnhancedConflictGenerator
        return EnhancedConflictGenerator(self.user_id, self.conversation_id)
    
    async def _load_stakeholder_system(self):
        """Load stakeholder autonomy system"""
        from logic.conflict_system.dynamic_stakeholder_agents import StakeholderAutonomySystem
        return StakeholderAutonomySystem(self.user_id, self.conversation_id)
    
    async def _load_integration_system(self):
        """Load conflict integration system"""
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        system = ConflictSystemIntegration(self.user_id, self.conversation_id)
        await system.initialize()
        return system
    
    async def _load_analysis_system(self):
        """Load conflict analysis system"""
        # This would load your conflict analysis agents
        from logic.conflict_system.conflict_agents import initialize_conflict_agents
        return await initialize_conflict_agents(self.user_id, self.conversation_id)
    
    async def _load_rewards_system(self):
        """Load player rewards system"""
        # This manages rewards from conflicts
        return {
            'calculate_rewards': self._calculate_conflict_rewards,
            'grant_rewards': self._grant_conflict_rewards
        }
    
    async def get_subsystem(self, name: str):
        """Get or load a subsystem"""
        if name not in self._subsystems:
            if name in self._subsystem_loaders:
                self._subsystems[name] = await self._subsystem_loaders[name]()
            else:
                raise ValueError(f"Unknown subsystem: {name}")
        return self._subsystems[name]
    
    # ========== Agent Properties ==========
    
    @property
    def orchestration_director(self) -> Agent:
        """Agent that directs orchestration strategy"""
        if self._orchestration_director is None:
            self._orchestration_director = Agent(
                name="Conflict Orchestration Director",
                instructions="""
                You orchestrate how different conflict subsystems work together.
                
                Your responsibilities:
                - Determine which modules to engage for each operation
                - Decide processing order (sequential/parallel/priority)
                - Identify dependencies between modules
                - Optimize for narrative coherence and performance
                - Prevent module conflicts and redundancies
                
                Consider the current game state and active conflicts when making decisions.
                """,
                model="gpt-5-nano",
            )
        return self._orchestration_director
    
    @property
    def state_coordinator(self) -> Agent:
        """Agent that maintains consistent state across modules"""
        if self._state_coordinator is None:
            self._state_coordinator = Agent(
                name="Conflict State Coordinator",
                instructions="""
                You maintain consistent state across all conflict modules.
                
                Your responsibilities:
                - Reconcile conflicting state updates from different modules
                - Ensure state consistency across subsystems
                - Track state transitions and validate them
                - Detect and resolve state conflicts
                - Maintain state history for rollback if needed
                
                Prioritize data consistency and narrative coherence.
                """,
                model="gpt-5-nano",
            )
        return self._state_coordinator
    
    @property
    def complexity_monitor(self) -> Agent:
        """Agent that monitors and manages system complexity"""
        if self._complexity_monitor is None:
            self._complexity_monitor = Agent(
                name="Complexity Monitor",
                instructions="""
                You monitor the complexity of the conflict system.
                
                Track:
                - Number of active conflicts
                - Interconnections between conflicts
                - Cognitive load on the player
                - System resource usage
                - Narrative complexity
                
                Alert when complexity becomes unmanageable and suggest simplifications.
                """,
                model="gpt-5-nano",
            )
        return self._complexity_monitor
    
    @property
    def cascade_analyzer(self) -> Agent:
        """Agent that analyzes cascade effects across modules"""
        if self._cascade_analyzer is None:
            self._cascade_analyzer = Agent(
                name="Cascade Effect Analyzer",
                instructions="""
                You analyze how actions in one module cascade to others.
                
                Identify:
                - Direct effects of module operations
                - Secondary effects across subsystems
                - Potential feedback loops
                - Unintended consequences
                - Opportunities for emergent gameplay
                
                Help prevent negative cascades while enabling interesting ones.
                """,
                model="gpt-5-nano",
            )
        return self._cascade_analyzer
    
    # ========== Core Orchestration Methods ==========
    
    async def create_conflict(self, conflict_data: Dict[str, Any]) -> OrchestrationResult:
        """
        Create a new conflict through orchestrated subsystems.
        
        This coordinates:
        1. Generation system for enhanced conflict data
        2. Resolution system for conflict structure
        3. Stakeholder system for NPC involvement
        4. Integration system for world state
        """
        try:
            # Determine orchestration strategy
            strategy = await self._determine_orchestration_strategy('create', conflict_data)
            
            module_results = {}
            cascade_effects = []
            
            # Phase 1: Generate enhanced conflict
            if 'generation' in strategy['modules']:
                generation = await self.get_subsystem('generation')
                enhanced_data = await generation.generate_conflict(conflict_data)
                module_results['generation'] = enhanced_data
                conflict_data.update(enhanced_data)
            
            # Phase 2: Create conflict structure
            if 'resolution' in strategy['modules']:
                resolution = await self.get_subsystem('resolution')
                conflict_id = await resolution.create_conflict(conflict_data)
                module_results['resolution'] = {'conflict_id': conflict_id}
            else:
                # Fallback to direct DB creation
                conflict_id = await self._create_basic_conflict(conflict_data)
            
            # Phase 3: Initialize stakeholders (parallel with phase 4)
            tasks = []
            
            if 'stakeholders' in strategy['modules']:
                stakeholders = await self.get_subsystem('stakeholders')
                tasks.append(stakeholders.initialize_stakeholders(
                    conflict_id, 
                    conflict_data.get('stakeholders', [])
                ))
            
            # Phase 4: Integrate with world state
            if 'integration' in strategy['modules']:
                integration = await self.get_subsystem('integration')
                tasks.append(integration.register_conflict(conflict_id, conflict_data))
            
            # Execute parallel tasks
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Subsystem error: {result}")
                    else:
                        module_results[f'parallel_{i}'] = result
            
            # Analyze cascade effects
            cascade_effects = await self._analyze_cascade_effects('create', conflict_id, module_results)
            
            # Cache the conflict state
            state = await self._get_unified_conflict_state(conflict_id)
            self.active_conflicts[conflict_id] = state
            
            # Update performance metrics
            self._update_performance_metrics(strategy['modules'], success=True)
            
            return OrchestrationResult(
                success=True,
                mode=OrchestrationMode(strategy['mode']),
                modules_engaged=strategy['modules'],
                primary_result={'conflict_id': conflict_id},
                module_results=module_results,
                cascade_effects=cascade_effects,
                state_changes={'created': conflict_id},
                narrative_impact=self._calculate_narrative_impact(module_results)
            )
            
        except Exception as e:
            logger.error(f"Error in orchestrated conflict creation: {e}")
            self._update_performance_metrics(strategy.get('modules', []), success=False)
            
            return OrchestrationResult(
                success=False,
                mode=OrchestrationMode.SEQUENTIAL,
                modules_engaged=[],
                primary_result={'error': str(e)},
                module_results={},
                cascade_effects=[],
                state_changes={},
                narrative_impact=0.0
            )
    
    async def resolve_conflict(
        self,
        conflict_id: int,
        resolution_data: Dict[str, Any]
    ) -> OrchestrationResult:
        """
        Resolve a conflict through orchestrated subsystems.
        
        Coordinates:
        1. Resolution system for outcome
        2. Stakeholder system for NPC reactions
        3. Rewards system for player rewards
        4. Integration system for world state updates
        """
        try:
            strategy = await self._determine_orchestration_strategy('resolve', {
                'conflict_id': conflict_id,
                'resolution_data': resolution_data
            })
            
            module_results = {}
            
            # Phase 1: Execute resolution
            if 'resolution' in strategy['modules']:
                resolution = await self.get_subsystem('resolution')
                result = await resolution.resolve_conflict(conflict_id, resolution_data)
                module_results['resolution'] = result
            
            # Phase 2: Update stakeholders
            if 'stakeholders' in strategy['modules']:
                stakeholders = await self.get_subsystem('stakeholders')
                reactions = await stakeholders.process_resolution(conflict_id, resolution_data)
                module_results['stakeholders'] = reactions
            
            # Phase 3: Calculate and grant rewards
            if 'rewards' in strategy['modules']:
                rewards = await self.get_subsystem('rewards')
                player_rewards = await rewards['calculate_rewards'](conflict_id, resolution_data)
                await rewards['grant_rewards'](self.user_id, self.conversation_id, player_rewards)
                module_results['rewards'] = player_rewards
            
            # Phase 4: Update world state
            if 'integration' in strategy['modules']:
                integration = await self.get_subsystem('integration')
                world_updates = await integration.handle_resolution(conflict_id, resolution_data)
                module_results['integration'] = world_updates
            
            # Analyze cascade effects
            cascade_effects = await self._analyze_cascade_effects('resolve', conflict_id, module_results)
            
            # Update conflict state
            if conflict_id in self.active_conflicts:
                self.active_conflicts[conflict_id].phase = ConflictPhase.AFTERMATH
            
            return OrchestrationResult(
                success=True,
                mode=OrchestrationMode(strategy['mode']),
                modules_engaged=strategy['modules'],
                primary_result={'resolved': True, 'conflict_id': conflict_id},
                module_results=module_results,
                cascade_effects=cascade_effects,
                state_changes={'resolved': conflict_id},
                narrative_impact=self._calculate_narrative_impact(module_results)
            )
            
        except Exception as e:
            logger.error(f"Error in orchestrated conflict resolution: {e}")
            return OrchestrationResult(
                success=False,
                mode=OrchestrationMode.SEQUENTIAL,
                modules_engaged=[],
                primary_result={'error': str(e)},
                module_results={},
                cascade_effects=[],
                state_changes={},
                narrative_impact=0.0
            )
    
    async def process_event(
        self,
        conflict_id: int,
        event: Dict[str, Any]
    ) -> OrchestrationResult:
        """
        Process an event through relevant conflict subsystems.
        
        Routes events to appropriate modules based on type and context.
        """
        try:
            event_type = event.get('type', 'unknown')
            
            # Determine which modules should handle this event
            strategy = await self._determine_orchestration_strategy('event', {
                'conflict_id': conflict_id,
                'event': event
            })
            
            module_results = {}
            
            # Route to appropriate modules
            for module_name in strategy['modules']:
                try:
                    module = await self.get_subsystem(module_name)
                    
                    # Each module has a process_event method
                    if hasattr(module, 'process_event'):
                        result = await module.process_event(conflict_id, event)
                        module_results[module_name] = result
                    elif hasattr(module, 'handle_event'):
                        result = await module.handle_event(conflict_id, event)
                        module_results[module_name] = result
                        
                except Exception as e:
                    logger.error(f"Module {module_name} failed to process event: {e}")
                    module_results[module_name] = {'error': str(e)}
            
            # Coordinate state updates across modules
            state_updates = await self._coordinate_state_updates(conflict_id, module_results)
            
            # Analyze cascade effects
            cascade_effects = await self._analyze_cascade_effects('event', conflict_id, module_results)
            
            return OrchestrationResult(
                success=True,
                mode=OrchestrationMode(strategy['mode']),
                modules_engaged=strategy['modules'],
                primary_result={'event_processed': True, 'type': event_type},
                module_results=module_results,
                cascade_effects=cascade_effects,
                state_changes=state_updates,
                narrative_impact=self._calculate_narrative_impact(module_results)
            )
            
        except Exception as e:
            logger.error(f"Error processing event through orchestration: {e}")
            return OrchestrationResult(
                success=False,
                mode=OrchestrationMode.SEQUENTIAL,
                modules_engaged=[],
                primary_result={'error': str(e)},
                module_results={},
                cascade_effects=[],
                state_changes={},
                narrative_impact=0.0
            )
    
    async def get_conflict_state(self, conflict_id: int) -> ConflictState:
        """Get unified state of a conflict across all modules"""
        
        # Check cache
        cache_key = f"state_{conflict_id}"
        if cache_key in self._state_cache:
            cached_time, cached_state = self._state_cache[cache_key]
            if (datetime.now() - cached_time).seconds < self._cache_ttl:
                return cached_state
        
        # Build unified state
        state = await self._get_unified_conflict_state(conflict_id)
        
        # Cache it
        self._state_cache[cache_key] = (datetime.now(), state)
        
        return state
    
    async def analyze_conflict_complexity(self) -> Dict[str, Any]:
        """Analyze overall complexity of active conflicts"""
        
        prompt = f"""
        Analyze the complexity of the current conflict system:
        
        Active Conflicts: {len(self.active_conflicts)}
        Conflict States: {json.dumps([c.phase.value for c in self.active_conflicts.values()])}
        
        Assess:
        - Overall complexity (0.0 to 1.0)
        - Player cognitive load
        - System resource usage
        - Narrative coherence
        - Recommendations for managing complexity
        
        Return JSON:
        {{
            "complexity_score": 0.0 to 1.0,
            "cognitive_load": "low/medium/high",
            "resource_usage": "low/medium/high",
            "narrative_coherence": 0.0 to 1.0,
            "recommendations": ["list of recommendations"],
            "warning_level": "none/low/medium/high"
        }}
        """
        
        response = await Runner.run(self.complexity_monitor, prompt)
        return json.loads(response.output)
    
    # ========== Internal Helper Methods ==========
    
    async def _determine_orchestration_strategy(
        self,
        operation: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine how to orchestrate modules for an operation"""
        
        prompt = f"""
        Determine orchestration strategy for operation: {operation}
        
        Context: {json.dumps(context)}
        Available Modules: {list(self._subsystem_loaders.keys())}
        
        Decide:
        - Which modules to engage
        - Processing mode (sequential/parallel/priority)
        - Dependencies between modules
        - Expected performance impact
        
        Return JSON:
        {{
            "modules": ["ordered list of modules to engage"],
            "mode": "sequential/parallel/priority/adaptive",
            "dependencies": {{"module": ["depends_on"]}},
            "estimated_time": "milliseconds",
            "priority_order": ["for priority mode"]
        }}
        """
        
        response = await Runner.run(self.orchestration_director, prompt)
        return json.loads(response.output)
    
    async def _coordinate_state_updates(
        self,
        conflict_id: int,
        module_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Coordinate state updates from multiple modules"""
        
        prompt = f"""
        Coordinate state updates from multiple conflict modules:
        
        Conflict ID: {conflict_id}
        Module Results: {json.dumps(module_results)}
        
        Reconcile any conflicts and produce unified state updates.
        
        Return JSON:
        {{
            "phase_change": "new phase if any",
            "intensity_change": -1.0 to 1.0,
            "progress_change": -100 to 100,
            "stakeholder_updates": {{}},
            "world_state_updates": {{}},
            "conflicts_resolved": ["how conflicts were resolved"]
        }}
        """
        
        response = await Runner.run(self.state_coordinator, prompt)
        updates = json.loads(response.output)
        
        # Apply updates to database
        if any([updates.get('phase_change'), updates.get('intensity_change'), updates.get('progress_change')]):
            async with get_db_connection_context() as conn:
                query_parts = []
                params = []
                param_count = 1
                
                if updates.get('phase_change'):
                    query_parts.append(f"phase = ${param_count}")
                    params.append(updates['phase_change'])
                    param_count += 1
                
                if updates.get('intensity_change'):
                    query_parts.append(f"intensity = intensity + ${param_count}")
                    params.append(updates['intensity_change'])
                    param_count += 1
                
                if updates.get('progress_change'):
                    query_parts.append(f"progress = progress + ${param_count}")
                    params.append(updates['progress_change'])
                    param_count += 1
                
                if query_parts:
                    params.append(conflict_id)
                    query = f"""
                        UPDATE Conflicts 
                        SET {', '.join(query_parts)}
                        WHERE conflict_id = ${param_count}
                    """
                    await conn.execute(query, *params)
        
        return updates
    
    async def _analyze_cascade_effects(
        self,
        operation: str,
        conflict_id: int,
        module_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze cascade effects across modules"""
        
        prompt = f"""
        Analyze cascade effects from conflict operation:
        
        Operation: {operation}
        Conflict ID: {conflict_id}
        Module Results: {json.dumps(module_results)}
        
        Identify:
        - Direct effects on other conflicts
        - Effects on NPCs not directly involved
        - World state changes
        - Player experience impacts
        - Potential future consequences
        
        Return JSON:
        {{
            "cascade_effects": [
                {{
                    "type": "conflict/npc/world/player",
                    "target": "entity affected",
                    "effect": "description of effect",
                    "magnitude": "low/medium/high",
                    "timing": "immediate/delayed",
                    "reversible": true/false
                }}
            ]
        }}
        """
        
        response = await Runner.run(self.cascade_analyzer, prompt)
        data = json.loads(response.output)
        return data.get('cascade_effects', [])
    
    async def _get_unified_conflict_state(self, conflict_id: int) -> ConflictState:
        """Build unified conflict state from all modules"""
        
        state_data = {}
        
        # Get base conflict data
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow("""
                SELECT * FROM Conflicts WHERE conflict_id = $1
            """, conflict_id)
            
            if not conflict:
                raise ValueError(f"Conflict {conflict_id} not found")
            
            # Get stakeholders
            stakeholders = await conn.fetch("""
                SELECT * FROM conflict_stakeholders WHERE conflict_id = $1
            """, conflict_id)
            
            # Get resolution paths
            paths = await conn.fetch("""
                SELECT * FROM conflict_resolution_paths WHERE conflict_id = $1
            """, conflict_id)
        
        # Build state object
        return ConflictState(
            conflict_id=conflict_id,
            phase=ConflictPhase(conflict['phase']),
            intensity=conflict['intensity'],
            complexity=conflict.get('complexity', 0.5),
            stakeholder_states={s['npc_id']: s for s in stakeholders},
            resolution_paths=[dict(p) for p in paths],
            active_events=[],  # Would be populated from event system
            module_states={}  # Would be populated from each module
        )
    
    async def _create_basic_conflict(self, conflict_data: Dict[str, Any]) -> int:
        """Fallback method to create basic conflict in database"""
        
        async with get_db_connection_context() as conn:
            conflict_id = await conn.fetchval("""
                INSERT INTO Conflicts (
                    user_id, conversation_id, conflict_name, conflict_type,
                    description, phase, intensity, progress, is_active
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING conflict_id
            """, 
            self.user_id, self.conversation_id,
            conflict_data.get('conflict_name', 'Unknown Conflict'),
            conflict_data.get('conflict_type', 'generic'),
            conflict_data.get('description', ''),
            'brewing', 0.5, 0.0, True)
        
        return conflict_id
    
    async def _calculate_conflict_rewards(
        self,
        conflict_id: int,
        resolution_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate rewards from conflict resolution"""
        
        # This would integrate with your rewards system
        return {
            'experience': 100,
            'items': [],
            'reputation_changes': {},
            'unlocks': []
        }
    
    async def _grant_conflict_rewards(
        self,
        user_id: int,
        conversation_id: int,
        rewards: Dict[str, Any]
    ) -> bool:
        """Grant calculated rewards to player"""
        
        # This would integrate with your inventory/player stats system
        return True
    
    def _calculate_narrative_impact(self, module_results: Dict[str, Any]) -> float:
        """Calculate the narrative impact of module results"""
        
        impact = 0.0
        
        # Weight different types of results
        weights = {
            'generation': 0.3,
            'resolution': 0.4,
            'stakeholders': 0.2,
            'integration': 0.1
        }
        
        for module, weight in weights.items():
            if module in module_results and not isinstance(module_results[module], dict):
                continue
            if module in module_results and 'narrative_impact' in module_results[module]:
                impact += module_results[module]['narrative_impact'] * weight
            elif module in module_results:
                # Estimate impact based on result size/complexity
                impact += min(len(str(module_results[module])) / 1000, 1.0) * weight * 0.5
        
        return min(impact, 1.0)
    
    def _update_performance_metrics(self, modules: List[str], success: bool):
        """Update performance metrics for modules"""
        
        for module in modules:
            self.module_performance[module]['calls'] += 1
            if success:
                self.module_performance[module]['successes'] += 1
            else:
                self.module_performance[module]['failures'] += 1
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report of the orchestration system"""
        
        report = {
            'active_conflicts': len(self.active_conflicts),
            'module_performance': dict(self.module_performance),
            'orchestration_history': len(self.orchestration_history),
            'cache_size': len(self._state_cache),
            'subsystems_loaded': list(self._subsystems.keys())
        }
        
        # Calculate success rates
        for module, stats in report['module_performance'].items():
            if stats['calls'] > 0:
                stats['success_rate'] = stats['successes'] / stats['calls']
            else:
                stats['success_rate'] = 0.0
        
        return report


# ===============================================================================
# INTEGRATION FUNCTIONS
# ===============================================================================

@function_tool
async def orchestrate_conflict_operation(
    ctx: RunContextWrapper,
    operation: str,
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main entry point for orchestrated conflict operations.
    
    Operations:
    - create: Create new conflict
    - resolve: Resolve existing conflict
    - event: Process conflict event
    - analyze: Analyze conflict state/complexity
    """
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    synthesizer = ConflictSynthesizer(user_id, conversation_id)
    
    if operation == 'create':
        result = await synthesizer.create_conflict(data)
    elif operation == 'resolve':
        conflict_id = data.get('conflict_id')
        resolution_data = data.get('resolution_data', {})
        result = await synthesizer.resolve_conflict(conflict_id, resolution_data)
    elif operation == 'event':
        conflict_id = data.get('conflict_id')
        event = data.get('event', {})
        result = await synthesizer.process_event(conflict_id, event)
    elif operation == 'analyze':
        result = await synthesizer.analyze_conflict_complexity()
    else:
        result = OrchestrationResult(
            success=False,
            mode=OrchestrationMode.SEQUENTIAL,
            modules_engaged=[],
            primary_result={'error': f'Unknown operation: {operation}'},
            module_results={},
            cascade_effects=[],
            state_changes={},
            narrative_impact=0.0
        )
    
    # Convert to dict for JSON serialization
    if isinstance(result, OrchestrationResult):
        return {
            'success': result.success,
            'mode': result.mode.value,
            'modules_engaged': result.modules_engaged,
            'primary_result': result.primary_result,
            'module_results': result.module_results,
            'cascade_effects': result.cascade_effects,
            'state_changes': result.state_changes,
            'narrative_impact': result.narrative_impact
        }
    
    return result


@function_tool
async def get_orchestration_status(ctx: RunContextWrapper) -> Dict[str, Any]:
    """Get status of the conflict orchestration system"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    synthesizer = ConflictSynthesizer(user_id, conversation_id)
    
    status = {
        'active_conflicts': len(synthesizer.active_conflicts),
        'loaded_subsystems': list(synthesizer._subsystems.keys()),
        'performance': await synthesizer.get_performance_report(),
        'complexity': await synthesizer.analyze_conflict_complexity()
    }
    
    return status
