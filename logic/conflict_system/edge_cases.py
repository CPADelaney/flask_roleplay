# logic/conflict_system/edge_cases.py
"""
Edge Case Handler with LLM-powered recovery and adaptation
Handles unusual, broken, or unexpected conflict situations gracefully
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

from agents import Agent, function_tool, ModelSettings, RunContextWrapper, Runner
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ===============================================================================
# EDGE CASE TYPES
# ===============================================================================

class EdgeCaseType(Enum):
    """Types of edge cases in conflict system"""
    ORPHANED_CONFLICT = "orphaned_conflict"  # Conflict with no stakeholders
    INFINITE_LOOP = "infinite_loop"  # Conflicts triggering each other
    CONTRADICTION = "contradiction"  # Mutually exclusive states
    STALE_CONFLICT = "stale_conflict"  # Conflict stuck too long
    COMPLEXITY_OVERLOAD = "complexity_overload"  # Too many simultaneous conflicts
    MISSING_CONTEXT = "missing_context"  # Required data missing
    PLAYER_DISCONNECT = "player_disconnect"  # Player not engaging
    NPC_UNAVAILABLE = "npc_unavailable"  # Key NPC removed/busy
    NARRATIVE_BREAK = "narrative_break"  # Story continuity broken
    SYSTEM_CONFLICT = "system_conflict"  # Different systems disagree


@dataclass
class EdgeCase:
    """Represents a detected edge case"""
    case_id: int
    case_type: EdgeCaseType
    affected_conflicts: List[int]
    severity: float  # 0-1
    description: str
    detection_context: Dict[str, Any]
    recovery_options: List[Dict[str, Any]]


# ===============================================================================
# EDGE CASE DETECTOR AND HANDLER
# ===============================================================================

class ConflictEdgeCaseHandler:
    """Detects and handles edge cases in conflict system"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Lazy-loaded agents
        self._anomaly_detector = None
        self._recovery_strategist = None
        self._narrative_healer = None
        self._graceful_degrader = None
        self._continuity_keeper = None
    
    # ========== Agent Properties ==========
    
    @property
    def anomaly_detector(self) -> Agent:
        if self._anomaly_detector is None:
            self._anomaly_detector = Agent(
                name="Anomaly Detector",
                instructions="""
                Detect unusual or problematic patterns in conflict systems.
                
                Identify:
                - Logical contradictions
                - Infinite loops
                - Orphaned elements
                - Stale progressions
                - Missing dependencies
                - Narrative breaks
                
                Focus on catching problems before they break the experience.
                """,
                model="gpt-5-nano",
            )
        return self._anomaly_detector
    
    @property
    def recovery_strategist(self) -> Agent:
        if self._recovery_strategist is None:
            self._recovery_strategist = Agent(
                name="Recovery Strategist",
                instructions="""
                Design recovery strategies for conflict system problems.
                
                Create strategies that:
                - Preserve narrative continuity
                - Minimize player disruption
                - Maintain conflict integrity
                - Enable graceful recovery
                - Prevent recurrence
                
                Turn problems into opportunities when possible.
                """,
                model="gpt-5-nano",
            )
        return self._recovery_strategist
    
    @property
    def narrative_healer(self) -> Agent:
        if self._narrative_healer is None:
            self._narrative_healer = Agent(
                name="Narrative Healer",
                instructions="""
                Heal narrative breaks and continuity issues.
                
                Create fixes that:
                - Explain inconsistencies
                - Bridge narrative gaps
                - Justify sudden changes
                - Maintain immersion
                - Feel intentional
                
                Make broken stories whole again.
                """,
                model="gpt-5-nano",
            )
        return self._narrative_healer
    
    @property
    def graceful_degrader(self) -> Agent:
        if self._graceful_degrader is None:
            self._graceful_degrader = Agent(
                name="Graceful Degradation Manager",
                instructions="""
                Manage graceful degradation when systems fail.
                
                Ensure:
                - Core experience preserved
                - Fallbacks feel natural
                - Complexity reduced smoothly
                - Player experience maintained
                - Recovery paths clear
                
                When things break, break beautifully.
                """,
                model="gpt-5-nano",
            )
        return self._graceful_degrader
    
    @property
    def continuity_keeper(self) -> Agent:
        if self._continuity_keeper is None:
            self._continuity_keeper = Agent(
                name="Continuity Keeper",
                instructions="""
                Maintain story continuity despite system issues.
                
                Preserve:
                - Character consistency
                - Timeline coherence
                - Relationship dynamics
                - World state logic
                - Player agency
                
                Keep the story making sense no matter what.
                """,
                model="gpt-5-nano",
            )
        return self._continuity_keeper
    
    # ========== Detection Methods ==========
    
    async def scan_for_edge_cases(self) -> List[EdgeCase]:
        """Scan system for edge cases"""
        
        edge_cases = []
        
        # Check for orphaned conflicts
        orphaned = await self._detect_orphaned_conflicts()
        edge_cases.extend(orphaned)
        
        # Check for infinite loops
        loops = await self._detect_infinite_loops()
        edge_cases.extend(loops)
        
        # Check for stale conflicts
        stale = await self._detect_stale_conflicts()
        edge_cases.extend(stale)
        
        # Check for complexity overload
        overload = await self._detect_complexity_overload()
        if overload:
            edge_cases.append(overload)
        
        # Check for narrative breaks
        breaks = await self._detect_narrative_breaks()
        edge_cases.extend(breaks)
        
        return edge_cases
    
    async def _detect_orphaned_conflicts(self) -> List[EdgeCase]:
        """Detect conflicts with no active stakeholders"""
        
        edge_cases = []
        
        async with get_db_connection_context() as conn:
            orphaned = await conn.fetch("""
                SELECT c.conflict_id, c.conflict_name, c.description
                FROM Conflicts c
                LEFT JOIN conflict_stakeholders cs ON c.conflict_id = cs.conflict_id
                WHERE c.user_id = $1 AND c.conversation_id = $2
                AND c.is_active = true
                GROUP BY c.conflict_id
                HAVING COUNT(cs.stakeholder_id) = 0
            """, self.user_id, self.conversation_id)
        
        for conflict in orphaned:
            recovery = await self._generate_orphan_recovery(conflict)
            
            edge_case = EdgeCase(
                case_id=await self._store_edge_case(
                    EdgeCaseType.ORPHANED_CONFLICT,
                    [conflict['conflict_id']],
                    0.7,
                    f"Conflict '{conflict['conflict_name']}' has no stakeholders"
                ),
                case_type=EdgeCaseType.ORPHANED_CONFLICT,
                affected_conflicts=[conflict['conflict_id']],
                severity=0.7,
                description=f"Orphaned conflict: {conflict['conflict_name']}",
                detection_context={'conflict': dict(conflict)},
                recovery_options=recovery
            )
            edge_cases.append(edge_case)
        
        return edge_cases
    
    async def _detect_infinite_loops(self) -> List[EdgeCase]:
        """Detect conflicts triggering each other infinitely"""
        
        edge_cases = []
        
        async with get_db_connection_context() as conn:
            # Get recent conflict events
            events = await conn.fetch("""
                SELECT conflict_id, triggered_by, event_type, created_at
                FROM conflict_events
                WHERE user_id = $1 AND conversation_id = $2
                AND created_at > NOW() - INTERVAL '1 hour'
                ORDER BY created_at DESC
                LIMIT 100
            """, self.user_id, self.conversation_id)
        
        # Analyze for circular triggers
        trigger_chains = {}
        for event in events:
            if event['triggered_by']:
                chain_key = f"{event['conflict_id']}-{event['triggered_by']}"
                if chain_key not in trigger_chains:
                    trigger_chains[chain_key] = 0
                trigger_chains[chain_key] += 1
        
        # Detect loops
        for chain, count in trigger_chains.items():
            if count > 5:  # More than 5 mutual triggers
                conflicts = [int(c) for c in chain.split('-')]
                
                recovery = await self._generate_loop_recovery(conflicts)
                
                edge_case = EdgeCase(
                    case_id=await self._store_edge_case(
                        EdgeCaseType.INFINITE_LOOP,
                        conflicts,
                        0.9,
                        f"Infinite loop detected between conflicts"
                    ),
                    case_type=EdgeCaseType.INFINITE_LOOP,
                    affected_conflicts=conflicts,
                    severity=0.9,
                    description="Conflicts triggering each other infinitely",
                    detection_context={'trigger_count': count},
                    recovery_options=recovery
                )
                edge_cases.append(edge_case)
        
        return edge_cases
    
    async def _detect_stale_conflicts(self) -> List[EdgeCase]:
        """Detect conflicts that haven't progressed"""
        
        edge_cases = []
        
        async with get_db_connection_context() as conn:
            stale = await conn.fetch("""
                SELECT conflict_id, conflict_name, phase, progress,
                       last_updated, CURRENT_TIMESTAMP - last_updated as stale_duration
                FROM Conflicts
                WHERE user_id = $1 AND conversation_id = $2
                AND is_active = true
                AND last_updated < CURRENT_TIMESTAMP - INTERVAL '7 days'
                AND progress < 100
            """, self.user_id, self.conversation_id)
        
        for conflict in stale:
            recovery = await self._generate_stale_recovery(conflict)
            
            edge_case = EdgeCase(
                case_id=await self._store_edge_case(
                    EdgeCaseType.STALE_CONFLICT,
                    [conflict['conflict_id']],
                    0.5,
                    f"Conflict stale for {conflict['stale_duration'].days} days"
                ),
                case_type=EdgeCaseType.STALE_CONFLICT,
                affected_conflicts=[conflict['conflict_id']],
                severity=min(0.9, conflict['stale_duration'].days / 14),
                description=f"Stale conflict: {conflict['conflict_name']}",
                detection_context={'stale_days': conflict['stale_duration'].days},
                recovery_options=recovery
            )
            edge_cases.append(edge_case)
        
        return edge_cases
    
    async def _detect_complexity_overload(self) -> Optional[EdgeCase]:
        """Detect if too many conflicts are active"""
        
        async with get_db_connection_context() as conn:
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM Conflicts
                WHERE user_id = $1 AND conversation_id = $2
                AND is_active = true
            """, self.user_id, self.conversation_id)
        
        if count > 10:  # More than 10 active conflicts
            conflicts = await conn.fetch("""
                SELECT conflict_id FROM Conflicts
                WHERE user_id = $1 AND conversation_id = $2
                AND is_active = true
            """, self.user_id, self.conversation_id)
            
            recovery = await self._generate_overload_recovery(count)
            
            return EdgeCase(
                case_id=await self._store_edge_case(
                    EdgeCaseType.COMPLEXITY_OVERLOAD,
                    [c['conflict_id'] for c in conflicts],
                    0.8,
                    f"{count} active conflicts causing overload"
                ),
                case_type=EdgeCaseType.COMPLEXITY_OVERLOAD,
                affected_conflicts=[c['conflict_id'] for c in conflicts],
                severity=min(1.0, count / 15),
                description=f"System overloaded with {count} active conflicts",
                detection_context={'active_count': count},
                recovery_options=recovery
            )
        
        return None
    
    async def _detect_narrative_breaks(self) -> List[EdgeCase]:
        """Detect narrative continuity breaks"""
        
        edge_cases = []
        
        # Detect conflicts with contradictory states
        async with get_db_connection_context() as conn:
            contradictions = await conn.fetch("""
                SELECT c1.conflict_id as conflict1, c2.conflict_id as conflict2,
                       c1.conflict_name as name1, c2.conflict_name as name2
                FROM Conflicts c1
                JOIN Conflicts c2 ON c1.user_id = c2.user_id
                WHERE c1.user_id = $1 AND c1.conversation_id = $2
                AND c1.is_active = true AND c2.is_active = true
                AND c1.conflict_id < c2.conflict_id
                AND EXISTS (
                    SELECT 1 FROM conflict_stakeholders cs1
                    JOIN conflict_stakeholders cs2 ON cs1.npc_id = cs2.npc_id
                    WHERE cs1.conflict_id = c1.conflict_id
                    AND cs2.conflict_id = c2.conflict_id
                    AND cs1.faction != cs2.faction
                )
            """, self.user_id, self.conversation_id)
        
        for contradiction in contradictions:
            recovery = await self._generate_contradiction_recovery(contradiction)
            
            edge_case = EdgeCase(
                case_id=await self._store_edge_case(
                    EdgeCaseType.CONTRADICTION,
                    [contradiction['conflict1'], contradiction['conflict2']],
                    0.6,
                    "Contradictory NPC positions in conflicts"
                ),
                case_type=EdgeCaseType.CONTRADICTION,
                affected_conflicts=[contradiction['conflict1'], contradiction['conflict2']],
                severity=0.6,
                description="NPC has contradictory roles in conflicts",
                detection_context={'conflicts': dict(contradiction)},
                recovery_options=recovery
            )
            edge_cases.append(edge_case)
        
        return edge_cases
    
    # ========== Recovery Generation ==========
    
    async def _generate_orphan_recovery(
        self,
        conflict: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recovery options for orphaned conflict"""
        
        prompt = f"""
        Generate recovery options for orphaned conflict:
        
        Conflict: {conflict['conflict_name']}
        Description: {conflict['description']}
        
        Create 3 recovery options:
        1. Graceful closure
        2. NPC assignment
        3. Player-centric pivot
        
        Return JSON:
        {{
            "options": [
                {{
                    "strategy": "close/assign/pivot",
                    "description": "How to recover",
                    "narrative": "Story explanation",
                    "implementation": ["steps to take"],
                    "risk": "low/medium/high"
                }}
            ]
        }}
        """
        
        response = await Runner.run(self.recovery_strategist, prompt)
        data = json.loads(response.output)
        return data['options']
    
    async def _generate_loop_recovery(
        self,
        conflicts: List[int]
    ) -> List[Dict[str, Any]]:
        """Generate recovery for infinite loop"""
        
        prompt = f"""
        Generate recovery for infinite conflict loop:
        
        Conflicts involved: {conflicts}
        
        Create solutions that:
        - Break the cycle
        - Preserve both conflicts if possible
        - Maintain narrative sense
        
        Return JSON:
        {{
            "options": [
                {{
                    "strategy": "break/merge/prioritize",
                    "description": "How to break loop",
                    "preserved_conflicts": [conflict_ids],
                    "narrative_bridge": "Story explanation",
                    "prevention": "How to prevent recurrence"
                }}
            ]
        }}
        """
        
        response = await Runner.run(self.recovery_strategist, prompt)
        data = json.loads(response.output)
        return data['options']
    
    async def _generate_stale_recovery(
        self,
        conflict: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recovery for stale conflict"""
        
        prompt = f"""
        Generate recovery for stale conflict:
        
        Conflict: {conflict['conflict_name']}
        Phase: {conflict['phase']}
        Progress: {conflict['progress']}%
        Stale for: {conflict['stale_duration'].days} days
        
        Create options to:
        - Revitalize with new development
        - Gracefully conclude
        - Transform into something new
        
        Return JSON:
        {{
            "options": [
                {{
                    "strategy": "revitalize/conclude/transform",
                    "description": "Recovery approach",
                    "narrative_event": "What happens in story",
                    "player_hook": "How to re-engage player",
                    "expected_outcome": "What this achieves"
                }}
            ]
        }}
        """
        
        response = await Runner.run(self.recovery_strategist, prompt)
        data = json.loads(response.output)
        return data['options']
    
    async def _generate_overload_recovery(
        self,
        count: int
    ) -> List[Dict[str, Any]]:
        """Generate recovery for complexity overload"""
        
        prompt = f"""
        Generate recovery for conflict overload:
        
        Active conflicts: {count}
        
        Create strategies to:
        - Reduce complexity gracefully
        - Merge related conflicts
        - Prioritize important conflicts
        - Create breathing room
        
        Return JSON:
        {{
            "options": [
                {{
                    "strategy": "consolidate/prioritize/pause/resolve",
                    "target_count": ideal number of conflicts,
                    "selection_criteria": "How to choose which conflicts",
                    "narrative_framing": "Story explanation",
                    "player_communication": "How to explain to player"
                }}
            ]
        }}
        """
        
        response = await Runner.run(self.graceful_degrader, prompt)
        data = json.loads(response.output)
        return data['options']
    
    async def _generate_contradiction_recovery(
        self,
        contradiction: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recovery for contradictory states"""
        
        prompt = f"""
        Generate recovery for contradictory NPC positions:
        
        Conflict 1: {contradiction['name1']}
        Conflict 2: {contradiction['name2']}
        
        NPC has contradictory roles in these conflicts.
        
        Create solutions that:
        - Explain the contradiction
        - Resolve the inconsistency
        - Maintain character integrity
        
        Return JSON:
        {{
            "options": [
                {{
                    "strategy": "explain/choose/split",
                    "narrative_explanation": "How to explain in story",
                    "character_development": "How this develops NPC",
                    "conflict_impact": {{
                        "conflict1": "impact on first conflict",
                        "conflict2": "impact on second conflict"
                    }}
                }}
            ]
        }}
        """
        
        response = await Runner.run(self.narrative_healer, prompt)
        data = json.loads(response.output)
        return data['options']
    
    # ========== Recovery Execution ==========
    
    async def execute_recovery(
        self,
        edge_case: EdgeCase,
        option_index: int = 0
    ) -> Dict[str, Any]:
        """Execute recovery strategy for edge case"""
        
        if option_index >= len(edge_case.recovery_options):
            return {'success': False, 'reason': 'Invalid recovery option'}
        
        recovery = edge_case.recovery_options[option_index]
        
        # Execute based on edge case type
        if edge_case.case_type == EdgeCaseType.ORPHANED_CONFLICT:
            result = await self._execute_orphan_recovery(
                edge_case.affected_conflicts[0],
                recovery
            )
        elif edge_case.case_type == EdgeCaseType.INFINITE_LOOP:
            result = await self._execute_loop_recovery(
                edge_case.affected_conflicts,
                recovery
            )
        elif edge_case.case_type == EdgeCaseType.STALE_CONFLICT:
            result = await self._execute_stale_recovery(
                edge_case.affected_conflicts[0],
                recovery
            )
        elif edge_case.case_type == EdgeCaseType.COMPLEXITY_OVERLOAD:
            result = await self._execute_overload_recovery(recovery)
        elif edge_case.case_type == EdgeCaseType.CONTRADICTION:
            result = await self._execute_contradiction_recovery(
                edge_case.affected_conflicts,
                recovery
            )
        else:
            result = {'success': False, 'reason': 'Unknown edge case type'}
        
        # Record recovery attempt
        await self._record_recovery(edge_case.case_id, recovery, result)
        
        return result
    
    async def _execute_orphan_recovery(
        self,
        conflict_id: int,
        recovery: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute recovery for orphaned conflict"""
        
        strategy = recovery['strategy']
        
        async with get_db_connection_context() as conn:
            if strategy == 'close':
                # Gracefully close the conflict
                await conn.execute("""
                    UPDATE Conflicts
                    SET is_active = false,
                        resolution_description = $1,
                        resolved_at = CURRENT_TIMESTAMP
                    WHERE conflict_id = $2
                """, recovery['narrative'], conflict_id)
                
                return {'success': True, 'action': 'closed_conflict'}
                
            elif strategy == 'assign':
                # Assign NPCs to the conflict
                # This would need actual NPC selection logic
                return {'success': True, 'action': 'assigned_npcs'}
                
            elif strategy == 'pivot':
                # Transform to player-centric conflict
                await conn.execute("""
                    UPDATE Conflicts
                    SET conflict_type = 'personal',
                        description = $1
                    WHERE conflict_id = $2
                """, recovery['narrative'], conflict_id)
                
                return {'success': True, 'action': 'pivoted_to_player'}
        
        return {'success': False}
    
    async def _execute_loop_recovery(
        self,
        conflicts: List[int],
        recovery: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute recovery for infinite loop"""
        
        strategy = recovery['strategy']
        
        async with get_db_connection_context() as conn:
            if strategy == 'break':
                # Break the trigger chain
                await conn.execute("""
                    DELETE FROM conflict_triggers
                    WHERE conflict_id = ANY($1)
                    AND triggered_conflict_id = ANY($1)
                """, conflicts)
                
                return {'success': True, 'action': 'broke_trigger_chain'}
                
            elif strategy == 'merge':
                # Merge conflicts into one
                # Keep first, deactivate others
                await conn.execute("""
                    UPDATE Conflicts
                    SET is_active = false
                    WHERE conflict_id = ANY($1[2:])
                """, conflicts)
                
                return {'success': True, 'action': 'merged_conflicts'}
        
        return {'success': False}
    
    async def _execute_stale_recovery(
        self,
        conflict_id: int,
        recovery: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute recovery for stale conflict"""
        
        # Generate narrative event
        narrative = recovery.get('narrative_event', 'The situation evolves unexpectedly')
        
        async with get_db_connection_context() as conn:
            # Progress the conflict
            await conn.execute("""
                UPDATE Conflicts
                SET progress = progress + 20,
                    phase = CASE 
                        WHEN progress + 20 >= 80 THEN 'resolution'
                        WHEN progress + 20 >= 60 THEN 'climax'
                        ELSE phase
                    END,
                    last_updated = CURRENT_TIMESTAMP
                WHERE conflict_id = $1
            """, conflict_id)
            
            # Add narrative event
            await conn.execute("""
                INSERT INTO conflict_events
                (conflict_id, event_type, description, created_at)
                VALUES ($1, 'recovery', $2, CURRENT_TIMESTAMP)
            """, conflict_id, narrative)
        
        return {'success': True, 'action': 'revitalized_conflict'}
    
    async def _execute_overload_recovery(
        self,
        recovery: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute recovery for complexity overload"""
        
        strategy = recovery['strategy']
        target_count = recovery.get('target_count', 5)
        
        async with get_db_connection_context() as conn:
            if strategy == 'prioritize':
                # Keep only highest intensity conflicts
                await conn.execute("""
                    UPDATE Conflicts
                    SET is_active = false
                    WHERE conflict_id IN (
                        SELECT conflict_id FROM Conflicts
                        WHERE user_id = $1 AND conversation_id = $2
                        AND is_active = true
                        ORDER BY 
                            CASE intensity
                                WHEN 'confrontation' THEN 5
                                WHEN 'opposition' THEN 4
                                WHEN 'friction' THEN 3
                                WHEN 'tension' THEN 2
                                ELSE 1
                            END ASC
                        OFFSET $3
                    )
                """, self.user_id, self.conversation_id, target_count)
                
                return {'success': True, 'action': 'prioritized_conflicts'}
        
        return {'success': False}
    
    async def _execute_contradiction_recovery(
        self,
        conflicts: List[int],
        recovery: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute recovery for contradictions"""
        
        # Create narrative explanation
        explanation = recovery.get('narrative_explanation', 'The situation clarifies')
        
        async with get_db_connection_context() as conn:
            # Add explanation event to both conflicts
            for conflict_id in conflicts:
                await conn.execute("""
                    INSERT INTO conflict_events
                    (conflict_id, event_type, description, created_at)
                    VALUES ($1, 'clarification', $2, CURRENT_TIMESTAMP)
                """, conflict_id, explanation)
        
        return {'success': True, 'action': 'explained_contradiction'}
    
    # ========== Storage Methods ==========
    
    async def _store_edge_case(
        self,
        case_type: EdgeCaseType,
        conflicts: List[int],
        severity: float,
        description: str
    ) -> int:
        """Store detected edge case"""
        
        async with get_db_connection_context() as conn:
            case_id = await conn.fetchval("""
                INSERT INTO conflict_edge_cases
                (case_type, affected_conflicts, severity, description, detected_at)
                VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                RETURNING case_id
            """, case_type.value, json.dumps(conflicts), severity, description)
        
        return case_id
    
    async def _record_recovery(
        self,
        case_id: int,
        recovery: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """Record recovery attempt"""
        
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO edge_case_recoveries
                (case_id, recovery_strategy, recovery_data, result, executed_at)
                VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
            """, case_id, recovery.get('strategy'), 
            json.dumps(recovery), json.dumps(result))


# ===============================================================================
# INTEGRATION FUNCTIONS
# ===============================================================================

@function_tool
async def scan_for_conflict_issues(
    ctx: RunContextWrapper
) -> Dict[str, Any]:
    """Scan for edge cases in conflict system"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    handler = ConflictEdgeCaseHandler(user_id, conversation_id)
    
    edge_cases = await handler.scan_for_edge_cases()
    
    return {
        'issues_found': len(edge_cases),
        'edge_cases': [
            {
                'type': case.case_type.value,
                'severity': case.severity,
                'description': case.description,
                'recovery_available': len(case.recovery_options) > 0
            }
            for case in edge_cases
        ]
    }


@function_tool
async def auto_recover_conflicts(
    ctx: RunContextWrapper
) -> Dict[str, Any]:
    """Automatically recover from detected edge cases"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    handler = ConflictEdgeCaseHandler(user_id, conversation_id)
    
    edge_cases = await handler.scan_for_edge_cases()
    
    recoveries = []
    for case in edge_cases:
        if case.recovery_options:
            # Execute first recovery option
            result = await handler.execute_recovery(case, 0)
            recoveries.append({
                'case_type': case.case_type.value,
                'success': result.get('success', False),
                'action': result.get('action', 'unknown')
            })
    
    return {
        'issues_found': len(edge_cases),
        'recoveries_attempted': len(recoveries),
        'recovery_results': recoveries
    }
