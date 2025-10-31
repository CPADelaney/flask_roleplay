# logic/conflict_system/edge_cases.py
"""
Edge Case Handler with LLM-powered recovery and adaptation
Integrated with ConflictSynthesizer as the central orchestrator

REFACTORED FOR PERFORMANCE:
- Expensive scanning operations are offloaded to a background Celery task.
- Scan results are cached in Redis to provide fast, non-blocking health checks.
- A Redis lock prevents redundant parallel scans.
- Individual detection methods run concurrently via asyncio.gather.
- Expensive LLM calls for recovery options are deferred until explicitly needed.
"""

import logging
import json
import random
import asyncio
import hashlib
import os
import redis.asyncio as redis
from typing import Dict, List, Any, Optional, Set, TypedDict
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from celery_config import celery_app

from agents import Agent, function_tool, RunContextWrapper
from nyx.tasks.background.conflict_edge_tasks import (
    RECOVERY_LOCK_TTL,
    dispatch_recovery_generation,
    recovery_cache_key,
    recovery_lock_key,
)
from db.connection import get_db_connection_context
logger = logging.getLogger(__name__)

# --- Configuration ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_redis_client: Optional[redis.Redis] = None

async def get_redis_client() -> redis.Redis:
    """Lazy-loads and returns a singleton async Redis client instance."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client

CACHE_KEY_TEMPLATE = "edge_case_scan:{user_id}:{conv_id}"
CACHE_TTL_SECONDS = 300  # 5 minutes

SCAN_LOCK_KEY_TEMPLATE = "lock:edge_case_scan:{user_id}:{conv_id}"
SCAN_LOCK_TIMEOUT_SECONDS = 120  # 2 minutes to prevent stale locks

# --- Helper Functions ---
def _severity_label(sev: float) -> str:
    if sev >= 0.85: return "critical"
    if sev >= 0.65: return "high"
    if sev >= 0.4:  return "medium"
    return "low"

# ===============================================================================
# EDGE CASE TYPES
# ===============================================================================

class EdgeCaseType(Enum):
    """Types of edge cases in conflict system"""
    ORPHANED_CONFLICT = "orphaned_conflict"
    INFINITE_LOOP = "infinite_loop"
    CONTRADICTION = "contradiction"
    STALE_CONFLICT = "stale_conflict"
    COMPLEXITY_OVERLOAD = "complexity_overload"
    MISSING_CONTEXT = "missing_context"
    PLAYER_DISCONNECT = "player_disconnect"
    NPC_UNAVAILABLE = "npc_unavailable"
    NARRATIVE_BREAK = "narrative_break"
    SYSTEM_CONFLICT = "system_conflict"

@dataclass
class EdgeCase:
    """Represents a detected edge case (recovery options generated on-demand)"""
    case_id: int
    case_type: EdgeCaseType
    affected_conflicts: List[int]
    severity: float
    description: str
    detection_context: Dict[str, Any]
    # Recovery options are now generated on-demand rather than during detection
    recovery_options: Optional[List[Dict[str, Any]]] = None

class EdgeCaseItem(TypedDict):
    subsystem: str
    issue: str
    severity: str
    recoverable: bool

class ScanIssuesResponse(TypedDict):
    issues_found: int
    edge_cases: List[EdgeCaseItem]
    status: str  # 'cached', 'scan_in_progress', or 'completed'
    error: str

class RecoveryResultItem(TypedDict):
    case_type: str
    success: bool
    action: str

class AutoRecoverResponse(TypedDict):
    issues_found: int
    recoveries_attempted: int
    recovery_results: List[RecoveryResultItem]
    error: str


# ===============================================================================
# EDGE CASE SUBSYSTEM (Refactored for Performance)
# ===============================================================================

class ConflictEdgeCaseSubsystem:
    """
    Edge case subsystem that integrates with ConflictSynthesizer.
    Detects and handles edge cases using a non-blocking, cache-first approach.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Lazy-loaded agents
        self._anomaly_detector = None
        self._recovery_strategist = None
        self._narrative_healer = None
        self._graceful_degrader = None
        self._continuity_keeper = None
        
        # Reference to synthesizer
        self.synthesizer = None
        
        # Edge case tracking (lightweight - full data in cache/DB)
        self._recovery_attempts = {}
    
    @property
    def subsystem_type(self):
        """Return the subsystem type"""
        from logic.conflict_system.conflict_synthesizer import SubsystemType
        return SubsystemType.EDGE_HANDLER
    
    @property
    def capabilities(self) -> Set[str]:
        """Return capabilities this subsystem provides"""
        return {
            'anomaly_detection',
            'recovery_strategy',
            'narrative_healing',
            'graceful_degradation',
            'continuity_maintenance',
            'system_protection'
        }
    
    @property
    def dependencies(self) -> Set:
        """Return other subsystems this depends on"""
        return set()
    
    @property
    def event_subscriptions(self) -> Set:
        """Return events this subsystem wants to receive"""
        from logic.conflict_system.conflict_synthesizer import EventType
        return {
            EventType.HEALTH_CHECK,
            EventType.EDGE_CASE_DETECTED,
        }
    
    async def initialize(self, synthesizer) -> bool:
        """Initialize the subsystem with synthesizer reference"""
        import weakref
        self.synthesizer = weakref.ref(synthesizer)
        # No initial scan - will be triggered by first health check
        return True
    
    async def handle_event(self, event) -> Any:
        """Handle an event from the synthesizer."""
        from logic.conflict_system.conflict_synthesizer import SubsystemResponse, EventType
    
        try:
            if event.event_type == EventType.HEALTH_CHECK:
                return await self._handle_health_check(event)
            
            if event.event_type == EventType.EDGE_CASE_DETECTED:
                return await self._handle_recovery_execution(event)
            
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={},
                side_effects=[]
            )
        except Exception as e:
            logger.error(f"Edge case subsystem error: {e}", exc_info=True)
            from logic.conflict_system.conflict_synthesizer import SubsystemResponse
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)},
                side_effects=[]
            )
    
    async def _handle_health_check(self, event) -> Any:
        """Handles health checks using the cache-first, background-task pattern."""
        from logic.conflict_system.conflict_synthesizer import SubsystemResponse
        
        cache_key = CACHE_KEY_TEMPLATE.format(user_id=self.user_id, conv_id=self.conversation_id)
        
        # 1. Fast Path: Check Redis Cache
        try:
            redis_client = await get_redis_client()
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                logger.debug(f"Edge case scan cache HIT for key: {cache_key}")
                scan_result = json.loads(cached_data)
                scan_result['status'] = 'cached'
                return SubsystemResponse(
                    subsystem=self.subsystem_type, event_id=event.event_id,
                    success=True, data=scan_result
                )
        except Exception as e:
            logger.warning(f"Redis cache check failed: {e}")
        
        # 2. Cache Miss: Try to trigger a background scan
        logger.debug(f"Edge case scan cache MISS for key: {cache_key}. Attempting background scan.")
        
        lock_key = SCAN_LOCK_KEY_TEMPLATE.format(user_id=self.user_id, conv_id=self.conversation_id)
        try:
            redis_client = await get_redis_client()
            lock_acquired = await redis_client.set(lock_key, "1", ex=SCAN_LOCK_TIMEOUT_SECONDS, nx=True)
            
            if lock_acquired:
                try:
                    # This name MUST match the name in the @celery_app.task decorator
                    celery_app.send_task('tasks.update_edge_case_scan', args=[self.user_id, self.conversation_id])
                    logger.info(f"Triggered background edge case scan for {self.user_id}:{self.conversation_id}")
                except ImportError:
                    logger.warning("Celery task not available, running scan inline as a fallback.")
                    asyncio.create_task(self.perform_full_scan_and_cache())
            else:
                logger.debug(f"Background scan already in progress for {self.user_id}:{self.conversation_id}")
        except Exception as e:
            logger.error(f"Failed to trigger background scan: {e}")
        
        # 3. Return a minimal response indicating scan is in progress
        # POLISH: Add a helpful message to the 'issues' list
        return SubsystemResponse(
            subsystem=self.subsystem_type, event_id=event.event_id, success=True,
            data={
                'healthy': True, 'edge_cases_found': 0, 'critical_cases': 0,
                'status': 'scan_in_progress',
                'issues': [{
                    'subsystem': 'edge_handler',
                    'issue': 'Scan is running in the background. Results will be available shortly.',
                    'severity': 'low',
                    'recoverable': False
                }],
            }
        )
    
    async def _handle_recovery_execution(self, event) -> Any:
        """Handle request to execute recovery for a specific edge case."""
        from logic.conflict_system.conflict_synthesizer import SubsystemResponse
        
        payload = event.payload or {}
        request = payload.get('request', '')
        
        if request != 'execute_recovery':
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': 'Unknown request type'},
                side_effects=[]
            )
        
        case_data = payload.get('case', {})
        case_type_str = case_data.get('case_type', '')
        
        try:
            case_type = EdgeCaseType(case_type_str)
        except ValueError:
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': f'Invalid case type: {case_type_str}'},
                side_effects=[]
            )
        
        # 1. Generate recovery options on-demand if not provided
        cached_options = await self._load_recovery_options(case_type, case_data, ensure_queue=False)

        options = cached_options or case_data.get('recovery_options')
        if not options:
            logger.info(f"Queueing recovery generation for {case_type.value}")
            await self._load_recovery_options(case_type, case_data, ensure_queue=True)
            options = self._fallback_recovery_options(case_type, case_data)
        elif cached_options is None or not cached_options:
            await self._load_recovery_options(case_type, case_data, ensure_queue=True)

        if not options:
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'success': False, 'action': 'failed_to_generate_options'},
                side_effects=[]
            )

        # 2. Execute the selected recovery option
        option_index = int(payload.get('option_index', 0))
        if option_index >= len(options):
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'success': False, 'action': 'invalid_option_index'},
                side_effects=[]
            )
        
        selected_option = options[option_index]
        affected_ids = [int(i) for i in case_data.get('affected_conflicts', [])]
        
        result = await self._execute_recovery_strategy(case_type, affected_ids, selected_option)
        
        # Track recovery attempt
        case_id = case_data.get('case_id')
        if case_id:
            self._recovery_attempts[case_id] = result
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=result.get('success', False),
            data=result,
            side_effects=[]
        )
    
    # ========== Background Scan Logic ==========
    
    async def perform_full_scan_and_cache(self):
        """
        Main function for background worker. Performs full scan and caches result.
        This is called by the Celery task or inline as fallback.
        """
        logger.info(f"Starting full edge case scan for {self.user_id}:{self.conversation_id}")
        
        # Gather all detection tasks to run them concurrently
        detection_tasks = [
            self._detect_orphaned_conflicts(),
            self._detect_infinite_loops(),
            self._detect_stale_conflicts(),
            self._detect_complexity_overload(),
            self._detect_narrative_breaks(),
        ]
        
        results = await asyncio.gather(*detection_tasks, return_exceptions=True)
        
        # Flatten the list of edge cases and filter out errors
        edge_cases: List[EdgeCase] = []
        for res in results:
            if isinstance(res, list):
                edge_cases.extend(res)
            elif isinstance(res, EdgeCase):
                edge_cases.append(res)
            elif isinstance(res, Exception):
                logger.error(f"Error during edge case detection: {res}", exc_info=res)
        
        # Serialize for caching (without recovery options)
        serializable_cases = []
        for case in edge_cases:
            serializable_cases.append({
                'case_id': case.case_id,
                'case_type': case.case_type.value,
                'severity': case.severity,
                'severity_label': _severity_label(case.severity),
                'affected_conflicts': case.affected_conflicts,
                'description': case.description,
                'detection_context': case.detection_context,
                'recoverable': True,  # Recovery options generated on-demand
                'subsystem': 'edge_handler',
                'issue': case.description,
            })
        
        # Prepare final cache object
        scan_result = {
            'healthy': all(c['severity'] < 0.8 for c in serializable_cases),
            'edge_cases_found': len(serializable_cases),
            'critical_cases': sum(1 for c in serializable_cases if c['severity'] >= 0.8),
            'edge_cases': serializable_cases,
            'issues': serializable_cases,  # Alias for tool compatibility
            'status': 'completed',
        }
        
        # Cache the result in Redis
        cache_key = CACHE_KEY_TEMPLATE.format(user_id=self.user_id, conv_id=self.conversation_id)
        try:
            redis_client = await get_redis_client()
            await redis_client.set(
                cache_key,
                json.dumps(scan_result),
                ex=CACHE_TTL_SECONDS
            )
            logger.info(f"Cached {len(serializable_cases)} edge cases for {self.user_id}:{self.conversation_id}")
        except Exception as e:
            logger.error(f"Failed to cache edge case scan results: {e}")
        finally:
            # Release the lock
            try:
                lock_key = SCAN_LOCK_KEY_TEMPLATE.format(
                    user_id=self.user_id,
                    conv_id=self.conversation_id
                )
                redis_client = await get_redis_client()
                await redis_client.delete(lock_key)
            except Exception as e:
                logger.error(f"Failed to release scan lock: {e}")
    
    # ========== Detection Methods (Run in Parallel) ==========
    
    async def _detect_orphaned_conflicts(self) -> List[EdgeCase]:
        """Detect conflicts with no active stakeholders."""
        edge_cases: List[EdgeCase] = []
        
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT c.conflict_id, c.conflict_name, COALESCE(c.description,'') AS description
                    FROM Conflicts c
                    LEFT JOIN stakeholders s
                      ON s.conflict_id = c.conflict_id 
                      AND s.user_id = $1 
                      AND s.conversation_id = $2
                    WHERE c.user_id = $1 
                      AND c.conversation_id = $2 
                      AND c.is_active = true
                    GROUP BY c.conflict_id, c.conflict_name, c.description
                    HAVING COUNT(s.stakeholder_id) = 0
                """, self.user_id, self.conversation_id)
            
            for r in rows:
                sev = 0.7
                case_id = await self._store_edge_case(
                    EdgeCaseType.ORPHANED_CONFLICT,
                    [int(r['conflict_id'])],
                    sev,
                    f"Conflict '{r['conflict_name']}' has no stakeholders"
                )
                edge_cases.append(EdgeCase(
                    case_id=case_id,
                    case_type=EdgeCaseType.ORPHANED_CONFLICT,
                    affected_conflicts=[int(r['conflict_id'])],
                    severity=sev,
                    description=f"Orphaned conflict: {r['conflict_name']}",
                    detection_context={'conflict': dict(r)},
                    recovery_options=None  # Generated on-demand
                ))
        except Exception as e:
            logger.error(f"Error detecting orphaned conflicts: {e}", exc_info=True)
            return [] # POLISH: Always return a list
        return edge_cases
        
        return edge_cases
    
    async def _detect_infinite_loops(self) -> List[EdgeCase]:
        """Detect conflicts triggering each other infinitely."""
        edge_cases = []
        
        try:
            async with get_db_connection_context() as conn:
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
                if event.get('triggered_by'):
                    chain_key = f"{event['conflict_id']}-{event['triggered_by']}"
                    if chain_key not in trigger_chains:
                        trigger_chains[chain_key] = 0
                    trigger_chains[chain_key] += 1
            
            # Detect loops
            for chain, count in trigger_chains.items():
                if count > 5:  # More than 5 mutual triggers
                    conflicts = [int(c) for c in chain.split('-')]
                    sev = 0.9
                    
                    case_id = await self._store_edge_case(
                        EdgeCaseType.INFINITE_LOOP,
                        conflicts,
                        sev,
                        f"Infinite loop detected between conflicts"
                    )
                    
                    edge_cases.append(EdgeCase(
                        case_id=case_id,
                        case_type=EdgeCaseType.INFINITE_LOOP,
                        affected_conflicts=conflicts,
                        severity=sev,
                        description="Conflicts triggering each other infinitely",
                        detection_context={'trigger_count': count},
                        recovery_options=None
                    ))
        except Exception as e:
            logger.error(f"Error detecting infinite loops: {e}", exc_info=True)
            return [] # POLISH: Always return a list
        return edge_cases
    
    async def _detect_stale_conflicts(self) -> List[EdgeCase]:
        """Detect conflicts that haven't progressed recently."""
        edge_cases: List[EdgeCase] = []
        
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT conflict_id, conflict_name, phase, progress,
                           last_updated, 
                           (CURRENT_TIMESTAMP - last_updated) AS stale_duration
                    FROM Conflicts
                    WHERE user_id = $1 
                      AND conversation_id = $2
                      AND is_active = true
                      AND last_updated < CURRENT_TIMESTAMP - INTERVAL '7 days'
                      AND progress < 100
                """, self.user_id, self.conversation_id)
            
            for r in rows:
                days = getattr(r['stale_duration'], 'days', None)
                stale_days = int(days) if days is not None else 7
                sev = min(0.9, stale_days / 14)
                
                case_id = await self._store_edge_case(
                    EdgeCaseType.STALE_CONFLICT,
                    [int(r['conflict_id'])],
                    sev,
                    f"Conflict stale for {stale_days} days"
                )
                
                edge_cases.append(EdgeCase(
                    case_id=case_id,
                    case_type=EdgeCaseType.STALE_CONFLICT,
                    affected_conflicts=[int(r['conflict_id'])],
                    severity=sev,
                    description=f"Stale conflict: {r['conflict_name']}",
                    detection_context={'stale_days': stale_days, 'conflict': dict(r)},
                    recovery_options=None
                ))
        except Exception as e:
            logger.error(f"Error detecting stale conflicts: {e}", exc_info=True)
        
        return edge_cases
    
    async def _detect_complexity_overload(self) -> List[EdgeCase]:
        """Detect if too many conflicts are active."""
        edge_cases: List[EdgeCase] = []
        
        try:
            async with get_db_connection_context() as conn:
                count = await conn.fetchval("""
                    SELECT COUNT(*) FROM Conflicts
                    WHERE user_id = $1 
                      AND conversation_id = $2 
                      AND is_active = true
                """, self.user_id, self.conversation_id)
                
                if (count or 0) <= 10:
                    return []
                
                conflicts = await conn.fetch("""
                    SELECT conflict_id FROM Conflicts
                    WHERE user_id = $1 
                      AND conversation_id = $2 
                      AND is_active = true
                """, self.user_id, self.conversation_id)
            
            conflict_ids = [int(c['conflict_id']) for c in conflicts]
            sev = min(1.0, float(count) / 15.0)
            
            case_id = await self._store_edge_case(
                EdgeCaseType.COMPLEXITY_OVERLOAD,
                conflict_ids,
                sev,
                f"{count} active conflicts causing overload"
            )
            
            edge_cases.append(EdgeCase(
                case_id=case_id,
                case_type=EdgeCaseType.COMPLEXITY_OVERLOAD,
                affected_conflicts=conflict_ids,
                severity=sev,
                description=f"System overloaded with {count} active conflicts",
                detection_context={'active_count': int(count)},
                recovery_options=None
            ))
        except Exception as e:
            logger.error(f"Error detecting complexity overload: {e}", exc_info=True)
        
        return edge_cases
    
    async def _detect_narrative_breaks(self) -> List[EdgeCase]:
        """Detect narrative continuity breaks via stakeholder role conflicts."""
        edge_cases: List[EdgeCase] = []
        
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT c1.conflict_id AS conflict1, 
                           c2.conflict_id AS conflict2,
                           c1.conflict_name AS name1, 
                           c2.conflict_name AS name2
                    FROM Conflicts c1
                    JOIN Conflicts c2
                      ON c1.user_id = c2.user_id
                     AND c1.conversation_id = c2.conversation_id
                     AND c1.conflict_id < c2.conflict_id
                    WHERE c1.user_id = $1 
                      AND c1.conversation_id = $2
                      AND c1.is_active = true 
                      AND c2.is_active = true
                      AND EXISTS (
                        SELECT 1
                        FROM stakeholders s1
                        JOIN stakeholders s2
                          ON s1.npc_id = s2.npc_id
                         AND s1.user_id = s2.user_id
                         AND s1.conversation_id = s2.conversation_id
                        WHERE s1.conflict_id = c1.conflict_id
                          AND s2.conflict_id = c2.conflict_id
                          AND (
                                COALESCE(s1.role, '') != COALESCE(s2.role, '')
                                OR COALESCE(s1.faction_id, 0) != COALESCE(s2.faction_id, 0)
                              )
                      )
                """, self.user_id, self.conversation_id)
            
            for r in rows:
                sev = 0.6
                case_id = await self._store_edge_case(
                    EdgeCaseType.CONTRADICTION,
                    [int(r["conflict1"]), int(r["conflict2"])],
                    sev,
                    "Contradictory NPC positions in conflicts"
                )
                
                edge_cases.append(EdgeCase(
                    case_id=case_id,
                    case_type=EdgeCaseType.CONTRADICTION,
                    affected_conflicts=[int(r["conflict1"]), int(r["conflict2"])],
                    severity=sev,
                    description="NPC has contradictory roles in conflicts",
                    detection_context={"conflicts": dict(r)},
                    recovery_options=None
                ))
        except Exception as e:
            logger.error(f"Error detecting narrative breaks: {e}", exc_info=True)
        
        return edge_cases
    
    # ========== On-Demand Recovery Generation ==========
    
    async def _generate_recovery_options_for_case(
        self,
        case_type: EdgeCaseType,
        case_data: Dict
    ) -> List[Dict]:
        """Fetch recovery options via cache-first background task flow."""

        options = await self._load_recovery_options(case_type, case_data, ensure_queue=True)
        if options:
            return options

        return self._fallback_recovery_options(case_type, case_data)

    async def _load_recovery_options(
        self,
        case_type: EdgeCaseType,
        case_data: Dict,
        *,
        ensure_queue: bool,
    ) -> List[Dict]:
        """Load cached recovery options and optionally queue background generation."""

        cache_key, lock_key, task_payload = self._build_recovery_task_payload(case_type, case_data)
        if cache_key:
            cached_payload = await self._fetch_cached_recovery(cache_key)
            if cached_payload:
                options = cached_payload.get('options', [])
                if isinstance(options, list) and options:
                    return options

        if ensure_queue and cache_key and task_payload:
            await self._ensure_recovery_task(case_type, cache_key, lock_key, task_payload)

        return []

    async def _fetch_cached_recovery(self, cache_key: str) -> Optional[Dict[str, Any]]:
        try:
            redis_client = await get_redis_client()
            cached = await redis_client.get(cache_key)
            if not cached:
                return None
            data = json.loads(cached)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.error("Failed to fetch cached recovery options: %s", exc)
        return None

    async def _ensure_recovery_task(
        self,
        case_type: EdgeCaseType,
        cache_key: str,
        lock_key: str,
        payload: Dict[str, Any],
    ) -> None:
        payload = dict(payload)
        payload['cache_key'] = cache_key
        payload['lock_key'] = lock_key

        try:
            redis_client = await get_redis_client()
            acquired = await redis_client.set(lock_key, 'queued', ex=RECOVERY_LOCK_TTL, nx=True)
        except Exception as exc:
            logger.error("Failed to set recovery lock for %s: %s", case_type.value, exc)
            dispatch_recovery_generation(case_type.value, payload)
            return

        if acquired:
            dispatch_recovery_generation(case_type.value, payload)

    def _build_recovery_task_payload(
        self,
        case_type: EdgeCaseType,
        case_data: Dict,
    ) -> tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        case_ref = self._case_reference(case_data)
        if not case_ref:
            return None, None, None

        cache_key = recovery_cache_key(
            self.user_id,
            self.conversation_id,
            case_type.value,
            case_ref,
        )
        lock_key = recovery_lock_key(
            self.user_id,
            self.conversation_id,
            case_type.value,
            case_ref,
        )

        detection_context = case_data.get('detection_context', {})
        payload: Dict[str, Any] = {
            'user_id': self.user_id,
            'conversation_id': self.conversation_id,
            'case_id': case_data.get('case_id'),
            'case_ref': case_ref,
        }

        if case_type == EdgeCaseType.ORPHANED_CONFLICT:
            conflict = detection_context.get('conflict') or {}
            if not conflict:
                return cache_key, lock_key, None
            payload['conflict'] = conflict
        elif case_type == EdgeCaseType.STALE_CONFLICT:
            conflict = detection_context.get('conflict') or {}
            if not conflict:
                return cache_key, lock_key, None
            payload['conflict'] = conflict
        elif case_type == EdgeCaseType.INFINITE_LOOP:
            payload['conflicts'] = case_data.get('affected_conflicts', [])
        elif case_type == EdgeCaseType.COMPLEXITY_OVERLOAD:
            payload['active_count'] = detection_context.get('active_count', len(case_data.get('affected_conflicts', [])))
        elif case_type == EdgeCaseType.CONTRADICTION:
            contradiction = detection_context.get('conflicts') or {}
            if not contradiction:
                return cache_key, lock_key, None
            payload['contradiction'] = contradiction
        else:
            return cache_key, lock_key, None

        return cache_key, lock_key, payload

    def _case_reference(self, case_data: Dict) -> str:
        case_id = case_data.get('case_id')
        if case_id:
            return str(case_id)

        fingerprint_source = {
            'context': case_data.get('detection_context', {}),
            'conflicts': case_data.get('affected_conflicts', []),
            'type': case_data.get('case_type'),
        }
        try:
            serialized = json.dumps(fingerprint_source, sort_keys=True, default=str)
        except Exception:
            serialized = repr(fingerprint_source)
        digest = hashlib.sha1(serialized.encode('utf-8')).hexdigest()
        return digest

    def _fallback_recovery_options(self, case_type: EdgeCaseType, case_data: Dict) -> List[Dict]:
        detection_context = case_data.get('detection_context', {})

        if case_type == EdgeCaseType.ORPHANED_CONFLICT:
            conflict = detection_context.get('conflict', {})
            return self._fallback_orphan_recovery(conflict)
        if case_type == EdgeCaseType.STALE_CONFLICT:
            conflict = detection_context.get('conflict', {})
            return self._fallback_stale_recovery(conflict)
        if case_type == EdgeCaseType.INFINITE_LOOP:
            conflicts = case_data.get('affected_conflicts', [])
            return self._fallback_loop_recovery(conflicts)
        if case_type == EdgeCaseType.COMPLEXITY_OVERLOAD:
            count = detection_context.get('active_count', 0)
            return self._fallback_overload_recovery(count)
        if case_type == EdgeCaseType.CONTRADICTION:
            contradiction = detection_context.get('conflicts', {})
            return self._fallback_contradiction_recovery(contradiction)
        return []

    def _fallback_orphan_recovery(self, conflict: Dict[str, Any]) -> List[Dict[str, Any]]:
        name = conflict.get('conflict_name', 'the conflict')
        description = conflict.get('description', 'this unresolved situation')
        return [
            {
                'strategy': 'close',
                'description': f"Resolve {name} quietly with a recap of outstanding threads.",
                'narrative': f"NPCs acknowledge that {description} has naturally concluded.",
                'implementation': [
                    'Notify conflict stakeholders of closure',
                    'Log a resolution summary in conflict history',
                ],
                'risk': 'low',
            },
            {
                'strategy': 'assign',
                'description': 'Introduce an aligned NPC to steward the conflict forward.',
                'narrative': 'A motivated ally steps in to take ownership and move things forward.',
                'implementation': [
                    'Select an NPC with relevant motivation',
                    'Create a handoff note for the new steward',
                    'Schedule a follow-up beat to confirm direction',
                ],
                'risk': 'medium',
            },
            {
                'strategy': 'pivot',
                'description': 'Reframe the conflict around the player to re-engage them.',
                'narrative': 'Consequences spill toward the player, inviting a response.',
                'implementation': [
                    'Draft a short scene where fallout reaches the player',
                    'Clarify what changes if the player leans in',
                ],
                'risk': 'medium',
            },
        ]

    def _fallback_stale_recovery(self, conflict: Dict[str, Any]) -> List[Dict[str, Any]]:
        name = conflict.get('conflict_name', 'the conflict')
        phase = conflict.get('phase', 'current phase')
        return [
            {
                'strategy': 'revitalize',
                'description': f'Introduce a twist that escalates {name} beyond the {phase}.',
                'narrative_event': 'A surprising revelation complicates existing stakes.',
                'player_hook': 'Offer the player a chance to exploit or contain the twist.',
                'expected_outcome': 'Conflict gains immediate urgency and direction.',
            },
            {
                'strategy': 'conclude',
                'description': f'Wrap {name} with a reflective epilogue that explains the slowdown.',
                'narrative_event': 'NPCs acknowledge lessons learned and the situation settles.',
                'player_hook': 'Invite the player to sign off or celebrate the resolution.',
                'expected_outcome': 'Conflict leaves the stage gracefully with continuity intact.',
            },
            {
                'strategy': 'transform',
                'description': f'Reshape {name} into a related but fresher objective.',
                'narrative_event': 'A new antagonist or goal emerges from the stalemate.',
                'player_hook': 'Present a new choice that reframes prior progress.',
                'expected_outcome': 'Conflict energy shifts into a new storyline without reset.',
            },
        ]

    def _fallback_loop_recovery(self, conflicts: List[int]) -> List[Dict[str, Any]]:
        conflict_list = ', '.join(str(c) for c in conflicts) or 'related conflicts'
        return [
            {
                'strategy': 'break',
                'description': 'Sever the automatic triggers and let the conflicts cool down.',
                'preserved_conflicts': conflicts,
                'narrative_bridge': 'Facilitators call a truce while the loop is audited.',
                'prevention': 'Add guardrails to prevent recursive triggers.',
            },
            {
                'strategy': 'merge',
                'description': f'Fuse {conflict_list} into a single escalation track.',
                'preserved_conflicts': conflicts[:1],
                'narrative_bridge': 'Participants recognize the overlap and align efforts.',
                'prevention': 'Track merged state to avoid re-splitting inadvertently.',
            },
            {
                'strategy': 'prioritize',
                'description': 'Designate a lead conflict and freeze the dependent one.',
                'preserved_conflicts': conflicts[:1],
                'narrative_bridge': 'One faction pauses until the main dispute resolves.',
                'prevention': 'Add queueing rules for any future mutual triggers.',
            },
        ]

    def _fallback_overload_recovery(self, count: int) -> List[Dict[str, Any]]:
        target = max(5, count - 3)
        return [
            {
                'strategy': 'consolidate',
                'target_count': target,
                'selection_criteria': 'Group conflicts with overlapping NPCs and stakes.',
                'narrative_framing': 'Leaders broker alliances to resolve redundant disputes.',
                'player_communication': 'Explain the consolidation as a negotiated ceasefire.',
            },
            {
                'strategy': 'prioritize',
                'target_count': max(3, target - 1),
                'selection_criteria': 'Keep conflicts tied directly to current player arcs.',
                'narrative_framing': 'Command issues urgent orders focusing on critical fronts.',
                'player_communication': 'Share a ranked list of conflicts requesting input.',
            },
            {
                'strategy': 'pause',
                'target_count': max(2, target - 2),
                'selection_criteria': 'Temporarily shelve conflicts with low momentum.',
                'narrative_framing': 'Civic authorities enforce a cooling-off period.',
                'player_communication': 'Explain the pause as time to breathe and regroup.',
            },
        ]

    def _fallback_contradiction_recovery(self, contradiction: Dict[str, Any]) -> List[Dict[str, Any]]:
        name1 = contradiction.get('name1', 'Conflict A')
        name2 = contradiction.get('name2', 'Conflict B')
        return [
            {
                'strategy': 'explain',
                'narrative_explanation': f'Reveal that the NPC acted under misinformation between {name1} and {name2}.',
                'character_development': 'Shows humility as the NPC admits the error.',
                'conflict_impact': {
                    'conflict1': 'Adjust stakes after clarification.',
                    'conflict2': 'Realign expectations with the corrected context.',
                },
            },
            {
                'strategy': 'choose',
                'narrative_explanation': 'NPC makes a decisive commitment to one side.',
                'character_development': 'Highlights loyalty or ambition driving the decision.',
                'conflict_impact': {
                    'conflict1': 'Becomes the primary focus with full support.',
                    'conflict2': 'Transitions to an opposition plot hook.',
                },
            },
            {
                'strategy': 'split',
                'narrative_explanation': 'NPC delegates one responsibility to a trusted ally.',
                'character_development': 'Expands the social web by introducing mentorship ties.',
                'conflict_impact': {
                    'conflict1': 'Gains a new supporting NPC to keep continuity.',
                    'conflict2': 'Retains the original NPC for key scenes.',
                },
            },
        ]
    
    # ========== Recovery Execution Methods ==========
    
    async def _execute_recovery_strategy(
        self,
        case_type: EdgeCaseType,
        affected_ids: List[int],
        option: Dict
    ) -> Dict[str, Any]:
        """Dispatches to the correct execution method based on strategy."""
        if case_type == EdgeCaseType.ORPHANED_CONFLICT and affected_ids:
            return await self._execute_orphan_recovery(affected_ids[0], option)
        
        elif case_type == EdgeCaseType.INFINITE_LOOP:
            return await self._execute_loop_recovery(affected_ids, option)
        
        elif case_type == EdgeCaseType.STALE_CONFLICT and affected_ids:
            return await self._execute_stale_recovery(affected_ids[0], option)
        
        elif case_type == EdgeCaseType.COMPLEXITY_OVERLOAD:
            return await self._execute_overload_recovery(option)
        
        elif case_type == EdgeCaseType.CONTRADICTION:
            return await self._execute_contradiction_recovery(affected_ids, option)
        
        return {'success': False, 'action': 'unknown_strategy'}
    
    async def _execute_orphan_recovery(self, conflict_id: int, recovery: Dict[str, Any]) -> Dict[str, Any]:
        """Execute recovery for orphaned conflict."""
        strategy = str(recovery.get('strategy', '')).lower()
        
        try:
            async with get_db_connection_context() as conn:
                if strategy == 'close':
                    await conn.execute("""
                        UPDATE Conflicts
                        SET is_active = false,
                            resolution_description = $1,
                            resolved_at = CURRENT_TIMESTAMP
                        WHERE conflict_id = $2 
                          AND user_id = $3 
                          AND conversation_id = $4
                    """, recovery.get('narrative', 'Conflict resolved'), 
                    conflict_id, self.user_id, self.conversation_id)
                    
                    return {'success': True, 'action': 'closed_conflict'}
                
                elif strategy == 'assign':
                    return {'success': True, 'action': 'assigned_npcs'}
                
                elif strategy == 'pivot':
                    await conn.execute("""
                        UPDATE Conflicts
                        SET conflict_type = 'personal',
                            description = $1
                        WHERE conflict_id = $2 
                          AND user_id = $3 
                          AND conversation_id = $4
                    """, recovery.get('narrative', 'Conflict transforms'), 
                    conflict_id, self.user_id, self.conversation_id)
                    
                    return {'success': True, 'action': 'pivoted_to_player'}
        except Exception as e:
            logger.error(f"Error executing orphan recovery: {e}")
            return {'success': False, 'action': 'error', 'error': str(e)}
        
        return {'success': False, 'action': 'unknown_strategy'}
    
    async def _execute_loop_recovery(self, conflicts: List[int], recovery: Dict[str, Any]) -> Dict[str, Any]:
        """Execute recovery for infinite loop"""
        strategy = str(recovery.get('strategy', '')).lower()
        
        try:
            async with get_db_connection_context() as conn:
                if strategy == 'break':
                    await conn.execute("""
                        DELETE FROM conflict_triggers
                        WHERE user_id = $1 
                          AND conversation_id = $2
                          AND conflict_id = ANY($3::int[])
                          AND triggered_conflict_id = ANY($3::int[])
                    """, self.user_id, self.conversation_id, conflicts)
                    
                    return {'success': True, 'action': 'broke_trigger_chain'}
                
                elif strategy == 'merge' and len(conflicts) > 1:
                    await conn.execute("""
                        UPDATE Conflicts
                        SET is_active = false
                        WHERE user_id = $1 
                          AND conversation_id = $2
                          AND conflict_id = ANY($3::int[])
                    """, self.user_id, self.conversation_id, conflicts[1:])
                    
                    return {'success': True, 'action': 'merged_conflicts'}
        except Exception as e:
            logger.error(f"Error executing loop recovery: {e}")
            return {'success': False, 'action': 'error', 'error': str(e)}
        
        return {'success': False, 'action': 'unknown_strategy'}
    
    async def _execute_stale_recovery(self, conflict_id: int, recovery: Dict[str, Any]) -> Dict[str, Any]:
        """Execute recovery for stale conflict"""
        narrative = recovery.get('narrative_event', 'The situation evolves unexpectedly')
        
        try:
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    UPDATE Conflicts
                    SET progress = LEAST(100, progress + 20),
                        phase = CASE 
                            WHEN progress + 20 >= 80 THEN 'resolution'
                            WHEN progress + 20 >= 60 THEN 'climax'
                            ELSE phase
                        END,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE conflict_id = $1 
                      AND user_id = $2 
                      AND conversation_id = $3
                """, conflict_id, self.user_id, self.conversation_id)
                
                await conn.execute("""
                    INSERT INTO conflict_events
                        (user_id, conversation_id, conflict_id, event_type, 
                         description, created_at)
                    VALUES ($1, $2, $3, 'recovery', $4, CURRENT_TIMESTAMP)
                """, self.user_id, self.conversation_id, conflict_id, narrative)
            
            return {'success': True, 'action': 'revitalized_conflict'}
        except Exception as e:
            logger.error(f"Error executing stale recovery: {e}")
            return {'success': False, 'action': 'error', 'error': str(e)}
    
    async def _execute_overload_recovery(self, recovery: Dict[str, Any]) -> Dict[str, Any]:
        """Execute recovery for complexity overload"""
        strategy = str(recovery.get('strategy', '')).lower()
        target_count = int(recovery.get('target_count', 5) or 5)
        
        if strategy != 'prioritize':
            return {'success': False, 'action': 'unsupported_strategy'}
        
        try:
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    UPDATE Conflicts
                    SET is_active = false
                    WHERE conflict_id IN (
                        SELECT conflict_id
                        FROM Conflicts
                        WHERE user_id = $1 
                          AND conversation_id = $2 
                          AND is_active = true
                        ORDER BY 
                          CASE intensity
                            WHEN 'confrontation' THEN 5
                            WHEN 'opposition' THEN 4
                            WHEN 'friction' THEN 3
                            WHEN 'tension' THEN 2
                            ELSE 1
                          END DESC
                        OFFSET $3
                    )
                """, self.user_id, self.conversation_id, target_count)
            
            return {'success': True, 'action': 'prioritized_conflicts'}
        except Exception as e:
            logger.error(f"Error executing overload recovery: {e}")
            return {'success': False, 'action': 'error', 'error': str(e)}
    
    async def _execute_contradiction_recovery(
        self, 
        conflicts: List[int], 
        recovery: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute recovery for contradiction"""
        explanation = recovery.get('narrative_explanation', 'The situation clarifies')
        
        try:
            async with get_db_connection_context() as conn:
                for cid in conflicts:
                    await conn.execute("""
                        INSERT INTO conflict_events
                            (user_id, conversation_id, conflict_id, event_type, 
                             description, created_at)
                        VALUES ($1, $2, $3, 'clarification', $4, CURRENT_TIMESTAMP)
                    """, self.user_id, self.conversation_id, int(cid), explanation)
            
            return {'success': True, 'action': 'explained_contradiction'}
        except Exception as e:
            logger.error(f"Error executing contradiction recovery: {e}")
            return {'success': False, 'action': 'error', 'error': str(e)}
    
    # ========== Helper Methods ==========
    
    async def _store_edge_case(
        self,
        case_type: EdgeCaseType,
        conflicts: List[int],
        severity: float,
        description: str
    ) -> int:
        """Store detected edge case"""
        try:
            async with get_db_connection_context() as conn:
                case_id = await conn.fetchval("""
                    INSERT INTO conflict_edge_cases
                    (user_id, conversation_id, case_type, affected_conflicts, 
                     severity, description, detected_at)
                    VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
                    RETURNING case_id
                """, self.user_id, self.conversation_id, case_type.value, 
                json.dumps(conflicts), severity, description)
                
                return int(case_id) if case_id else 0
        except Exception as e:
            logger.error(f"Error storing edge case: {e}")
            return 0
    
    async def health_check(self) -> Dict[str, Any]:
        """Return health status of the subsystem"""
        # This is a lightweight check, actual scanning happens via cache/background
        try:
            cache_key = CACHE_KEY_TEMPLATE.format(
                user_id=self.user_id,
                conv_id=self.conversation_id
            )
            redis_client = await get_redis_client()
            cached_data = await redis_client.get(cache_key)
            
            if cached_data:
                scan_result = json.loads(cached_data)
                return {
                    'healthy': scan_result.get('healthy', True),
                    'edge_cases': scan_result.get('edge_cases_found', 0),
                    'critical_cases': scan_result.get('critical_cases', 0),
                    'status': 'cached'
                }
        except Exception:
            pass
        
        return {
            'healthy': True,
            'edge_cases': 0,
            'critical_cases': 0,
            'status': 'unknown'
        }
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        """Get edge case data for a specific conflict"""
        # Check cached scan results for this conflict
        try:
            cache_key = CACHE_KEY_TEMPLATE.format(
                user_id=self.user_id,
                conv_id=self.conversation_id
            )
            redis_client = await get_redis_client()
            cached_data = await redis_client.get(cache_key)
            
            if cached_data:
                scan_result = json.loads(cached_data)
                edge_cases = []
                for case in scan_result.get('issues', []):
                    if conflict_id in case.get('affected_conflicts', []):
                        edge_cases.append({
                            'type': case.get('case_type'),
                            'severity': case.get('severity'),
                            'description': case.get('description')
                        })
                
                return {
                    'has_edge_cases': len(edge_cases) > 0,
                    'edge_cases': edge_cases
                }
        except Exception:
            pass
        
        return {'has_edge_cases': False, 'edge_cases': []}
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current state of edge case system"""
        return {
            'recovery_attempts': len(self._recovery_attempts),
            'monitoring_active': True
        }
    
    async def is_relevant_to_scene(self, scene_context: Dict[str, Any]) -> bool:
        """Check if edge case system is relevant to scene"""
        return True  # Always relevant for monitoring
    
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


# ===============================================================================
# PUBLIC API FUNCTIONS
# ===============================================================================

@function_tool
async def scan_for_conflict_issues(
    ctx: RunContextWrapper
) -> ScanIssuesResponse:
    """Scan for edge cases in conflict system through synthesizer."""
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    from logic.conflict_system.conflict_synthesizer import (
        get_synthesizer, SystemEvent, EventType, SubsystemType
    )
    
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    # Ask EDGE_HANDLER to run a scan (health check request)
    event = SystemEvent(
        event_id=f"scan_{datetime.now().timestamp()}",
        event_type=EventType.HEALTH_CHECK,
        source_subsystem=SubsystemType.SLICE_OF_LIFE,
        payload={'request': 'edge_case_scan'},
        target_subsystems={SubsystemType.EDGE_HANDLER},
        requires_response=True,
        priority=3,
    )
    
    items: List[EdgeCaseItem] = []
    status = "completed"
    error = ""
    
    responses = await synthesizer.emit_event(event)
    if responses:
        for r in responses:
            if r.subsystem == SubsystemType.EDGE_HANDLER:
                data = r.data or {}
                status = data.get('status', 'completed')
                
                # Accept a variety of shapes and normalize to a fixed list
                raw_cases = data.get('edge_cases') or data.get('issues') or []
                for rc in raw_cases:
                    subsystem = str(rc.get('subsystem', 'unknown'))
                    issue = str(rc.get('issue', rc.get('description', 'unknown')))
                    severity = str(rc.get('severity_label', rc.get('severity', 'medium')))
                    recoverable = bool(rc.get('recoverable', False))
                    
                    items.append({
                        'subsystem': subsystem,
                        'issue': issue,
                        'severity': severity,
                        'recoverable': recoverable,
                    })
                break
    else:
        error = "Edge handler did not respond"
    
    return {
        'issues_found': len(items),
        'edge_cases': items,
        'status': status,
        'error': error,
    }


@function_tool
async def auto_recover_conflicts(
    ctx: RunContextWrapper
) -> AutoRecoverResponse:
    """Automatically recover from detected edge cases via the synthesizer."""
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    from logic.conflict_system.conflict_synthesizer import (
        get_synthesizer, SystemEvent, EventType, SubsystemType
    )
    
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    # 1) Ask EDGE_HANDLER to scan for edge cases
    scan_evt = SystemEvent(
        event_id=f"scan_{datetime.now().timestamp()}",
        event_type=EventType.HEALTH_CHECK,
        source_subsystem=SubsystemType.SLICE_OF_LIFE,
        payload={'request': 'edge_case_scan'},
        target_subsystems={SubsystemType.EDGE_HANDLER},
        requires_response=True,
        priority=3,
    )
    
    scan_responses = await synthesizer.emit_event(scan_evt)
    
    edge_cases = []
    if scan_responses:
        for r in scan_responses:
            if r.subsystem == SubsystemType.EDGE_HANDLER:
                data = r.data or {}
                edge_cases = data.get('edge_cases') or data.get('issues') or []
                break
    
    recovery_results: List[RecoveryResultItem] = []
    error = ""
    
    # 2) For each case, request recovery execution
    for case in edge_cases:
        case_type = str(case.get('case_type', case.get('type', 'unknown')))
        
        # Request recovery execution (options will be generated on-demand)
        recover_evt = SystemEvent(
            event_id=f"recover_{case_type}_{datetime.now().timestamp()}",
            event_type=EventType.EDGE_CASE_DETECTED,
            source_subsystem=SubsystemType.EDGE_HANDLER,
            payload={
                'request': 'execute_recovery',
                'case': case,
                'option_index': 0,  # Use first recovery option
            },
            target_subsystems={SubsystemType.EDGE_HANDLER},
            requires_response=True,
            priority=2,
        )
        
        recover_resps = await synthesizer.emit_event(recover_evt)
        
        # Default outcome
        success = False
        action_label = 'unknown'
        
        if recover_resps:
            for rr in recover_resps:
                if rr.subsystem == SubsystemType.EDGE_HANDLER:
                    rd = rr.data or {}
                    success = bool(rd.get('success', False))
                    action_label = str(rd.get('action', action_label))
                    break
        
        recovery_results.append({
            'case_type': case_type,
            'success': success,
            'action': action_label,
        })
    
    return {
        'issues_found': len(edge_cases),
        'recoveries_attempted': len(recovery_results),
        'recovery_results': recovery_results,
        'error': error,
    }


