# logic/conflict_system/conflict_canon.py
"""
Conflict Canon System integrated with Core Lore Canon
Manages how conflicts become part of world lore through the unified canon system.
"""

import logging
import json
import random
import hashlib
from typing import Dict, List, Any, Optional, Set, TypedDict
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from celery import current_app

from logic.conflict_system.dynamic_conflict_template import extract_runner_response
from agents import Agent, function_tool, RunContextWrapper, Runner
from db.connection import get_db_connection_context

# Orchestrator interface
from logic.conflict_system.conflict_synthesizer import (
    ConflictSubsystem,
    SubsystemType,
    EventType,
    SystemEvent,
    SubsystemResponse,
)

# Core canon system imports
from lore.core.canon import (
    log_canonical_event,
)
from lore.core.context import CanonicalContext
from embedding.vector_store import generate_embedding

logger = logging.getLogger(__name__)

# ===============================================================================
# CANON TYPES
# ===============================================================================

class CanonEventType(Enum):
    """Types of canonical events from conflicts"""
    HISTORICAL_PRECEDENT = "historical_precedent"
    CULTURAL_SHIFT = "cultural_shift"
    RELATIONSHIP_MILESTONE = "relationship_milestone"
    POWER_RESTRUCTURING = "power_restructuring"
    SOCIAL_EVOLUTION = "social_evolution"
    LEGENDARY_MOMENT = "legendary_moment"
    TRADITION_BORN = "tradition_born"
    TABOO_BROKEN = "taboo_broken"


@dataclass
class CanonicalEvent:
    """An event that becomes part of world canon"""
    event_id: int
    conflict_id: int
    event_type: CanonEventType
    name: str
    description: str
    significance: float  # 0-1, how important to world lore
    cultural_impact: Dict[str, Any]
    referenced_by: List[str]
    creates_precedent: bool
    legacy: str


class CanonizationInputDTO(TypedDict, total=False):
    resolution_type: str
    outcome: str
    summary: str
    significance: float
    tags: List[str]
    notable_consequences: List[str]
    victory_achieved: bool


class CanonizationResponse(TypedDict):
    became_canonical: bool
    canonical_event_id: int
    reason: str
    significance: float
    tags: List[str]
    snapshot_id: Optional[int]
    pending: bool


class LoreAlignmentResponse(TypedDict, total=False):
    is_compliant: bool
    conflicts: List[str]
    matching_event_ids: List[int]
    matching_tradition_ids: List[int]
    suggestions: List[str]
    suggestions_pending: bool
    cache_id: Optional[int]


class CanonicalPrecedent(TypedDict):
    event: str
    tags: List[str]
    significance: float
    established: str  # ISO-8601 string


# ===============================================================================
# CONFLICT CANON SUBSYSTEM
# ===============================================================================

class ConflictCanonSubsystem(ConflictSubsystem):
    """
    Canon subsystem integrated with Core Lore Canon System.
    Manages how conflicts become part of world lore through the unified canon.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.ctx = CanonicalContext(user_id, conversation_id)

        # Lazy-loaded agents
        self._lore_integrator = None
        self._precedent_analyzer = None
        self._cultural_interpreter = None
        self._legacy_writer = None
        self._reference_generator = None

        # Reference to synthesizer (weakref set during initialize)
        self.synthesizer = None

        self._resolution_queue_table = "conflict_canon_resolution_queue"
        self._compliance_cache_table = "conflict_lore_compliance_cache"
        self._reference_cache_table = "conflict_canon_reference_cache"
        self._reference_cache_ttl = timedelta(hours=12)
        self._compliance_cache_ttl = timedelta(hours=6)
    
    @property
    def subsystem_type(self) -> SubsystemType:
        return SubsystemType.CANON
    
    @property
    def capabilities(self) -> Set[str]:
        return {
            'lore_integration',
            'precedent_tracking',
            'cultural_impact',
            'legacy_creation',
            'reference_generation',
            'tradition_establishment'
        }
    
    @property
    def dependencies(self) -> Set[SubsystemType]:
        return set()
    
    @property
    def event_subscriptions(self) -> Set[EventType]:
        # Subscribe to STATE_SYNC so function tools can target requests to CANON
        return {
            EventType.CONFLICT_RESOLVED,
            EventType.PHASE_TRANSITION,
            EventType.HEALTH_CHECK,
            EventType.CANON_ESTABLISHED,
            EventType.STATE_SYNC,
        }
    
    async def initialize(self, synthesizer) -> bool:
        import weakref
        self.synthesizer = weakref.ref(synthesizer)

        # Ensure at least some founding canon exists
        async with get_db_connection_context() as conn:
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM CanonicalEvents
                WHERE user_id = $1 AND conversation_id = $2
                  AND tags ? 'conflict'
            """, self.user_id, self.conversation_id)
            
            if (count or 0) == 0:
                await self._create_initial_lore()
        return True

    # ------------------------------------------------------------------
    # Internal storage helpers
    # ------------------------------------------------------------------

    async def _ensure_resolution_queue_table(self) -> None:
        async with get_db_connection_context() as conn:
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._resolution_queue_table} (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    conflict_id INTEGER NOT NULL,
                    resolution JSONB,
                    status TEXT NOT NULL DEFAULT 'pending',
                    result JSONB,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP
                )
                """
            )

    async def _store_resolution_snapshot(
        self,
        conflict_id: int,
        resolution_data: Dict[str, Any]
    ) -> int:
        await self._ensure_resolution_queue_table()
        async with get_db_connection_context() as conn:
            snapshot_id = await conn.fetchval(
                f"""
                INSERT INTO {self._resolution_queue_table}
                    (user_id, conversation_id, conflict_id, resolution, status)
                VALUES ($1, $2, $3, $4::jsonb, 'pending')
                RETURNING id
                """,
                self.user_id,
                self.conversation_id,
                conflict_id,
                json.dumps(resolution_data or {}),
            )
        return int(snapshot_id)

    async def _mark_snapshot_status(
        self,
        snapshot_id: int,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        await self._ensure_resolution_queue_table()
        async with get_db_connection_context() as conn:
            await conn.execute(
                f"""
                UPDATE {self._resolution_queue_table}
                   SET status = $2,
                       result = $3::jsonb,
                       error = $4,
                       processed_at = CASE WHEN $2 = 'pending' THEN processed_at ELSE NOW() END
                 WHERE id = $1
                """,
                snapshot_id,
                status,
                json.dumps(result or {}) if result is not None else None,
                error,
            )

    async def _fetch_resolution_snapshot(
        self,
        snapshot_id: int
    ) -> Optional[Dict[str, Any]]:
        await self._ensure_resolution_queue_table()
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT conflict_id, resolution
                  FROM {self._resolution_queue_table}
                 WHERE id = $1
                   AND user_id = $2
                   AND conversation_id = $3
                """,
                snapshot_id,
                self.user_id,
                self.conversation_id,
            )
        if not row:
            return None
        data = dict(row)
        resolution = data.get('resolution') or {}
        if isinstance(resolution, str):
            try:
                resolution = json.loads(resolution)
            except json.JSONDecodeError:
                resolution = {}
        return {
            'conflict_id': int(data.get('conflict_id') or 0),
            'resolution': resolution,
        }

    async def process_canonization_snapshot(
        self,
        snapshot_id: int,
        conflict_id: int,
    ) -> None:
        snapshot = await self._fetch_resolution_snapshot(snapshot_id)
        if not snapshot:
            await self._mark_snapshot_status(
                snapshot_id,
                'failed',
                error='Resolution snapshot not found',
            )
            return

        snapshot_conflict_id = int(snapshot.get('conflict_id') or 0)
        resolved_conflict_id = int(conflict_id or snapshot_conflict_id)
        if resolved_conflict_id <= 0:
            await self._mark_snapshot_status(
                snapshot_id,
                'failed',
                result={'became_canonical': False},
                error='Invalid conflict id for canonization',
            )
            return

        resolution_data = snapshot.get('resolution') or {}
        try:
            result = await self._perform_canon_evaluation(resolved_conflict_id, resolution_data)
            await self._mark_snapshot_status(snapshot_id, 'completed', result=result)
        except Exception as exc:
            logger.exception("Canon snapshot processing failed", exc_info=exc)
            await self._mark_snapshot_status(
                snapshot_id,
                'failed',
                result={'became_canonical': False},
                error=str(exc),
            )
            raise

    async def _perform_canon_evaluation(
        self,
        conflict_id: int,
        resolution_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow(
                """
                SELECT *, id as conflict_id FROM Conflicts WHERE id = $1
                """,
                conflict_id,
            )
            if not conflict:
                return {
                    'became_canonical': False,
                    'reason': 'Conflict not found',
                    'significance': 0.0,
                    'tags': [],
                    'canonical_event_id': 0,
                }
            stakeholders = await conn.fetch(
                """
                SELECT * FROM ConflictStakeholders WHERE conflict_id = $1
                """,
                conflict_id,
            )

        prompt = f"""
Evaluate if this conflict resolution should become canonical:

Conflict: {conflict['conflict_name']}
Type: {conflict['conflict_type']}
Resolution: {json.dumps(resolution_data)}
Stakeholders: {len(stakeholders)}

Return JSON:
{{
  "should_be_canonical": true/false,
  "reason": "Why this matters (or doesn't)",
  "event_type": "historical_precedent|cultural_shift|relationship_milestone|power_restructuring|social_evolution|legendary_moment|tradition_born|taboo_broken",
  "significance": 0.0
}}
"""
        response = await Runner.run(self.lore_integrator, prompt)
        data = json.loads(extract_runner_response(response))
        should = bool(data.get('should_be_canonical', False))
        reason = data.get('reason', '')
        significance = float(data.get('significance', 0.0) or 0.0)

        if not should:
            return {
                'became_canonical': False,
                'reason': reason,
                'significance': significance,
                'tags': [],
                'canonical_event_id': 0,
            }

        event_type_str = data.get('event_type', CanonEventType.LEGENDARY_MOMENT.value)
        event_type = CanonEventType(event_type_str)
        event, tags = await self._create_canonical_event(
            conflict_id, conflict, resolution_data, event_type, significance
        )

        return {
            'became_canonical': True,
            'reason': reason,
            'significance': significance,
            'tags': tags,
            'canonical_event_id': int(event.event_id),
            'legacy': event.legacy,
            'name': event.name,
            'creates_precedent': event.creates_precedent,
        }

    async def _ensure_compliance_cache_table(self) -> None:
        async with get_db_connection_context() as conn:
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._compliance_cache_table} (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    conflict_type TEXT NOT NULL,
                    context_hash TEXT NOT NULL,
                    context JSONB,
                    is_compliant BOOLEAN DEFAULT TRUE,
                    matching_event_ids INTEGER[],
                    matching_tradition_ids INTEGER[],
                    suggestions JSONB,
                    suggestions_status TEXT DEFAULT 'pending',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (user_id, conversation_id, context_hash)
                )
                """
            )

    async def _get_compliance_cache_row(self, context_hash: str) -> Optional[Dict[str, Any]]:
        await self._ensure_compliance_cache_table()
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT * FROM {self._compliance_cache_table}
                 WHERE user_id = $1 AND conversation_id = $2 AND context_hash = $3
                """,
                self.user_id,
                self.conversation_id,
                context_hash,
            )
        return dict(row) if row else None

    async def _upsert_compliance_cache(
        self,
        conflict_type: str,
        context_hash: str,
        context_payload: Dict[str, Any],
        is_compliant: bool,
        matching_event_ids: List[int],
        matching_tradition_ids: List[int],
        suggestions_status: str,
    ) -> int:
        await self._ensure_compliance_cache_table()
        async with get_db_connection_context() as conn:
            cache_id = await conn.fetchval(
                f"""
                INSERT INTO {self._compliance_cache_table}
                    (user_id, conversation_id, conflict_type, context_hash, context,
                     is_compliant, matching_event_ids, matching_tradition_ids, suggestions_status, updated_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9, NOW())
                ON CONFLICT (user_id, conversation_id, context_hash)
                DO UPDATE SET
                    conflict_type = EXCLUDED.conflict_type,
                    context = EXCLUDED.context,
                    is_compliant = EXCLUDED.is_compliant,
                    matching_event_ids = EXCLUDED.matching_event_ids,
                    matching_tradition_ids = EXCLUDED.matching_tradition_ids,
                    suggestions_status = CASE
                        WHEN {self._compliance_cache_table}.suggestions_status = 'ready'
                             AND EXCLUDED.suggestions_status = 'pending' THEN 'ready'
                        ELSE EXCLUDED.suggestions_status
                    END,
                    updated_at = NOW()
                RETURNING id
                """,
                self.user_id,
                self.conversation_id,
                conflict_type,
                context_hash,
                json.dumps(context_payload or {}),
                is_compliant,
                matching_event_ids,
                matching_tradition_ids,
                suggestions_status,
            )
        return int(cache_id)

    async def _queue_compliance_suggestions_task(
        self,
        cache_id: int,
        conflict_type: str,
        conflict_context: Dict[str, Any],
        matching_event_ids: List[int],
    ) -> None:
        try:
            current_app.send_task(
                "canon.generate_lore_suggestions",
                args=[
                    self.user_id,
                    self.conversation_id,
                    cache_id,
                    conflict_type,
                    conflict_context,
                    matching_event_ids,
                ],
                queue="background",
            )
        except Exception as exc:
            logger.exception("Failed to enqueue lore suggestion task", exc_info=exc)

    async def update_compliance_suggestions(
        self,
        cache_id: int,
        suggestions: List[str],
        status: str = 'ready',
        error: Optional[str] = None,
    ) -> None:
        await self._ensure_compliance_cache_table()
        async with get_db_connection_context() as conn:
            await conn.execute(
                f"""
                UPDATE {self._compliance_cache_table}
                   SET suggestions = $2::jsonb,
                       suggestions_status = $3,
                       updated_at = NOW()
                 WHERE id = $1 AND user_id = $4 AND conversation_id = $5
                """,
                cache_id,
                json.dumps(suggestions) if suggestions else (None if status != 'ready' else json.dumps([])),
                status,
                self.user_id,
                self.conversation_id,
            )
        if error:
            logger.warning("Lore compliance suggestions error for cache %s: %s", cache_id, error)

    async def _ensure_reference_cache_table(self) -> None:
        async with get_db_connection_context() as conn:
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._reference_cache_table} (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    event_id INTEGER NOT NULL,
                    context TEXT NOT NULL,
                    references JSONB,
                    status TEXT NOT NULL DEFAULT 'pending',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (user_id, conversation_id, event_id, context)
                )
                """
            )

    async def _get_reference_cache(
        self,
        event_id: int,
        context: str,
    ) -> Optional[Dict[str, Any]]:
        await self._ensure_reference_cache_table()
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT * FROM {self._reference_cache_table}
                 WHERE user_id = $1 AND conversation_id = $2
                   AND event_id = $3 AND context = $4
                """,
                self.user_id,
                self.conversation_id,
                event_id,
                context,
            )
        return dict(row) if row else None

    async def _create_reference_cache_entry(
        self,
        event_id: int,
        context: str,
    ) -> int:
        await self._ensure_reference_cache_table()
        async with get_db_connection_context() as conn:
            cache_id = await conn.fetchval(
                f"""
                INSERT INTO {self._reference_cache_table}
                    (user_id, conversation_id, event_id, context, status, updated_at)
                VALUES ($1, $2, $3, $4, 'pending', NOW())
                ON CONFLICT (user_id, conversation_id, event_id, context)
                DO UPDATE SET status = 'pending', updated_at = NOW()
                RETURNING id
                """,
                self.user_id,
                self.conversation_id,
                event_id,
                context,
            )
        return int(cache_id)

    async def _queue_reference_generation(
        self,
        cache_id: int,
        event_id: int,
        context: str,
    ) -> None:
        try:
            current_app.send_task(
                "canon.generate_canon_references",
                args=[
                    self.user_id,
                    self.conversation_id,
                    cache_id,
                    event_id,
                    context,
                ],
                queue="background",
            )
        except Exception as exc:
            logger.exception("Failed to enqueue canon reference task", exc_info=exc)

    async def update_reference_cache(
        self,
        cache_id: int,
        references: List[str],
        status: str = 'ready',
    ) -> None:
        await self._ensure_reference_cache_table()
        async with get_db_connection_context() as conn:
            await conn.execute(
                f"""
                UPDATE {self._reference_cache_table}
                   SET references = $2::jsonb,
                       status = $3,
                       updated_at = NOW()
                 WHERE id = $1 AND user_id = $4 AND conversation_id = $5
                """,
                cache_id,
                json.dumps(references or []),
                status,
                self.user_id,
                self.conversation_id,
            )

    async def build_reference_cache(
        self,
        cache_id: int,
        event_id: int,
        context: str,
    ) -> None:
        try:
            async with get_db_connection_context() as conn:
                event = await conn.fetchrow(
                    """
                    SELECT * FROM CanonicalEvents
                    WHERE user_id = $1 AND conversation_id = $2
                      AND id = $3
                    """,
                    self.user_id,
                    self.conversation_id,
                    event_id,
                )

                if not event:
                    details = await conn.fetchrow(
                        """
                        SELECT * FROM conflict_canon_details WHERE id = $1
                        """,
                        event_id,
                    )
                    if not details:
                        await self.update_reference_cache(cache_id, [], status='failed')
                        return
                    event_dict = {
                        'event_text': (details.get('metadata') or {}).get('name', 'Unknown Event'),
                        'tags': ['conflict', details.get('event_type', '')],
                        'significance': 5,
                    }
                else:
                    event_dict = dict(event)

            prompt = f"""
Generate NPC references to this canonical event:

Event: {event_dict.get('event_text','')}
Tags: {event_dict.get('tags', [])}
Significance: {event_dict.get('significance', 5)}
Context: {context}

Return JSON:
{{ "references": [{{"text": "..."}}] }}
"""
            response = await Runner.run(self.reference_generator, prompt)
            data = json.loads(extract_runner_response(response))
            refs = [ref.get('text', '') for ref in data.get('references', [])]
            await self.update_reference_cache(cache_id, refs, status='ready')
        except Exception as exc:
            logger.exception("Failed to build reference cache", exc_info=exc)
            await self.update_reference_cache(cache_id, [], status='failed')

    async def build_compliance_suggestions(
        self,
        cache_id: int,
        conflict_type: str,
        conflict_context: Dict[str, Any],
        matching_event_ids: List[int],
    ) -> None:
        try:
            async with get_db_connection_context() as conn:
                related_events = []
                if matching_event_ids:
                    related_events = await conn.fetch(
                        """
                        SELECT id, event_text, tags, significance
                          FROM CanonicalEvents
                         WHERE user_id = $1 AND conversation_id = $2
                           AND id = ANY($3::int[])
                        """,
                        self.user_id,
                        self.conversation_id,
                        matching_event_ids,
                    )

            prompt = f"""
Assess lore guidance for the following conflict.

Conflict Type: {conflict_type}
Context: {json.dumps(conflict_context)}
Matching Canonical Events: {json.dumps([dict(row) for row in related_events])}

Return JSON: {{"suggestions": ["specific player-facing suggestion"]}}
"""
            response = await Runner.run(self.precedent_analyzer, prompt)
            data = json.loads(extract_runner_response(response))
            suggestions = [str(s) for s in (data.get('suggestions') or [])]
            await self.update_compliance_suggestions(cache_id, suggestions, status='ready')
        except Exception as exc:
            logger.exception("Failed to build compliance suggestions", exc_info=exc)
            await self.update_compliance_suggestions(
                cache_id,
                [],
                status='failed',
                error=str(exc),
            )
    
    async def handle_event(self, event) -> SubsystemResponse:
        try:
            if event.event_type == EventType.CONFLICT_RESOLVED:
                conflict_id = event.payload.get('conflict_id')
                resolution_data = event.payload.get('context', {}) or {}

                # Evaluate + possibly create canon, with metadata
                eval_result = await self.evaluate_for_canon(conflict_id, resolution_data)
                canonical_event_id = eval_result.get('canonical_event_id')
                became_canonical = bool(eval_result.get('became_canonical', False))

                side_effects = []
                if became_canonical and canonical_event_id:
                    side_effects.append(SystemEvent(
                        event_id=f"canon_{event.event_id}",
                        event_type=EventType.CANON_ESTABLISHED,
                        source_subsystem=self.subsystem_type,
                        payload={
                            'canonical_event': canonical_event_id,
                            'name': eval_result.get('name'),
                            'significance': eval_result.get('significance'),
                            'creates_precedent': eval_result.get('creates_precedent')
                        },
                        priority=3
                    ))

                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={
                        'became_canonical': became_canonical,
                        'canonical_event': canonical_event_id,
                        'legacy': eval_result.get('legacy'),
                        'snapshot_id': eval_result.get('snapshot_id'),
                        'pending': eval_result.get('pending', not became_canonical),
                        # Provide metadata for tools
                        'reason': eval_result.get('reason', ''),
                        'significance': float(eval_result.get('significance', 0.0) or 0.0),
                        'tags': eval_result.get('tags', []),
                    },
                    side_effects=side_effects
                )
            
            elif event.event_type == EventType.PHASE_TRANSITION:
                if event.payload.get('phase') == 'resolution':
                    conflict_id = event.payload.get('conflict_id')
                    significance = await self._assess_conflict_significance(conflict_id)
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={'monitoring': True, 'significance': significance, 'potential_canon': significance > 0.7},
                        side_effects=[]
                    )
            
            elif event.event_type == EventType.HEALTH_CHECK:
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data=await self.health_check(),
                    side_effects=[]
                )
            
            elif event.event_type == EventType.STATE_SYNC:
                # Orchestrator-routed requests (targeted)
                req = (event.payload or {}).get('request')
                if req == 'check_lore_compliance':
                    conflict_type = event.payload.get('conflict_type', 'unknown')
                    context = {
                        'participants': event.payload.get('participants', []) or [],
                        'location': event.payload.get('location', 'unknown'),
                        'notes': event.payload.get('notes', ''),
                    }
                    result = await self.check_lore_compliance(conflict_type, context)
                    # Normalize minimal shape for tool
                    out = {
                        'is_compliant': bool(result.get('is_compliant', True)),
                        'conflicts': [str(x) for x in (result.get('matching_event_ids') or [])],
                        'matching_event_ids': [int(x) for x in (result.get('matching_event_ids') or [])],
                        'matching_tradition_ids': [int(x) for x in (result.get('matching_tradition_ids') or [])],
                        'suggestions': [str(x) for x in (result.get('suggestions') or [])],
                        'suggestions_pending': bool(result.get('suggestions_pending', False)),
                        'cache_id': result.get('cache_id'),
                    }
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data=out,
                        side_effects=[]
                    )
                elif req == 'generate_canon_references':
                    ev_id = int(event.payload.get('event_id', 0) or 0)
                    style = event.payload.get('context', 'casual') or 'casual'
                    refs = await self.generate_canon_references(ev_id, context=style)
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data=refs,
                        side_effects=[]
                    )
                elif req == 'generate_mythology':
                    conflict_id = int(event.payload.get('conflict_id', 0) or 0)
                    mythology = await self._generate_mythology_text(conflict_id)
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={'mythology': mythology},
                        side_effects=[]
                    )
                # Unknown STATE_SYNC request
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'status': 'no_action_taken'},
                    side_effects=[]
                )
            
            # Default no-op
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={},
                side_effects=[]
            )
        
        except Exception as e:
            logger.error(f"Canon subsystem error: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)},
                side_effects=[]
            )
    
    async def health_check(self) -> Dict[str, Any]:
        async with get_db_connection_context() as conn:
            canon_count = await conn.fetchval("""
                SELECT COUNT(*) FROM CanonicalEvents
                WHERE user_id = $1 AND conversation_id = $2
                  AND tags ? 'conflict'
            """, self.user_id, self.conversation_id)
            
            contradictions = await conn.fetchval("""
                SELECT COUNT(*) FROM (
                    SELECT event_text, COUNT(*) as cnt
                    FROM CanonicalEvents
                    WHERE user_id = $1 AND conversation_id = $2
                      AND tags ? 'precedent'
                    GROUP BY event_text
                    HAVING COUNT(*) > 1
                ) as duplicates
            """, self.user_id, self.conversation_id)
        
        return {
            'healthy': contradictions == 0,
            'canonical_events': int(canon_count or 0),
            'contradictions': int(contradictions or 0),
            'issue': 'Contradictory precedents' if (contradictions or 0) > 0 else None
        }
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        async with get_db_connection_context() as conn:
            canonical = await conn.fetch("""
                SELECT * FROM CanonicalEvents 
                WHERE user_id = $1 AND conversation_id = $2
                  AND tags ? $3
                ORDER BY significance DESC
            """, self.user_id, self.conversation_id, f"conflict_id_{conflict_id}")
            
            precedents = await conn.fetch("""
                SELECT * FROM CanonicalEvents
                WHERE user_id = $1 AND conversation_id = $2
                  AND tags ? 'precedent'
                  AND significance >= 7
                ORDER BY significance DESC
                LIMIT 5
            """, self.user_id, self.conversation_id)
        
        return {
            'canonical_events': [dict(c) for c in canonical],
            'precedents_available': [dict(p) for p in precedents]
        }
    
    async def get_state(self) -> Dict[str, Any]:
        async with get_db_connection_context() as conn:
            total = await conn.fetchval("""
                SELECT COUNT(*) FROM CanonicalEvents
                WHERE user_id = $1 AND conversation_id = $2
                  AND tags ? 'conflict'
            """, self.user_id, self.conversation_id)
            
            precedents = await conn.fetchval("""
                SELECT COUNT(*) FROM CanonicalEvents
                WHERE user_id = $1 AND conversation_id = $2
                  AND tags ? 'precedent'
            """, self.user_id, self.conversation_id)
            
            recent = await conn.fetch("""
                SELECT event_text, significance, timestamp FROM CanonicalEvents
                WHERE user_id = $1 AND conversation_id = $2
                  AND tags ? 'conflict'
                ORDER BY timestamp DESC
                LIMIT 3
            """, self.user_id, self.conversation_id)
        
        return {
            'total_canonical_events': int(total or 0),
            'active_precedents': int(precedents or 0),
            'recent_canon': [dict(r) for r in recent]
        }
    
    async def is_relevant_to_scene(self, scene_context: Dict[str, Any]) -> bool:
        if scene_context.get('resolving_conflict'):
            return True
        if scene_context.get('activity') in ['ceremony', 'ritual', 'court', 'judgment']:
            return True
        return random.random() < 0.2
    
    # ========== Agent Properties ==========
    
    @property
    def lore_integrator(self) -> Agent:
        if self._lore_integrator is None:
            self._lore_integrator = Agent(
                name="Lore Integration Specialist",
                instructions="""
                Integrate conflicts into world lore and canon.

                Ensure that:
                - Conflicts respect established lore
                - Important events become canonical
                - Cultural consistency is maintained
                - History feels organic and interconnected
                """,
                model="gpt-5-nano",
            )
        return self._lore_integrator
    
    @property
    def precedent_analyzer(self) -> Agent:
        if self._precedent_analyzer is None:
            self._precedent_analyzer = Agent(
                name="Precedent Analyzer",
                instructions="""
                Analyze how conflicts create precedents for future events.
                """,
                model="gpt-5-nano",
            )
        return self._precedent_analyzer
    
    @property
    def cultural_interpreter(self) -> Agent:
        if self._cultural_interpreter is None:
            self._cultural_interpreter = Agent(
                name="Cultural Impact Interpreter",
                instructions="""
                Interpret the cultural significance of conflict resolutions.
                """,
                model="gpt-5-nano",
            )
        return self._cultural_interpreter
    
    @property
    def legacy_writer(self) -> Agent:
        if self._legacy_writer is None:
            self._legacy_writer = Agent(
                name="Legacy Writer",
                instructions="""
                Write the lasting legacy of significant conflicts.
                """,
                model="gpt-5-nano",
            )
        return self._legacy_writer
    
    @property
    def reference_generator(self) -> Agent:
        if self._reference_generator is None:
            self._reference_generator = Agent(
                name="Canon Reference Generator",
                instructions="""
                Generate how NPCs and systems reference canonical events.
                """,
                model="gpt-5-nano",
            )
        return self._reference_generator
    
    # ========== Canon Creation Methods ==========
    
    async def evaluate_for_canon(
        self,
        conflict_id: int,
        resolution_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Store snapshot + queue async canonization. Returns immediate placeholder.
        """
        snapshot_id = await self._store_resolution_snapshot(conflict_id, resolution_data)

        try:
            current_app.send_task(
                "canon.canonize_conflict",
                args=[
                    self.user_id,
                    self.conversation_id,
                    int(conflict_id),
                    snapshot_id,
                ],
                queue="background",
            )
        except Exception as exc:
            logger.exception("Failed to enqueue canonization task", exc_info=exc)
            await self._mark_snapshot_status(
                snapshot_id,
                'failed',
                result={'became_canonical': False},
                error=str(exc),
            )

        return {
            'event': None,
            'reason': 'Canon evaluation queued',
            'significance': 0.0,
            'tags': [],
            'became_canonical': False,
            'snapshot_id': snapshot_id,
            'pending': True,
            'canonical_event_id': None,
            'creates_precedent': None,
            'name': None,
            'legacy': None,
        }
    
    async def _create_canonical_event(
        self,
        conflict_id: int,
        conflict: Dict[str, Any],
        resolution_data: Dict[str, Any],
        event_type: CanonEventType,
        significance: float
    ) -> (CanonicalEvent, List[str]):
        """Create a new canonical event using the core canon system (returns event, tags)."""
        prompt = f"""
Create a canonical description for this event:

Conflict: {conflict['conflict_name']}
Resolution: {json.dumps(resolution_data)}
Event Type: {event_type.value}
Significance: {significance:.2f}

Return JSON:
{{
  "canonical_name": "How history will remember this",
  "canonical_description": "2-3 sentence historical record",
  "cultural_impact": {{
    "immediate": "How society reacts",
    "long_term": "Cultural changes over time",
    "traditions_affected": ["existing traditions impacted"],
    "new_traditions": ["potential new traditions"]
  }},
  "creates_precedent": true/false,
  "precedent_description": "What precedent if any"
}}
"""
        response = await Runner.run(self.cultural_interpreter, prompt)
        data = json.loads(extract_runner_response(response))
        
        legacy = await self._generate_legacy(conflict, resolution_data, data)
        
        tags = [
            'conflict',
            'resolution',
            event_type.value,
            conflict['conflict_type'],
            f"conflict_id_{conflict_id}",
            'precedent' if data.get('creates_precedent') else 'event'
        ]
        
        async with get_db_connection_context() as conn:
            await log_canonical_event(
                self.ctx, conn,
                f"{data['canonical_name']}: {data['canonical_description']}",
                tags=tags,
                significance=int(max(1, min(10, round(significance * 10))))  # clamp to 1-10
            )
            # Store detailed record for cross-linking
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conflict_canon_details (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    conflict_id INTEGER,
                    event_type TEXT,
                    cultural_impact JSONB,
                    creates_precedent BOOLEAN,
                    legacy TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            event_id = await conn.fetchval("""
                INSERT INTO conflict_canon_details
                (user_id, conversation_id, conflict_id, event_type, cultural_impact, 
                 creates_precedent, legacy, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            """, self.user_id, self.conversation_id, conflict_id, event_type.value, 
            json.dumps(data.get('cultural_impact', {})), bool(data.get('creates_precedent', False)), 
            legacy, json.dumps({'name': data.get('canonical_name', '')}))
        
        return CanonicalEvent(
            event_id=event_id,
            conflict_id=conflict_id,
            event_type=event_type,
            name=data.get('canonical_name', ''),
            description=data.get('canonical_description', ''),
            significance=significance,
            cultural_impact=data.get('cultural_impact', {}),
            referenced_by=[],
            creates_precedent=bool(data.get('creates_precedent', False)),
            legacy=legacy
        ), tags
    
    async def _generate_legacy(
        self,
        conflict: Dict[str, Any],
        resolution: Dict[str, Any],
        cultural_data: Dict[str, Any]
    ) -> str:
        """Generate the lasting legacy of a canonical event."""
        prompt = f"""
Write the lasting legacy of this canonical event:

Event: {cultural_data.get('canonical_name','')}
Description: {cultural_data.get('canonical_description','')}
Cultural Impact: {json.dumps(cultural_data.get('cultural_impact', {}))}
"""
        response = await Runner.run(self.legacy_writer, prompt)
        return extract_runner_response(response)
    
    async def check_lore_compliance(
        self,
        conflict_type: str,
        conflict_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check lore alignment via vector similarity; queue suggestion generation."""

        context_payload = conflict_context or {}
        context_signature = json.dumps(
            {'type': conflict_type, 'context': context_payload},
            sort_keys=True,
        )
        context_hash = hashlib.sha256(context_signature.encode("utf-8")).hexdigest()

        existing = await self._get_compliance_cache_row(context_hash)

        stale_cache = False
        if existing and existing.get('updated_at'):
            updated_at = existing['updated_at']
            reference_time: Optional[datetime] = None
            if isinstance(updated_at, datetime):
                reference_time = updated_at
                if updated_at.tzinfo is not None:
                    reference_time = updated_at.astimezone(timezone.utc).replace(tzinfo=None)
            elif isinstance(updated_at, str):
                try:
                    parsed = datetime.fromisoformat(updated_at)
                    if parsed.tzinfo is not None:
                        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
                    reference_time = parsed
                except ValueError:
                    reference_time = None
            if reference_time:
                stale_cache = datetime.utcnow() - reference_time > self._compliance_cache_ttl

        conflict_embedding = await generate_embedding(
            f"{conflict_type} {json.dumps(context_payload)}"
        )

        async with get_db_connection_context() as conn:
            similar_events = await conn.fetch(
                """
                SELECT id, event_text, tags, significance,
                       1 - (embedding <=> $1) AS similarity
                  FROM CanonicalEvents
                 WHERE user_id = $2 AND conversation_id = $3
                   AND embedding IS NOT NULL
                   AND 1 - (embedding <=> $1) > 0.7
                 ORDER BY embedding <=> $1
                 LIMIT 5
                """,
                conflict_embedding,
                self.user_id,
                self.conversation_id,
            )

            traditions = await conn.fetch(
                """
                SELECT id, event_text, tags
                  FROM CanonicalEvents
                 WHERE user_id = $1 AND conversation_id = $2
                   AND tags ? 'tradition'
                 ORDER BY timestamp DESC
                 LIMIT 5
                """,
                self.user_id,
                self.conversation_id,
            )

        matching_event_ids = [int(row['id']) for row in similar_events]
        matching_tradition_ids = [int(row['id']) for row in traditions]
        is_compliant = bool(matching_event_ids or matching_tradition_ids)

        suggestions_status = (existing or {}).get('suggestions_status', 'pending') or 'pending'
        suggestions = []
        if stale_cache:
            suggestions_status = 'pending'
        if suggestions_status == 'ready' and not stale_cache:
            stored = existing.get('suggestions') if existing else []
            if isinstance(stored, list):
                suggestions = [str(s) for s in stored]
        else:
            suggestions_status = 'pending'

        cache_id = await self._upsert_compliance_cache(
            conflict_type,
            context_hash,
            context_payload,
            is_compliant,
            matching_event_ids,
            matching_tradition_ids,
            suggestions_status,
        )

        if suggestions_status != 'ready':
            await self._queue_compliance_suggestions_task(
                cache_id,
                conflict_type,
                context_payload,
                matching_event_ids,
            )

        return {
            'is_compliant': is_compliant,
            'matching_event_ids': matching_event_ids,
            'matching_tradition_ids': matching_tradition_ids,
            'suggestions': suggestions,
            'suggestions_pending': suggestions_status != 'ready',
            'cache_id': cache_id,
        }
    
    async def generate_canon_references(
        self,
        event_id: int,
        context: str = "casual"
    ) -> Dict[str, Any]:
        """Return cached canon references, enqueue generation if missing/stale."""

        normalized_context = context or "casual"
        cache = await self._get_reference_cache(event_id, normalized_context)

        references: List[str] = []
        cache_id: Optional[int] = cache.get('id') if cache else None
        status = (cache or {}).get('status', 'pending')
        updated_at = (cache or {}).get('updated_at')
        stale = False
        reference_time: Optional[datetime] = None
        if updated_at:
            if isinstance(updated_at, datetime):
                reference_time = updated_at
                if updated_at.tzinfo is not None:
                    reference_time = updated_at.astimezone(timezone.utc).replace(tzinfo=None)
            elif isinstance(updated_at, str):
                try:
                    parsed = datetime.fromisoformat(updated_at)
                    if parsed.tzinfo is not None:
                        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
                    reference_time = parsed
                except ValueError:
                    stale = True
        if reference_time is not None and not stale:
            stale = datetime.utcnow() - reference_time > self._reference_cache_ttl

        stored_refs = (cache or {}).get('references')
        if isinstance(stored_refs, list):
            references = [str(r) for r in stored_refs]

        should_refresh = cache is None or stale or status in {'pending', 'failed'}

        if cache is None or should_refresh:
            cache_id = await self._create_reference_cache_entry(event_id, normalized_context)
            status = 'pending'
            await self._queue_reference_generation(cache_id, event_id, normalized_context)
        else:
            # If cache exists but not yet ready, ensure generation is queued
            if status != 'ready':
                await self._queue_reference_generation(cache_id, event_id, normalized_context)

        pending = status != 'ready' or should_refresh

        return {
            'references': references,
            'pending': pending,
            'cache_id': cache_id,
        }
    
    async def _create_initial_lore(self):
        """Create initial canonical events using core canon system."""
        async with get_db_connection_context() as conn:
            founding_events = [
                ("The First Accord", "Ancient agreement establishing conflict resolution through dialogue and mutual respect", 10),
                ("The Great Schism", "Historical conflict that shaped modern power structures and social hierarchies", 9),
                ("The Reconciliation", "Legendary peace treaty that created lasting traditions of forgiveness", 8),
                ("The Breaking of Chains", "Revolutionary moment when old oppressive systems were overthrown", 9),
                ("The Council of Equals", "Establishment of fair representation in conflict resolution", 7)
            ]
            for name, description, significance in founding_events:
                await log_canonical_event(
                    self.ctx, conn,
                    f"{name}: {description}",
                    tags=['founding_myth', 'precedent', 'historical', 'conflict'],
                    significance=significance,
                    persist_memory=False,
                )
    
    async def _assess_conflict_significance(self, conflict_id: int) -> float:
        """Assess how significant a conflict is for canon."""
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow("""
                SELECT *, id as conflict_id FROM Conflicts WHERE id = $1
            """, conflict_id)
            stakeholders = await conn.fetchval("""
                SELECT COUNT(*) FROM ConflictStakeholders 
                WHERE conflict_id = $1
            """, conflict_id)
        
        if not conflict:
            return 0.0
        
        base_significance = 0.3
        type_scores = {
            'political': 0.3,
            'faction': 0.25,
            'power': 0.25,
            'social': 0.15,
            'personal': 0.1,
            'background': 0.05
        }
        base_significance += type_scores.get(conflict.get('conflict_type', ''), 0.1)
        base_significance += min(0.2, float(stakeholders or 0) * 0.05)
        progress = float(conflict.get('progress', 0) or 0)
        if progress >= 80:
            base_significance += 0.1
        
        # Proper existence check on tags for this conflict id
        async with get_db_connection_context() as conn:
            canonical_refs = await conn.fetchval("""
                SELECT COUNT(*) FROM CanonicalEvents
                WHERE user_id = $1 AND conversation_id = $2
                  AND tags ? $3
            """, self.user_id, self.conversation_id, f"conflict_id_{conflict_id}")
        if (canonical_refs or 0) > 0:
            base_significance += 0.1
        
        return min(1.0, base_significance)
    
    async def _generate_mythology_text(self, conflict_id: int) -> str:
        """Internal: generate mythological interpretation for a conflict."""
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow("""
                SELECT *, id as conflict_id FROM Conflicts WHERE id = $1
            """, conflict_id)
            canonical_events = await conn.fetch("""
                SELECT event_text, significance FROM CanonicalEvents
                WHERE user_id = $1 AND conversation_id = $2
                  AND tags ? $3
                ORDER BY significance DESC
            """, self.user_id, self.conversation_id, f"conflict_id_{conflict_id}")
        
        if not canonical_events:
            return "This conflict has not yet become part of the canonical lore."
        
        prompt = f"""
Generate the mythological interpretation of this conflict:

Conflict: {conflict['conflict_name'] if conflict else f'Conflict {conflict_id}'}
Type: {conflict['conflict_type'] if conflict else 'Unknown'}
Canonical Events: {json.dumps([dict(e) for e in canonical_events])}

Write 2-3 paragraphs of authentic folklore.
"""
        response = await Runner.run(self.cultural_interpreter, prompt)
        return extract_runner_response(response)


# ===============================================================================
# PUBLIC API FUNCTIONS (via orchestrator)
# ===============================================================================

@function_tool
async def canonize_conflict_resolution(
    ctx: RunContextWrapper,
    conflict_id: int,
    resolution: CanonizationInputDTO,
) -> CanonizationResponse:
    """Evaluate and potentially canonize a conflict resolution (routes through orchestrator)."""
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')

    from logic.conflict_system.conflict_synthesizer import (
        get_synthesizer, SystemEvent, EventType, SubsystemType
    )
    synthesizer = await get_synthesizer(user_id, conversation_id)

    event = SystemEvent(
        event_id=f"canonize_{conflict_id}",
        event_type=EventType.CONFLICT_RESOLVED,
        source_subsystem=SubsystemType.RESOLUTION,
        payload={'conflict_id': conflict_id, 'context': dict(resolution or {})},
        target_subsystems={SubsystemType.CANON},
        requires_response=True,
    )
    responses = await synthesizer.emit_event(event)

    if responses:
        for response in responses:
            if response.subsystem == SubsystemType.CANON:
                data = response.data or {}
                return {
                    'became_canonical': bool(data.get('became_canonical', False)),
                    'canonical_event_id': int(data.get('canonical_event', 0) or 0),
                    'reason': str(data.get('reason', "")),
                    'significance': float(data.get('significance', 0.0) or 0.0),
                    'tags': [str(t) for t in (data.get('tags') or [])],
                    'snapshot_id': data.get('snapshot_id'),
                    'pending': bool(data.get('pending', False)),
                }

    return {
        'became_canonical': False,
        'canonical_event_id': 0,
        'reason': 'Canon system did not respond',
        'significance': 0.0,
        'tags': [],
        'snapshot_id': None,
        'pending': True,
    }


@function_tool
async def check_conflict_lore_alignment(
    ctx: RunContextWrapper,
    conflict_type: str,
    participants: List[int],
) -> LoreAlignmentResponse:
    """Check if a potential conflict aligns with established lore (routes through orchestrator)."""
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')

    from logic.conflict_system.conflict_synthesizer import (
        get_synthesizer, SystemEvent, EventType, SubsystemType
    )
    synthesizer = await get_synthesizer(user_id, conversation_id)

    event = SystemEvent(
        event_id=f"lore_align_{conflict_type}_{datetime.now().timestamp()}",
        event_type=EventType.STATE_SYNC,
        source_subsystem=SubsystemType.CANON,
        payload={
            'request': 'check_lore_compliance',
            'conflict_type': conflict_type,
            'participants': participants,
            'location': ctx.data.get('location', 'unknown'),
        },
        target_subsystems={SubsystemType.CANON},
        requires_response=True,
    )
    responses = await synthesizer.emit_event(event)

    if responses:
        for response in responses:
            if response.subsystem == SubsystemType.CANON:
                data = response.data or {}
                return {
                    'is_compliant': bool(data.get('is_compliant', True)),
                    'conflicts': [str(x) for x in (data.get('conflicts') or [])],
                    'matching_event_ids': [int(x) for x in (data.get('matching_event_ids') or [])],
                    'matching_tradition_ids': [int(x) for x in (data.get('matching_tradition_ids') or [])],
                    'suggestions': [str(x) for x in (data.get('suggestions') or [])],
                    'suggestions_pending': bool(data.get('suggestions_pending', False)),
                    'cache_id': data.get('cache_id'),
                }

    return {
        'is_compliant': True,
        'conflicts': [],
        'matching_event_ids': [],
        'matching_tradition_ids': [],
        'suggestions': ['Canon system not available'],
        'suggestions_pending': True,
        'cache_id': None,
    }


@function_tool
async def get_canonical_precedents(
    ctx: RunContextWrapper,
    situation_type: str,
) -> List[CanonicalPrecedent]:
    """Get relevant canonical precedents for a situation."""
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')

    precedents: List[CanonicalPrecedent] = []

    async with get_db_connection_context() as conn:
        rows = await conn.fetch("""
            SELECT event_text, tags, significance, timestamp
            FROM CanonicalEvents
            WHERE user_id = $1 AND conversation_id = $2
              AND tags ? 'precedent'
              AND (
                tags ? $3 OR
                event_text ILIKE $4
              )
            ORDER BY significance DESC, timestamp DESC
            LIMIT 5
        """, user_id, conversation_id, situation_type, f"%{situation_type}%")

    for p in rows:
        tags = p['tags'] or []
        if not isinstance(tags, list):
            tags = [str(tags)]
        precedents.append({
            'event': str(p['event_text'] or ""),
            'tags': [str(t) for t in tags],
            'significance': float(p['significance'] or 0.0),
            'established': (p['timestamp'].isoformat() if p['timestamp'] else ""),
        })

    return precedents


@function_tool
async def generate_conflict_mythology(
    ctx: RunContextWrapper,
    conflict_id: int,
) -> str:
    """Generate how a conflict has entered mythology (routes through orchestrator)."""
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')

    from logic.conflict_system.conflict_synthesizer import (
        get_synthesizer, SystemEvent, EventType, SubsystemType
    )
    synthesizer = await get_synthesizer(user_id, conversation_id)

    event = SystemEvent(
        event_id=f"myth_{conflict_id}_{datetime.now().timestamp()}",
        event_type=EventType.STATE_SYNC,
        source_subsystem=SubsystemType.CANON,
        payload={'request': 'generate_mythology', 'conflict_id': conflict_id},
        target_subsystems={SubsystemType.CANON},
        requires_response=True,
    )
    responses = await synthesizer.emit_event(event)
    if responses:
        for r in responses:
            if r.subsystem == SubsystemType.CANON:
                return str((r.data or {}).get('mythology', "")) or "The conflict has not yet entered mythology."
    return "The conflict has not yet entered mythology."
