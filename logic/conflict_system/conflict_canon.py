# logic/conflict_system/conflict_canon.py
"""
Conflict Canon System integrated with Core Lore Canon
Manages how conflicts become part of world lore through the unified canon system.
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Set, TypedDict
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

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


class LoreAlignmentResponse(TypedDict):
    is_compliant: bool
    conflicts: List[str]
    suggestions: List[str]


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
    
    async def handle_event(self, event) -> SubsystemResponse:
        try:
            if event.event_type == EventType.CONFLICT_RESOLVED:
                conflict_id = event.payload.get('conflict_id')
                resolution_data = event.payload.get('context', {}) or {}
                
                # Evaluate + possibly create canon, with metadata
                eval_result = await self.evaluate_for_canon(conflict_id, resolution_data)
                canonical_event = eval_result.get('event')
                
                side_effects = []
                if canonical_event:
                    side_effects.append(SystemEvent(
                        event_id=f"canon_{event.event_id}",
                        event_type=EventType.CANON_ESTABLISHED,
                        source_subsystem=self.subsystem_type,
                        payload={
                            'canonical_event': canonical_event.event_id,
                            'name': canonical_event.name,
                            'significance': canonical_event.significance,
                            'creates_precedent': canonical_event.creates_precedent
                        },
                        priority=3
                    ))
                
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={
                        'became_canonical': canonical_event is not None,
                        'canonical_event': canonical_event.event_id if canonical_event else None,
                        'legacy': canonical_event.legacy if canonical_event else None,
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
                        'conflicts': [str(x) for x in (result.get('conflicts') or [])],
                        'suggestions': [str(x) for x in (result.get('suggestions') or [])],
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
                        data={'references': refs},
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
        Evaluate if a conflict resolution should become canonical.
        Returns dict: {'event': CanonicalEvent|None, 'reason': str, 'significance': float, 'tags': List[str]}
        """
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow("""
                SELECT *, id as conflict_id FROM Conflicts WHERE id = $1
            """, conflict_id)
            if not conflict:
                return {'event': None, 'reason': 'Conflict not found', 'significance': 0.0, 'tags': []}
            stakeholders = await conn.fetch("""
                SELECT * FROM ConflictStakeholders WHERE conflict_id = $1
            """, conflict_id)
        
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
            return {'event': None, 'reason': reason, 'significance': significance, 'tags': []}
        
        event_type_str = data.get('event_type', CanonEventType.LEGENDARY_MOMENT.value)
        event_type = CanonEventType(event_type_str)
        # Create canonical event and capture tags used
        event, tags = await self._create_canonical_event(
            conflict_id, conflict, resolution_data, event_type, significance
        )
        return {'event': event, 'reason': reason, 'significance': significance, 'tags': tags}
    
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
        """Check if a conflict aligns with established lore using core canon."""
        async with get_db_connection_context() as conn:
            canonical_events = await conn.fetch("""
                SELECT * FROM CanonicalEvents
                WHERE user_id = $1 AND conversation_id = $2
                  AND tags @> '["precedent"]'::jsonb
                  AND significance >= 7
                ORDER BY significance DESC
                LIMIT 5
            """, self.user_id, self.conversation_id)
            
            conflict_embedding = await generate_embedding(
                f"{conflict_type} {json.dumps(conflict_context)}"
            )
            
            similar_events = await conn.fetch("""
                SELECT event_text, tags, significance,
                       1 - (embedding <=> $1) AS similarity
                FROM CanonicalEvents
                WHERE user_id = $2 AND conversation_id = $3
                  AND embedding IS NOT NULL
                  AND 1 - (embedding <=> $1) > 0.7
                ORDER BY embedding <=> $1
                LIMIT 5
            """, conflict_embedding, self.user_id, self.conversation_id)
            
            traditions = await conn.fetch("""
                SELECT event_text, tags FROM CanonicalEvents
                WHERE user_id = $1 AND conversation_id = $2
                  AND tags ? 'tradition'
                ORDER BY timestamp DESC
                LIMIT 5
            """, self.user_id, self.conversation_id)
        
        prompt = f"""
Check if this conflict aligns with established lore:

Conflict Type: {conflict_type}
Context: {json.dumps(conflict_context)}
Related Canon: {json.dumps([dict(e) for e in similar_events])}
High-Significance Events: {json.dumps([dict(e) for e in canonical_events[:3]])}
Active Traditions: {json.dumps([dict(t) for t in traditions])}

Return JSON:
{{
  "is_compliant": true/false,
  "conflicts": ["any lore conflicts"],
  "suggestions": ["how to better align with lore"]
}}
"""
        response = await Runner.run(self.precedent_analyzer, prompt)
        return json.loads(extract_runner_response(response))
    
    async def generate_canon_references(
        self,
        event_id: int,
        context: str = "casual"
    ) -> List[str]:
        """Generate how NPCs might reference a canonical event."""
        async with get_db_connection_context() as conn:
            event = await conn.fetchrow("""
                SELECT * FROM CanonicalEvents 
                WHERE user_id = $1 AND conversation_id = $2
                  AND id = $3
            """, self.user_id, self.conversation_id, event_id)
            
            if not event:
                details = await conn.fetchrow("""
                    SELECT * FROM conflict_canon_details WHERE id = $1
                """, event_id)
                if not details:
                    return []
                event_dict = {
                    'event_text': (details.get('metadata') or {}).get('name', 'Unknown Event'),
                    'tags': ['conflict', details.get('event_type', '')],
                    'significance': 5
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
        return [ref.get('text', '') for ref in data.get('references', [])]
    
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
                    significance=significance
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
                }

    return {
        'became_canonical': False,
        'canonical_event_id': 0,
        'reason': 'Canon system did not respond',
        'significance': 0.0,
        'tags': [],
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
                    'suggestions': [str(x) for x in (data.get('suggestions') or [])],
                }

    return {
        'is_compliant': True,
        'conflicts': [],
        'suggestions': ['Canon system not available'],
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
