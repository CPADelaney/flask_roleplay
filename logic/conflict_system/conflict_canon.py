# logic/conflict_system/conflict_canon.py
"""
Conflict Canon System integrated with Core Lore Canon
Manages how conflicts become part of world lore through the unified canon system.
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass

from agents import Agent, function_tool, ModelSettings, RunContextWrapper, Runner
from db.connection import get_db_connection_context

# Core canon system imports
from lore.core.canon import (
    log_canonical_event,
    ensure_canonical_context,
    find_or_create_entity,
    update_entity_with_governance
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
    referenced_by: List[str]  # NPCs or systems that reference this
    creates_precedent: bool
    legacy: str


# ===============================================================================
# CONFLICT CANON SUBSYSTEM (Integrated with Core Canon)
# ===============================================================================

class ConflictCanonSubsystem:
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
        
        # Reference to synthesizer
        self.synthesizer = None
    
    @property
    def subsystem_type(self):
        """Return the subsystem type"""
        from logic.conflict_system.conflict_synthesizer import SubsystemType
        return SubsystemType.CANON
    
    @property
    def capabilities(self) -> Set[str]:
        """Return capabilities this subsystem provides"""
        return {
            'lore_integration',
            'precedent_tracking',
            'cultural_impact',
            'legacy_creation',
            'reference_generation',
            'tradition_establishment'
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
            EventType.CONFLICT_RESOLVED,
            EventType.PHASE_TRANSITION,
            EventType.HEALTH_CHECK,
            EventType.CANON_ESTABLISHED
        }
    
    async def initialize(self, synthesizer) -> bool:
        """Initialize the subsystem with synthesizer reference"""
        import weakref
        self.synthesizer = weakref.ref(synthesizer)
        
        # Check for existing canonical events in the unified system
        async with get_db_connection_context() as conn:
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM CanonicalEvents
                WHERE user_id = $1 AND conversation_id = $2
                AND 'conflict' = ANY(tags)
            """, self.user_id, self.conversation_id)
            
            if count == 0:
                # Create initial lore seeds
                await self._create_initial_lore()
        
        return True
    
    async def handle_event(self, event) -> Any:
        """Handle an event from the synthesizer"""
        from logic.conflict_system.conflict_synthesizer import SubsystemResponse, SystemEvent, EventType
        
        try:
            if event.event_type == EventType.CONFLICT_RESOLVED:
                # Evaluate if resolution should become canonical
                conflict_id = event.payload.get('conflict_id')
                resolution_data = event.payload.get('context', {})
                
                canonical_event = await self.evaluate_for_canon(
                    conflict_id, resolution_data
                )
                
                side_effects = []
                if canonical_event:
                    # Notify all systems of new canon
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
                        'legacy': canonical_event.legacy if canonical_event else None
                    },
                    side_effects=side_effects
                )
                
            elif event.event_type == EventType.PHASE_TRANSITION:
                # Check if phase transition is significant enough for canon
                if event.payload.get('phase') == 'resolution':
                    conflict_id = event.payload.get('conflict_id')
                    
                    # Check significance
                    significance = await self._assess_conflict_significance(conflict_id)
                    
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={
                            'monitoring': True,
                            'significance': significance,
                            'potential_canon': significance > 0.7
                        }
                    )
                    
            elif event.event_type == EventType.HEALTH_CHECK:
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data=await self.health_check()
                )
            
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={}
            )
            
        except Exception as e:
            logger.error(f"Canon subsystem error: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)}
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Return health status of the subsystem"""
        async with get_db_connection_context() as conn:
            # Check canonical events in unified system
            canon_count = await conn.fetchval("""
                SELECT COUNT(*) FROM CanonicalEvents
                WHERE user_id = $1 AND conversation_id = $2
                AND 'conflict' = ANY(tags)
            """, self.user_id, self.conversation_id)
            
            # Check for contradictions
            contradictions = await conn.fetchval("""
                SELECT COUNT(*) FROM (
                    SELECT event_text, COUNT(*) as cnt
                    FROM CanonicalEvents
                    WHERE user_id = $1 AND conversation_id = $2
                    AND 'precedent' = ANY(tags)
                    GROUP BY event_text
                    HAVING COUNT(*) > 1
                ) as duplicates
            """, self.user_id, self.conversation_id)
        
        return {
            'healthy': contradictions == 0,
            'canonical_events': canon_count,
            'contradictions': contradictions,
            'issue': 'Contradictory precedents' if contradictions > 0 else None
        }
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        """Get canon-related data for a specific conflict"""
        async with get_db_connection_context() as conn:
            # Get canonical events related to this conflict
            canonical = await conn.fetch("""
                SELECT * FROM CanonicalEvents 
                WHERE user_id = $1 AND conversation_id = $2
                AND $3 = ANY(tags)
                ORDER BY significance DESC
            """, self.user_id, self.conversation_id, f"conflict_id_{conflict_id}")
            
            # Get precedents
            precedents = await conn.fetch("""
                SELECT * FROM CanonicalEvents
                WHERE user_id = $1 AND conversation_id = $2
                AND 'precedent' = ANY(tags)
                AND significance >= 7
                ORDER BY significance DESC
                LIMIT 5
            """, self.user_id, self.conversation_id)
        
        return {
            'canonical_events': [dict(c) for c in canonical],
            'precedents_available': [dict(p) for p in precedents]
        }
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current state of canon system"""
        async with get_db_connection_context() as conn:
            total = await conn.fetchval("""
                SELECT COUNT(*) FROM CanonicalEvents
                WHERE user_id = $1 AND conversation_id = $2
                AND 'conflict' = ANY(tags)
            """, self.user_id, self.conversation_id)
            
            precedents = await conn.fetchval("""
                SELECT COUNT(*) FROM CanonicalEvents
                WHERE user_id = $1 AND conversation_id = $2
                AND 'precedent' = ANY(tags)
            """, self.user_id, self.conversation_id)
            
            recent = await conn.fetch("""
                SELECT event_text, significance FROM CanonicalEvents
                WHERE user_id = $1 AND conversation_id = $2
                AND 'conflict' = ANY(tags)
                ORDER BY timestamp DESC
                LIMIT 3
            """, self.user_id, self.conversation_id)
        
        return {
            'total_canonical_events': total,
            'active_precedents': precedents,
            'recent_canon': [dict(r) for r in recent]
        }
    
    async def is_relevant_to_scene(self, scene_context: Dict[str, Any]) -> bool:
        """Check if canon system is relevant to scene"""
        # Canon is relevant when:
        # - Conflicts are being resolved
        # - NPCs might reference past events
        # - Cultural traditions are involved
        
        if scene_context.get('resolving_conflict'):
            return True
            
        if scene_context.get('activity') in ['ceremony', 'ritual', 'court', 'judgment']:
            return True
            
        # Random chance for NPCs to reference canon
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
                - Matriarchal themes are reinforced
                - History feels organic and interconnected
                
                Create lore that feels ancient even when newly created.
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
                
                Identify:
                - New social norms established
                - Behavioral patterns legitimized
                - Power dynamics codified
                - Boundaries redefined
                - Traditions started
                
                Show how today's conflicts become tomorrow's traditions.
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
                
                Analyze:
                - How society views the outcome
                - Cultural meanings assigned
                - Mythological interpretations
                - Social lessons drawn
                - Collective memory formation
                
                Transform personal conflicts into cultural touchstones.
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
                
                Create legacies that:
                - Feel historically significant
                - Influence future generations
                - Create cultural echoes
                - Establish new archetypes
                - Generate folklore and stories
                
                Make conflicts feel like they matter beyond the moment.
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
                
                Create references that:
                - Feel natural in conversation
                - Show different perspectives
                - Evolve over time
                - Build mythology
                - Connect to current events
                
                Make the past feel alive in the present.
                """,
                model="gpt-5-nano",
            )
        return self._reference_generator
    
    # ========== Canon Creation Methods ==========
    
    async def evaluate_for_canon(
        self,
        conflict_id: int,
        resolution_data: Dict[str, Any]
    ) -> Optional[CanonicalEvent]:
        """Evaluate if a conflict resolution should become canonical"""
        
        # Get conflict details
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow("""
                SELECT * FROM Conflicts WHERE conflict_id = $1
            """, conflict_id)
            
            if not conflict:
                return None
            
            stakeholders = await conn.fetch("""
                SELECT * FROM conflict_stakeholders WHERE conflict_id = $1
            """, conflict_id)
        
        prompt = f"""
        Evaluate if this conflict resolution should become canonical:
        
        Conflict: {conflict['conflict_name']}
        Type: {conflict['conflict_type']}
        Resolution: {json.dumps(resolution_data)}
        Stakeholders: {len(stakeholders)}
        
        Determine if this creates:
        - Historical precedent
        - Cultural shift
        - Social evolution
        - Power restructuring
        
        Return JSON:
        {{
            "should_be_canonical": true/false,
            "reason": "Why this matters (or doesn't)",
            "event_type": "canonical event type if applicable",
            "significance": 0.0 to 1.0,
            "potential_legacy": "What this could mean long-term"
        }}
        """
        
        response = await Runner.run(self.lore_integrator, prompt)
        data = json.loads(response.output)
        
        if not data['should_be_canonical']:
            return None
        
        # Create canonical event
        return await self._create_canonical_event(
            conflict_id,
            conflict,
            resolution_data,
            CanonEventType(data['event_type']),
            data['significance']
        )
    
    async def _create_canonical_event(
        self,
        conflict_id: int,
        conflict: Dict[str, Any],
        resolution_data: Dict[str, Any],
        event_type: CanonEventType,
        significance: float
    ) -> CanonicalEvent:
        """Create a new canonical event using the core canon system"""
        
        # Generate canonical description
        prompt = f"""
        Create a canonical description for this event:
        
        Conflict: {conflict['conflict_name']}
        Resolution: {json.dumps(resolution_data)}
        Event Type: {event_type.value}
        Significance: {significance:.1%}
        
        Write as if this is a historical event being recorded.
        
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
            "precedent_description": "What precedent if any",
            "mythological_interpretation": "How this enters folklore"
        }}
        """
        
        response = await Runner.run(self.cultural_interpreter, prompt)
        data = json.loads(response.output)
        
        # Generate legacy
        legacy = await self._generate_legacy(conflict, resolution_data, data)
        
        async with get_db_connection_context() as conn:
            # Use the core canon system to log the event
            await log_canonical_event(
                self.ctx, conn,
                f"{data['canonical_name']}: {data['canonical_description']}",
                tags=[
                    'conflict',
                    'resolution',
                    event_type.value,
                    conflict['conflict_type'],
                    f"conflict_id_{conflict_id}",
                    'precedent' if data['creates_precedent'] else 'event'
                ],
                significance=int(significance * 10)  # Convert 0-1 to 1-10 scale
            )
            
            # Store additional conflict-specific data
            # First ensure the table exists
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            
            event_id = await conn.fetchval("""
                INSERT INTO conflict_canon_details
                (user_id, conversation_id, conflict_id, event_type, cultural_impact, 
                 creates_precedent, legacy, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            """, self.user_id, self.conversation_id, conflict_id, event_type.value, 
            json.dumps(data['cultural_impact']), data['creates_precedent'], 
            legacy, json.dumps({'name': data['canonical_name']}))
        
        return CanonicalEvent(
            event_id=event_id,
            conflict_id=conflict_id,
            event_type=event_type,
            name=data['canonical_name'],
            description=data['canonical_description'],
            significance=significance,
            cultural_impact=data['cultural_impact'],
            referenced_by=[],
            creates_precedent=data['creates_precedent'],
            legacy=legacy
        )
    
    async def _generate_legacy(
        self,
        conflict: Dict[str, Any],
        resolution: Dict[str, Any],
        cultural_data: Dict[str, Any]
    ) -> str:
        """Generate the lasting legacy of a canonical event"""
        
        prompt = f"""
        Write the lasting legacy of this canonical event:
        
        Event: {cultural_data['canonical_name']}
        Description: {cultural_data['canonical_description']}
        Cultural Impact: {json.dumps(cultural_data['cultural_impact'])}
        Mythological: {cultural_data.get('mythological_interpretation', '')}
        
        Write a powerful paragraph about how this event echoes through time.
        Focus on:
        - How future generations remember it
        - Lessons society draws from it
        - How it changes social dynamics
        - Its place in cultural memory
        
        Make it feel like a pivotal moment in history.
        """
        
        response = await Runner.run(self.legacy_writer, prompt)
        return response.output
    
    async def check_lore_compliance(
        self,
        conflict_type: str,
        conflict_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if a conflict aligns with established lore using core canon"""
        
        async with get_db_connection_context() as conn:
            # Use the core canon system's canonical events table
            canonical_events = await conn.fetch("""
                SELECT * FROM CanonicalEvents
                WHERE user_id = $1 AND conversation_id = $2
                AND significance >= 7
                ORDER BY significance DESC, timestamp DESC
                LIMIT 10
            """, self.user_id, self.conversation_id)
            
            # Check for related conflicts using semantic search
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
            
            # Get active traditions from core canon
            traditions = await conn.fetch("""
                SELECT event_text, tags FROM CanonicalEvents
                WHERE user_id = $1 AND conversation_id = $2
                AND 'tradition' = ANY(tags)
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
        
        Analyze:
        - Does this respect established precedents?
        - Does it honor cultural traditions?
        - Are there lore conflicts?
        - How can it build on existing canon?
        
        Return JSON:
        {{
            "is_compliant": true/false,
            "conflicts": ["any lore conflicts"],
            "precedents_referenced": ["relevant precedents"],
            "traditions_involved": ["relevant traditions"],
            "suggestions": ["how to better align with lore"],
            "enrichment_opportunities": ["ways to deepen lore connection"]
        }}
        """
        
        response = await Runner.run(self.precedent_analyzer, prompt)
        return json.loads(response.output)
    
    async def generate_canon_references(
        self,
        event_id: int,
        context: str = "casual"
    ) -> List[str]:
        """Generate how NPCs might reference a canonical event"""
        
        # Get event details from unified canon
        async with get_db_connection_context() as conn:
            event = await conn.fetchrow("""
                SELECT * FROM CanonicalEvents 
                WHERE user_id = $1 AND conversation_id = $2
                AND id = $3
            """, self.user_id, self.conversation_id, event_id)
            
            if not event:
                # Try conflict_canon_details as fallback
                details = await conn.fetchrow("""
                    SELECT * FROM conflict_canon_details WHERE id = $1
                """, event_id)
                
                if not details:
                    return []
                    
                # Use details to generate references
                event = {
                    'event_text': details.get('metadata', {}).get('name', 'Unknown Event'),
                    'tags': ['conflict', details.get('event_type', '')],
                    'significance': 5
                }
        
        prompt = f"""
        Generate NPC references to this canonical event:
        
        Event: {event['event_text']}
        Tags: {event.get('tags', [])}
        Significance: {event.get('significance', 5)}
        Context: {context}
        
        Create 5 different ways NPCs might reference this:
        - Casual mentions
        - Teaching moments
        - Warnings or lessons
        - Mythologized versions
        - Personal interpretations
        
        Return JSON:
        {{
            "references": [
                {{
                    "style": "casual/formal/mythological/warning/nostalgic",
                    "text": "What the NPC says",
                    "subtext": "What they really mean"
                }}
            ]
        }}
        """
        
        response = await Runner.run(self.reference_generator, prompt)
        data = json.loads(response.output)
        
        return [ref['text'] for ref in data['references']]
    
    # ========== Helper Methods ==========
    
    async def _create_initial_lore(self):
        """Create initial canonical events using core canon system"""
        
        async with get_db_connection_context() as conn:
            # Create founding precedents using the core canon system
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
        """Assess how significant a conflict is for canon"""
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow("""
                SELECT * FROM Conflicts WHERE conflict_id = $1
            """, conflict_id)
            
            stakeholders = await conn.fetchval("""
                SELECT COUNT(*) FROM conflict_stakeholders 
                WHERE conflict_id = $1
            """, conflict_id)
        
        if not conflict:
            return 0.0
        
        # Calculate significance
        base_significance = 0.3
        
        # Type-based significance
        type_scores = {
            'political': 0.3,
            'faction': 0.25,
            'power': 0.25,
            'social': 0.15,
            'personal': 0.1,
            'background': 0.05
        }
        base_significance += type_scores.get(conflict.get('conflict_type', ''), 0.1)
        
        # Multiple stakeholders add significance
        base_significance += min(0.2, stakeholders * 0.05)
        
        # Progress/resolution adds significance
        progress = conflict.get('progress', 0)
        if progress >= 80:
            base_significance += 0.1
        
        # Check for canonical references
        canonical_refs = await conn.fetchval("""
            SELECT COUNT(*) FROM CanonicalEvents
            WHERE user_id = $1 AND conversation_id = $2
            AND $3 = ANY(tags)
        """, self.user_id, self.conversation_id, f"conflict_id_{conflict_id}")
        
        if canonical_refs > 0:
            base_significance += 0.1
        
        return min(1.0, base_significance)


# ===============================================================================
# PUBLIC API FUNCTIONS
# ===============================================================================

@function_tool
async def canonize_conflict_resolution(
    ctx: RunContextWrapper,
    conflict_id: int,
    resolution_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate and potentially canonize a conflict resolution"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    # Get synthesizer and emit event
    from logic.conflict_system.conflict_synthesizer import get_synthesizer, SystemEvent, EventType, SubsystemType
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    event = SystemEvent(
        event_id=f"canonize_{conflict_id}",
        event_type=EventType.CONFLICT_RESOLVED,
        source_subsystem=SubsystemType.RESOLUTION,
        payload={
            'conflict_id': conflict_id,
            'context': resolution_data
        },
        target_subsystems={SubsystemType.CANON},
        requires_response=True
    )
    
    responses = await synthesizer.emit_event(event)
    
    if responses:
        for response in responses:
            if response.subsystem == SubsystemType.CANON:
                return response.data
    
    return {
        'became_canonical': False,
        'reason': 'Canon system did not respond'
    }


@function_tool
async def check_conflict_lore_alignment(
    ctx: RunContextWrapper,
    conflict_type: str,
    participants: List[int]
) -> Dict[str, Any]:
    """Check if a potential conflict aligns with established lore"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    # Get canon subsystem through synthesizer
    from logic.conflict_system.conflict_synthesizer import get_synthesizer, SubsystemType
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    canon_subsystem = synthesizer._subsystems.get(SubsystemType.CANON)
    if canon_subsystem:
        context = {
            'conflict_type': conflict_type,
            'participants': participants,
            'location': ctx.data.get('location', 'unknown')
        }
        
        return await canon_subsystem.check_lore_compliance(conflict_type, context)
    
    return {
        'is_compliant': True,
        'conflicts': [],
        'suggestions': ['Canon system not available']
    }


@function_tool
async def get_canonical_precedents(
    ctx: RunContextWrapper,
    situation_type: str
) -> List[Dict[str, Any]]:
    """Get relevant canonical precedents for a situation"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    async with get_db_connection_context() as conn:
        # Search in unified CanonicalEvents table
        precedents = await conn.fetch("""
            SELECT event_text, tags, significance, timestamp
            FROM CanonicalEvents
            WHERE user_id = $1 AND conversation_id = $2
            AND 'precedent' = ANY(tags)
            AND (
                $3 = ANY(tags) OR
                event_text ILIKE $4
            )
            ORDER BY significance DESC, timestamp DESC
            LIMIT 5
        """, user_id, conversation_id, situation_type, f"%{situation_type}%")
    
    return [
        {
            'event': p['event_text'],
            'tags': p['tags'],
            'significance': p['significance'],
            'established': p['timestamp'].isoformat() if p['timestamp'] else 'Unknown'
        }
        for p in precedents
    ]


@function_tool
async def generate_conflict_mythology(
    ctx: RunContextWrapper,
    conflict_id: int
) -> str:
    """Generate how a conflict has entered mythology"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    # Get canon subsystem
    from logic.conflict_system.conflict_synthesizer import get_synthesizer, SubsystemType
    synthesizer = await get_synthesizer(user_id, conversation_id)
    
    canon_subsystem = synthesizer._subsystems.get(SubsystemType.CANON)
    if not canon_subsystem:
        return "The conflict has not yet entered mythology."
    
    # Get conflict details
    async with get_db_connection_context() as conn:
        conflict = await conn.fetchrow("""
            SELECT * FROM Conflicts WHERE conflict_id = $1
        """, conflict_id)
        
        # Get canonical events for this conflict
        canonical_events = await conn.fetch("""
            SELECT event_text, significance FROM CanonicalEvents
            WHERE user_id = $1 AND conversation_id = $2
            AND $3 = ANY(tags)
            ORDER BY significance DESC
        """, user_id, conversation_id, f"conflict_id_{conflict_id}")
    
    if not canonical_events:
        return "This conflict has not yet become part of the canonical lore."
    
    prompt = f"""
    Generate the mythological interpretation of this conflict:
    
    Conflict: {conflict['conflict_name'] if conflict else f'Conflict {conflict_id}'}
    Type: {conflict['conflict_type'] if conflict else 'Unknown'}
    Canonical Events: {json.dumps([dict(e) for e in canonical_events])}
    
    Write 2-3 paragraphs about how this conflict has entered mythology.
    Include:
    - How common people tell the story
    - What lessons or morals are drawn
    - How it has been embellished over time
    - What symbols or metaphors represent it
    
    Make it feel like authentic folklore.
    """
    
    response = await Runner.run(canon_subsystem.cultural_interpreter, prompt)
    return response.output
