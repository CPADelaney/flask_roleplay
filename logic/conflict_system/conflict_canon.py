# logic/conflict_system/conflict_canon.py
"""
Conflict Canon System with LLM-generated lore integration
Ensures conflicts align with established world lore and create canonical events
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from agents import Agent, function_tool, ModelSettings, RunContextWrapper, Runner
from db.connection import get_db_connection_context

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
# CONFLICT CANON MANAGER
# ===============================================================================

class ConflictCanonManager:
    """Manages how conflicts become part of world lore"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Lazy-loaded agents
        self._lore_integrator = None
        self._precedent_analyzer = None
        self._cultural_interpreter = None
        self._legacy_writer = None
        self._reference_generator = None
    
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
            
            stakeholders = await conn.fetch("""
                SELECT * FROM conflict_stakeholders WHERE conflict_id = $1
            """, conflict_id)
        
        prompt = f"""
        Evaluate if this conflict resolution should become canonical:
        
        Conflict: {conflict['conflict_name']}
        Type: {conflict['conflict_type']}
        Intensity: {conflict['intensity']}
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
        """Create a new canonical event"""
        
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
        
        # Store in database
        async with get_db_connection_context() as conn:
            event_id = await conn.fetchval("""
                INSERT INTO canonical_events
                (conflict_id, event_type, name, description,
                 significance, cultural_impact, creates_precedent, legacy)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING event_id
            """, conflict_id, event_type.value, data['canonical_name'],
            data['canonical_description'], significance,
            json.dumps(data['cultural_impact']),
            data['creates_precedent'], legacy)
        
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
        """Check if a conflict aligns with established lore"""
        
        # Get relevant lore
        async with get_db_connection_context() as conn:
            canonical_events = await conn.fetch("""
                SELECT * FROM canonical_events
                WHERE creates_precedent = true
                ORDER BY significance DESC
                LIMIT 10
            """)
            
            traditions = await conn.fetch("""
                SELECT * FROM cultural_traditions
                WHERE is_active = true
            """)
        
        prompt = f"""
        Check if this conflict aligns with established lore:
        
        Conflict Type: {conflict_type}
        Context: {json.dumps(conflict_context)}
        Established Precedents: {json.dumps([dict(e) for e in canonical_events])}
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

    async def process_event(self, conflict_id: int, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process conflict events for this subsystem"""
        event_type = event.get('type', 'unknown')
        
        # Route to appropriate handler
        if event_type == 'your_specific_type':
            return await self.handle_specific_event(conflict_id, event)
        
        return {'processed': True, 'subsystem': 'module_name'}
    
    async def generate_canon_references(
        self,
        event_id: int,
        context: str = "casual"
    ) -> List[str]:
        """Generate how NPCs might reference a canonical event"""
        
        # Get event details
        async with get_db_connection_context() as conn:
            event = await conn.fetchrow("""
                SELECT * FROM canonical_events WHERE event_id = $1
            """, event_id)
        
        prompt = f"""
        Generate NPC references to this canonical event:
        
        Event: {event['name']}
        Description: {event['description']}
        Legacy: {event['legacy']}
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
    
    async def evolve_canon_over_time(
        self,
        event_id: int,
        time_passed: int  # Days since event
    ) -> Dict[str, Any]:
        """Evolve how a canonical event is remembered over time"""
        
        # Get event
        async with get_db_connection_context() as conn:
            event = await conn.fetchrow("""
                SELECT * FROM canonical_events WHERE event_id = $1
            """, event_id)
        
        prompt = f"""
        Show how this canonical event has evolved in memory:
        
        Original Event: {event['name']}
        Original Description: {event['description']}
        Time Passed: {time_passed} days
        
        Generate how the story has changed:
        - What details got embellished
        - What got forgotten
        - What new meanings emerged
        - How different groups remember it
        
        Return JSON:
        {{
            "current_version": "How it's told now",
            "embellishments": ["details that grew"],
            "forgotten_aspects": ["what was lost"],
            "new_interpretations": ["modern meanings"],
            "competing_narratives": [
                {{
                    "group": "who tells it this way",
                    "version": "their version"
                }}
            ],
            "mythological_evolution": "How it entered mythology"
        }}
        """
        
        response = await Runner.run(self.cultural_interpreter, prompt)
        evolution = json.loads(response.output)
        
        # Update the canon
        await conn.execute("""
            UPDATE canonical_events
            SET evolved_description = $1,
                last_evolution = CURRENT_TIMESTAMP
            WHERE event_id = $2
        """, evolution['current_version'], event_id)
        
        return evolution


# ===============================================================================
# INTEGRATION FUNCTIONS
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
    
    manager = ConflictCanonManager(user_id, conversation_id)
    
    # Evaluate for canon
    canonical_event = await manager.evaluate_for_canon(conflict_id, resolution_data)
    
    if canonical_event:
        # Generate initial references
        references = await manager.generate_canon_references(
            canonical_event.event_id,
            "formal"
        )
        
        return {
            'became_canonical': True,
            'event_name': canonical_event.name,
            'significance': canonical_event.significance,
            'legacy': canonical_event.legacy,
            'sample_references': references[:3]
        }
    else:
        return {
            'became_canonical': False,
            'reason': 'Not significant enough for canon'
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
    
    manager = ConflictCanonManager(user_id, conversation_id)
    
    context = {
        'conflict_type': conflict_type,
        'participants': participants,
        'location': ctx.data.get('location', 'unknown')
    }
    
    compliance = await manager.check_lore_compliance(conflict_type, context)
    
    return compliance


@function_tool
async def get_canonical_precedents(
    ctx: RunContextWrapper,
    situation_type: str
) -> List[Dict[str, Any]]:
    """Get relevant canonical precedents for a situation"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    async with get_db_connection_context() as conn:
        precedents = await conn.fetch("""
            SELECT ce.*, c.conflict_type
            FROM canonical_events ce
            JOIN Conflicts c ON ce.conflict_id = c.conflict_id
            WHERE ce.creates_precedent = true
            AND (c.conflict_type LIKE $1 OR ce.event_type LIKE $1)
            ORDER BY ce.significance DESC
            LIMIT 5
        """, f"%{situation_type}%")
    
    return [
        {
            'event': p['name'],
            'description': p['description'],
            'precedent': p['legacy'],
            'significance': p['significance']
        }
        for p in precedents
    ]
