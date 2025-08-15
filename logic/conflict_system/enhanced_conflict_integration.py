# logic/conflict_system/enhanced_conflict_integration.py
"""
Enhanced conflict system integration with LLM-generated dynamic content.
Works through ConflictSynthesizer as the central orchestrator.
"""

import logging
import json
import asyncio
import random
from typing import Dict, List, Any, Optional, Tuple, Set, TypedDict
from datetime import datetime, timedelta

from agents import Agent, ModelSettings, function_tool, RunContextWrapper, Runner
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

class ChoiceItem(TypedDict):
    label: str
    action: str
    priority: int

class NpcBehaviorItem(TypedDict):
    npc_id: int
    behavior: str

class ProcessConflictInSceneResponse(TypedDict):
    processed: bool
    conflicts_active: bool
    conflicts_detected: List[int]
    manifestations: List[str]
    choices: List[ChoiceItem]
    npc_behaviors: List[NpcBehaviorItem]
    error: str

class TensionSignal(TypedDict):
    source: str
    level: float

class AnalyzeSceneResponse(TypedDict):
    tension_score: float
    should_generate_conflict: bool
    primary_dynamic: str
    potential_conflict_types: List[str]
    tension_signals: List[TensionSignal]
    error: str

class IntegrateDailyConflictsResponse(TypedDict):
    conflicts_active: bool
    activity_proceeds_normally: bool
    manifestations: List[str]
    player_choices: List[ChoiceItem]
    npc_reactions: List[NpcBehaviorItem]
    atmosphere: List[str]
    error: str


# ===============================================================================
# ENHANCED INTEGRATION SUBSYSTEM (Works through Synthesizer)
# ===============================================================================

class EnhancedIntegrationSubsystem:
    """
    Enhanced integration subsystem that works through ConflictSynthesizer.
    Provides LLM-powered tension analysis and conflict generation.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Reference to synthesizer
        self.synthesizer = None
        
        # LLM agents
        self._tension_analyzer = None
        self._conflict_generator = None
        self._integration_narrator = None
        
        # Connections to other game systems (lazy loaded)
        self._relationship_manager = None
        self._npc_handler = None
        self._world_director = None
        self._lore_system = None
        self._memory_manager = None
    
    @property
    def subsystem_type(self):
        """Return the subsystem type"""
        from logic.conflict_system.conflict_synthesizer import SubsystemType
        return SubsystemType.SLICE_OF_LIFE  # This enhances slice-of-life conflicts
    
    @property
    def capabilities(self) -> Set[str]:
        """Return capabilities this subsystem provides"""
        return {
            'tension_analysis',
            'contextual_generation',
            'daily_integration',
            'pattern_detection',
            'narrative_weaving'
        }
    
    @property
    def dependencies(self) -> Set:
        """Return other subsystems this depends on"""
        from logic.conflict_system.conflict_synthesizer import SubsystemType
        return {
            SubsystemType.DETECTION,
            SubsystemType.TENSION,
            SubsystemType.FLOW
        }
    
    @property
    def event_subscriptions(self) -> Set:
        """Return events this subsystem wants to receive"""
        from logic.conflict_system.conflict_synthesizer import EventType
        return {
            EventType.STATE_SYNC,
            EventType.PLAYER_CHOICE,
            EventType.NPC_REACTION,
            EventType.HEALTH_CHECK
        }
    
    async def initialize(self, synthesizer) -> bool:
        """Initialize the subsystem with synthesizer reference"""
        import weakref
        self.synthesizer = weakref.ref(synthesizer)
        return True
    
    async def handle_event(self, event) -> Any:
        """Handle an event from the synthesizer"""
        from logic.conflict_system.conflict_synthesizer import SubsystemResponse, SystemEvent, EventType
        
        try:
            if event.event_type == EventType.STATE_SYNC:
                # Analyze for emerging tensions
                scene_context = event.payload
                tensions = await self.analyze_scene_tensions(scene_context)
                
                side_effects = []
                if tensions and tensions['should_generate_conflict']:
                    # Request conflict creation through synthesizer
                    side_effects.append(SystemEvent(
                        event_id=f"tension_{event.event_id}",
                        event_type=EventType.CONFLICT_CREATED,
                        source_subsystem=self.subsystem_type,
                        payload={
                            'conflict_type': tensions['suggested_type'],
                            'context': tensions['context'],
                            'tension_source': tensions['source']
                        },
                        priority=5
                    ))
                
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={
                        'tensions_found': len(tensions.get('tensions', [])),
                        'manifestation': tensions.get('manifestation', [])
                    },
                    side_effects=side_effects
                )
                
            elif event.event_type == EventType.PLAYER_CHOICE:
                # Process how choice affects conflicts
                choice_impact = await self._analyze_choice_impact(event.payload)
                
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data=choice_impact
                )
                
            elif event.event_type == EventType.NPC_REACTION:
                # Integrate NPC reactions into conflicts
                reaction_integration = await self._integrate_npc_reaction(event.payload)
                
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data=reaction_integration
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
            logger.error(f"Enhanced integration error: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)}
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Return health status of the subsystem"""
        return {
            'healthy': True,
            'agents_loaded': bool(self._tension_analyzer or self._conflict_generator),
            'connections_available': True
        }
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        """Get enhanced integration data for a specific conflict"""
        # Get related tensions and patterns
        tensions = await self._get_conflict_tensions(conflict_id)
        patterns = await self._get_conflict_patterns(conflict_id)
        
        return {
            'active_tensions': tensions,
            'detected_patterns': patterns
        }
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current state of enhanced integration"""
        return {
            'integration_active': True,
            'llm_agents_ready': bool(self._tension_analyzer)
        }
    
    async def is_relevant_to_scene(self, scene_context: Dict[str, Any]) -> bool:
        """Check if enhanced integration is relevant to scene"""
        # Always relevant for tension analysis
        return True
    
    # ========== LLM Agent Properties ==========
    
    @property
    def tension_analyzer(self) -> Agent:
        """Agent for analyzing tensions from various sources"""
        if self._tension_analyzer is None:
            self._tension_analyzer = Agent(
                name="Tension Analyzer",
                instructions="""
                Analyze game state for emerging tensions and conflicts.
                
                Consider multiple sources:
                - Relationship dynamics and power imbalances
                - NPC narrative progression and mask integrity
                - Matriarchal society lore and cultural tensions
                - Recent player patterns and behaviors
                
                Generate contextual, nuanced tensions that feel organic.
                Focus on slice-of-life conflicts rather than dramatic confrontations.
                Consider how different tension sources interact and compound.
                """,
                model="gpt-5-nano",
            )
        return self._tension_analyzer
    
    @property
    def conflict_generator(self) -> Agent:
        """Agent for generating conflicts from tensions"""
        if self._conflict_generator is None:
            self._conflict_generator = Agent(
                name="Conflict Generator",
                instructions="""
                Transform identified tensions into active conflicts.
                
                Create conflicts that:
                - Feel natural to the current scene and activity
                - Involve present NPCs meaningfully
                - Build on established patterns
                - Have clear but subtle stakes
                - Can manifest through daily activities
                
                Generate specific, contextual descriptions.
                Make conflicts feel like natural extensions of relationships.
                """,
                model="gpt-5-nano",
            )
        return self._conflict_generator
    
    @property
    def integration_narrator(self) -> Agent:
        """Agent for narrating conflict integration"""
        if self._integration_narrator is None:
            self._integration_narrator = Agent(
                name="Conflict Integration Narrator",
                instructions="""
                Narrate how conflicts weave into daily activities.
                
                Focus on:
                - Subtle manifestations in routine moments
                - NPC behaviors that reflect underlying tensions
                - Environmental cues and atmosphere
                - Player choice opportunities
                - The accumulation of small moments
                
                Keep narration grounded and slice-of-life.
                Make every detail feel purposeful but not heavy-handed.
                """,
                model="gpt-5-nano",
            )
        return self._integration_narrator
    
    # ========== Analysis Methods ==========
    
    async def analyze_scene_tensions(self, scene_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a scene for tensions using LLM"""
        
        # Gather context from various sources
        npcs = scene_context.get('present_npcs', [])
        location = scene_context.get('location', 'unknown')
        activity = scene_context.get('activity', 'unknown')
        
        # Get additional context if available
        relationship_data = await self._gather_relationship_data(npcs)
        npc_data = await self._gather_npc_progression_data(npcs)
        
        context = {
            'location': location,
            'activity': activity,
            'npcs_present': len(npcs),
            'relationships': self._summarize_for_llm(relationship_data),
            'npc_states': self._summarize_for_llm(npc_data)
        }
        
        prompt = f"""
        Analyze this scene for emerging tensions:
        
        Context:
        {json.dumps(context, indent=2)}
        
        Identify:
        1. Tension sources (relationship/cultural/personal)
        2. Conflict potential (0.0-1.0)
        3. Suggested conflict type
        4. How it might manifest
        
        Return JSON:
        {{
            "tensions": [
                {{
                    "source": "tension source",
                    "level": 0.0 to 1.0,
                    "description": "specific tension"
                }}
            ],
            "should_generate_conflict": true/false,
            "suggested_type": "conflict type",
            "manifestation": ["how it shows up"],
            "context": {{additional context}}
        }}
        """
        
        response = await Runner.run(self.tension_analyzer, prompt)
        
        try:
            return json.loads(response.output)
        except json.JSONDecodeError:
            return {
                'tensions': [],
                'should_generate_conflict': False
            }
    
    async def generate_contextual_conflict(
        self,
        tension_data: Dict[str, Any],
        npcs: List[int]
    ) -> Dict[str, Any]:
        """Generate a conflict from analyzed tensions"""
        
        prompt = f"""
        Generate a conflict from these tensions:
        
        Tensions: {json.dumps(tension_data, indent=2)}
        NPCs Involved: {npcs}
        
        Create:
        1. Conflict name
        2. Description (2-3 sentences)
        3. Initial intensity
        4. Stakes for player
        5. How it starts
        
        Return JSON:
        {{
            "name": "conflict name",
            "description": "detailed description",
            "intensity": "subtle/tension/friction",
            "stakes": "what player risks/gains",
            "opening": "how it begins"
        }}
        """
        
        response = await Runner.run(self.conflict_generator, prompt)
        
        try:
            return json.loads(response.output)
        except json.JSONDecodeError:
            return {
                'name': "Emerging Tension",
                'description': "A subtle conflict begins",
                'intensity': 'tension',
                'stakes': "Personal dynamics",
                'opening': "Tension fills the air"
            }
    
    async def integrate_conflicts_with_activity(
        self,
        activity: str,
        active_conflicts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate how conflicts integrate into an activity"""
        
        if not active_conflicts:
            return {'conflicts_active': False}
        
        prompt = f"""
        Integrate these conflicts into the activity:
        
        Activity: {activity}
        Conflicts: {json.dumps(active_conflicts[:3], indent=2)}
        
        Generate:
        1. How conflicts subtly manifest
        2. Environmental cues
        3. NPC behavior changes
        4. Player choice opportunities
        
        Return JSON:
        {{
            "manifestations": ["specific details"],
            "environmental_cues": ["atmosphere changes"],
            "npc_behaviors": {{"npc_id": "behavior"}},
            "choices": [
                {{
                    "text": "choice text",
                    "subtext": "hidden meaning"
                }}
            ]
        }}
        """
        
        response = await Runner.run(self.integration_narrator, prompt)
        
        try:
            result = json.loads(response.output)
            return {
                'conflicts_active': True,
                **result
            }
        except json.JSONDecodeError:
            return {
                'conflicts_active': True,
                'manifestations': ["Tension colors the interaction"]
            }
    
    # ========== Helper Methods ==========
    
    async def _analyze_choice_impact(self, choice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how a player choice impacts conflicts"""
        
        choice = choice_data.get('choice', '')
        context = choice_data.get('context', {})
        
        # Simple analysis - could be enhanced with LLM
        impact = {
            'tension_change': random.uniform(-0.1, 0.1),
            'relationship_impact': {},
            'conflict_progression': random.uniform(0, 0.2)
        }
        
        return impact
    
    async def _integrate_npc_reaction(self, reaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate NPC reaction into conflict system"""
        
        npc_id = reaction_data.get('npc_id')
        reaction = reaction_data.get('reaction', '')
        
        # Simple integration - could be enhanced
        return {
            'reaction_integrated': True,
            'tension_modifier': random.uniform(-0.05, 0.05)
        }
    
    async def _gather_relationship_data(self, npcs: List[int]) -> Dict:
        """Gather relationship data for LLM context"""
        data = {}
        
        # Would connect to relationship system
        # For now, return mock data
        for npc_id in npcs[:3]:
            data[npc_id] = {
                'trust': random.uniform(0, 1),
                'power': random.uniform(-1, 1)
            }
        
        return data
    
    async def _gather_npc_progression_data(self, npcs: List[int]) -> Dict:
        """Gather NPC progression data for LLM context"""
        data = {}
        
        async with get_db_connection_context() as conn:
            for npc_id in npcs[:3]:
                # Mock query - would use actual NPC progression table
                data[npc_id] = {
                    'narrative_stage': 'developing',
                    'relationship_level': random.randint(1, 5)
                }
        
        return data
    
    def _summarize_for_llm(self, data: Any) -> str:
        """Create concise summary for LLM prompts"""
        if isinstance(data, dict):
            if not data:
                return "No significant data"
            items = []
            for key, value in list(data.items())[:5]:
                items.append(f"{key}: {value}")
            return "; ".join(items)
        elif isinstance(data, list):
            return "; ".join(str(item) for item in data[:5])
        else:
            return str(data)[:200]
    
    async def _get_conflict_tensions(self, conflict_id: int) -> List[Dict[str, Any]]:
        """Get tensions related to a conflict"""
        # Would query tension events for this conflict
        return []
    
    async def _get_conflict_patterns(self, conflict_id: int) -> List[str]:
        """Get patterns detected in a conflict"""
        # Would analyze conflict events for patterns
        return []


# ===============================================================================
# PUBLIC API FUNCTIONS (Work through Synthesizer)
# ===============================================================================

@function_tool
async def process_conflict_in_scene(
    ctx: RunContextWrapper,
    scene_type: str,
    activity: str,
    present_npcs: List[int]
) -> ProcessConflictInSceneResponse:
    """Process conflicts within a scene through synthesizer (strict schema)."""

    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')

    from logic.conflict_system.conflict_synthesizer import (
        get_synthesizer, SystemEvent, EventType, SubsystemType
    )
    synthesizer = await get_synthesizer(user_id, conversation_id)

    scene_context = {
        'scene_type': scene_type,
        'activity': activity,
        'present_npcs': present_npcs,
        'npcs': present_npcs,  # many subsystems look for `npcs`
        'timestamp': datetime.now().isoformat(),
    }

    # 1) Process the scene for core manifestations/choices
    synth_result = await synthesizer.process_scene(scene_context) or {}

    # 2) Ask Stakeholder system explicitly for NPC behaviors (normalized)
    behavior_evt = SystemEvent(
        event_id=f"behaviors_{datetime.now().timestamp()}",
        event_type=EventType.STATE_SYNC,
        source_subsystem=SubsystemType.SLICE_OF_LIFE,
        payload=scene_context,
        target_subsystems={SubsystemType.STAKEHOLDER},
        requires_response=True,
        priority=5,
    )
    behavior_resps = await synthesizer.emit_event(behavior_evt) or []

    npc_behaviors: List[NpcBehaviorItem] = []
    for r in behavior_resps:
        if r.subsystem == SubsystemType.STAKEHOLDER:
            data = r.data or {}
            raw = data.get('npc_behaviors', {})  # might be a dict {npc_id: behavior}
            # normalize to list
            for k, v in (raw.items() if isinstance(raw, dict) else []):
                try:
                    npc_id = int(k)
                except Exception:
                    continue
                npc_behaviors.append({'npc_id': npc_id, 'behavior': str(v)})

    # Normalize choices into a strict list
    choices_src = synth_result.get('choices', []) or []
    choices: List[ChoiceItem] = []
    for c in choices_src:
        # defensive coercion
        label = str(c.get('label', c.get('text', 'Choice')))
        action = str(c.get('action', c.get('id', 'unknown')))
        try:
            priority = int(c.get('priority', 5))
        except Exception:
            priority = 5
        choices.append({'label': label, 'action': action, 'priority': priority})

    return {
        'processed': bool(synth_result.get('scene_processed', synth_result.get('processed', True))),
        'conflicts_active': bool(synth_result.get('conflicts_active', False)),
        'conflicts_detected': list(synth_result.get('conflicts_detected', []) or []),
        'manifestations': list(synth_result.get('manifestations', []) or []),
        'choices': choices,
        'npc_behaviors': npc_behaviors,
        'error': "",
    }


@function_tool
async def analyze_scene_for_conflict_potential(
    ctx: RunContextWrapper,
    scene_description: str,
    npcs_present: List[int],
    recent_events: List[str]
) -> AnalyzeSceneResponse:
    """Analyze a scene for potential conflict generation (strict schema)."""

    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')

    from logic.conflict_system.conflict_synthesizer import (
        get_synthesizer, SystemEvent, EventType, SubsystemType
    )
    synthesizer = await get_synthesizer(user_id, conversation_id)

    evt = SystemEvent(
        event_id=f"analyze_{datetime.now().timestamp()}",
        event_type=EventType.STATE_SYNC,
        source_subsystem=SubsystemType.SLICE_OF_LIFE,
        payload={
            'scene_description': scene_description,
            'present_npcs': npcs_present,
            'npcs': npcs_present,
            'recent_events': recent_events,
            'request_analysis': True,
        },
        target_subsystems={SubsystemType.SLICE_OF_LIFE},
        requires_response=True,
        priority=5,
    )

    responses = await synthesizer.emit_event(evt) or []

    best_signals: List[TensionSignal] = []
    should_generate = False
    primary_dynamic = "none"
    suggested_types: List[str] = []
    error = "no_response"

    for r in responses:
        if r.subsystem == SubsystemType.SLICE_OF_LIFE:
            d = r.data or {}
            tensions = d.get('tensions', []) or []
            # Normalize signals
            for t in tensions:
                src = str(t.get('source', 'unknown'))
                lvl = float(t.get('level', 0.0))
                best_signals.append({'source': src, 'level': max(0.0, min(1.0, lvl))})
            # Suggested type(s) and primary dynamic
            suggested = d.get('suggested_type')
            if suggested:
                suggested_types.append(str(suggested))
            if tensions:
                # pick highest level as primary
                primary = max(tensions, key=lambda x: float(x.get('level', 0.0)))
                primary_dynamic = str(primary.get('source', primary_dynamic))
            should_generate = bool(d.get('should_generate_conflict', False))
            error = ""

    # Compute tension score
    if best_signals:
        tension_score = sum(s['level'] for s in best_signals) / len(best_signals)
    else:
        tension_score = 0.0

    return {
        'tension_score': float(tension_score),
        'should_generate_conflict': should_generate,
        'primary_dynamic': primary_dynamic,
        'potential_conflict_types': list(dict.fromkeys(suggested_types)),
        'tension_signals': best_signals,
        'error': error,
    }


@function_tool
async def integrate_daily_conflicts(
    ctx: RunContextWrapper,
    activity_type: str,
    activity_description: str
) -> IntegrateDailyConflictsResponse:
    """Integrate conflicts into daily activities through synthesizer (strict schema)."""

    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')

    from logic.conflict_system.conflict_synthesizer import (
        get_synthesizer, SystemEvent, EventType, SubsystemType
    )
    synthesizer = await get_synthesizer(user_id, conversation_id)

    # Check if there are any active conflicts first
    sys_state = await synthesizer.get_system_state() or {}
    active_conflict_ids = sys_state.get('active_conflicts', []) or []
    conflicts_active = len(active_conflict_ids) > 0

    if not conflicts_active:
        return {
            'conflicts_active': False,
            'activity_proceeds_normally': True,
            'manifestations': [],
            'player_choices': [],
            'npc_reactions': [],
            'atmosphere': [],
            'error': "",
        }

    scene_context = {
        'scene_type': 'daily',
        'activity': activity_type,
        'activity_type': activity_type,
        'scene_description': activity_description,
        'integrating_conflicts': True,
        'timestamp': datetime.now().isoformat(),
    }

    synth_result = await synthesizer.process_scene(scene_context) or {}

    # Ask Stakeholder system for NPC reactions for this activity
    behavior_evt = SystemEvent(
        event_id=f"daily_behaviors_{datetime.now().timestamp()}",
        event_type=EventType.STATE_SYNC,
        source_subsystem=SubsystemType.SLICE_OF_LIFE,
        payload=scene_context,
        target_subsystems={SubsystemType.STAKEHOLDER},
        requires_response=True,
        priority=5,
    )
    behavior_resps = await synthesizer.emit_event(behavior_evt) or []

    npc_reactions: List[NpcBehaviorItem] = []
    for r in behavior_resps:
        if r.subsystem == SubsystemType.STAKEHOLDER:
            data = r.data or {}
            raw = data.get('npc_behaviors', {})
            for k, v in (raw.items() if isinstance(raw, dict) else []):
                try:
                    npc_id = int(k)
                except Exception:
                    continue
                npc_reactions.append({'npc_id': npc_id, 'behavior': str(v)})

    # Normalize choices
    choices_src = synth_result.get('choices', []) or []
    player_choices: List[ChoiceItem] = []
    for c in choices_src:
        label = str(c.get('label', c.get('text', 'Choice')))
        action = str(c.get('action', c.get('id', 'unknown')))
        try:
            priority = int(c.get('priority', 5))
        except Exception:
            priority = 5
        player_choices.append({'label': label, 'action': action, 'priority': priority})

    manifestations = list(synth_result.get('manifestations', []) or [])
    atmosphere = list(
        synth_result.get('atmospheric_elements', synth_result.get('atmosphere', [])) or []
    )

    # If nothing interesting happened, the activity may proceed normally even with conflicts present
    activity_proceeds_normally = (not manifestations) and (not player_choices) and (not npc_reactions)

    return {
        'conflicts_active': True,
        'activity_proceeds_normally': activity_proceeds_normally,
        'manifestations': manifestations,
        'player_choices': player_choices,
        'npc_reactions': npc_reactions,
        'atmosphere': atmosphere,
        'error': "",
    }
