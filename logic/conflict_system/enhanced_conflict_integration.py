# logic/conflict_system/enhanced_conflict_integration.py
"""
Enhanced conflict system integration with LLM-generated dynamic content.
Works through ConflictSynthesizer as the central orchestrator.
"""

import logging
import json
import asyncio
import random
import time
from collections import OrderedDict
from typing import Dict, List, Any, Optional, Tuple, Set, TypedDict
from datetime import datetime, timedelta
from logic.conflict_system.dynamic_conflict_template import extract_runner_response

from agents import Agent, ModelSettings, function_tool, RunContextWrapper, Runner
from db.connection import get_db_connection_context
from monitoring.metrics import record_cache_operation
from utils.cache_manager import CacheManager

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

        # Cached tension summaries populated asynchronously
        self._tension_cache = CacheManager(
            name=f"enhanced_conflict_tension_{user_id}_{conversation_id}",
            max_size=64,
            ttl=3600
        )
        self._known_scene_contexts: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._scope_memory_limit = 32
        self._pending_refresh_tasks: Set[asyncio.Task] = set()
        self._refresh_lock = asyncio.Lock()
        self._table_lock = asyncio.Lock()
        self._tension_table_ready = False
        self._refresh_batch_size = 12

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
            EventType.HEALTH_CHECK,
            EventType.DAY_TRANSITION
        }
    
    async def initialize(self, synthesizer) -> bool:
        """Initialize the subsystem with synthesizer reference"""
        import weakref
        self.synthesizer = weakref.ref(synthesizer)
        return True
    
    async def handle_event(self, event) -> Any:
        """Handle an event from the synthesizer (orchestrator-aware)."""
        from logic.conflict_system.conflict_synthesizer import SubsystemResponse, SystemEvent, EventType
    
        try:
            if event.event_type == EventType.STATE_SYNC:
                # Accept either raw scene-context shape or {scene_context: {...}}
                payload = event.payload or {}
                scene_context = payload.get('scene_context') or payload

                tensions = await self.analyze_scene_tensions(scene_context)
    
                side_effects = []
                if tensions and tensions.get('should_generate_conflict'):
                    # Suggest conflict creation as a side-effect (orchestrator will route)
                    try:
                        suggested_type = tensions.get('suggested_type') or 'slice_of_life'
                        side_effects.append(SystemEvent(
                            event_id=f"tension_{event.event_id}",
                            event_type=EventType.CONFLICT_CREATED,
                            source_subsystem=self.subsystem_type,
                            payload={
                                'conflict_type': suggested_type,
                                'context': tensions.get('context', {}),
                                'tension_source': tensions.get('tensions', []),
                            },
                            priority=6
                        ))
                    except Exception:
                        pass
    
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={
                        'tensions': tensions.get('tensions', []),
                        'should_generate_conflict': bool(tensions.get('should_generate_conflict', False)),
                        'suggested_type': tensions.get('suggested_type'),
                        'manifestations': tensions.get('manifestation', []),
                    },
                    side_effects=side_effects
                )

            if event.event_type == EventType.DAY_TRANSITION:
                await self._handle_day_transition_event(event.payload or {})
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'refresh_scheduled': True},
                    side_effects=[]
                )

            if event.event_type == EventType.PLAYER_CHOICE:
                choice_impact = await self._analyze_choice_impact(event.payload or {})
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data=choice_impact,
                    side_effects=[]
                )
    
            if event.event_type == EventType.NPC_REACTION:
                reaction_integration = await self._integrate_npc_reaction(event.payload or {})
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data=reaction_integration,
                    side_effects=[]
                )
    
            if event.event_type == EventType.HEALTH_CHECK:
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data=await self.health_check(),
                    side_effects=[]
                )
    
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={},
                side_effects=[]
            )
    
        except Exception as e:
            logger.error(f"Enhanced integration error: {e}")
            from logic.conflict_system.conflict_synthesizer import SubsystemResponse
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)},
                side_effects=[]
            )

    async def get_scene_bundle(self, scope) -> Dict[str, Any]:
        """
        Provide a small, cheap bundle for the scene. The synthesizer will merge these.
        """
        try:
            raw_scene_context = {
                'location': getattr(scope, "location_id", None),
                'present_npcs': getattr(scope, "npc_ids", []) or [],
                'npcs': getattr(scope, "npc_ids", []) or [],
                'topics': getattr(scope, "topics", []) or [],
                'scene_type': getattr(scope, "scene_type", "unknown"),
                'activity': getattr(scope, "activity", None) or getattr(scope, "current_activity", None),
            }
            scene_context = self._normalize_scene_context(raw_scene_context)
            scope_key = self._scope_key_from_context(scene_context)
            self._remember_scope(scope_key, scene_context)

            cached_summary = await self._get_cached_tension_summary(scope_key)
            if cached_summary:
                await record_cache_operation(
                    cache_type="enhanced_conflict_tension",
                    hit=True,
                    size_bytes=len(json.dumps(cached_summary))
                )
                logger.debug(
                    "Serving cached tension summary for scope %s", scope_key
                )
            else:
                await record_cache_operation(
                    cache_type="enhanced_conflict_tension",
                    hit=False
                )
                logger.info(
                    "No cached tension summary for scope %s; scheduling background refresh",
                    scope_key
                )
                self._schedule_refresh({scope_key: scene_context}, reason="cache_miss")
                cached_summary = {}

            manifestations = list(cached_summary.get('manifestation', []) or [])

            # Surface subtle ambient effects for slice-of-life vibes
            ambient = []
            for m in manifestations[:3]:
                ambient.append(f"subtle_{str(m).lower().replace(' ', '_')}")
            opportunity = []
            if cached_summary.get('should_generate_conflict'):
                stype = cached_summary.get('suggested_type') or 'slice_of_life'
                opportunity.append({
                    'type': f"tension_{stype}",
                    'description': 'Emerging tension could become a small conflict',
                })

            return {
                'manifestations': manifestations,
                'ambient_effects': ambient,
                'opportunities': opportunity,
                'tension_summary_cached_at': cached_summary.get('cached_at'),
                'summary_source': 'cached' if cached_summary else 'pending',
                'last_changed_at': datetime.now().timestamp(),
            }
        except Exception as e:
            logger.debug(f"get_scene_bundle failed: {e}")
            return {}

    def _normalize_scene_context(self, scene_context: Dict[str, Any]) -> Dict[str, Any]:
        present_npcs = scene_context.get('present_npcs') or scene_context.get('npcs') or []
        if isinstance(present_npcs, set):
            present_npcs = list(present_npcs)
        normalized_npcs: Set[Any] = set()
        for npc in present_npcs:
            if npc is None:
                continue
            if isinstance(npc, int):
                normalized_npcs.add(npc)
                continue
            try:
                normalized_npcs.add(int(npc))
            except (TypeError, ValueError):
                normalized_npcs.add(str(npc))

        topics = scene_context.get('topics') or []
        if isinstance(topics, set):
            topics = list(topics)
        normalized = {
            'location': scene_context.get('location') or scene_context.get('location_id') or 'unknown',
            'present_npcs': sorted(normalized_npcs, key=lambda x: str(x)),
            'topics': sorted({str(t) for t in topics if t is not None}),
            'scene_type': scene_context.get('scene_type') or 'unknown',
            'activity': scene_context.get('activity') or scene_context.get('current_activity'),
        }
        normalized['npcs'] = normalized['present_npcs']
        return normalized

    def _scope_key_from_context(self, scene_context: Dict[str, Any]) -> str:
        location = str(scene_context.get('location') or 'unknown')
        scene_type = str(scene_context.get('scene_type') or 'unknown')
        npcs = ','.join(str(n) for n in scene_context.get('present_npcs', []))
        topics = ','.join(scene_context.get('topics', []))
        return f"{location}|{scene_type}|{npcs}|{topics}"

    def _remember_scope(self, scope_key: str, scene_context: Dict[str, Any]) -> None:
        if not scope_key:
            return
        self._known_scene_contexts[scope_key] = scene_context
        self._known_scene_contexts.move_to_end(scope_key)
        while len(self._known_scene_contexts) > self._scope_memory_limit:
            self._known_scene_contexts.popitem(last=False)

    def _schedule_refresh(self, contexts: Dict[str, Dict[str, Any]], reason: str) -> None:
        if not contexts:
            return

        async def _runner():
            try:
                await self._refresh_cached_tensions(contexts, reason=reason)
            except Exception as exc:
                logger.error("Background tension refresh failed (%s): %s", reason, exc)

        task = asyncio.create_task(_runner())
        self._pending_refresh_tasks.add(task)
        task.add_done_callback(lambda t: self._pending_refresh_tasks.discard(t))

    async def _refresh_cached_tensions(self, contexts: Dict[str, Dict[str, Any]], reason: str) -> None:
        started = time.perf_counter()
        async with self._refresh_lock:
            refreshed = 0
            for scope_key, scene_context in contexts.items():
                try:
                    summary = await self.analyze_scene_tensions(scene_context)
                    summary = summary or {}
                    summary['cached_at'] = datetime.utcnow().isoformat()
                    await self._tension_cache.set(scope_key, summary, ttl=3600)
                    await self._persist_tension_summary(scope_key, summary)
                    self._remember_scope(scope_key, scene_context)
                    refreshed += 1
                    logger.info(
                        "Refreshed tension summary for scope %s via %s", scope_key, reason
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to refresh tension summary for scope %s: %s",
                        scope_key,
                        exc
                    )
        duration = time.perf_counter() - started
        logger.info(
            "Completed tension refresh (%s) for %d scope(s) in %.2fs",
            reason,
            refreshed,
            duration
        )

    async def _get_cached_tension_summary(self, scope_key: str) -> Dict[str, Any]:
        if not scope_key:
            return {}
        cached = await self._tension_cache.get(scope_key)
        if cached:
            return cached

        async with get_db_connection_context() as conn:
            await self._ensure_tension_summary_table(conn)
            row = await conn.fetchrow(
                """
                SELECT summary, last_updated
                FROM conflict_scene_tension_summaries
                WHERE user_id = $1 AND conversation_id = $2 AND scope_key = $3
                """,
                self.user_id,
                self.conversation_id,
                scope_key
            )
            if not row:
                return {}
            summary = row['summary'] or {}
            if isinstance(summary, str):
                try:
                    summary = json.loads(summary)
                except json.JSONDecodeError:
                    summary = {}
            if row.get('last_updated') and isinstance(summary, dict):
                summary.setdefault('cached_at', row['last_updated'].isoformat())
            await self._tension_cache.set(scope_key, summary, ttl=3600)
            return summary

    async def _persist_tension_summary(self, scope_key: str, summary: Dict[str, Any]) -> None:
        async with get_db_connection_context() as conn:
            await self._ensure_tension_summary_table(conn)
            await conn.execute(
                """
                INSERT INTO conflict_scene_tension_summaries (
                    user_id,
                    conversation_id,
                    scope_key,
                    summary,
                    last_updated
                )
                VALUES ($1, $2, $3, $4::jsonb, $5)
                ON CONFLICT (user_id, conversation_id, scope_key)
                DO UPDATE SET
                    summary = EXCLUDED.summary,
                    last_updated = EXCLUDED.last_updated
                """,
                self.user_id,
                self.conversation_id,
                scope_key,
                json.dumps(summary),
                datetime.utcnow()
            )

    async def _ensure_tension_summary_table(self, conn) -> None:
        if self._tension_table_ready:
            return
        async with self._table_lock:
            if self._tension_table_ready:
                return
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conflict_scene_tension_summaries (
                    user_id BIGINT NOT NULL,
                    conversation_id BIGINT NOT NULL,
                    scope_key TEXT NOT NULL,
                    summary JSONB NOT NULL,
                    last_updated TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (user_id, conversation_id, scope_key)
                )
                """
            )
            self._tension_table_ready = True

    async def _handle_day_transition_event(self, payload: Dict[str, Any]) -> None:
        contexts: Dict[str, Dict[str, Any]] = {}
        candidate_contexts = []

        if isinstance(payload, dict):
            if isinstance(payload.get('scene_contexts'), list):
                candidate_contexts.extend(payload.get('scene_contexts') or [])
            processing_result = payload.get('processing_result') or {}
            if isinstance(processing_result.get('scene_contexts'), list):
                candidate_contexts.extend(processing_result.get('scene_contexts') or [])
            if isinstance(processing_result.get('active_scenes'), list):
                candidate_contexts.extend(processing_result.get('active_scenes') or [])

        for ctx in candidate_contexts:
            normalized = self._normalize_scene_context(ctx or {})
            scope_key = self._scope_key_from_context(normalized)
            if scope_key and scope_key not in contexts:
                contexts[scope_key] = normalized

        if not contexts:
            keys = list(self._known_scene_contexts.keys())[-self._refresh_batch_size:]
            for key in keys:
                contexts[key] = self._known_scene_contexts[key]

        if contexts:
            logger.info(
                "Scheduling tension refresh for %d scope(s) due to day transition",
                len(contexts)
            )
            self._schedule_refresh(contexts, reason="day_transition")
        
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
            return json.loads(extract_runner_response(response))
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
            return json.loads(extract_runner_response(response))
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
    
        Return JSON:
        {{
          "manifestations": ["specific details"],
          "environmental_cues": ["atmosphere changes"],
          "npc_behaviors": {{"npc_id": "behavior"}},
          "choices": [{{"text": "choice text","subtext": "hidden meaning"}}]
        }}
        """
        response = await Runner.run(self.integration_narrator, prompt)
        try:
            result = json.loads(extract_runner_response(response))
            return {'conflicts_active': True, **result}
        except Exception:
            return {'conflicts_active': True, 'manifestations': ["Tension colors the interaction"]}
    
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

    # Check if there are any active conflicts via DB (since synthesizer.get_system_state() is not available)
    async with get_db_connection_context() as conn:
        has_active = await conn.fetchval("""
            SELECT EXISTS (
              SELECT 1 FROM Conflicts
              WHERE user_id = $1 AND conversation_id = $2 AND is_active = true
            )
        """, user_id, conversation_id)
    conflicts_active = bool(has_active)

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
