# logic/conflict_system/slice_of_life_conflicts.py
"""
Slice-of-life conflict system with LLM-generated dynamic content.
Refactored to work as a ConflictSubsystem with the synthesizer (circular-safe).
"""

import logging
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from db.connection import get_db_connection_context
from logic.conflict_system.tension import TensionType

logger = logging.getLogger(__name__)

# ===============================================================================
# Lazy orchestrator type access (avoid circular imports at module load)
# ===============================================================================

def _orch():
    from logic.conflict_system.conflict_synthesizer import (
        SubsystemType, EventType, SystemEvent, SubsystemResponse
    )
    return SubsystemType, EventType, SystemEvent, SubsystemResponse


# ===============================================================================
# ENUMS
# ===============================================================================

class SliceOfLifeConflictType(Enum):
    """Types of subtle daily conflicts"""
    PERMISSION_PATTERNS = "permission_patterns"
    ROUTINE_DOMINANCE = "routine_dominance"
    BOUNDARY_EROSION = "boundary_erosion"
    SOCIAL_PECKING_ORDER = "social_pecking_order"
    FRIENDSHIP_BOUNDARIES = "friendship_boundaries"
    ROLE_EXPECTATIONS = "role_expectations"
    FINANCIAL_CONTROL = "financial_control"
    SUBTLE_RIVALRY = "subtle_rivalry"
    PREFERENCE_SUBMISSION = "preference_submission"
    CARE_DEPENDENCY = "care_dependency"
    INDEPENDENCE_STRUGGLE = "independence_struggle"
    PASSIVE_AGGRESSION = "passive_aggression"
    EMOTIONAL_LABOR = "emotional_labor"
    DECISION_FATIGUE = "decision_fatigue"
    MASK_SLIPPAGE = "mask_slippage"
    DOMESTIC_HIERARCHY = "domestic_hierarchy"
    GROOMING_PATTERNS = "grooming_patterns"
    GASLIGHTING_GENTLE = "gaslighting_gentle"
    SOCIAL_ISOLATION = "social_isolation"
    CONDITIONING_RESISTANCE = "conditioning_resistance"

class ConflictIntensity(Enum):
    """How overtly the conflict manifests"""
    SUBTEXT = "subtext"
    TENSION = "tension"
    PASSIVE = "passive"
    DIRECT = "direct"
    CONFRONTATION = "confrontation"

class ResolutionApproach(Enum):
    """How conflicts resolve in slice-of-life"""
    GRADUAL_ACCEPTANCE = "gradual_acceptance"
    SUBTLE_RESISTANCE = "subtle_resistance"
    NEGOTIATED_COMPROMISE = "negotiated_compromise"
    ESTABLISHED_PATTERN = "established_pattern"
    THIRD_PARTY_INFLUENCE = "third_party_influence"
    TIME_EROSION = "time_erosion"

@dataclass
class SliceOfLifeStake:
    """What's at stake in mundane conflicts"""
    stake_type: str
    description: str
    daily_impact: str
    relationship_impact: str
    accumulation_factor: float

@dataclass
class DailyConflictEvent:
    """A conflict moment embedded in daily routine"""
    activity_type: str
    conflict_manifestation: str
    choice_presented: bool
    accumulation_impact: float
    npc_reactions: Dict[int, str]


# ===============================================================================
# SLICE OF LIFE CONFLICT SUBSYSTEM (duck-typed, circular-safe)
# ===============================================================================

class SliceOfLifeConflictSubsystem:
    """
    Slice-of-life conflict subsystem integrated with synthesizer.
    Combines all slice-of-life components into one subsystem.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.synthesizer = None  # weakref set in initialize

        # Components
        self.detector = EmergentConflictDetector(user_id, conversation_id)
        self.manager = SliceOfLifeConflictManager(user_id, conversation_id)
        self.resolver = PatternBasedResolution(user_id, conversation_id)
        self.daily_integration = ConflictDailyIntegration(user_id, conversation_id)

        # Local observation buffers (purely numeric/pattern based)
        self._observations: Deque[Tuple[float, str, str, float, str]] = deque(maxlen=240)
        self._subject_labels: Dict[str, str] = {}
        self._active_patterns: Dict[str, Dict[str, Any]] = {}
        self._pattern_conflict_cooldowns: Dict[str, float] = {}
        self._recent_memory_ids: Deque[int] = deque(maxlen=400)
        self._seen_memory_ids: Set[int] = set()
        self._last_memory_refresh: float = 0.0
    
    # ========== Subsystem Interface ==========
    
    @property
    def subsystem_type(self):
        SubsystemType, _, _, _ = _orch()
        return SubsystemType.SLICE_OF_LIFE
    
    @property
    def capabilities(self) -> Set[str]:
        return {
            'detect_emerging_tensions',
            'embed_in_daily_activities',
            'pattern_based_resolution',
            'generate_slice_of_life_conflicts',
            'subtle_conflict_manifestation'
        }
    
    @property
    def dependencies(self) -> Set:
        SubsystemType, _, _, _ = _orch()
        return {SubsystemType.FLOW, SubsystemType.SOCIAL}
    
    @property
    def event_subscriptions(self) -> Set:
        _, EventType, _, _ = _orch()
        return {
            EventType.STATE_SYNC,
            EventType.PLAYER_CHOICE,
            EventType.PHASE_TRANSITION,
            EventType.CONFLICT_CREATED
        }
    
    async def initialize(self, synthesizer) -> bool:
        """Initialize with synthesizer reference"""
        self.synthesizer = weakref.ref(synthesizer)
        return True
    
    async def handle_event(self, event):
        """Handle events from synthesizer"""
        _, EventType, _, SubsystemResponse = _orch()
        try:
            if event.event_type == EventType.STATE_SYNC:
                return await self._handle_state_sync(event)
            if event.event_type == EventType.PLAYER_CHOICE:
                return await self._handle_player_choice(event)
            if event.event_type == EventType.PHASE_TRANSITION:
                return await self._handle_phase_transition(event)
            if event.event_type == EventType.CONFLICT_CREATED:
                return await self._handle_conflict_creation(event)
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'handled': False},
                side_effects=[]
            )
        except Exception as e:
            logger.error(f"SliceOfLife error handling event: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)},
                side_effects=[]
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of subsystem"""
        try:
            tensions = await self.detector.detect_brewing_tensions()
            return {
                'healthy': True,
                'active_tensions': len(tensions),
                'components': {
                    'detector': 'operational',
                    'manager': 'operational',
                    'resolver': 'operational',
                    'daily_integration': 'operational'
                }
            }
        except Exception as e:
            return {'healthy': False, 'issue': str(e)}
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        """Get slice-of-life specific conflict data"""
        resolution = await self.resolver.check_resolution_by_pattern(conflict_id)
        return {
            'subsystem': 'slice_of_life',
            'pattern_resolution_available': resolution is not None,
            'resolution_details': resolution
        }
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        tensions = await self.detector.detect_brewing_tensions()
        return {
            'emerging_tensions': len(tensions),
            'tension_types': [t.get('type', 'unknown') for t in tensions],
            'daily_integration_active': True
        }
    
    # Optional: provide a small scene bundle to orchestrator for parallel merges
    async def get_scene_bundle(self, scope) -> Dict[str, Any]:
        """
        Cheap bundle: surface subtle ambient effects and opportunities.
        """
        try:
            tensions = await self.detector.detect_brewing_tensions()
            ambient = []
            for t in tensions[:3]:
                label = str(getattr(t.get('intensity'), 'value', t.get('intensity', 'tension')))
                ambient.append(f"slicing_{label}")
            opportunities = []
            for t in tensions[:2]:
                opportunities.append({
                    'type': 'slice_opportunity',
                    'description': str(t.get('description', 'subtle pattern')),
                })
            return {
                'ambient_effects': ambient,
                'opportunities': opportunities,
                'last_changed_at': datetime.now().timestamp(),
            }
        except Exception as e:
            logger.debug(f"slice_of_life get_scene_bundle failed: {e}")
            return {}
    
    # ========== Event Handlers ==========
    
    async def _handle_state_sync(self, event):
        """Handle scene state synchronization"""
        SubsystemType, EventType, SystemEvent, SubsystemResponse = _orch()
        payload = event.payload or {}
        scene_context = payload.get('scene_context') or payload

        self._refresh_subject_labels(scene_context)
        await self._ingest_recent_memories()
        self._ingest_scene_observations(scene_context)

        evaluation, side_effects = self._evaluate_patterns(scene_context, reason="state_sync")

        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data=evaluation,
            side_effects=side_effects
        )

    async def _handle_player_choice(self, event):
        """Handle player choices"""
        _, EventType, SystemEvent, SubsystemResponse = _orch()
        payload = event.payload or {}
        conflict_id = payload.get('conflict_id')

        scene_context = payload.get('scene_context') or payload.get('context') or {}
        self._refresh_subject_labels(scene_context)

        choice_descriptor = str(payload.get('choice') or payload.get('selected_option') or '').lower()
        choice_tags = payload.get('tags') or payload.get('choice_tags') or []
        target = payload.get('target_npc') or payload.get('npc_id') or payload.get('npc')
        if isinstance(target, dict):
            target = target.get('id') or target.get('npc_id')

        self._ingest_choice_observation(choice_descriptor, choice_tags, target)

        evaluation, pattern_side_effects = self._evaluate_patterns(scene_context, reason="player_choice")

        resolution = None
        side_effects = list(pattern_side_effects)

        if conflict_id:
            await self.detector.detect_brewing_tensions(eager=True)
            resolution = await self.resolver.check_resolution_by_pattern(int(conflict_id))
            if resolution:
                side_effects.append(SystemEvent(
                    event_id=f"resolve_{conflict_id}",
                    event_type=EventType.CONFLICT_RESOLVED,
                    source_subsystem=self.subsystem_type,
                    payload={
                        'conflict_id': int(conflict_id),
                        'resolution_type': 'pattern_based',
                        'context': {'resolution': resolution}
                    },
                    priority=4
                ))

        evaluation.setdefault('choice_processed', True)
        evaluation['resolution'] = resolution

        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data=evaluation,
            side_effects=side_effects
        )
    
    async def _handle_phase_transition(self, event):
        """Handle phase transitions"""
        _, _, _, SubsystemResponse = _orch()
        new_phase = (event.payload or {}).get('new_phase')
        await self.detector.detect_brewing_tensions(eager=True)
        data = {
            'integration_adjustment': 'reducing' if new_phase in ['resolution', 'aftermath'] else 'maintaining',
            'phase_acknowledged': new_phase
        }
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data=data,
            side_effects=[]
        )
    
    async def _handle_conflict_creation(self, event):
        """Handle new conflict creation (post-create embedding if applicable)"""
        _, _, _, SubsystemResponse = _orch()
        payload = event.payload or {}
        conflict_type = str(payload.get('conflict_type', ''))
        
        is_slice_of_life = conflict_type in [t.value for t in SliceOfLifeConflictType]
        if not is_slice_of_life:
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'not_slice_of_life': True},
                side_effects=[]
            )
        
        context = payload.get('context', {}) or {}
        await self.detector.detect_brewing_tensions(eager=True)

        conflict_id = payload.get('conflict_id')  # May be absent during initial create event
        if conflict_id:
            event_result = await self.manager.embed_conflict_in_activity(
                int(conflict_id),
                context.get('activity', 'conversation'),
                list(context.get('npcs', []) or [])
            )
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={
                    'conflict_type': 'slice_of_life',
                    'initial_manifestation': event_result.conflict_manifestation,
                    'daily_integration': True
                },
                side_effects=[]
            )
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'pending_conflict_embedding': True},
            side_effects=[]
        )


    # ========== Observation & Pattern Helpers ==========

    def _normalize_subject(self, raw: Any) -> Optional[str]:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return str(int(raw))
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            if not lowered:
                return None
            if lowered in {'player', 'pc', 'you', 'self'}:
                return 'player'
            if lowered.startswith('npc:'):
                token = lowered.split(':', 1)[1]
                return token or lowered
            if lowered.isdigit():
                return lowered
            return raw
        if isinstance(raw, dict):
            for key in ('id', 'npc_id', 'entity_id'):
                if key in raw and raw[key] is not None:
                    return self._normalize_subject(raw[key])
        return None

    def _refresh_subject_labels(self, scene_context: Dict[str, Any]) -> None:
        if not isinstance(scene_context, dict):
            return

        mapping: Dict[str, str] = {}

        def register(subject: Any, label: Any) -> None:
            key = self._normalize_subject(subject)
            if key is None:
                return
            try:
                mapping[key] = str(label)
            except Exception:
                mapping[key] = str(subject)

        candidates = [
            scene_context.get('present_npcs'),
            scene_context.get('npcs'),
            scene_context.get('npc_states'),
            scene_context.get('actors'),
        ]
        for candidate in candidates:
            if isinstance(candidate, dict):
                for key, value in candidate.items():
                    if isinstance(value, dict):
                        register(value.get('id', key), value.get('name') or value.get('display_name') or key)
                    else:
                        register(key, value)
            elif isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, dict):
                        register(
                            item.get('id') or item.get('npc_id') or item.get('entity_id'),
                            item.get('name') or item.get('display_name') or item.get('alias') or item.get('label')
                        )
                    else:
                        register(item, item)

        names_map = scene_context.get('npc_names')
        if isinstance(names_map, dict):
            for subject, label in names_map.items():
                register(subject, label)

        player_label = scene_context.get('player_name') or scene_context.get('pc_name')
        if player_label:
            mapping['player'] = str(player_label)

        if mapping:
            self._subject_labels.update(mapping)

    async def _ingest_recent_memories(self) -> None:
        now = time.time()
        if now - self._last_memory_refresh < 60:
            return

        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, entity_id, memory_text, tags, created_at
                    FROM enhanced_memories
                    WHERE user_id = $1 AND conversation_id = $2
                    ORDER BY created_at DESC
                    LIMIT 60
                    """,
                    self.user_id,
                    self.conversation_id,
                )
        except Exception as exc:
            logger.debug("slice_of_life memory ingest failed: %s", exc)
            self._last_memory_refresh = now
            return

        for row in rows or []:
            memory_id = row.get('id')
            if memory_id is None:
                continue
            if memory_id in self._seen_memory_ids:
                continue
            if len(self._recent_memory_ids) == self._recent_memory_ids.maxlen:
                oldest = self._recent_memory_ids.popleft()
                self._seen_memory_ids.discard(oldest)
            self._recent_memory_ids.append(memory_id)
            self._seen_memory_ids.add(memory_id)

            created_at = row.get('created_at')
            timestamp = None
            if hasattr(created_at, 'timestamp'):
                try:
                    timestamp = float(created_at.timestamp())
                except Exception:
                    timestamp = None

            tags_raw = row.get('tags') or []
            tag_values: Set[str] = set()
            if isinstance(tags_raw, (list, tuple, set)):
                tag_values = {str(tag).lower() for tag in tags_raw if tag is not None}
            elif isinstance(tags_raw, dict):
                tag_values = {str(key).lower() for key in tags_raw.keys() if key is not None}

            text = str(row.get('memory_text') or '')
            subject = row.get('entity_id')

            if self._matches_chore_signal(text, tag_values):
                self._record_observation(
                    'routine_dominance',
                    subject,
                    weight=1.0,
                    evidence=text[:160],
                    timestamp=timestamp,
                )
            if self._matches_ignore_signal(text, tag_values):
                self._record_observation(
                    'social_isolation',
                    subject,
                    weight=1.0,
                    evidence=text[:160],
                    timestamp=timestamp,
                )

        self._last_memory_refresh = now

    @staticmethod
    def _matches_chore_signal(text: str, tags: Set[str]) -> bool:
        text_lower = text.lower()
        keywords = {'chore', 'chores', 'clean', 'cleaning', 'laundry', 'cook', 'cooking', 'errand', 'service', 'task'}
        if tags & keywords:
            return True
        return any(token in text_lower for token in keywords)

    @staticmethod
    def _matches_ignore_signal(text: str, tags: Set[str]) -> bool:
        text_lower = text.lower()
        keywords = {'ignore', 'ignored', 'snub', 'cold shoulder', 'dismiss', 'overlook', 'brush off', 'avoid'}
        if tags & keywords:
            return True
        return any(token in text_lower for token in keywords)

    def _record_observation(
        self,
        pattern: str,
        subject: Any,
        *,
        weight: float,
        evidence: str = '',
        timestamp: Optional[float] = None,
    ) -> None:
        subject_key = self._normalize_subject(subject) or 'unknown'
        try:
            weight_value = float(weight)
        except Exception:
            weight_value = 0.0
        if weight_value == 0.0:
            return
        ts = float(timestamp) if timestamp is not None else time.time()
        evidence_str = str(evidence) if evidence is not None else ''
        self._observations.append((ts, pattern, subject_key, weight_value, evidence_str))

    def _ingest_scene_observations(self, scene_context: Dict[str, Any]) -> None:
        if not isinstance(scene_context, dict):
            return

        recent_actions = scene_context.get('recent_actions') or scene_context.get('action_history') or []
        if isinstance(recent_actions, list):
            for action in recent_actions[-8:]:
                if not isinstance(action, dict):
                    continue
                actor = action.get('actor_id') or action.get('npc_id') or action.get('actor')
                action_label = str(action.get('type') or action.get('action') or action.get('name') or '').lower()
                tags_raw = action.get('tags') or []
                if isinstance(tags_raw, dict):
                    tags = {str(k).lower() for k in tags_raw.keys() if k is not None}
                else:
                    tags = {str(t).lower() for t in tags_raw} if isinstance(tags_raw, (list, set, tuple)) else set()
                evidence = action.get('description') or action.get('summary') or action_label

                if self._is_chore_action(action_label, tags):
                    weight = 1.0
                    try:
                        weight += 0.2 * float(action.get('intensity') or 0)
                    except Exception:
                        pass
                    self._record_observation('routine_dominance', actor, weight=weight, evidence=evidence)

                if self._is_ignore_action(action):
                    target = action.get('target') or action.get('target_id') or actor
                    self._record_observation('social_isolation', target, weight=1.2, evidence=evidence)

        recent_turns = scene_context.get('recent_turns') or scene_context.get('recent_interactions') or []
        if isinstance(recent_turns, list):
            npc_counts: Dict[str, int] = defaultdict(int)
            player_turns = 0
            recent_slice = recent_turns[-8:]
            for turn in recent_slice:
                if not isinstance(turn, dict):
                    continue
                sender = turn.get('sender') or turn.get('speaker')
                subject_key = self._normalize_subject(sender)
                if subject_key == 'player':
                    player_turns += 1
                elif subject_key:
                    npc_counts[subject_key] += 1

            for subject_key, count in npc_counts.items():
                if count < 3:
                    continue
                if player_turns >= max(2, count // 2):
                    continue
                last_line = ''
                for turn in reversed(recent_slice):
                    if not isinstance(turn, dict):
                        continue
                    sender = self._normalize_subject(turn.get('sender') or turn.get('speaker'))
                    if sender == subject_key:
                        last_line = str(turn.get('content') or turn.get('text') or '')
                        if last_line:
                            break
                evidence = last_line or 'Repeated attempts to engage receive little response.'
                self._record_observation('social_isolation', subject_key, weight=1.0 + 0.1 * count, evidence=evidence)

    @staticmethod
    def _is_chore_action(action_label: str, tags: Set[str]) -> bool:
        keywords = {
            'chore', 'chores', 'clean', 'cleaning', 'laundry', 'cook', 'cooking',
            'prepare', 'tidy', 'organize', 'service', 'errand', 'task', 'maintenance', 'serve'
        }
        if tags & keywords:
            return True
        return any(token in action_label for token in keywords)

    @staticmethod
    def _is_ignore_action(action: Dict[str, Any]) -> bool:
        fields = [
            str(action.get('outcome') or '').lower(),
            str(action.get('result') or '').lower(),
            str(action.get('response') or '').lower(),
            str(action.get('status') or '').lower(),
        ]
        keywords = ('ignored', 'ignore', 'dismiss', 'rebuff', 'cold', 'snub', 'brush')
        return any(any(keyword in field for keyword in keywords) for field in fields)

    def _ingest_choice_observation(self, descriptor: str, tags: List[Any], target: Any) -> None:
        normalized_descriptor = descriptor.lower() if descriptor else ''
        tag_set = {str(tag).lower() for tag in tags if tag is not None}

        if not normalized_descriptor and not tag_set:
            return

        negative_keywords = {'ignore', 'leave', 'avoid', 'refuse', 'dismiss', 'stay quiet', 'silence'}
        supportive_keywords = {'help', 'assist', 'support', 'apologize', 'listen', 'share load', 'take over', 'pitch in'}

        subject = target or 'player'

        if any(keyword in normalized_descriptor for keyword in negative_keywords) or (tag_set & {'ignore', 'refuse', 'dismiss'}):
            evidence = descriptor or 'Player declined engagement.'
            self._record_observation('social_isolation', subject, weight=1.5, evidence=evidence)
        elif any(keyword in normalized_descriptor for keyword in supportive_keywords) or (tag_set & {'support', 'assist'}):
            evidence = descriptor or 'Player offered support.'
            self._record_observation('social_relief', subject, weight=1.2, evidence=evidence)
            self._record_observation('routine_relief', subject, weight=1.0, evidence=evidence)

    def _aggregate_observations(self, window: float = 900.0) -> Tuple[Dict[str, Dict[str, float]], Dict[Tuple[str, str], List[str]]]:
        now = time.time()
        counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        evidence_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        filtered: Deque[Tuple[float, str, str, float, str]] = deque(maxlen=self._observations.maxlen)

        while self._observations:
            ts, pattern, subject, weight, evidence = self._observations.popleft()
            if now - ts > window:
                continue
            filtered.append((ts, pattern, subject, weight, evidence))
            counts[pattern][subject] += weight
            if evidence:
                key = (pattern, subject)
                bucket = evidence_map[key]
                if len(bucket) < 5:
                    bucket.append(evidence)

        self._observations = filtered
        return counts, evidence_map

    @staticmethod
    def _pattern_key(pattern: str, subject: str) -> str:
        return f"{pattern}|{subject}"

    @staticmethod
    def _is_high_stakes_scene(scene_context: Dict[str, Any]) -> bool:
        if not isinstance(scene_context, dict):
            return False
        phase = str(scene_context.get('phase') or scene_context.get('scene_phase') or scene_context.get('conflict_phase') or '').lower()
        if not phase:
            return False
        high_phases = {'climax', 'finale', 'showdown', 'crisis'}
        return phase in high_phases

    def _should_escalate_pattern(
        self,
        pattern_key: str,
        details: Dict[str, Any],
        score: float,
        scene_context: Dict[str, Any],
    ) -> bool:
        now = time.time()
        if self._is_high_stakes_scene(scene_context):
            return False
        cooldown = self._pattern_conflict_cooldowns.get(pattern_key)
        if cooldown and now - cooldown < 900:
            return False
        started_at = details.get('started_at', now)
        if now - started_at < 600:
            return False
        return score >= 6.0

    def _evaluate_patterns(
        self,
        scene_context: Dict[str, Any],
        *,
        reason: str,
    ) -> Tuple[Dict[str, Any], List[Any]]:
        _, EventType, SystemEvent, _ = _orch()
        counts, evidence_map = self._aggregate_observations()

        routine_counts = counts.get('routine_dominance', {})
        routine_relief = counts.get('routine_relief', {})
        social_counts = counts.get('social_isolation', {})
        social_relief = counts.get('social_relief', {})

        active_patterns: List[Dict[str, Any]] = []
        resolved_patterns: List[Dict[str, Any]] = []
        side_effects: List[Any] = []
        tracked_now = time.time()
        updated_keys: Set[str] = set()

        def subject_label(subject_key: str) -> str:
            return self._subject_labels.get(subject_key, subject_key)

        # Routine dominance detection
        for subject_key, raw_score in routine_counts.items():
            net_score = max(0.0, raw_score - routine_relief.get(subject_key, 0.0))
            if net_score <= 1.5:
                continue
            pattern_key = self._pattern_key('routine_dominance', subject_key)
            status = 'sustained' if pattern_key in self._active_patterns else 'new'
            evidence = evidence_map.get(('routine_dominance', subject_key), [])
            details = self._active_patterns.get(pattern_key, {})
            started_at = details.get('started_at', tracked_now)
            self._active_patterns[pattern_key] = {
                'pattern': 'routine_dominance',
                'subject': subject_key,
                'score': round(net_score, 2),
                'started_at': started_at,
                'last_update': tracked_now,
                'npc_name': subject_label(subject_key),
                'evidence': evidence,
            }
            updated_keys.add(pattern_key)
            active_patterns.append({
                'pattern': SliceOfLifeConflictType.ROUTINE_DOMINANCE.value,
                'subject': subject_key,
                'npc_name': subject_label(subject_key),
                'score': round(net_score, 2),
                'status': status,
                'evidence': evidence,
            })

            delta = min(0.08, 0.02 + min(1.0, net_score / 8.0) * 0.06)
            side_effects.append(SystemEvent(
                event_id=f"sol_tension_routine_{int(tracked_now * 1000)}_{subject_key}",
                event_type=EventType.TENSION_CHANGED,
                source_subsystem=self.subsystem_type,
                payload={
                    'tension_type': TensionType.SOCIAL.value,
                    'change': float(delta),
                    'reason': 'slice_of_life_routine_dominance'
                },
                priority=3
            ))

            if self._should_escalate_pattern(pattern_key, self._active_patterns[pattern_key], net_score, scene_context):
                activity = scene_context.get('activity') or scene_context.get('scene_type') or 'daily_routine'
                conflict_payload = {
                    'conflict_type': SliceOfLifeConflictType.ROUTINE_DOMINANCE.value,
                    'template_hint': 'slice_of_life_low',
                    'pattern': 'routine_dominance',
                    'context': {
                        'activity': activity,
                        'npc': subject_label(subject_key),
                        'score': round(net_score, 2),
                        'evidence': evidence,
                        'reason': reason,
                    }
                }
                side_effects.append(SystemEvent(
                    event_id=f"sol_conflict_seed_{int(tracked_now * 1000)}_{subject_key}",
                    event_type=EventType.CONFLICT_CREATED,
                    source_subsystem=self.subsystem_type,
                    payload=conflict_payload,
                    priority=4
                ))
                self._pattern_conflict_cooldowns[pattern_key] = tracked_now

        # Social isolation detection
        for subject_key, raw_score in social_counts.items():
            net_score = max(0.0, raw_score - social_relief.get(subject_key, 0.0))
            if net_score <= 1.0:
                continue
            pattern_key = self._pattern_key('social_isolation', subject_key)
            status = 'sustained' if pattern_key in self._active_patterns else 'new'
            evidence = evidence_map.get(('social_isolation', subject_key), [])
            details = self._active_patterns.get(pattern_key, {})
            started_at = details.get('started_at', tracked_now)
            self._active_patterns[pattern_key] = {
                'pattern': 'social_isolation',
                'subject': subject_key,
                'score': round(net_score, 2),
                'started_at': started_at,
                'last_update': tracked_now,
                'npc_name': subject_label(subject_key),
                'evidence': evidence,
            }
            updated_keys.add(pattern_key)
            active_patterns.append({
                'pattern': SliceOfLifeConflictType.SOCIAL_ISOLATION.value,
                'subject': subject_key,
                'npc_name': subject_label(subject_key),
                'score': round(net_score, 2),
                'status': status,
                'evidence': evidence,
            })

            delta = min(0.07, 0.015 + min(1.0, net_score / 6.0) * 0.05)
            side_effects.append(SystemEvent(
                event_id=f"sol_tension_social_{int(tracked_now * 1000)}_{subject_key}",
                event_type=EventType.TENSION_CHANGED,
                source_subsystem=self.subsystem_type,
                payload={
                    'tension_type': TensionType.SOCIAL.value,
                    'change': float(delta),
                    'reason': 'slice_of_life_social_isolation'
                },
                priority=3
            ))

            if self._should_escalate_pattern(pattern_key, self._active_patterns[pattern_key], net_score, scene_context):
                conflict_payload = {
                    'conflict_type': SliceOfLifeConflictType.SOCIAL_ISOLATION.value,
                    'template_hint': 'slice_of_life_low',
                    'pattern': 'social_isolation',
                    'context': {
                        'npc': subject_label(subject_key),
                        'score': round(net_score, 2),
                        'evidence': evidence,
                        'reason': reason,
                    }
                }
                side_effects.append(SystemEvent(
                    event_id=f"sol_conflict_hint_{int(tracked_now * 1000)}_{subject_key}",
                    event_type=EventType.CONFLICT_CREATED,
                    source_subsystem=self.subsystem_type,
                    payload=conflict_payload,
                    priority=4
                ))
                self._pattern_conflict_cooldowns[pattern_key] = tracked_now

        # Resolve inactive patterns and ease tension
        for pattern_key, details in list(self._active_patterns.items()):
            if pattern_key in updated_keys:
                continue
            resolved_patterns.append({
                'pattern': details.get('pattern'),
                'subject': details.get('subject'),
                'npc_name': details.get('npc_name'),
                'score': details.get('score'),
                'status': 'resolved',
                'evidence': details.get('evidence') or [],
            })
            side_effects.append(SystemEvent(
                event_id=f"sol_tension_relief_{int(tracked_now * 1000)}_{pattern_key}",
                event_type=EventType.TENSION_CHANGED,
                source_subsystem=self.subsystem_type,
                payload={
                    'tension_type': TensionType.SOCIAL.value,
                    'change': -0.03,
                    'reason': 'slice_of_life_relief'
                },
                priority=2
            ))
            self._active_patterns.pop(pattern_key, None)

        # Cleanup stale cooldowns
        for pattern_key, ts in list(self._pattern_conflict_cooldowns.items()):
            if tracked_now - ts > 1800:
                self._pattern_conflict_cooldowns.pop(pattern_key, None)

        def format_metrics(primary: Dict[str, float], relief: Dict[str, float]) -> List[Dict[str, Any]]:
            summary: List[Dict[str, Any]] = []
            for subject_key, value in primary.items():
                net_value = max(0.0, value - relief.get(subject_key, 0.0))
                if net_value <= 0.0:
                    continue
                summary.append({
                    'subject': subject_key,
                    'npc_name': subject_label(subject_key),
                    'score': round(net_value, 2)
                })
            summary.sort(key=lambda item: item['score'], reverse=True)
            return summary

        evaluation = {
            'patterns_active': active_patterns,
            'patterns_resolved': resolved_patterns,
            'pattern_metrics': {
                'routine_dominance': format_metrics(routine_counts, routine_relief),
                'social_isolation': format_metrics(social_counts, social_relief),
            },
            'reason': reason,
        }

        return evaluation, side_effects


# ===============================================================================
# ORIGINAL COMPONENTS (schema-safe + LLM parsing hardened)
# ===============================================================================

class EmergentConflictDetector:
    """Detects conflicts emerging from daily interactions using LLM analysis"""

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def detect_brewing_tensions(self, *, eager: bool = False) -> List[Dict[str, Any]]:
        """Analyze recent interactions for emerging conflicts using cache-first helper."""
        from logic.conflict_system.slice_of_life_conflicts_hotpath import (
            get_detected_tensions,
        )

        return await get_detected_tensions(
            self.user_id, self.conversation_id, eager=eager
        )

    async def collect_tension_inputs(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Fetch memory and relationship slices for downstream processing."""
        async with get_db_connection_context() as conn:
            memory_rows = await conn.fetch(
                """
                SELECT entity_id, entity_type, memory_text, emotional_valence, tags
                FROM enhanced_memories
                WHERE user_id = $1 AND conversation_id = $2
                  AND created_at > NOW() - INTERVAL '3 days'
                ORDER BY created_at DESC
                LIMIT 100
                """,
                self.user_id,
                self.conversation_id,
            )

            relationship_rows = await conn.fetch(
                """
                SELECT entity1_id, entity2_id, dimension, current_value, recent_delta
                FROM relationship_dimensions
                WHERE user_id = $1 AND conversation_id = $2
                  AND dimension IN ('dominance', 'control', 'dependency', 'resistance')
                """,
                self.user_id,
                self.conversation_id,
            )
        memories = [dict(row) for row in memory_rows]
        relationships = [dict(row) for row in relationship_rows]
        return memories, relationships

    async def _detect_tensions(self) -> List[Dict[str, Any]]:
        """Slow-path implementation that now returns heuristic tensions."""
        memories, relationships = await self.collect_tension_inputs()

        if not memories and not relationships:
            return []

        tensions = await self._analyze_patterns_with_llm(memories, relationships)

        sanitized: List[Dict[str, Any]] = []
        for tension in tensions:
            sanitized.append(
                {
                    "type": getattr(tension.get("type"), "value", tension.get("type", "subtle_rivalry")),
                    "intensity": getattr(
                        tension.get("intensity"), "value", tension.get("intensity", "tension")
                    ),
                    "description": tension.get("description", "A subtle tension emerges"),
                    "evidence": list(tension.get("evidence", [])),
                    "tension_level": float(tension.get("tension_level", 0.5)),
                }
            )
        return sanitized

    async def _analyze_patterns_with_llm(
        self,
        memories: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Heuristic approximation of tension analysis used as a fallback."""

        tensions: List[Dict[str, Any]] = []

        relationship_map = {
            "dominance": SliceOfLifeConflictType.ROUTINE_DOMINANCE,
            "control": SliceOfLifeConflictType.FINANCIAL_CONTROL,
            "dependency": SliceOfLifeConflictType.CARE_DEPENDENCY,
            "resistance": SliceOfLifeConflictType.INDEPENDENCE_STRUGGLE,
        }

        for rel in relationships[:5]:
            dimension = str(rel.get("dimension", "")).lower()
            current_value = float(rel.get("current_value") or 0.0)
            if abs(current_value) < 0.25:
                continue

            ctype = relationship_map.get(dimension, SliceOfLifeConflictType.SUBTLE_RIVALRY)
            magnitude = min(max(abs(current_value), 0.1), 1.0)
            if magnitude > 0.75:
                intensity = ConflictIntensity.CONFRONTATION
            elif magnitude > 0.55:
                intensity = ConflictIntensity.DIRECT
            elif magnitude > 0.4:
                intensity = ConflictIntensity.PASSIVE
            else:
                intensity = ConflictIntensity.TENSION

            description = (
                f"{dimension.title()} dynamics are trending toward {('imbalance' if current_value > 0 else 'withdrawal')}"
                if dimension
                else "Subtle relationship pressure is forming"
            )

            evidence = []
            if relationships and dimension:
                evidence.append(
                    f"{dimension.title()} score shifted by {float(rel.get('recent_delta') or 0.0):+.2f}"
                )
            if memories:
                evidence.append(memories[0].get("memory_text", "Recent interaction felt strained."))

            tensions.append(
                {
                    "type": ctype,
                    "intensity": intensity,
                    "description": description,
                    "evidence": evidence,
                    "tension_level": round(magnitude, 2),
                }
            )

        if not tensions and memories:
            sample = memories[0]
            sentiment = str(sample.get("emotional_valence", "neutral")).lower()
            intensity = ConflictIntensity.SUBTEXT if sentiment == "positive" else ConflictIntensity.TENSION
            tensions.append(
                {
                    "type": SliceOfLifeConflictType.SUBTLE_RIVALRY,
                    "intensity": intensity,
                    "description": sample.get(
                        "memory_text", "A low-grade disagreement is brewing in daily routines."
                    ),
                    "evidence": [sample.get("memory_text", "")],
                    "tension_level": 0.35,
                }
            )

        return tensions[:3]
    
    def _summarize_memories(self, memories: List) -> str:
        """Create a summary of memories for LLM context"""
        summary = []
        for m in memories[:10]:
            summary.append(f"- {m['memory_text']} (emotion: {m.get('emotional_valence', 'neutral')})")
        return "\n".join(summary) if summary else "No recent significant memories"
    
    def _summarize_relationships(self, relationships: List) -> str:
        """Create a summary of relationship dynamics"""
        summary = []
        for r in relationships[:5]:
            summary.append(
                f"- {r['dimension']}: {float(r['current_value']):.2f} "
                f"(recent change: {float(r.get('recent_delta', 0) or 0):.2f})"
            )
        return "\n".join(summary) if summary else "No significant relationship dynamics"


class SliceOfLifeConflictManager:
    """Manages conflicts through daily activities with LLM generation"""

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def embed_conflict_in_activity(
        self,
        conflict_id: int,
        activity_type: str,
        participating_npcs: List[int]
    ) -> DailyConflictEvent:
        """Cache-first helper that returns a daily conflict event."""
        from logic.conflict_system.slice_of_life_conflicts_hotpath import (
            get_activity_manifestation,
        )

        return await get_activity_manifestation(
            self.user_id,
            self.conversation_id,
            conflict_id,
            activity_type,
            participating_npcs,
        )

    async def collect_activity_context(
        self,
        conflict_id: int,
        participating_npcs: List[int],
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """Load conflict record and NPC descriptors for downstream use."""
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow(
                """SELECT * FROM Conflicts WHERE id = $1""",
                int(conflict_id),
            )

            npc_details = []
            for npc_id in participating_npcs[:3]:
                npc = await conn.fetchrow(
                    """SELECT name, personality_traits FROM NPCs WHERE id = $1""",
                    int(npc_id),
                )
                if npc:
                    name = npc["name"]
                    traits = (
                        npc.get("personality_traits", "unknown")
                        if isinstance(npc, dict)
                        else npc["personality_traits"]
                    )
                    npc_details.append(f"{name} ({traits})")
        return dict(conflict) if conflict else None, npc_details

    async def _generate_conflict_manifestation(
        self,
        conflict: Optional[Dict[str, Any]],
        activity_type: str,
        participating_npcs: List[int],
        npc_descriptors: Optional[List[str]] = None,
    ) -> DailyConflictEvent:
        """Heuristic manifestation generator used when LLM support is unavailable."""

        npc_descriptors = npc_descriptors or []
        intensity = str((conflict or {}).get("intensity", "tension")).lower()
        phase = str((conflict or {}).get("phase", "active")).lower()

        if intensity in {"confrontation", "direct"}:
            manifestation = (
                f"{activity_type.title()} is interrupted by a pointed exchange"
                f" between the involved parties."
            )
            impact = 0.25
            choice = True
        elif phase == "cooldown":
            manifestation = (
                f"Residual tension lingers during {activity_type}, but everyone keeps things polite."
            )
            impact = 0.05
            choice = False
        else:
            npc_clause = (
                f" involving {', '.join(npc_descriptors[:2])}" if npc_descriptors else ""
            )
            manifestation = (
                f"{activity_type.title()} carries an undercurrent of hesitation{npc_clause}."
            )
            impact = 0.12
            choice = False

        npc_reactions: Dict[int, str] = {}
        for idx, npc_id in enumerate(participating_npcs[: len(npc_descriptors or [])]):
            npc_reactions[int(npc_id)] = (
                f"{npc_descriptors[idx].split('(')[0].strip()} keeps their distance."
            )

        return DailyConflictEvent(
            activity_type=activity_type,
            conflict_manifestation=manifestation,
            choice_presented=choice,
            accumulation_impact=impact,
            npc_reactions=npc_reactions,
        )

    async def _embed_conflict_in_activity(
        self,
        conflict_id: int,
        activity_type: str,
        participating_npcs: List[int],
    ) -> DailyConflictEvent:
        """Heuristic-only embedding retained for compatibility with cache warmers."""

        conflict, npc_details = await self.collect_activity_context(
            conflict_id, participating_npcs
        )
        return await self._generate_conflict_manifestation(
            conflict, activity_type, participating_npcs, npc_details
        )


class PatternBasedResolution:
    """Resolves conflicts based on accumulated patterns using LLM"""

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def check_resolution_by_pattern(self, conflict_id: int) -> Optional[Dict[str, Any]]:
        """Cache-first helper that returns pattern-based resolution if available."""
        from logic.conflict_system.slice_of_life_conflicts_hotpath import (
            get_resolution_recommendation,
        )

        return await get_resolution_recommendation(
            self.user_id, self.conversation_id, conflict_id
        )

    async def collect_resolution_inputs(
        self, conflict_id: int
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Gather conflict record and tagged memories for evaluation."""
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow(
                """SELECT * FROM Conflicts WHERE id = $1""",
                int(conflict_id),
            )
            memories = await conn.fetch(
                """
                SELECT memory_text, emotional_valence, created_at
                FROM enhanced_memories
                WHERE user_id = $1 AND conversation_id = $2
                  AND tags @> ARRAY[$3::text]
                ORDER BY created_at DESC
                LIMIT 20
                """,
                self.user_id,
                self.conversation_id,
                f"conflict_{int(conflict_id)}",
            )
        return dict(conflict) if conflict else None, [dict(row) for row in memories]

    async def _evaluate_resolution(self, conflict_id: int) -> Optional[Dict[str, Any]]:
        """Heuristic evaluation that approximates the previous LLM decision."""

        conflict, memories = await self.collect_resolution_inputs(conflict_id)

        if not conflict or not memories:
            return None

        progress = float(conflict.get("progress", 0) or 0)
        phase = str(conflict.get("phase", "active")).lower()
        positive_memories = [
            m for m in memories if str(m.get("emotional_valence", "")).lower() in {"positive", "hopeful"}
        ]

        if progress >= 85 or phase in {"cooldown", "dormant"}:
            resolution_type = ResolutionApproach.TIME_EROSION.value
        elif progress >= 65 and positive_memories:
            resolution_type = ResolutionApproach.NEGOTIATED_COMPROMISE.value
        elif len(memories) >= 5 and all(
            str(m.get("emotional_valence", "")).lower() == "negative" for m in memories[:3]
        ):
            resolution_type = ResolutionApproach.SUBTLE_RESISTANCE.value
        else:
            return None

        description = "Conflict momentum suggests a natural winding down."
        if resolution_type == ResolutionApproach.NEGOTIATED_COMPROMISE.value:
            description = "Recent interactions show parties testing small compromises."
        elif resolution_type == ResolutionApproach.SUBTLE_RESISTANCE.value:
            description = "Persistence of negative beats indicates resistance rather than escalation."

        return {
            "resolution_type": resolution_type,
            "description": description,
            "new_patterns": [m.get("memory_text", "") for m in positive_memories[:2]],
            "final_state": "resolved",
        }
    
    def _format_memory_pattern(self, memories: List) -> str:
        """Format memories for LLM analysis"""
        return "\n".join(f"- {m['memory_text']}" for m in memories)


class ConflictDailyIntegration:
    """Integrates conflicts with daily routines using LLM"""

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def get_conflicts_for_time_of_day(self, time_of_day: str) -> List[Dict]:
        """Get conflicts appropriate for current time"""
        async with get_db_connection_context() as conn:
            conflicts = await conn.fetch("""
                SELECT c.*, COALESCE(array_agg(s.npc_id) FILTER (WHERE s.npc_id IS NOT NULL), '{}') AS stakeholder_npcs
                FROM Conflicts c
                LEFT JOIN stakeholders s
                  ON s.conflict_id = c.id
                 AND s.user_id = $1
                 AND s.conversation_id = $2
                WHERE c.user_id = $1 AND c.conversation_id = $2
                  AND c.is_active = true
                GROUP BY c.id
            """, self.user_id, self.conversation_id)
        
        appropriate = []
        for c in conflicts:
            if await self._is_appropriate_for_time(dict(c), time_of_day):
                appropriate.append(dict(c))
        return appropriate
    
    async def _is_appropriate_for_time(self, conflict: Dict, time_of_day: str) -> bool:
        """Cache-first helper for time-of-day suitability."""
        from logic.conflict_system.slice_of_life_conflicts_hotpath import (
            is_conflict_appropriate_for_time,
        )

        return await is_conflict_appropriate_for_time(
            self.user_id, self.conversation_id, conflict, time_of_day
        )

    async def _evaluate_time_appropriateness(
        self, conflict: Dict, time_of_day: str
    ) -> bool:
        """Heuristic classifier for time-of-day suitability."""

        if not conflict:
            return False

        intensity = str(conflict.get("intensity", "tension")).lower()
        phase = str(conflict.get("phase", "active")).lower()
        time_key = time_of_day.lower()

        quiet_hours = {"sleep", "rest", "late_night", "midnight"}
        routine_hours = {"morning", "commute", "work", "afternoon"}

        if intensity in {"confrontation", "direct"} and time_key in quiet_hours:
            return False
        if phase == "dormant" and time_key not in routine_hours:
            return False
        if intensity == "subtext":
            return True

        return time_key not in {"sleep", "rest"}
