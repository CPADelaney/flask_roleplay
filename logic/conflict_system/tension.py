# logic/conflict_system/tension.py
"""
Dynamic Tension System with LLM-generated content.
Refactored to work as a subsystem under the Conflict Synthesizer orchestrator (circular-safe).
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Set, TypedDict
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import weakref

from agents import Agent, Runner, function_tool, RunContextWrapper
from db.connection import get_db_connection_context
from logic.conflict_system.dynamic_conflict_template import extract_runner_response

logger = logging.getLogger(__name__)

# Lazy orchestrator types (avoid circular import at module load)
def _orch():
    from logic.conflict_system.conflict_synthesizer import (
        SubsystemType, EventType, SystemEvent, SubsystemResponse
    )
    return SubsystemType, EventType, SystemEvent, SubsystemResponse


# ===============================================================================
# TENSION TYPES AND STRUCTURES
# ===============================================================================

class TensionObservation(TypedDict):
    type: str
    level: float
    source: str
    note: str

class AnalyzeSceneTensionsResponse(TypedDict):
    tension_score: float
    should_generate_conflict: bool
    primary_dynamic: str
    observations: List[TensionObservation]
    error: str

class ModifyTensionResponse(TypedDict):
    success: bool
    tension_type: str
    applied_change: float
    new_level: float
    clamped: bool
    reason: str
    side_effects: List[str]
    error: str

class TensionCategory(TypedDict):
    name: str
    level: float

class TensionReportResponse(TypedDict):
    total_categories: int
    categories: List[TensionCategory]
    overall_score: float
    hotspots: List[str]
    last_updated_iso: str
    error: str


class TensionType(Enum):
    POWER = "power"
    SOCIAL = "social"
    SEXUAL = "sexual"
    EMOTIONAL = "emotional"
    ADDICTION = "addiction"
    VITAL = "vital"
    ECONOMIC = "economic"
    IDEOLOGICAL = "ideological"
    TERRITORIAL = "territorial"

class TensionLevel(Enum):
    ABSENT = 0.0
    SUBTLE = 0.2
    NOTICEABLE = 0.4
    PALPABLE = 0.6
    INTENSE = 0.8
    BREAKING = 1.0

@dataclass
class TensionSource:
    source_type: str
    source_id: Any
    contribution: float
    description: str

@dataclass
class TensionManifestation:
    tension_type: TensionType
    level: float
    physical_cues: List[str]
    dialogue_modifications: List[str]
    environmental_changes: List[str]
    player_sensations: List[str]


# ===============================================================================
# TENSION SUBSYSTEM (duck-typed; orchestrator-friendly)
# ===============================================================================

class TensionSystem:
    """
    Manages tension dynamics as a subsystem under the orchestrator.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._synthesizer = None  # weakref set in initialize
        
        # Tension state
        self._current_tensions: Dict[TensionType, float] = {}
        self._tension_sources: List[TensionSource] = []
        
        # LLM agents
        self._tension_analyzer = None
        self._manifestation_generator = None
        self._escalation_narrator = None
    
    # ----- Subsystem interface -----
    
    @property
    def subsystem_type(self):
        SubsystemType, _, _, _ = _orch()
        return SubsystemType.TENSION
    
    @property
    def capabilities(self) -> Set[str]:
        return {
            'calculate_tensions',
            'build_tension',
            'resolve_tension',
            'generate_manifestation',
            'check_breaking_point',
            'analyze_tension_sources',
            'create_tension_narrative'
        }
    
    @property
    def dependencies(self) -> Set:
        SubsystemType, _, _, _ = _orch()
        return {SubsystemType.STAKEHOLDER, SubsystemType.FLOW, SubsystemType.SOCIAL}
    
    @property
    def event_subscriptions(self) -> Set:
        _, EventType, _, _ = _orch()
        return {
            EventType.CONFLICT_CREATED,
            EventType.CONFLICT_UPDATED,
            EventType.STAKEHOLDER_ACTION,
            EventType.PHASE_TRANSITION,
            EventType.PLAYER_CHOICE,
            EventType.NPC_REACTION,
            EventType.STATE_SYNC,
            EventType.TENSION_CHANGED,  # so modify_tension tools can route
            EventType.HEALTH_CHECK
        }
    
    async def initialize(self, synthesizer) -> bool:
        self._synthesizer = weakref.ref(synthesizer)
        await self._load_tension_state()
        return True
    
    async def handle_event(self, event):
        SubsystemType, EventType, SystemEvent, SubsystemResponse = _orch()
        try:
            et = event.event_type
            if et == EventType.CONFLICT_CREATED:
                return await self._on_conflict_created(event)
            if et == EventType.STAKEHOLDER_ACTION:
                return await self._on_stakeholder_action(event)
            if et == EventType.PHASE_TRANSITION:
                return await self._on_phase_transition(event)
            if et == EventType.PLAYER_CHOICE:
                return await self._on_player_choice(event)
            if et == EventType.NPC_REACTION:
                return await self._on_npc_reaction(event)
            if et == EventType.STATE_SYNC:
                return await self._on_state_sync(event)
            if et == EventType.TENSION_CHANGED:
                return await self._on_tension_changed(event)
            if et == EventType.HEALTH_CHECK:
                return await self._on_health_check(event)
            return SubsystemResponse(
                subsystem=self.subsystem_type, event_id=event.event_id,
                success=True, data={'status': 'no_action_taken'}, side_effects=[]
            )
        except Exception as e:
            logger.error(f"Tension system error handling event: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type, event_id=event.event_id,
                success=False, data={'error': str(e)}, side_effects=[]
            )
    
    async def health_check(self) -> Dict[str, Any]:
        total_tension = sum(self._current_tensions.values())
        return {
            'healthy': total_tension < 5.0,
            'active_tensions': len([t for t, v in self._current_tensions.items() if v > 0.1]),
            'total_tension': total_tension,
            'critical_tensions': [t.value for t, v in self._current_tensions.items() if v > 0.8],
            'status': 'operational'
        }
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        return {
            'tensions': {t.value: v for t, v in self._current_tensions.items()},
            'dominant_tension': max(self._current_tensions.items(), key=lambda x: x[1])[0].value
                if self._current_tensions else None,
            'total_tension': sum(self._current_tensions.values())
        }
    
    async def get_state(self) -> Dict[str, Any]:
        return {
            'current_tensions': {t.value: v for t, v in self._current_tensions.items()},
            'active_sources': len(self._tension_sources),
            'breaking_points': [t.value for t, v in self._current_tensions.items() if v >= 0.9]
        }
    
    async def get_scene_bundle(self, scope) -> Dict[str, Any]:
        """Optional: provide small bundle for orchestrator merge."""
        try:
            # Ambient effects hint proportional to aggregate tension
            agg = sum(self._current_tensions.values())
            ambient = []
            if agg > 1.5:
                ambient.append("air_feels_heavy")
            if agg > 3.0:
                ambient.append("voices_tighten")
            return {
                'ambient_effects': ambient,
                'tensions': {t.value: v for t, v in self._current_tensions.items()},
                'last_changed_at': datetime.now().timestamp()
            }
        except Exception:
            return {}
    
    async def is_relevant_to_scene(self, scene_context: Dict[str, Any]) -> bool:
        if any(v > 0.1 for v in self._current_tensions.values()):
            return True
        activity = (scene_context or {}).get('activity', '').lower()
        return any(word in activity for word in ['argument', 'confrontation', 'negotiation', 'intimate'])
    
    # ----- Event handlers -----
    
    async def _on_conflict_created(self, event):
        SubsystemType, EventType, SystemEvent, SubsystemResponse = _orch()
        payload = event.payload or {}
        conflict_type = payload.get('conflict_type', '')
        context = payload.get('context', {}) or {}
        
        initial_tensions = await self._determine_initial_tensions(conflict_type, context)
        side_effects = []
        for tension_type, level in initial_tensions.items():
            self._current_tensions[tension_type] = float(level)
            if level > 0.3:
                side_effects.append(SystemEvent(
                    event_id=f"tension_{tension_type.value}_{event.event_id}",
                    event_type=EventType.TENSION_CHANGED,
                    source_subsystem=self.subsystem_type,
                    payload={'tension_type': tension_type.value, 'level': float(level), 'source': 'conflict_creation'},
                    priority=5
                ))
        await self._save_tension_state()
        return SubsystemResponse(
            subsystem=self.subsystem_type, event_id=event.event_id, success=True,
            data={
                'tensions_initialized': {t.value: l for t, l in initial_tensions.items()},
                'dominant_tension': max(initial_tensions.items(), key=lambda x: x[1])[0].value
                    if initial_tensions else None
            },
            side_effects=side_effects
        )
    
    async def _on_stakeholder_action(self, event):
        SubsystemType, EventType, SystemEvent, SubsystemResponse = _orch()
        payload = event.payload or {}
        action_type = str(payload.get('action_type', ''))
        intensity = float(payload.get('intensity', 0.5) or 0.5)
        
        tension_changes = self._map_action_to_tension_changes(action_type, intensity)
        side_effects = []
        for ttype, change in tension_changes.items():
            old_level = float(self._current_tensions.get(ttype, 0.0))
            new_level = max(0.0, min(1.0, old_level + float(change)))
            self._current_tensions[ttype] = new_level
            if new_level >= TensionLevel.BREAKING.value and old_level < TensionLevel.BREAKING.value:
                side_effects.append(SystemEvent(
                    event_id=f"breaking_{ttype.value}_{event.event_id}",
                    event_type=EventType.EDGE_CASE_DETECTED,
                    source_subsystem=self.subsystem_type,
                    payload={'edge_case': 'tension_breaking_point', 'tension_type': ttype.value,
                             'level': new_level, 'requires_immediate_resolution': True},
                    priority=1
                ))
        await self._save_tension_state()
        return SubsystemResponse(
            subsystem=self.subsystem_type, event_id=event.event_id, success=True,
            data={
                'tensions_modified': {t.value: c for t, c in tension_changes.items()},
                'current_tensions': {t.value: v for t, v in self._current_tensions.items()}
            },
            side_effects=side_effects
        )
    
    async def _on_phase_transition(self, event):
        _, _, _, SubsystemResponse = _orch()
        payload = event.payload or {}
        from_phase = payload.get('from_phase', '')
        to_phase = payload.get('to_phase', payload.get('phase', ''))
        adjustments = self._calculate_phase_tension_adjustments(from_phase, to_phase)
        for t, adj in adjustments.items():
            current = float(self._current_tensions.get(t, 0.0))
            self._current_tensions[t] = max(0.0, min(1.0, current + float(adj)))
        await self._save_tension_state()
        return SubsystemResponse(
            subsystem=self.subsystem_type, event_id=event.event_id, success=True,
            data={'tensions_adjusted': {t.value: a for t, a in adjustments.items()}, 'phase_impact': 'tensions_shifted'},
            side_effects=[]
        )
    
    async def _on_player_choice(self, event):
        _, _, _, SubsystemResponse = _orch()
        payload = event.payload or {}
        choice_type = str(payload.get('choice_type', ''))
        narrative = await self._generate_choice_tension_narrative(choice_type)
        impact = self._calculate_choice_tension_impact(choice_type)
        for t, change in impact.items():
            current = float(self._current_tensions.get(t, 0.0))
            self._current_tensions[t] = max(0.0, min(1.0, current + float(change)))
        await self._save_tension_state()
        return SubsystemResponse(
            subsystem=self.subsystem_type, event_id=event.event_id, success=True,
            data={'tension_narrative': narrative, 'tension_changes': {t.value: c for t, c in impact.items()},
                  'player_impact': 'acknowledged'},
            side_effects=[]
        )
    
    async def _on_npc_reaction(self, event):
        _, _, _, SubsystemResponse = _orch()
        payload = event.payload or {}
        emotional_state = (payload.get('emotional_state') or '').lower()
        if emotional_state in ['angry', 'distressed', 'fearful']:
            self._current_tensions[TensionType.EMOTIONAL] = min(
                1.0, float(self._current_tensions.get(TensionType.EMOTIONAL, 0.0)) + 0.1
            )
            await self._save_tension_state()
        return SubsystemResponse(
            subsystem=self.subsystem_type, event_id=event.event_id, success=True,
            data={'tension_adjusted': True}, side_effects=[]
        )
    
    async def _on_state_sync(self, event):
        _, _, _, SubsystemResponse = _orch()
        # Accept either raw scene payload or {'scene_context': {...}}
        payload = event.payload or {}
        scene_context = payload.get('scene_context') or payload
        request = str(payload.get('request', '')).lower()
        
        # Special: analysis/report requested by tools
        if request == 'tension_analysis':
            await self.calculate_current_tensions()
            score = max(0.0, min(1.0, sum(self._current_tensions.values()) / max(1, len(self._current_tensions))))
            primary = ''
            if self._current_tensions:
                primary = max(self._current_tensions.items(), key=lambda x: x[1])[0].value
            observations: List[Dict[str, Any]] = [
                {'type': t.value, 'level': float(lvl), 'source': 'aggregate', 'note': ''}
                for t, lvl in self._current_tensions.items()
            ]
            return SubsystemResponse(
                subsystem=self.subsystem_type, event_id=event.event_id, success=True,
                data={
                    'tension_score': float(score),
                    'should_generate_conflict': bool(score > 0.6),
                    'primary_dynamic': primary,
                    'tensions': observations
                },
                side_effects=[]
            )
        
        if request == 'tension_report':
            await self.calculate_current_tensions()
            categories = [{'name': t.value, 'level': float(l)} for t, l in self._current_tensions.items()]
            overall = max(0.0, min(1.0, sum(l for _, l in self._current_tensions.items()) / max(1, len(self._current_tensions))))
            hotspots = [t for t, l in self._current_tensions.items() if l > 0.7]
            return SubsystemResponse(
                subsystem=self.subsystem_type, event_id=event.event_id, success=True,
                data={
                    'categories': categories,
                    'overall_score': float(overall),
                    'hotspots': [t.value for t in hotspots],
                    'last_updated': datetime.now().isoformat()
                },
                side_effects=[]
            )
        
        # Default scene manifestation path
        manifestation = await self.generate_tension_manifestation(scene_context)
        breaking_point = await self.check_tension_breaking_point()
        return SubsystemResponse(
            subsystem=self.subsystem_type, event_id=event.event_id, success=True,
            data={
                'manifestation': {
                    'type': manifestation.tension_type.value,
                    'level': manifestation.level,
                    'physical_cues': manifestation.physical_cues,
                    'dialogue_mods': manifestation.dialogue_modifications,
                    'environment': manifestation.environmental_changes,
                    'sensations': manifestation.player_sensations
                },
                'breaking_point': breaking_point,
                'current_tensions': {t.value: v for t, v in self._current_tensions.items()}
            },
            side_effects=[]
        )
    
    async def _on_tension_changed(self, event):
        """Handle direct tension adjustments (from modify_tension tool)."""
        _, EventType, _, SubsystemResponse = _orch()
        payload = event.payload or {}
        ttype = str(payload.get('tension_type', 'emotional')).lower()
        change = float(payload.get('change', 0.0))
        reason = str(payload.get('reason', ''))
        
        # Resolve enum robustly
        try:
            t_enum = TensionType[ttype.upper()]
        except Exception:
            t_enum = TensionType.EMOTIONAL
        
        old = float(self._current_tensions.get(t_enum, 0.0))
        new = old + change
        clamped = False
        if new < 0.0:
            new = 0.0
            clamped = True
        if new > 1.0:
            new = 1.0
            clamped = True
        
        self._current_tensions[t_enum] = float(new)
        await self._save_tension_state()
        
        return SubsystemResponse(
            subsystem=self.subsystem_type, event_id=event.event_id, success=True,
            data={
                'success': True,
                'tension_type': t_enum.value,
                'applied_change': float(change),
                'new_level': float(new),
                'clamped': bool(clamped),
                'side_effects': [f"tension_{t_enum.value}_set_{new:.2f}", f"reason:{reason}"]
            },
            side_effects=[]
        )
    
    async def _on_health_check(self, event):
        _, _, _, SubsystemResponse = _orch()
        return SubsystemResponse(
            subsystem=self.subsystem_type, event_id=event.event_id, success=True,
            data=await self.health_check(), side_effects=[]
        )
    
    # ----- Core tension management -----
    
    async def calculate_current_tensions(self) -> Dict[TensionType, float]:
        tensions = {t: 0.0 for t in TensionType}
        sources = await self._gather_tension_sources()
        prompt = f"""
Analyze these tension sources and calculate overall tension levels (0.0-1.0) per type.

Sources:
{json.dumps([self._source_to_dict(s) for s in sources], indent=2)}

Return JSON mapping: {{
  "power": 0.2, "social": 0.1, "emotional": 0.3, ...
}}
"""
        try:
            response = await Runner.run(self.tension_analyzer, prompt)
            parsed = extract_runner_response(response)
            result = json.loads(parsed) if parsed else {}
            for tension_type in TensionType:
                if tension_type.value in result:
                    tensions[tension_type] = max(0.0, min(1.0, float(result[tension_type.value])))
        except Exception as e:
            logger.warning(f"Failed to parse tension levels: {e}")
            for src in sources:
                t = self._map_source_to_tension_type(src)
                tensions[t] = max(0.0, min(1.0, tensions[t] + float(src.contribution) * 0.2))
        
        self._current_tensions = tensions
        await self._save_tension_state()
        
        # Best-effort: emit events for high tensions
        SubsystemType, EventType, SystemEvent, _ = _orch()
        try:
            synth = self._synthesizer() if self._synthesizer else None
            if synth:
                for t, level in tensions.items():
                    if level > 0.7:
                        await synth.emit_event(SystemEvent(
                            event_id=f"high_tension_{t.value}",
                            event_type=EventType.TENSION_CHANGED,
                            source_subsystem=self.subsystem_type,
                            payload={'tension_type': t.value, 'level': float(level)}
                        ))
        except Exception:
            pass
        
        return tensions
    
    async def build_tension(self, tension_type: TensionType, amount: float, source: str,
                            context: Optional[Dict] = None) -> Dict[str, Any]:
        old_level = float(self._current_tensions.get(tension_type, 0.0))
        new_level = max(0.0, min(1.0, old_level + float(amount)))
        self._current_tensions[tension_type] = new_level
        
        prompt = f"""
Narrate tension building:

Type: {tension_type.value}
Current Level: {old_level:.2f} → {new_level:.2f}
Source: {source}

Return JSON: {{ "description": "...", "cues": ["..."] }}
"""
        try:
            response = await Runner.run(self.escalation_narrator, prompt)
            narrative = json.loads(extract_runner_response(response) or '{}')
        except Exception:
            narrative = {'description': f"The {tension_type.value} tension grows", 'cues': ['A subtle shift in atmosphere']}
        
        await self._save_tension_state()
        return {
            'tension_type': tension_type.value,
            'old_level': old_level,
            'new_level': new_level,
            'narrative': narrative,
            'threshold_crossed': self._check_threshold_crossed(old_level, new_level)
        }
    
    async def resolve_tension(self, tension_type: TensionType, amount: float, resolution_type: str,
                              context: Optional[Dict] = None) -> Dict[str, Any]:
        old_level = float(self._current_tensions.get(tension_type, 0.0))
        new_level = max(0.0, min(1.0, old_level - float(amount)))
        self._current_tensions[tension_type] = new_level
        
        prompt = f"""
Narrate tension resolution:

Type: {tension_type.value}
Current Level: {old_level:.2f} → {new_level:.2f}
Resolution Type: {resolution_type}

Return JSON: {{ "release": "...", "aftermath": "..." }}
"""
        try:
            response = await Runner.run(self.escalation_narrator, prompt)
            narrative = json.loads(extract_runner_response(response) or '{}')
        except Exception:
            narrative = {'release': f"The {tension_type.value} tension eases", 'aftermath': 'calm'}
        
        await self._save_tension_state()
        return {
            'tension_type': tension_type.value,
            'old_level': old_level,
            'new_level': new_level,
            'narrative': narrative,
            'fully_resolved': new_level < 0.1
        }
    
    async def generate_tension_manifestation(self, scene_context: Dict[str, Any]) -> TensionManifestation:
        if not self._current_tensions:
            return self._create_no_tension_manifestation()
        dominant_type, dominant_level = max(self._current_tensions.items(), key=lambda x: x[1])
        if dominant_level < 0.1:
            return self._create_no_tension_manifestation()
        prompt = f"""
Generate tension manifestations:

Dominant Tension: {dominant_type.value} ({dominant_level:.2f})
Scene: {json.dumps(scene_context, indent=2)}

Return JSON with arrays:
{{
  "physical_cues": ["..."],
  "dialogue_modifications": ["..."],
  "environmental_changes": ["..."],
  "player_sensations": ["..."]
}}
"""
        try:
            response = await Runner.run(self.manifestation_generator, prompt)
            result = json.loads(extract_runner_response(response) or '{}')
            return TensionManifestation(
                tension_type=dominant_type,
                level=float(dominant_level),
                physical_cues=list(result.get('physical_cues', [])),
                dialogue_modifications=list(result.get('dialogue_modifications', [])),
                environmental_changes=list(result.get('environmental_changes', [])),
                player_sensations=list(result.get('player_sensations', []))
            )
        except Exception:
            return self._create_fallback_manifestation(dominant_type, float(dominant_level))
    
    async def check_tension_breaking_point(self) -> Optional[Dict[str, Any]]:
        breaking = {t: l for t, l in self._current_tensions.items() if l >= TensionLevel.BREAKING.value}
        if not breaking:
            return None
        breaking_type = max(breaking.items(), key=lambda x: x[1])[0]
        prompt = f"""
A tension has reached breaking point:

Breaking Tension: {breaking_type.value}
Level: {breaking[breaking_type]:.2f}

Return JSON:
{{ "trigger": "...", "consequences": ["..."], "choices": ["..."] }}
"""
        try:
            response = await Runner.run(self.escalation_narrator, prompt)
            result = json.loads(extract_runner_response(response) or '{}')
            return {
                'breaking_tension': breaking_type.value,
                'trigger': result.get('trigger', 'The tension snaps'),
                'consequences': result.get('consequences', []),
                'player_choices': result.get('choices', [])
            }
        except Exception:
            return {
                'breaking_tension': breaking_type.value,
                'trigger': 'The tension reaches a breaking point',
                'consequences': ['Things cannot continue as they were']
            }
    
    # ----- Helper methods -----
    
    async def _load_tension_state(self):
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT tension_type, level
                FROM TensionLevels
                WHERE user_id = $1 AND conversation_id = $2
            """, self.user_id, self.conversation_id)
        for r in rows or []:
            try:
                self._current_tensions[TensionType(str(r['tension_type']))] = float(r['level'])
            except Exception:
                pass
    
    async def _save_tension_state(self):
        async with get_db_connection_context() as conn:
            for ttype, level in self._current_tensions.items():
                await conn.execute("""
                    INSERT INTO TensionLevels (user_id, conversation_id, tension_type, level)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (user_id, conversation_id, tension_type)
                    DO UPDATE SET level = EXCLUDED.level, updated_at = NOW()
                """, self.user_id, self.conversation_id, ttype.value, float(level))
    
    async def _determine_initial_tensions(self, conflict_type: str, context: Dict[str, Any]) -> Dict[TensionType, float]:
        patterns = {
            'power': {TensionType.POWER: 0.6, TensionType.SOCIAL: 0.3},
            'social': {TensionType.SOCIAL: 0.7, TensionType.EMOTIONAL: 0.4},
            'romantic': {TensionType.SEXUAL: 0.5, TensionType.EMOTIONAL: 0.5},
            'economic': {TensionType.ECONOMIC: 0.8, TensionType.POWER: 0.3},
            'ideological': {TensionType.IDEOLOGICAL: 0.7, TensionType.SOCIAL: 0.4}
        }
        for key, pattern in patterns.items():
            if key in (conflict_type or '').lower():
                return pattern
        return {TensionType.EMOTIONAL: 0.4, TensionType.SOCIAL: 0.3}
    
    def _map_action_to_tension_changes(self, action_type: str, intensity: float) -> Dict[TensionType, float]:
        action = (action_type or '').lower()
        changes: Dict[TensionType, float] = {}
        if 'aggressive' in action:
            changes[TensionType.POWER] = 0.2 * intensity
            changes[TensionType.EMOTIONAL] = 0.1 * intensity
        elif 'diplomatic' in action:
            changes[TensionType.SOCIAL] = -0.1 * intensity
            changes[TensionType.POWER] = -0.05 * intensity
        elif 'manipulative' in action:
            changes[TensionType.SOCIAL] = 0.15 * intensity
            changes[TensionType.EMOTIONAL] = 0.1 * intensity
        return changes
    
    def _calculate_phase_tension_adjustments(self, from_phase: str, to_phase: str) -> Dict[TensionType, float]:
        adjustments: Dict[TensionType, float] = {}
        if to_phase == 'climax':
            for t in TensionType:
                adjustments[t] = 0.2
        elif to_phase == 'resolution':
            for t in TensionType:
                adjustments[t] = -0.3
        elif from_phase == 'emerging' and to_phase == 'rising':
            adjustments[TensionType.EMOTIONAL] = 0.1
            adjustments[TensionType.SOCIAL] = 0.1
        return adjustments
    
    async def _generate_choice_tension_narrative(self, choice_type: str) -> str:
        prompt = f"""
Generate a brief narrative for how this choice affects tension (one sentence):
Choice Type: {choice_type}
"""
        try:
            response = await Runner.run(self.escalation_narrator, prompt)
            return (extract_runner_response(response) or '').strip()
        except Exception:
            return ""
    
    def _calculate_choice_tension_impact(self, choice_type: str) -> Dict[TensionType, float]:
        choice = (choice_type or '').lower()
        impacts: Dict[TensionType, float] = {}
        if 'submit' in choice:
            impacts[TensionType.POWER] = -0.1
            impacts[TensionType.EMOTIONAL] = 0.05
        elif 'resist' in choice:
            impacts[TensionType.POWER] = 0.15
            impacts[TensionType.SOCIAL] = 0.1
        elif 'negotiate' in choice:
            impacts[TensionType.SOCIAL] = -0.05
        return impacts
    
    async def _gather_tension_sources(self) -> List[TensionSource]:
        sources: List[TensionSource] = []
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT id, conflict_type, intensity, progress
                FROM Conflicts
                WHERE user_id = $1 AND conversation_id = $2 AND is_active = true
            """, self.user_id, self.conversation_id)
        for r in rows or []:
            try:
                progress = float(r.get('progress', 0.0))
                sources.append(TensionSource(
                    source_type="conflict",
                    source_id=int(r['id']),
                    contribution=(progress / 100.0) * 0.5,
                    description=f"{r.get('conflict_type', 'conflict')} conflict"
                ))
            except Exception:
                continue
        return sources
    
    def _source_to_dict(self, source: TensionSource) -> Dict:
        return {
            'type': source.source_type,
            'id': str(source.source_id),
            'contribution': float(source.contribution),
            'description': source.description
        }
    
    def _map_source_to_tension_type(self, source: TensionSource) -> TensionType:
        mappings = {
            'conflict': TensionType.POWER,
            'npc': TensionType.EMOTIONAL,
            'environment': TensionType.VITAL,
            'activity': TensionType.SOCIAL
        }
        return mappings.get(source.source_type, TensionType.EMOTIONAL)
    
    def _check_threshold_crossed(self, old_level: float, new_level: float) -> Optional[str]:
        thresholds = [(0.2, "subtle"), (0.4, "noticeable"), (0.6, "palpable"), (0.8, "intense"), (1.0, "breaking")]
        for th, name in thresholds:
            if old_level < th <= new_level:
                return f"entered_{name}"
            if new_level < th <= old_level:
                return f"left_{name}"
        return None
    
    def _create_no_tension_manifestation(self) -> TensionManifestation:
        return TensionManifestation(
            tension_type=TensionType.EMOTIONAL, level=0.0,
            physical_cues=["Relaxed postures"],
            dialogue_modifications=["Natural speech"],
            environmental_changes=["Comfortable atmosphere"],
            player_sensations=["A sense of ease"]
        )
    
    def _create_fallback_manifestation(self, tension_type: TensionType, level: float) -> TensionManifestation:
        return TensionManifestation(
            tension_type=tension_type, level=level,
            physical_cues=[f"Subtle {tension_type.value} tension"],
            dialogue_modifications=["Careful words"],
            environmental_changes=["Charged atmosphere"],
            player_sensations=["Underlying tension"]
        )
    
    # Optional API used by orchestrator for bundle fast-paths
    async def get_npc_tensions(self, npc_ids: List[int]) -> Dict[Tuple[int, int], float]:
        """Lightweight pairwise tension estimation from relationship tables."""
        tensions: Dict[Tuple[int, int], float] = {}
        if not npc_ids or len(npc_ids) < 2:
            return tensions
        pairs = []
        base = list(set(int(n) for n in npc_ids))
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                pairs.append((base[i], base[j]))
        # Try to read a 'tension' or 'conflict' dimension; fall back to 0.3
        async with get_db_connection_context() as conn:
            for a, b in pairs:
                row = await conn.fetchrow("""
                    SELECT AVG(current_value) AS val
                    FROM relationship_dimensions
                    WHERE (entity1_id = $1 AND entity2_id = $2)
                       OR (entity1_id = $2 AND entity2_id = $1)
                      AND dimension IN ('tension','conflict','dominance')
                """, a, b)
                val = float(row['val']) if row and row['val'] is not None else 0.3
                tensions[(a, b)] = max(0.0, min(1.0, val))
        return tensions
    
    # ----- LLM Agents -----
    
    @property
    def tension_analyzer(self) -> Agent:
        if self._tension_analyzer is None:
            self._tension_analyzer = Agent(
                name="Tension Analyzer",
                instructions="""
                Analyze various sources to determine tension levels and types.
                """,
                model="gpt-5-nano",
            )
        return self._tension_analyzer
    
    @property
    def manifestation_generator(self) -> Agent:
        if self._manifestation_generator is None:
            self._manifestation_generator = Agent(
                name="Tension Manifestation Generator",
                instructions="""
                Generate specific, sensory manifestations of tension.
                """,
                model="gpt-5-nano",
            )
        return self._manifestation_generator
    
    @property
    def escalation_narrator(self) -> Agent:
        if self._escalation_narrator is None:
            self._escalation_narrator = Agent(
                name="Tension Escalation Narrator",
                instructions="""
                Narrate how tensions build, peak, and release.
                """,
                model="gpt-5-nano",
            )
        return self._escalation_narrator


# Keep orchestrator import path stable
class TensionSubsystem(TensionSystem):
    pass


# ===============================================================================
# PUBLIC API (routes through orchestrator)
# ===============================================================================

@function_tool
async def analyze_scene_tensions(
    ctx: RunContextWrapper,
    scene_description: str,
    npcs_present: List[int],
    current_activity: str
) -> AnalyzeSceneTensionsResponse:
    from logic.conflict_system.conflict_synthesizer import get_synthesizer
    SubsystemType, EventType, SystemEvent, _ = _orch()
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    synthesizer = await get_synthesizer(user_id, conversation_id)
    event = SystemEvent(
        event_id=f"tension_analyze_{datetime.now().timestamp()}",
        event_type=EventType.STATE_SYNC,
        source_subsystem=SubsystemType.SLICE_OF_LIFE,
        payload={
            'scene_context': {
                'scene_description': scene_description,
                'npcs': npcs_present,
                'present_npcs': npcs_present,
                'activity': current_activity
            },
            'request': 'tension_analysis'
        },
        target_subsystems={SubsystemType.TENSION},
        requires_response=True
    )
    responses = await synthesizer.emit_event(event) or []
    data = {}
    for r in responses:
        if r.subsystem == SubsystemType.TENSION:
            data = r.data or {}
            break
    score = float(max(0.0, min(1.0, data.get('tension_score', 0.0))))
    raw_obs = data.get('tensions', []) or []
    observations: List[TensionObservation] = []
    for t in raw_obs[:20]:
        level = float(max(0.0, min(1.0, t.get('level', 0.0))))
        observations.append({
            'type': str(t.get('type', 'ambient')),
            'level': level,
            'source': str(t.get('source', 'unknown')),
            'note': str(t.get('notes', t.get('note', '')))
        })
    return {
        'tension_score': score,
        'should_generate_conflict': bool(data.get('should_generate_conflict', False)),
        'primary_dynamic': str(data.get('primary_dynamic', 'none')),
        'observations': observations,
        'error': "" if data else "No response from tension system",
    }


@function_tool
async def modify_tension(
    ctx: RunContextWrapper,
    tension_type: str,
    change: float,
    reason: str
) -> ModifyTensionResponse:
    from logic.conflict_system.conflict_synthesizer import get_synthesizer
    SubsystemType, EventType, SystemEvent, _ = _orch()
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    synthesizer = await get_synthesizer(user_id, conversation_id)
    event = SystemEvent(
        event_id=f"manual_tension_{tension_type}",
        event_type=EventType.TENSION_CHANGED,
        source_subsystem=SubsystemType.TENSION,
        payload={'tension_type': tension_type, 'change': float(change), 'reason': reason},
        target_subsystems={SubsystemType.TENSION},
        requires_response=True
    )
    responses = await synthesizer.emit_event(event) or []
    data = responses[0].data if responses else {}
    new_level = float(max(0.0, min(1.0, data.get('new_level', 0.0))))
    side_effects = data.get('side_effects', [])
    if not isinstance(side_effects, list):
        side_effects = [str(side_effects)]
    return {
        'success': bool(data.get('success', bool(responses))),
        'tension_type': str(data.get('tension_type', tension_type)),
        'applied_change': float(data.get('applied_change', change)),
        'new_level': new_level,
        'clamped': bool(data.get('clamped', False)),
        'reason': str(reason),
        'side_effects': [str(s) for s in side_effects[:20]],
        'error': "" if responses else "No response from tension system",
    }


@function_tool
async def get_tension_report(
    ctx: RunContextWrapper
) -> TensionReportResponse:
    from logic.conflict_system.conflict_synthesizer import get_synthesizer
    SubsystemType, EventType, SystemEvent, _ = _orch()
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    synthesizer = await get_synthesizer(user_id, conversation_id)
    event = SystemEvent(
        event_id=f"tension_report_{datetime.now().timestamp()}",
        event_type=EventType.STATE_SYNC,
        source_subsystem=SubsystemType.SLICE_OF_LIFE,
        payload={'request': 'tension_report'},
        target_subsystems={SubsystemType.TENSION},
        requires_response=True
    )
    responses = await synthesizer.emit_event(event) or []
    data = {}
    for r in responses:
        if r.subsystem == SubsystemType.TENSION:
            data = r.data or {}
            break
    categories_raw = data.get('categories', data.get('tensions', {}))
    categories: List[TensionCategory] = []
    if isinstance(categories_raw, dict):
        for name, lvl in list(categories_raw.items())[:50]:
            val = float(max(0.0, min(1.0, lvl)))
            categories.append({'name': str(name), 'level': val})
    elif isinstance(categories_raw, list):
        for item in categories_raw[:50]:
            name = str(item.get('name', item.get('type', 'unknown')))
            val = float(max(0.0, min(1.0, item.get('level', 0.0))))
            categories.append({'name': name, 'level': val})
    overall = float(max(0.0, min(1.0, data.get('overall_score', data.get('tension_score', 0.0)))))
    hotspots_raw = data.get('hotspots', [])
    if not isinstance(hotspots_raw, list):
        hotspots_raw = [str(hotspots_raw)]
    hotspots = [str(h) for h in hotspots_raw[:20]]
    last_updated = data.get('last_updated') or datetime.now().isoformat()
    return {
        'total_categories': len(categories),
        'categories': categories,
        'overall_score': overall,
        'hotspots': hotspots,
        'last_updated_iso': str(last_updated),
        'error': "" if data else "No response from tension system",
    }
