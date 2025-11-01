"""Background tasks for conflict flow orchestration.

These tasks execute the slower prompt-based steps for conflict flow management
that were moved off the hot path:

* Initial flow bootstrapping (`initialize_conflict_flow`)
* Event analysis and pacing adjustments (`analyze_flow_event`)
* Dramatic beat synthesis (`generate_dramatic_beat`)
* Phase transition narration (`narrate_phase_transition`)
* Beat narration embellishment (`generate_beat_description`)

Each task writes its output to Redis so the synchronous hot path can serve
cached results. Tasks also reconcile finalized payloads back into the
`conflict_flows` tables to keep persisted state aligned with the async work.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from celery import shared_task

from agents import Agent
from db.connection import get_db_connection_context
from infra.cache import set_json
from logic.conflict_system.dynamic_conflict_template import extract_runner_response
from logic.conflict_system.conflict_flow_hotpath import (
    _beat_cache_key,
    _event_analysis_cache_key,
    _flow_init_cache_key,
    _transition_cache_key,
)
from nyx.tasks.utils import with_retry, run_coro
from nyx.utils.idempotency import idempotent
from nyx.gateway import llm_gateway
from nyx.gateway.llm_gateway import LLMRequest

logger = logging.getLogger(__name__)


_pacing_director_agent: Optional[Agent] = None
_flow_analyzer_agent: Optional[Agent] = None
_beat_generator_agent: Optional[Agent] = None
_transition_narrator_agent: Optional[Agent] = None


def _get_pacing_director() -> Agent:
    global _pacing_director_agent
    if _pacing_director_agent is None:
        _pacing_director_agent = Agent(
            name="Pacing Director",
            instructions="""
            Direct the pacing and rhythm of conflicts.
            Keep pacing natural, avoid rushing or dragging.
            """,
            model="gpt-5-nano",
        )
    return _pacing_director_agent


def _get_flow_analyzer() -> Agent:
    global _flow_analyzer_agent
    if _flow_analyzer_agent is None:
        _flow_analyzer_agent = Agent(
            name="Flow Analyzer",
            instructions="""
            Analyze flow effectiveness and advise on intensity, momentum, and transitions.
            """,
            model="gpt-5-nano",
        )
    return _flow_analyzer_agent


def _get_beat_generator() -> Agent:
    global _beat_generator_agent
    if _beat_generator_agent is None:
        _beat_generator_agent = Agent(
            name="Dramatic Beat Generator",
            instructions="""
            Generate dramatic beats that advance conflict meaningfully.
            """,
            model="gpt-5-nano",
        )
    return _beat_generator_agent


def _get_transition_narrator() -> Agent:
    global _transition_narrator_agent
    if _transition_narrator_agent is None:
        _transition_narrator_agent = Agent(
            name="Transition Narrator",
            instructions="""
            Narrate transitions between conflict phases with clear, earned shifts.
            """,
            model="gpt-5-nano",
        )
    return _transition_narrator_agent


async def _run_prompt(agent: Agent, prompt: str) -> str:
    response = await llm_gateway.execute(
        LLMRequest(
            prompt=prompt,
            agent=agent,
        )
    )
    return extract_runner_response(response)


async def _persist_flow_state(
    user_id: int,
    conversation_id: int,
    conflict_id: int,
    *,
    phase: Optional[str] = None,
    pacing: Optional[str] = None,
    intensity: Optional[float] = None,
    momentum: Optional[float] = None,
    phase_progress: Optional[float] = None,
) -> None:
    async with get_db_connection_context() as conn:
        await conn.execute(
            """
            UPDATE conflict_flows
            SET current_phase = COALESCE($1, current_phase),
                pacing_style = COALESCE($2, pacing_style),
                intensity = COALESCE($3, intensity),
                momentum = COALESCE($4, momentum),
                phase_progress = COALESCE($5, phase_progress)
            WHERE user_id = $6 AND conversation_id = $7 AND conflict_id = $8
            """,
            phase,
            pacing,
            intensity,
            momentum,
            phase_progress,
            user_id,
            conversation_id,
            conflict_id,
        )


async def _record_transition(
    user_id: int,
    conversation_id: int,
    conflict_id: int,
    from_phase: Optional[str],
    to_phase: str,
    transition_type: str,
    narrative: str,
    trigger: str,
) -> None:
    async with get_db_connection_context() as conn:
        await conn.execute(
            """
            INSERT INTO phase_transitions
                (user_id, conversation_id, conflict_id, from_phase, to_phase,
                 transition_type, trigger, narrative, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)
            ON CONFLICT DO NOTHING
            """,
            user_id,
            conversation_id,
            conflict_id,
            from_phase,
            to_phase,
            transition_type,
            trigger,
            narrative,
        )


async def _record_dramatic_beat(
    user_id: int,
    conversation_id: int,
    conflict_id: int,
    beat_type: str,
    description: str,
    impact: float,
) -> None:
    async with get_db_connection_context() as conn:
        await conn.execute(
            """
            INSERT INTO dramatic_beats
                (user_id, conversation_id, conflict_id, beat_type, description, impact, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
            ON CONFLICT DO NOTHING
            """,
            user_id,
            conversation_id,
            conflict_id,
            beat_type,
            description,
            impact,
        )


def _idempotency_key_transition(payload: Dict[str, Any]) -> str:
    conflict_id = payload.get("conflict_id")
    from_phase = payload.get("from_phase") or ""
    to_phase = payload.get("to_phase") or ""
    return f"flow_transition:{conflict_id}:{from_phase}:{to_phase}"


def _idempotency_key_beat(payload: Dict[str, Any]) -> str:
    conflict_id = payload.get("conflict_id")
    beat_type = payload.get("beat_meta", {}).get("type", "")
    beat_id = payload.get("beat_meta", {}).get("id", "")
    return f"flow_beat:{conflict_id}:{beat_type}:{beat_id}"


async def _narrate_transition_async(
    conflict_id: int, from_phase: str, to_phase: str, context: Dict[str, Any]
) -> str:
    prompt = f"""
Narrate phase transition:

From: {from_phase}
To: {to_phase}
Trigger: {json.dumps(context, indent=2)}
Return JSON:
{{
  "type": "natural|triggered|forced|stalled|reversed",
  "narrative": "2-3 sentences"
}}
"""

    raw = await _run_prompt(_get_transition_narrator(), prompt)
    return raw


async def _narrate_beat_async(conflict_id: int, beat_meta: Dict[str, Any]) -> str:
    prompt = f"""
Generate dramatic beat description:

Conflict ID: {conflict_id}
Beat Meta: {json.dumps(beat_meta, indent=2)}

Return JSON:
{{
  "description": "..."
}}
"""

    raw = await _run_prompt(_get_transition_narrator(), prompt)
    return raw


async def _initialize_flow_async(
    conflict_type: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = f"""
Initialize conflict flow:

Conflict Type: {conflict_type}
Context: {json.dumps(context, indent=2)}

Return JSON:
{{
  "phase": "seeds|emerging|rising",
  "pacing": "slow_burn|rapid_escalation|waves|steady|erratic",
  "intensity": 0.0,
  "momentum": 0.0,
  "conditions": ["..."]
}}
"""

    raw = await _run_prompt(_get_pacing_director(), prompt)
    try:
        return json.loads(raw)
    except Exception:
        return {}


async def _analyze_event_async(flow_state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
Update conflict flow:

Current State:
- Phase: {flow_state.get('phase')}
- Intensity: {flow_state.get('intensity')}
- Momentum: {flow_state.get('momentum')}
- Progress: {flow_state.get('phase_progress')}

Event: {json.dumps(event, indent=2)}

Return JSON:
{{
  "intensity": 0.0,
  "momentum": 0.0,
  "progress_change": 0.0,
  "should_transition": true,
  "narrative_impact": "..."
}}
"""

    raw = await _run_prompt(_get_flow_analyzer(), prompt)
    try:
        return json.loads(raw)
    except Exception:
        return {}


async def _generate_beat_async(flow_state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
Generate dramatic beat:

Current Phase: {flow_state.get('phase')}
Intensity: {flow_state.get('intensity')}
Momentum: {flow_state.get('momentum')}
Context: {json.dumps(context, indent=2)}

Return JSON:
{{
  "type": "revelation|betrayal|escalation|reconciliation|twist|moment",
  "description": "...",
  "impact": 0.0,
  "characters": [1,2,3]
}}
"""

    raw = await _run_prompt(_get_beat_generator(), prompt)
    try:
        return json.loads(raw)
    except Exception:
        return {}


@shared_task(
    name="nyx.tasks.background.flow_tasks.initialize_conflict_flow",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=lambda payload: f"flow_init:{payload.get('conflict_id')}")
def initialize_conflict_flow(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    conflict_id = payload.get("conflict_id")
    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")
    conflict_type = payload.get("conflict_type", "unknown")
    context = payload.get("context", {})
    ttl = payload.get("ttl", 3600)

    if not conflict_id or user_id is None or conversation_id is None:
        raise ValueError("conflict_id, user_id, and conversation_id are required")

    logger.info(
        "Running conflict flow initialization for conflict=%s (user=%s convo=%s)",
        conflict_id,
        user_id,
        conversation_id,
    )

    result = run_coro(_initialize_flow_async(conflict_type, context))

    phase = (result.get("phase") or "emerging").lower()
    pacing = (result.get("pacing") or "steady").lower()
    intensity = float(result.get("intensity", 0.3) or 0.3)
    momentum = float(result.get("momentum", 0.2) or 0.2)
    conditions = result.get("conditions", []) or []

    cache_payload = {
        "phase": phase,
        "pacing": pacing,
        "intensity": intensity,
        "momentum": momentum,
        "conditions": conditions,
        "source": "llm",
        "timestamp": datetime.utcnow().isoformat(),
    }
    set_json(_flow_init_cache_key(conflict_id), cache_payload, ex=ttl)

    run_coro(
        _persist_flow_state(
            user_id,
            conversation_id,
            conflict_id,
            phase=phase,
            pacing=pacing,
            intensity=intensity,
            momentum=momentum,
            phase_progress=0.0,
        )
    )

    return {
        "status": "initialized",
        "conflict_id": conflict_id,
        "phase": phase,
        "pacing": pacing,
    }


@shared_task(
    name="nyx.tasks.background.flow_tasks.analyze_flow_event",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=lambda payload: _event_analysis_cache_key(payload.get("conflict_id"), payload.get("event", {}).get("event_id")))
def analyze_flow_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    conflict_id = payload.get("conflict_id")
    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")
    event = payload.get("event", {})
    flow_state = payload.get("flow_state", {})
    ttl = payload.get("ttl", 900)

    if not conflict_id or user_id is None or conversation_id is None:
        raise ValueError("conflict_id, user_id, and conversation_id are required")

    event_id = event.get("event_id") or event.get("id") or "latest"
    logger.info("Analyzing conflict flow event %s for conflict %s", event_id, conflict_id)

    analysis = run_coro(_analyze_event_async(flow_state, event))

    intensity = analysis.get("intensity")
    momentum = analysis.get("momentum")
    progress_change = analysis.get("progress_change")
    should_transition = bool(analysis.get("should_transition"))
    narrative_impact = analysis.get("narrative_impact", "")

    cache_payload = {
        "intensity": intensity,
        "momentum": momentum,
        "progress_change": progress_change,
        "should_transition": should_transition,
        "narrative_impact": narrative_impact,
        "event_id": event_id,
        "timestamp": datetime.utcnow().isoformat(),
    }
    set_json(_event_analysis_cache_key(conflict_id, event_id), cache_payload, ex=ttl)

    new_intensity = float(intensity) if intensity is not None else None
    new_momentum = float(momentum) if momentum is not None else None
    base_progress = float(flow_state.get("phase_progress") or 0.0)
    if progress_change is not None:
        new_progress = max(0.0, min(1.0, base_progress + float(progress_change)))
    else:
        new_progress = None

    run_coro(
        _persist_flow_state(
            user_id,
            conversation_id,
            conflict_id,
            intensity=new_intensity,
            momentum=new_momentum,
            phase_progress=new_progress,
        )
    )

    return {
        "status": "analyzed",
        "conflict_id": conflict_id,
        "event_id": event_id,
        "should_transition": should_transition,
    }


@shared_task(
    name="nyx.tasks.background.flow_tasks.generate_dramatic_beat",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=lambda payload: _beat_cache_key(payload.get("conflict_id"), payload.get("beat_id", "")))
def generate_dramatic_beat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    conflict_id = payload.get("conflict_id")
    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")
    beat_id = payload.get("beat_id")
    flow_state = payload.get("flow_state", {})
    context = payload.get("context", {})
    ttl = payload.get("ttl", 1800)

    if not conflict_id or not beat_id or user_id is None or conversation_id is None:
        raise ValueError("conflict_id, beat_id, user_id, and conversation_id are required")

    logger.info("Generating dramatic beat %s for conflict %s", beat_id, conflict_id)

    beat_result = run_coro(_generate_beat_async(flow_state, context))
    beat_type = beat_result.get("type", "moment")
    description = beat_result.get("description", "A significant moment occurs")
    impact = float(beat_result.get("impact", 0.2) or 0.2)
    characters = beat_result.get("characters", []) or []

    cache_payload = {
        "beat_meta": {
            "type": beat_type,
            "id": beat_id,
            "characters": characters,
        },
        "impact": impact,
        "text": description,
        "timestamp": datetime.utcnow().isoformat(),
    }
    set_json(_beat_cache_key(conflict_id, beat_id), cache_payload, ex=ttl)

    run_coro(
        _record_dramatic_beat(
            user_id,
            conversation_id,
            conflict_id,
            beat_type,
            description,
            impact,
        )
    )

    return {
        "status": "generated",
        "conflict_id": conflict_id,
        "beat_id": beat_id,
        "beat_type": beat_type,
    }


@shared_task(
    name="nyx.tasks.background.flow_tasks.narrate_phase_transition",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_transition)
def narrate_phase_transition(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    conflict_id = payload.get("conflict_id")
    from_phase = payload.get("from_phase") or ""
    to_phase = payload.get("to_phase", "")
    context = payload.get("context", {})
    ttl = payload.get("ttl", 3600)
    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")

    if (
        not conflict_id
        or not to_phase
        or user_id is None
        or conversation_id is None
    ):
        raise ValueError("conflict_id, to_phase, user_id, and conversation_id are required")

    logger.info(
        "Generating transition narration for conflict %s: %s -> %s",
        conflict_id,
        from_phase,
        to_phase,
    )

    raw = run_coro(_narrate_transition_async(conflict_id, from_phase, to_phase, context))
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {}

    prose = parsed.get("narrative", "The conflict shifts.")
    transition_type = parsed.get("type", "natural")

    cache_payload = {
        "text": prose,
        "transition_type": transition_type,
        "from_phase": from_phase,
        "to_phase": to_phase,
        "timestamp": datetime.utcnow().isoformat(),
    }

    set_json(_transition_cache_key(conflict_id), cache_payload, ex=ttl)

    run_coro(
        _record_transition(
            user_id,
            conversation_id,
            conflict_id,
            from_phase or None,
            to_phase,
            transition_type,
            prose,
            json.dumps(context),
        )
    )

    return {
        "status": "narrated",
        "conflict_id": conflict_id,
        "transition_type": transition_type,
        "prose_length": len(prose),
    }


@shared_task(
    name="nyx.tasks.background.flow_tasks.generate_beat_description",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_beat)
def generate_beat_description(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    conflict_id = payload.get("conflict_id")
    beat_meta = payload.get("beat_meta", {})
    ttl = payload.get("ttl", 1800)

    if not conflict_id or not beat_meta:
        raise ValueError("conflict_id and beat_meta are required")

    beat_id = beat_meta.get("id", "")
    beat_type = beat_meta.get("type", "unknown")

    logger.info(
        "Generating beat description for conflict %s, beat %s (%s)",
        conflict_id,
        beat_id,
        beat_type,
    )

    raw = run_coro(_narrate_beat_async(conflict_id, beat_meta))
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {}

    prose = parsed.get("description", "A significant moment occurs")

    key = _beat_cache_key(conflict_id, beat_id)
    cached = {
        "text": prose,
        "beat_meta": beat_meta,
        "timestamp": datetime.utcnow().isoformat(),
    }
    set_json(key, cached, ex=ttl)

    return {
        "status": "narrated",
        "conflict_id": conflict_id,
        "cache_key": key,
        "beat_id": beat_id,
    }


__all__ = [
    "initialize_conflict_flow",
    "analyze_flow_event",
    "generate_dramatic_beat",
    "narrate_phase_transition",
    "generate_beat_description",
]
