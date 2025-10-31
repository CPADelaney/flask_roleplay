"""Background tasks for conflict victory narrative generation."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Sequence, Tuple

from celery import shared_task

from agents import Agent, Runner
from logic.conflict_system import conflict_victory_hotpath
from logic.conflict_system.dynamic_conflict_template import extract_runner_response
from nyx.tasks.utils import run_coro
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)


_victory_generator_agent: Agent | None = None
_achievement_narrator_agent: Agent | None = None
_consequence_calculator_agent: Agent | None = None
_epilogue_writer_agent: Agent | None = None


def _victory_generator() -> Agent:
    global _victory_generator_agent
    if _victory_generator_agent is None:
        _victory_generator_agent = Agent(
            name="Victory Condition Generator",
            instructions="""
            Generate nuanced victory conditions for conflicts.
            """,
            model="gpt-5-nano",
        )
    return _victory_generator_agent


def _achievement_narrator() -> Agent:
    global _achievement_narrator_agent
    if _achievement_narrator_agent is None:
        _achievement_narrator_agent = Agent(
            name="Achievement Narrator",
            instructions="""
            Narrate victory achievements with emotional depth.
            """,
            model="gpt-5-nano",
        )
    return _achievement_narrator_agent


def _consequence_calculator() -> Agent:
    global _consequence_calculator_agent
    if _consequence_calculator_agent is None:
        _consequence_calculator_agent = Agent(
            name="Victory Consequence Calculator",
            instructions="""
            Analyse the immediate and long-term consequences of conflict victories.
            """,
            model="gpt-5-nano",
        )
    return _consequence_calculator_agent


def _epilogue_writer() -> Agent:
    global _epilogue_writer_agent
    if _epilogue_writer_agent is None:
        _epilogue_writer_agent = Agent(
            name="Conflict Epilogue Writer",
            instructions="""
            Weave closing chapters for resolved conflicts while hinting at future tension.
            """,
            model="gpt-5-nano",
        )
    return _epilogue_writer_agent


async def _update_victory_conditions(
    updates: Sequence[Tuple[int, Dict[str, Any]]],
    task_id: str,
) -> None:
    if not updates:
        return
    serialized_requirements = [
        (
            condition_id,
            update.get('victory_type') or 'narrative',
            update.get('description') or '',
            json.dumps(update.get('requirements') or {}),
        )
        for condition_id, update in updates
    ]
    async with get_db_connection_context() as conn:
        await conn.executemany(
            """
            UPDATE victory_conditions
               SET victory_type = $2,
                   description = $3,
                   requirements = $4,
                   updated_at = CURRENT_TIMESTAMP
             WHERE condition_id = $1
            """,
            serialized_requirements,
        )

    metadata_updates = [
        (
            condition_id,
            conflict_victory_hotpath.build_entry(
                'ready',
                task_id=task_id,
                result={
                    'victory_type': update.get('victory_type'),
                    'description': update.get('description'),
                    'requirements': update.get('requirements'),
                    'impact': update.get('impact'),
                },
            ),
        )
        for condition_id, update in updates
    ]
    await conflict_victory_hotpath.write_many_condition_metadata(
        metadata_updates,
        conflict_victory_hotpath.TaskKey.GENERATOR,
    )


def _parse_conditions(payload: Dict[str, Any], raw: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(raw or '{}')
    except json.JSONDecodeError:
        data = {}
    conditions = data.get('conditions')
    if isinstance(conditions, list):
        parsed: List[Dict[str, Any]] = []
        for entry in conditions:
            if isinstance(entry, dict):
                parsed.append(entry)
        if parsed:
            return parsed
    stakeholder = payload.get('stakeholder') or {}
    conflict_type = payload.get('conflict_type') or 'narrative'
    return conflict_victory_hotpath.fallback_victory_conditions(conflict_type, stakeholder)


@shared_task(
    name="nyx.tasks.background.conflict_victory.generate_victory_conditions",
    bind=True,
    acks_late=True,
)
def generate_victory_conditions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    condition_ids = [int(cid) for cid in payload.get('condition_ids') or [] if int(cid) > 0]
    if not condition_ids:
        return {'status': 'skipped', 'reason': 'missing_condition_ids'}

    prompt = f"""
Generate victory conditions for this stakeholder:

Conflict Type: {payload.get('conflict_type')}
Stakeholder ID: {(payload.get('stakeholder') or {}).get('id')}
Role: {(payload.get('stakeholder') or {}).get('role', 'participant')}
Involvement: {(payload.get('stakeholder') or {}).get('involvement', 'primary')}

Return JSON:
{{
  "conditions": [
    {{
      "victory_type": "dominance|submission|compromise|pyrrhic|tactical|moral|escape|transformation|stalemate|narrative",
      "description": "What this victory looks like",
      "requirements": {{"specific": {{}}, "narrative": {{}}}},
      "impact": {{"relationship": 0.0, "power": 0.0, "satisfaction": 0.0}}
    }}
  ]
}}
"""

    try:
        response = run_coro(Runner.run(_victory_generator(), prompt))
        raw = extract_runner_response(response)
    except Exception:
        logger.exception("Victory condition generation failed for condition_ids=%s", condition_ids)
        raw = ''

    parsed_conditions = _parse_conditions(payload, raw)
    updates: List[Tuple[int, Dict[str, Any]]] = []
    for idx, condition_id in enumerate(condition_ids):
        source = parsed_conditions[min(idx, len(parsed_conditions) - 1)] if parsed_conditions else {}
        update = {
            'victory_type': str(source.get('victory_type') or 'narrative'),
            'description': str(source.get('description') or ''),
            'requirements': source.get('requirements') or {},
            'impact': source.get('impact') or {},
        }
        updates.append((condition_id, update))

    run_coro(_update_victory_conditions(updates, self.request.id))
    return {'status': 'updated', 'updated_conditions': len(updates)}


@shared_task(
    name="nyx.tasks.background.conflict_victory.generate_achievement_summary",
    bind=True,
    acks_late=True,
)
def generate_achievement_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    condition_id = int(payload.get('condition_id') or 0)
    if not condition_id:
        return {'status': 'skipped', 'reason': 'missing_condition_id'}

    condition_stub = {
        'condition_id': condition_id,
        'victory_type': payload.get('victory_type'),
        'description': payload.get('description'),
    }
    current_state = payload.get('current_state') or {}

    prompt = f"""
Narrate this victory achievement:

Victory Type: {payload.get('victory_type')}
Description: {payload.get('description')}
Context: {json.dumps(current_state)}
Write a powerful 2-3 paragraph narration.
"""

    try:
        response = run_coro(Runner.run(_achievement_narrator(), prompt))
        summary = extract_runner_response(response).strip()
    except Exception:
        logger.exception("Achievement summary generation failed for condition_id=%s", condition_id)
        summary = ''

    if not summary:
        summary = conflict_victory_hotpath.fallback_achievement_summary(condition_stub, current_state)

    entry = conflict_victory_hotpath.build_entry(
        'ready',
        task_id=self.request.id,
        result=summary,
    )
    run_coro(
        conflict_victory_hotpath.write_condition_metadata(
            condition_id,
            conflict_victory_hotpath.TaskKey.SUMMARY,
            entry,
        )
    )
    return {'status': 'ready', 'condition_id': condition_id, 'summary': summary}


@shared_task(
    name="nyx.tasks.background.conflict_victory.calculate_victory_consequences",
    bind=True,
    acks_late=True,
)
def calculate_victory_consequences(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    condition_id = int(payload.get('condition_id') or 0)
    if not condition_id:
        return {'status': 'skipped', 'reason': 'missing_condition_id'}

    condition_stub = {
        'condition_id': condition_id,
        'victory_type': payload.get('victory_type'),
        'description': payload.get('description'),
    }
    current_state = payload.get('current_state') or {}

    prompt = f"""
Calculate consequences of this victory:

Victory Type: {payload.get('victory_type')}
Description: {payload.get('description')}
Current State: {json.dumps(current_state)}

Return JSON:
{{"immediate": {{}}, "long_term": {{}}, "hidden_consequences": []}}
"""

    try:
        response = run_coro(Runner.run(_consequence_calculator(), prompt))
        raw = extract_runner_response(response)
        consequences = json.loads(raw)
    except Exception:
        logger.exception("Consequence calculation failed for condition_id=%s", condition_id)
        consequences = conflict_victory_hotpath.fallback_victory_consequences(
            condition_stub,
            current_state,
        )
    else:
        if not isinstance(consequences, dict):
            consequences = conflict_victory_hotpath.fallback_victory_consequences(
                condition_stub,
                current_state,
            )

    entry = conflict_victory_hotpath.build_entry(
        'ready',
        task_id=self.request.id,
        result=consequences,
    )
    run_coro(
        conflict_victory_hotpath.write_condition_metadata(
            condition_id,
            conflict_victory_hotpath.TaskKey.CONSEQUENCES,
            entry,
        )
    )
    return {'status': 'ready', 'condition_id': condition_id}


@shared_task(
    name="nyx.tasks.background.conflict_victory.generate_conflict_epilogue",
    bind=True,
    acks_late=True,
)
def generate_conflict_epilogue(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    condition_ids = [int(cid) for cid in payload.get('condition_ids') or [] if int(cid) > 0]
    conflict = payload.get('conflict') or {}
    achievements = payload.get('achievements') or []
    resolution = payload.get('resolution') or {}

    prompt = f"""
Write an epilogue for this resolved conflict:

Conflict: {conflict.get('conflict_name')}
Type: {conflict.get('conflict_type')}
Description: {conflict.get('description', '')}
Achievements: {json.dumps(achievements)}
Resolution: {json.dumps(resolution)}

Write 3-4 paragraphs that feel like the end of a chapter, not the end of the story.
"""

    try:
        response = run_coro(Runner.run(_epilogue_writer(), prompt))
        epilogue = extract_runner_response(response).strip()
    except Exception:
        logger.exception("Conflict epilogue generation failed for conflict_id=%s", payload.get('conflict_id'))
        epilogue = ''

    if not epilogue:
        epilogue = conflict_victory_hotpath.fallback_conflict_epilogue(
            conflict,
            achievements,
            resolution,
        )

    if condition_ids:
        run_coro(
            conflict_victory_hotpath.mark_task_ready(
                condition_ids,
                conflict_victory_hotpath.TaskKey.EPILOGUE,
                epilogue,
                self.request.id,
            )
        )

    return {'status': 'ready', 'epilogue': epilogue}


@shared_task(
    name="nyx.tasks.background.conflict_victory.generate_consolation",
    bind=True,
    acks_late=True,
)
def generate_consolation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    condition_id = int(payload.get('condition_id') or 0)
    if not condition_id:
        return {'status': 'skipped', 'reason': 'missing_condition_id'}

    condition_stub = {
        'condition_id': condition_id,
        'victory_type': payload.get('victory_type'),
        'description': payload.get('description'),
        'progress': payload.get('progress'),
    }

    prompt = f"""
Generate a consolation for this partial victory:

Victory Type: {payload.get('victory_type')}
Progress: {float(payload.get('progress') or 0.0)*100:.0f}%
Description: {payload.get('description')}

Write a single encouraging paragraph.
"""

    try:
        response = run_coro(Runner.run(_achievement_narrator(), prompt))
        consolation = extract_runner_response(response).strip()
    except Exception:
        logger.exception("Consolation generation failed for condition_id=%s", condition_id)
        consolation = ''

    if not consolation:
        consolation = conflict_victory_hotpath.fallback_consolation(condition_stub)

    entry = conflict_victory_hotpath.build_entry(
        'ready',
        task_id=self.request.id,
        result=consolation,
    )
    run_coro(
        conflict_victory_hotpath.write_condition_metadata(
            condition_id,
            conflict_victory_hotpath.TaskKey.CONSOLATION,
            entry,
        )
    )
    return {'status': 'ready', 'condition_id': condition_id}


__all__ = [
    'generate_victory_conditions',
    'generate_achievement_summary',
    'calculate_victory_consequences',
    'generate_conflict_epilogue',
    'generate_consolation',
]
