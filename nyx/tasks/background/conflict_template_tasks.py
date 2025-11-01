"""Background tasks for conflict template generation and caching."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from celery import shared_task

from agents import Agent
from db.connection import get_db_connection_context
from logic.conflict_system.dynamic_conflict_template import extract_runner_response
from monitoring.metrics import metrics
from nyx.tasks.utils import run_coro, with_retry
from nyx.gateway import llm_gateway
from nyx.gateway.llm_gateway import LLMRequest

logger = logging.getLogger(__name__)

_CACHE_STAGE_COLUMNS: Dict[str, tuple[str, str]] = {
    "variation": ("variation", "variation_status"),
    "adaptation": ("adaptation", "adaptation_status"),
    "unique": ("unique_payload", "unique_status"),
    "hooks": ("hooks", "hooks_status"),
}

_STAGE_DEPENDENCIES: Dict[str, str] = {
    "adaptation": "variation",
    "unique": "adaptation",
    "hooks": "unique",
}


def _safe_metric(stage: str, result: str) -> None:
    try:
        metrics().CONFLICT_TEMPLATE_WARMUPS.labels(stage=stage, result=result).inc()
    except Exception:  # pragma: no cover - metrics are optional
        pass


def _safe_pending_metric(pending: Dict[str, int]) -> None:
    try:
        gauge = metrics().CONFLICT_TEMPLATE_CACHE_PENDING
    except Exception:  # pragma: no cover
        return

    for stage, count in pending.items():
        try:
            gauge.labels(stage=stage).set(count)
        except Exception:  # pragma: no cover
            continue


_TEMPLATE_CREATOR: Optional[Agent] = None
_VARIATION_AGENT: Optional[Agent] = None
_ADAPTATION_AGENT: Optional[Agent] = None
_UNIQUE_AGENT: Optional[Agent] = None
_HOOK_AGENT: Optional[Agent] = None


def _template_creator_agent() -> Agent:
    global _TEMPLATE_CREATOR
    if _TEMPLATE_CREATOR is None:
        _TEMPLATE_CREATOR = Agent(
            name="Template Creator",
            instructions="""
            Create flexible conflict templates that can generate countless variations.

            Design templates that:
            - Have clear core structures
            - Include variable elements
            - Allow contextual adaptation
            - Support complexity scaling
            - Enable emergent storytelling

            Templates should be seeds for infinite stories, not rigid patterns.

            IMPORTANT: Always respond with valid JSON only, no explanatory text.
            """,
            model="gpt-5-nano",
        )
    return _TEMPLATE_CREATOR


def _variation_agent() -> Agent:
    global _VARIATION_AGENT
    if _VARIATION_AGENT is None:
        _VARIATION_AGENT = Agent(
            name="Variation Generator",
            instructions="""
            Generate unique variations from conflict templates.

            Create variations that:
            - Feel fresh and original
            - Respect template structure
            - Add surprising elements
            - Fit the context perfectly
            - Create memorable experiences

            Each variation should feel like a unique story, not a copy.

            IMPORTANT: Always respond with valid JSON only, no explanatory text.
            """,
            model="gpt-5-nano",
        )
    return _VARIATION_AGENT


def _adaptation_agent() -> Agent:
    global _ADAPTATION_AGENT
    if _ADAPTATION_AGENT is None:
        _ADAPTATION_AGENT = Agent(
            name="Context Adapter",
            instructions="""
            Adapt conflict templates to specific contexts.

            Ensure adaptations:
            - Fit the current situation
            - Respect character personalities
            - Match location atmosphere
            - Align with ongoing narratives
            - Feel organic to the world

            Make templated conflicts feel bespoke to the moment.

            IMPORTANT: Always respond with valid JSON only, no explanatory text.
            """,
            model="gpt-5-nano",
        )
    return _ADAPTATION_AGENT


def _unique_agent() -> Agent:
    global _UNIQUE_AGENT
    if _UNIQUE_AGENT is None:
        _UNIQUE_AGENT = Agent(
            name="Uniqueness Engine",
            instructions="""
            Ensure each generated conflict feels unique and memorable.

            Add elements that:
            - Create distinctive moments
            - Generate quotable lines
            - Produce unexpected twists
            - Build character-specific drama
            - Leave lasting impressions

            Every conflict should have something players remember.

            IMPORTANT: Always respond with valid JSON only, no explanatory text.
            """,
            model="gpt-5-nano",
        )
    return _UNIQUE_AGENT


def _hook_agent() -> Agent:
    global _HOOK_AGENT
    if _HOOK_AGENT is None:
        _HOOK_AGENT = Agent(
            name="Narrative Hook Generator",
            instructions="""
            Generate compelling hooks that draw players into conflicts.

            Create hooks that:
            - Grab immediate attention
            - Create emotional investment
            - Promise interesting outcomes
            - Connect to player history
            - Build anticipation

            Make players WANT to engage with the conflict.

            IMPORTANT: Always respond with valid JSON only, no explanatory text.
            """,
            model="gpt-5-nano",
        )
    return _HOOK_AGENT


async def _ensure_cache_table() -> None:
    async with get_db_connection_context() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conflict_template_cache (
                cache_id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                conversation_id INTEGER NOT NULL,
                template_id INTEGER NOT NULL REFERENCES conflict_templates(template_id) ON DELETE CASCADE,
                context_hash TEXT NOT NULL,
                context JSONB NOT NULL DEFAULT '{}'::jsonb,
                variation JSONB,
                variation_status TEXT NOT NULL DEFAULT 'pending',
                adaptation JSONB,
                adaptation_status TEXT NOT NULL DEFAULT 'pending',
                unique_payload JSONB,
                unique_status TEXT NOT NULL DEFAULT 'pending',
                hooks JSONB,
                hooks_status TEXT NOT NULL DEFAULT 'pending',
                last_error TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (template_id, context_hash)
            );
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conflict_template_cache_lookup
                ON conflict_template_cache(user_id, conversation_id, template_id, context_hash);
            """
        )


async def _set_template_metadata_status(template_id: int, status: str, error: Optional[str] = None) -> None:
    async with get_db_connection_context() as conn:
        metadata = await conn.fetchval(
            """
            SELECT metadata FROM conflict_templates
             WHERE template_id = $1
            """,
            template_id,
        )
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        payload = dict(metadata or {})
        payload["status"] = status
        if error is None:
            payload.pop("last_error", None)
        else:
            payload["last_error"] = error
        await conn.execute(
            """
            UPDATE conflict_templates
               SET metadata = $2::jsonb,
                   updated_at = NOW()
             WHERE template_id = $1
            """,
            template_id,
            json.dumps(payload),
        )


async def _get_template_record(template_id: int) -> Optional[Dict[str, Any]]:
    async with get_db_connection_context() as conn:
        row = await conn.fetchrow(
            """
            SELECT template_id, category, name, base_structure, variable_elements,
                   contextual_modifiers, complexity_min, complexity_max
              FROM conflict_templates
             WHERE template_id = $1
            """,
            template_id,
        )
    if not row:
        return None
    return {
        "template_id": int(row["template_id"]),
        "category": row["category"],
        "name": row["name"],
        "base_structure": dict(row["base_structure"] or {}),
        "variable_elements": list(row["variable_elements"] or []),
        "contextual_modifiers": dict(row["contextual_modifiers"] or {}),
        "complexity_min": float(row["complexity_min"] or 0.2),
        "complexity_max": float(row["complexity_max"] or 0.9),
    }


async def _update_template_record(template_id: int, data: Dict[str, Any]) -> None:
    async with get_db_connection_context() as conn:
        await conn.execute(
            """
            UPDATE conflict_templates
               SET name = $2,
                   base_structure = $3::jsonb,
                   variable_elements = $4::jsonb,
                   contextual_modifiers = $5::jsonb,
                   complexity_min = $6,
                   complexity_max = $7,
                   updated_at = NOW()
             WHERE template_id = $1
            """,
            template_id,
            data["name"],
            json.dumps(data["base_structure"], default=str),
            json.dumps(data["variable_elements"], default=str),
            json.dumps(data["contextual_modifiers"], default=str),
            float(data.get("complexity_range", {}).get("minimum", data.get("complexity_min", 0.2))),
            float(data.get("complexity_range", {}).get("maximum", data.get("complexity_max", 0.9))),
        )


async def _set_stage_running(
    user_id: int,
    conversation_id: int,
    template_id: int,
    context_hash: str,
    stage: str,
) -> None:
    await _ensure_cache_table()
    _, status_column = _CACHE_STAGE_COLUMNS[stage]
    async with get_db_connection_context() as conn:
        await conn.execute(
            f"""
            UPDATE conflict_template_cache
               SET {status_column} = 'running',
                   updated_at = NOW()
             WHERE user_id = $1 AND conversation_id = $2
               AND template_id = $3 AND context_hash = $4
            """,
            user_id,
            conversation_id,
            template_id,
            context_hash,
        )


async def _finalize_stage(
    user_id: int,
    conversation_id: int,
    template_id: int,
    context_hash: str,
    stage: str,
    *,
    status: str,
    data: Optional[Any] = None,
    error: Optional[str] = None,
) -> None:
    await _ensure_cache_table()
    column, status_column = _CACHE_STAGE_COLUMNS[stage]
    async with get_db_connection_context() as conn:
        if data is None:
            current = await conn.fetchval(
                f"""
                SELECT {column} FROM conflict_template_cache
                 WHERE user_id = $1 AND conversation_id = $2
                   AND template_id = $3 AND context_hash = $4
                """,
                user_id,
                conversation_id,
                template_id,
                context_hash,
            )
        else:
            current = data

        await conn.execute(
            f"""
            UPDATE conflict_template_cache
               SET {status_column} = $5,
                   {column} = $6::jsonb,
                   last_error = $7,
                   updated_at = NOW()
             WHERE user_id = $1 AND conversation_id = $2
               AND template_id = $3 AND context_hash = $4
            """,
            user_id,
            conversation_id,
            template_id,
            context_hash,
            status,
            json.dumps(current, default=str) if current is not None else None,
            error,
        )


async def _mark_stage_pending(
    user_id: int,
    conversation_id: int,
    template_id: int,
    context_hash: str,
    stage: str,
) -> bool:
    await _ensure_cache_table()
    _, status_column = _CACHE_STAGE_COLUMNS[stage]
    async with get_db_connection_context() as conn:
        row = await conn.fetchrow(
            f"""
            UPDATE conflict_template_cache
               SET {status_column} = 'pending',
                   updated_at = NOW()
             WHERE user_id = $1 AND conversation_id = $2
               AND template_id = $3 AND context_hash = $4
               AND {status_column} NOT IN ('pending', 'running')
            RETURNING cache_id
            """,
            user_id,
            conversation_id,
            template_id,
            context_hash,
        )
    return bool(row)


async def _get_cache_entry(
    user_id: int,
    conversation_id: int,
    template_id: int,
    context_hash: str,
) -> Optional[Dict[str, Any]]:
    await _ensure_cache_table()
    async with get_db_connection_context() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM conflict_template_cache
             WHERE user_id = $1 AND conversation_id = $2
               AND template_id = $3 AND context_hash = $4
            """,
            user_id,
            conversation_id,
            template_id,
            context_hash,
        )
    return dict(row) if row else None


async def _fetch_pending_counts() -> Dict[str, int]:
    await _ensure_cache_table()
    async with get_db_connection_context() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                SUM(CASE WHEN variation_status IN ('pending', 'fallback', 'error') THEN 1 ELSE 0 END) AS variation,
                SUM(CASE WHEN adaptation_status IN ('pending', 'fallback', 'error') THEN 1 ELSE 0 END) AS adaptation,
                SUM(CASE WHEN unique_status IN ('pending', 'fallback', 'error') THEN 1 ELSE 0 END) AS unique,
                SUM(CASE WHEN hooks_status IN ('pending', 'fallback', 'error') THEN 1 ELSE 0 END) AS hooks
              FROM conflict_template_cache
            """
        )
    if not row:
        return {stage: 0 for stage in _CACHE_STAGE_COLUMNS.keys()}
    return {
        "variation": int(row["variation"] or 0),
        "adaptation": int(row["adaptation"] or 0),
        "unique": int(row["unique"] or 0),
        "hooks": int(row["hooks"] or 0),
    }


def _fallback_template(category: str, base_concept: str) -> Dict[str, Any]:
    return {
        "name": f"{category.replace('_', ' ').title()} Template",
        "base_structure": {
            "core_tension": f"A {category} conflict",
            "stakeholder_roles": ["protagonist", "antagonist", "mediator"],
            "progression_phases": ["setup", "escalation", "climax", "resolution"],
            "resolution_conditions": ["victory", "compromise", "defeat", "stalemate"],
        },
        "variable_elements": [
            "Setting location",
            "Time of day",
            "Number of participants",
            "Stakes level",
            "Public vs private",
            "Emotional intensity",
            "Physical vs verbal",
            "Resource type",
            "Authority involvement",
            "Witness presence",
        ],
        "contextual_modifiers": {
            "personality_axes": ["aggressive-passive", "cooperative-competitive"],
            "environmental_factors": ["crowded-isolated", "formal-casual"],
            "cultural_variables": ["traditional-modern", "hierarchical-egalitarian"],
            "power_modifiers": ["equal-unequal", "official-unofficial"],
        },
        "generation_rules": {
            "required_elements": ["core_tension", "stakeholders"],
            "optional_elements": ["witnesses", "mediators"],
            "exclusions": ["violence", "illegal_activity"],
        },
        "complexity_range": {
            "minimum": 0.2,
            "maximum": 0.9,
        },
    }


def _ensure_template_fields(
    data: Dict[str, Any],
    category: str,
    base_concept: str,
) -> Dict[str, Any]:
    fallback = _fallback_template(category, base_concept)
    for key, value in fallback.items():
        data.setdefault(key, value)
    return data


def _fallback_variation(template: Dict[str, Any]) -> Dict[str, Any]:
    base_structure = template.get("base_structure", {})
    variable_elements = template.get("variable_elements", [])
    return {
        "template_id": template.get("template_id"),
        "seed": "fallback",
        "core_tension": base_structure.get("core_tension", "A conflict emerges"),
        "stakeholder_configuration": {
            "roles": base_structure.get("stakeholder_roles", ["participant"]),
            "relationships": ["neutral"],
        },
        "chosen_variables": {var: "default" for var in variable_elements[:3]},
        "progression_path": base_structure.get("progression_phases", ["start", "middle", "end"]),
        "resolution_options": base_structure.get("resolution_conditions", ["resolved"]),
        "twist_potential": "An unexpected turn of events",
    }


def _fallback_adaptation(context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "location_integration": f"Takes place in {context.get('location', 'the current location')}",
        "npc_motivations": {},
        "temporal_factors": "Happens at an opportune moment",
        "continuity_connections": [],
        "environmental_obstacles": [],
        "atmospheric_elements": ["tense", "uncertain"],
    }


def _fallback_unique() -> Dict[str, Any]:
    return {
        "unique_elements": [
            "An unexpected alliance forms",
            "A hidden truth is revealed",
            "The stakes suddenly increase",
        ],
        "memorable_quote": "This changes everything.",
        "signature_moment": "A dramatic confrontation",
        "sensory_detail": "The tension is palpable",
        "conversation_piece": "The unexpected twist",
    }


@shared_task(
    name="nyx.tasks.background.conflict_template_tasks.create_template_for_category",
    bind=True,
)
@with_retry
def create_template_for_category(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    template_id = int(payload["template_id"])
    category = payload.get("category", "general")
    base_concept = payload.get("base_concept", "")

    run_coro(_set_template_metadata_status(template_id, "running"))
    try:
        agent = _template_creator_agent()
        prompt = f"""
        Create a flexible conflict template:

        Category: {category}
        Base Concept: {base_concept}

        Design a template that can generate hundreds of unique conflicts.

        Return ONLY valid JSON (no other text):
        {{
            "name": "Template name",
            "base_structure": {{
                "core_tension": "Fundamental conflict",
                "stakeholder_roles": ["role types needed"],
                "progression_phases": ["typical phases"],
                "resolution_conditions": ["ways it can end"]
            }},
            "variable_elements": [
                "List of 10+ elements that can change between instances"
            ],
            "contextual_modifiers": {{
                "personality_axes": ["relevant personality traits"],
                "environmental_factors": ["location/setting influences"],
                "cultural_variables": ["social/cultural elements"],
                "power_modifiers": ["hierarchy/authority factors"]
            }},
            "generation_rules": {{
                "required_elements": ["must-have components"],
                "optional_elements": ["can-have components"],
                "exclusions": ["incompatible elements"]
            }},
            "complexity_range": {{
                "minimum": 0.2,
                "maximum": 0.9
            }}
        }}
        """
        response = run_coro(
            llm_gateway.execute(
                LLMRequest(
                    prompt=prompt,
                    agent=agent,
                )
            )
        )
        response_text = extract_runner_response(response).strip()
        if response_text.startswith("```"):
            response_text = response_text.strip("`").lstrip("json").strip()
        data = json.loads(response_text) if response_text else {}
        data = _ensure_template_fields(data, category, base_concept)
    except Exception as exc:
        logger.exception("Template creation failed", exc_info=exc)
        data = _ensure_template_fields({}, category, base_concept)
        run_coro(_set_template_metadata_status(template_id, "error", error=str(exc)))
        _safe_metric("template", "error")
        raise

    run_coro(_update_template_record(template_id, data))
    run_coro(_set_template_metadata_status(template_id, "ready"))
    _safe_metric("template", "success")
    return {"template_id": template_id, "status": "ready"}


@shared_task(
    name="nyx.tasks.background.conflict_template_tasks.generate_variation",
    bind=True,
)
@with_retry
def generate_variation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    user_id = int(payload["user_id"])
    conversation_id = int(payload["conversation_id"])
    template_id = int(payload["template_id"])
    context_hash = payload["context_hash"]
    context = payload.get("context", {})

    run_coro(_set_stage_running(user_id, conversation_id, template_id, context_hash, "variation"))
    template = run_coro(_get_template_record(template_id))
    if not template:
        error_msg = f"Template {template_id} not found"
        run_coro(
            _finalize_stage(
                user_id,
                conversation_id,
                template_id,
                context_hash,
                "variation",
                status="error",
                data=None,
                error=error_msg,
            )
        )
        _safe_metric("variation", "error")
        raise ValueError(error_msg)

    try:
        agent = _variation_agent()
        prompt = f"""
        Generate a unique variation from this template:

        Template: {template['name']}
        Base Structure: {json.dumps(template['base_structure'], default=str)}
        Variable Elements: {json.dumps(template['variable_elements'], default=str)}
        Context: {json.dumps(context, default=str)}

        Create a variation that uses the base structure and varies 3-5 variable elements.

        Return ONLY valid JSON:
        {{
            "seed": "Unique identifier for this variation",
            "core_tension": "Specific tension for this instance",
            "stakeholder_configuration": {{
                "roles": ["specific roles"],
                "relationships": ["specific relationships"]
            }},
            "chosen_variables": {{
                "variable_name": "specific value"
            }},
            "progression_path": ["specific phases"],
            "resolution_options": ["specific endings"],
            "twist_potential": "Unexpected element"
        }}
        """
        response = run_coro(
            llm_gateway.execute(
                LLMRequest(
                    prompt=prompt,
                    agent=agent,
                )
            )
        )
        response_text = extract_runner_response(response).strip()
        if response_text.startswith("```"):
            response_text = response_text.strip("`").lstrip("json").strip()
        variation = json.loads(response_text) if response_text else {}
        if not variation:
            variation = _fallback_variation(template)
    except Exception as exc:
        logger.exception("Variation generation failed", exc_info=exc)
        variation = _fallback_variation(template)
        run_coro(
            _finalize_stage(
                user_id,
                conversation_id,
                template_id,
                context_hash,
                "variation",
                status="error",
                data=variation,
                error=str(exc),
            )
        )
        _safe_metric("variation", "error")
        raise

    variation["template_id"] = template_id
    run_coro(
        _finalize_stage(
            user_id,
            conversation_id,
            template_id,
            context_hash,
            "variation",
            status="ready",
            data=variation,
            error=None,
        )
    )
    _safe_metric("variation", "success")
    return {"template_id": template_id, "context_hash": context_hash}


@shared_task(
    name="nyx.tasks.background.conflict_template_tasks.adapt_variation",
    bind=True,
)
@with_retry
def adapt_variation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    user_id = int(payload["user_id"])
    conversation_id = int(payload["conversation_id"])
    template_id = int(payload["template_id"])
    context_hash = payload["context_hash"]
    context = payload.get("context", {})

    cache = run_coro(_get_cache_entry(user_id, conversation_id, template_id, context_hash))
    if not cache or cache.get("variation_status") != "ready" or not cache.get("variation"):
        raise self.retry(countdown=30)

    run_coro(_set_stage_running(user_id, conversation_id, template_id, context_hash, "adaptation"))
    try:
        agent = _adaptation_agent()
        prompt = f"""
        Adapt this conflict variation to the specific context:

        Variation: {json.dumps(cache['variation'], default=str)}
        Current Location: {context.get('location', 'unknown')}
        Present NPCs: {json.dumps(context.get('npcs', []), default=str)}
        Time of Day: {context.get('time', 'unknown')}
        Recent Events: {json.dumps(context.get('recent_events', []), default=str)}

        Return ONLY valid JSON:
        {{
            "location_integration": "How location shapes conflict",
            "npc_motivations": {{}},
            "temporal_factors": "How timing affects it",
            "continuity_connections": ["links to recent events"],
            "environmental_obstacles": ["location-specific challenges"],
            "atmospheric_elements": ["mood and tone elements"]
        }}
        """
        response = run_coro(
            llm_gateway.execute(
                LLMRequest(
                    prompt=prompt,
                    agent=agent,
                )
            )
        )
        response_text = extract_runner_response(response).strip()
        if response_text.startswith("```"):
            response_text = response_text.strip("`").lstrip("json").strip()
        adaptation = json.loads(response_text) if response_text else {}
        if not adaptation:
            adaptation = _fallback_adaptation(context)
    except Exception as exc:
        logger.exception("Context adaptation failed", exc_info=exc)
        adaptation = _fallback_adaptation(context)
        run_coro(
            _finalize_stage(
                user_id,
                conversation_id,
                template_id,
                context_hash,
                "adaptation",
                status="error",
                data=adaptation,
                error=str(exc),
            )
        )
        _safe_metric("adaptation", "error")
        raise

    run_coro(
        _finalize_stage(
            user_id,
            conversation_id,
            template_id,
            context_hash,
            "adaptation",
            status="ready",
            data=adaptation,
            error=None,
        )
    )
    _safe_metric("adaptation", "success")
    return {"template_id": template_id, "context_hash": context_hash}


@shared_task(
    name="nyx.tasks.background.conflict_template_tasks.add_unique_elements",
    bind=True,
)
@with_retry
def add_unique_elements(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    user_id = int(payload["user_id"])
    conversation_id = int(payload["conversation_id"])
    template_id = int(payload["template_id"])
    context_hash = payload["context_hash"]
    context = payload.get("context", {})

    cache = run_coro(_get_cache_entry(user_id, conversation_id, template_id, context_hash))
    if not cache or cache.get("adaptation_status") != "ready" or not cache.get("adaptation"):
        raise self.retry(countdown=30)

    merged_conflict = dict(cache.get("variation") or {})
    merged_conflict["context_adaptation"] = cache.get("adaptation")

    run_coro(_set_stage_running(user_id, conversation_id, template_id, context_hash, "unique"))
    try:
        agent = _unique_agent()
        prompt = f"""
        Add unique elements to make this conflict memorable:

        Conflict: {json.dumps(merged_conflict, default=str)}
        Player History: {json.dumps(context.get('player_history', []), default=str)}

        Return ONLY valid JSON:
        {{
            "unique_elements": [
                "List of 3-5 unique elements"
            ],
            "memorable_quote": "Something an NPC might say",
            "signature_moment": "A scene players will remember",
            "sensory_detail": "Something visceral",
            "conversation_piece": "What players will discuss later"
        }}
        """
        response = run_coro(
            llm_gateway.execute(
                LLMRequest(
                    prompt=prompt,
                    agent=agent,
                )
            )
        )
        response_text = extract_runner_response(response).strip()
        if response_text.startswith("```"):
            response_text = response_text.strip("`").lstrip("json").strip()
        unique_data = json.loads(response_text) if response_text else {}
        if not unique_data:
            unique_data = _fallback_unique()
    except Exception as exc:
        logger.exception("Unique element generation failed", exc_info=exc)
        unique_data = _fallback_unique()
        run_coro(
            _finalize_stage(
                user_id,
                conversation_id,
                template_id,
                context_hash,
                "unique",
                status="error",
                data=unique_data,
                error=str(exc),
            )
        )
        _safe_metric("unique", "error")
        raise

    run_coro(
        _finalize_stage(
            user_id,
            conversation_id,
            template_id,
            context_hash,
            "unique",
            status="ready",
            data=unique_data,
            error=None,
        )
    )
    _safe_metric("unique", "success")
    return {"template_id": template_id, "context_hash": context_hash}


@shared_task(
    name="nyx.tasks.background.conflict_template_tasks.generate_hooks",
    bind=True,
)
@with_retry
def generate_hooks(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    user_id = int(payload["user_id"])
    conversation_id = int(payload["conversation_id"])
    template_id = int(payload["template_id"])
    context_hash = payload["context_hash"]
    context = payload.get("context", {})

    cache = run_coro(_get_cache_entry(user_id, conversation_id, template_id, context_hash))
    if not cache or cache.get("unique_status") != "ready" or not cache.get("unique_payload"):
        raise self.retry(countdown=30)

    conflict_data = dict(cache.get("variation") or {})
    conflict_data["unique_elements"] = cache.get("unique_payload", {}).get("unique_elements", [])
    conflict_data["signature_content"] = cache.get("unique_payload", {})

    run_coro(_set_stage_running(user_id, conversation_id, template_id, context_hash, "hooks"))
    try:
        agent = _hook_agent()
        prompt = f"""
        Generate narrative hooks for this conflict:

        Core Tension: {conflict_data.get('core_tension', '')}
        Unique Elements: {json.dumps(conflict_data.get('unique_elements', []), default=str)}
        Signature Moment: {conflict_data.get('signature_content', {}).get('signature_moment', '')}

        Create 3-5 hooks that grab attention and create investment.

        Return ONLY valid JSON:
        {{
            "hooks": [
                "List of compelling one-sentence hooks"
            ]
        }}
        """
        response = run_coro(
            llm_gateway.execute(
                LLMRequest(
                    prompt=prompt,
                    agent=agent,
                )
            )
        )
        response_text = extract_runner_response(response).strip()
        if response_text.startswith("```"):
            response_text = response_text.strip("`").lstrip("json").strip()
        data = json.loads(response_text) if response_text else {}
        hooks = data.get("hooks") if isinstance(data, dict) else None
        if not hooks:
            hooks = [
                "A new challenge emerges that tests everyone involved.",
                "Old tensions resurface in unexpected ways.",
                "What starts small quickly escalates beyond control.",
            ]
    except Exception as exc:
        logger.exception("Hook generation failed", exc_info=exc)
        hooks = [
            "A new challenge emerges that tests everyone involved.",
            "Old tensions resurface in unexpected ways.",
            "What starts small quickly escalates beyond control.",
        ]
        run_coro(
            _finalize_stage(
                user_id,
                conversation_id,
                template_id,
                context_hash,
                "hooks",
                status="error",
                data=hooks,
                error=str(exc),
            )
        )
        _safe_metric("hooks", "error")
        raise

    run_coro(
        _finalize_stage(
            user_id,
            conversation_id,
            template_id,
            context_hash,
            "hooks",
            status="ready",
            data=hooks,
            error=None,
        )
    )
    _safe_metric("hooks", "success")
    return {"template_id": template_id, "context_hash": context_hash}


async def _fetch_warmup_entries(limit: int) -> List[Dict[str, Any]]:
    await _ensure_cache_table()
    async with get_db_connection_context() as conn:
        rows = await conn.fetch(
            """
            SELECT user_id, conversation_id, template_id, context_hash, context,
                   variation_status, adaptation_status, unique_status, hooks_status
              FROM conflict_template_cache
             WHERE variation_status IN ('fallback', 'error')
                OR adaptation_status IN ('pending', 'fallback', 'error')
                OR unique_status IN ('pending', 'fallback', 'error')
                OR hooks_status IN ('pending', 'fallback', 'error')
             ORDER BY updated_at ASC
             LIMIT $1
            """,
            limit,
        )
    return [dict(row) for row in rows]


def _queue_stage(stage: str, entry: Dict[str, Any]) -> bool:
    user_id = int(entry["user_id"])
    conversation_id = int(entry["conversation_id"])
    template_id = int(entry["template_id"])
    context_hash = entry["context_hash"]
    payload = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "template_id": template_id,
        "context_hash": context_hash,
        "context": entry.get("context", {}),
    }

    if not run_coro(
        _mark_stage_pending(user_id, conversation_id, template_id, context_hash, stage)
    ):
        return False

    if stage == "variation":
        generate_variation.apply_async(args=[payload], queue="background")
    elif stage == "adaptation":
        adapt_variation.apply_async(args=[payload], queue="background")
    elif stage == "unique":
        add_unique_elements.apply_async(args=[payload], queue="background")
    elif stage == "hooks":
        generate_hooks.apply_async(args=[payload], queue="background")
    else:
        return False

    return True


@shared_task(
    name="nyx.tasks.background.conflict_template_tasks.warm_template_cache",
    bind=True,
)
@with_retry
def warm_template_cache(self, limit: int = 20) -> Dict[str, Any]:
    entries = run_coro(_fetch_warmup_entries(limit))
    queued = 0
    for entry in entries:
        for stage in ("variation", "adaptation", "unique", "hooks"):
            status_key = _CACHE_STAGE_COLUMNS[stage][1]
            if entry.get(status_key) in {"ready", "running", "pending"}:
                continue
            dependency = _STAGE_DEPENDENCIES.get(stage)
            if dependency and entry.get(_CACHE_STAGE_COLUMNS[dependency][1]) != "ready":
                continue
            if _queue_stage(stage, entry):
                queued += 1
            break

    pending = run_coro(_fetch_pending_counts())
    _safe_pending_metric(pending)
    return {"queued": queued, "pending": pending}


__all__ = [
    "create_template_for_category",
    "generate_variation",
    "adapt_variation",
    "add_unique_elements",
    "generate_hooks",
    "warm_template_cache",
]
