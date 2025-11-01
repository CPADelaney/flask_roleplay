"""Celery tasks for stakeholder background processing."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, Optional

from nyx.tasks.base import NyxTask, app
from openai import AsyncOpenAI

from db.connection import get_db_connection_context
from nyx.tasks.utils import run_coro, with_retry
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)


_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI()
    return _client


_json_block_re = re.compile(r"\{[\s\S]*\}|\[[\s\S]*\]")


def _extract_json(text: str) -> str:
    if not text:
        return "{}"

    text = text.strip()
    if (text.startswith("{") and text.endswith("}")) or (
        text.startswith("[") and text.endswith("]")
    ):
        return text

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fence:
        inner = fence.group(1).strip()
        if inner:
            return inner

    match = _json_block_re.search(text)
    if match:
        return match.group(0)

    return "{}"


async def llm_json(prompt: str) -> Dict[str, Any]:
    try:
        client = _get_client()
        response = await client.responses.create(model="gpt-5-nano", input=prompt)
        text = getattr(response, "output_text", None)
        if not text:
            try:
                parts = []
                for item in getattr(response, "output", []) or []:
                    for chunk in getattr(item, "content", []) or []:
                        value = getattr(chunk, "text", None)
                        if value:
                            parts.append(value)
                text = "\n".join(parts).strip()
            except Exception:
                text = ""

        payload = _extract_json(text)
        return json.loads(payload)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("llm_json failed: %s", exc)
        return {}


def _compute_context_hash(context: Dict[str, Any]) -> str:
    stable_context = json.dumps(context, sort_keys=True)
    return hashlib.sha256(stable_context.encode()).hexdigest()[:16]


def _idempotency_key_action(payload: Dict[str, Any]) -> str:
    stakeholder_id = payload.get("stakeholder_id")
    context_hash = payload.get("context_hash", "")
    return f"stakeholder_action:{stakeholder_id}:{context_hash}"


def _idempotency_key_reaction(payload: Dict[str, Any]) -> str:
    stakeholder_id = payload.get("stakeholder_id")
    event_id = payload.get("event_id", "")
    return f"stakeholder_reaction:{stakeholder_id}:{event_id}"


def _idempotency_key_populate(payload: Dict[str, Any]) -> str:
    stakeholder_id = payload.get("stakeholder_id")
    return f"stakeholder_populate:{stakeholder_id}"


def _idempotency_key_role(payload: Dict[str, Any]) -> str:
    stakeholder_id = payload.get("stakeholder_id")
    context_hash = payload.get("context_hash", "")
    return f"stakeholder_role:{stakeholder_id}:{context_hash}"


async def _fetch_npc_details(npc_id: int) -> Dict[str, Any]:
    async with get_db_connection_context() as conn:
        row = await conn.fetchrow("SELECT * FROM NPCs WHERE npc_id = $1", npc_id)
    return dict(row) if row else {}


async def _store_planned_action(
    stakeholder_id: int,
    scene_context: Dict[str, Any],
    kind: str,
    payload: Dict[str, Any],
    priority: int,
    context_hash: Optional[str] = None,
) -> int:
    async with get_db_connection_context() as conn:
        action_id = await conn.fetchval(
            """
            INSERT INTO planned_stakeholder_actions
            (stakeholder_id, scene_id, scene_hash, conflict_id, kind, payload,
             status, priority, context_hash, available_at)
            VALUES ($1, $2, $3, $4, $5, $6, 'ready', $7, $8, NOW())
            RETURNING id
            """,
            stakeholder_id,
            scene_context.get("scene_id"),
            scene_context.get("scene_hash"),
            scene_context.get("conflict_id"),
            kind,
            json.dumps(payload),
            priority,
            context_hash,
        )
    logger.info(
        "Stored planned %s %s for stakeholder %s", kind, action_id, stakeholder_id
    )
    return action_id


async def _create_stakeholder_profile_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    stakeholder_id = payload["stakeholder_id"]
    npc_id = payload["npc_id"]
    conflict_id = payload["conflict_id"]
    initial_role = payload.get("initial_role")

    npc_details = await _fetch_npc_details(npc_id)

    prompt = f"""
Create stakeholder profile as JSON:
NPC: {npc_details.get('name', 'Unknown')}
Personality: {npc_details.get('personality_traits', 'Unknown')}
Conflict Context: Conflict #{conflict_id}
Suggested Role: {initial_role or 'determine based on personality'}

Return JSON:
{{
  "role": "bystander|instigator|defender|mediator|opportunist|victim|escalator|peacemaker",
  "decision_style": "reactive|rational|emotional|instinctive|calculating|principled",
  "goals": ["..."],
  "resources": {{"influence": 0.0, "wealth": 0.0}},
  "stress_level": 0.0,
  "commitment_level": 0.0
}}
"""

    llm_result = await llm_json(prompt)

    role = llm_result.get("role", initial_role or "bystander")
    decision_style = llm_result.get("decision_style", "reactive")
    stress = float(llm_result.get("stress_level", 0.3) or 0.3)
    commitment = float(llm_result.get("commitment_level", 0.5) or 0.5)

    async with get_db_connection_context() as conn:
        await conn.execute(
            """
            UPDATE stakeholders
            SET role = $1,
                decision_style = $2,
                stress_level = $3,
                commitment_level = $4
            WHERE stakeholder_id = $5
            """,
            role,
            decision_style,
            stress,
            commitment,
            stakeholder_id,
        )

    logger.info(
        "Updated stakeholder %s role=%s decision_style=%s via LLM",
        stakeholder_id,
        role,
        decision_style,
    )

    return {
        "stakeholder_id": stakeholder_id,
        "role": role,
        "decision_style": decision_style,
        "goals": llm_result.get("goals", []),
        "resources": llm_result.get("resources", {}),
        "stress_level": stress,
        "commitment_level": commitment,
    }


async def _generate_action_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    stakeholder_id = payload["stakeholder_id"]
    stakeholder_snapshot = payload.get("stakeholder_snapshot", {})
    conflict_state = payload.get("conflict_state", {})
    available_options = payload.get("available_options")
    scene_context = payload.get("scene_context", {})
    context_hash = payload.get("context_hash") or _compute_context_hash(
        {
            "stakeholder_id": stakeholder_id,
            "conflict_state": conflict_state,
            "available_options": available_options,
        }
    )
    priority = payload.get("priority", 5)

    prompt = f"""
Make decision as JSON:
Character: {stakeholder_snapshot.get('name')}
Personality: {stakeholder_snapshot.get('personality_traits')}
Role: {stakeholder_snapshot.get('role')}
Decision Style: {stakeholder_snapshot.get('decision_style')}
Stress Level: {stakeholder_snapshot.get('stress_level')}

Conflict State: {json.dumps(conflict_state, indent=2)}
Available Options: {json.dumps(available_options) if available_options else "default"}

Return JSON:
{{
  "action_type": "observant|aggressive|defensive|diplomatic|manipulative|supportive|evasive|strategic",
  "description": "What they do",
  "target": 123,
  "resources": {{}},
  "success_probability": 0.0,
  "consequences": {{}}
}}
"""

    llm_result = await llm_json(prompt)

    action_payload = {
        "action_type": llm_result.get("action_type", "observant"),
        "description": llm_result.get("description", "Observes the situation"),
        "target": llm_result.get("target"),
        "resources": llm_result.get("resources", {}),
        "success_probability": float(llm_result.get("success_probability", 0.5) or 0.5),
        "consequences": llm_result.get("consequences", {}),
        "stakeholder_snapshot": stakeholder_snapshot,
        "conflict_state": conflict_state,
        "available_options": available_options,
    }

    action_id = await _store_planned_action(
        stakeholder_id=stakeholder_id,
        scene_context=scene_context,
        kind="action",
        payload=action_payload,
        priority=priority,
        context_hash=context_hash,
    )

    action_payload["planned_action_id"] = action_id
    action_payload["context_hash"] = context_hash

    return {
        "status": "generated",
        "action_id": action_id,
        "stakeholder_id": stakeholder_id,
        "payload": action_payload,
    }


async def _generate_reaction_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    stakeholder_id = payload["stakeholder_id"]
    stakeholder_snapshot = payload.get("stakeholder_snapshot", {})
    triggering_action = payload.get("triggering_action", {})
    action_context = payload.get("action_context", {})
    event_id = payload.get("event_id", "")
    priority = payload.get("priority", 6)

    prompt = f"""
Generate reaction as JSON:
Reacting Character: {stakeholder_snapshot.get('name')}
Personality: {stakeholder_snapshot.get('personality_traits')}
Current Stress: {stakeholder_snapshot.get('stress_level')}

Triggering Action: {triggering_action.get('description')} ({triggering_action.get('action_type')})

Return JSON:
{{
  "reaction_type": "counter|support|ignore|escalate|de-escalate",
  "description": "What they do",
  "emotional_response": "neutral|angry|fearful|surprised|relieved",
  "relationship_impact": 0.0,
  "stress_impact": 0.0
}}
"""

    llm_result = await llm_json(prompt)

    reaction_payload = {
        "reaction_type": llm_result.get("reaction_type", "observe"),
        "description": llm_result.get("description", "Reacts to the action"),
        "emotional_response": llm_result.get("emotional_response", "neutral"),
        "relationship_impact": llm_result.get("relationship_impact", 0.0),
        "stress_impact": llm_result.get("stress_impact", 0.0),
        "event_id": event_id,
        "stakeholder_snapshot": stakeholder_snapshot,
        "triggering_action": triggering_action,
        "action_context": action_context,
    }

    scene_context = payload.get("scene_context", {})

    reaction_id = await _store_planned_action(
        stakeholder_id=stakeholder_id,
        scene_context=scene_context,
        kind="reaction",
        payload=reaction_payload,
        priority=priority,
        context_hash=event_id,
    )

    reaction_payload["planned_action_id"] = reaction_id

    return {
        "status": "generated",
        "action_id": reaction_id,
        "stakeholder_id": stakeholder_id,
        "payload": reaction_payload,
    }


async def _evaluate_role_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    stakeholder_id = payload["stakeholder_id"]
    stakeholder_snapshot = payload.get("stakeholder_snapshot", {})
    changing_conditions = payload.get("changing_conditions", {})
    context_hash = payload.get("context_hash") or _compute_context_hash(
        {"stakeholder_id": stakeholder_id, "conditions": changing_conditions}
    )
    priority = payload.get("priority", 4)

    prompt = f"""
Evaluate role adaptation as JSON:
Character: {stakeholder_snapshot.get('name')}
Current Role: {stakeholder_snapshot.get('role')}
Stress: {stakeholder_snapshot.get('stress_level')}

Changing Conditions: {json.dumps(changing_conditions, indent=2)}

Return JSON:
{{
  "change_role": true/false,
  "new_role": "bystander|instigator|defender|mediator|opportunist|victim|escalator|peacemaker",
  "reason": "..."
}}
"""

    llm_result = await llm_json(prompt)

    should_change = bool(llm_result.get("change_role", False))
    new_role = llm_result.get("new_role") or stakeholder_snapshot.get("role")
    reason = llm_result.get("reason", "Circumstances changed")

    if should_change:
        async with get_db_connection_context() as conn:
            await conn.execute(
                "UPDATE stakeholders SET role = $1 WHERE stakeholder_id = $2",
                new_role,
                stakeholder_id,
            )

    payload_record = {
        "stakeholder_id": stakeholder_id,
        "change_role": should_change,
        "new_role": new_role,
        "reason": reason,
        "context_hash": context_hash,
        "conditions": changing_conditions,
    }

    await _store_planned_action(
        stakeholder_id=stakeholder_id,
        scene_context=payload.get("scene_context", {}),
        kind="role_adaptation",
        payload=payload_record,
        priority=priority,
        context_hash=context_hash,
    )

    return payload_record


@app.task(base=NyxTask, name="nyx.tasks.background.stakeholder_tasks.create_stakeholder_profile",
    bind=True,
    max_retries=2,
    acks_late=True,)
@with_retry
@idempotent(key_fn=_idempotency_key_populate)
def create_stakeholder_profile(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not payload.get("stakeholder_id"):
        raise ValueError("stakeholder_id is required")
    if not payload.get("npc_id"):
        raise ValueError("npc_id is required")
    if not payload.get("conflict_id"):
        raise ValueError("conflict_id is required")

    logger.info(
        "Creating stakeholder profile for stakeholder=%s npc=%s",
        payload.get("stakeholder_id"),
        payload.get("npc_id"),
    )

    return run_coro(_create_stakeholder_profile_async(payload))


@app.task(base=NyxTask, name="nyx.tasks.background.stakeholder_tasks.generate_stakeholder_action",
    bind=True,
    max_retries=2,
    acks_late=True,)
@with_retry
@idempotent(key_fn=_idempotency_key_action)
def generate_stakeholder_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not payload.get("stakeholder_id"):
        raise ValueError("stakeholder_id is required")

    logger.info(
        "Generating autonomous action for stakeholder %s",
        payload.get("stakeholder_id"),
    )

    return run_coro(_generate_action_async(payload))


@app.task(base=NyxTask, name="nyx.tasks.background.stakeholder_tasks.generate_stakeholder_reaction",
    bind=True,
    max_retries=2,
    acks_late=True,)
@with_retry
@idempotent(key_fn=_idempotency_key_reaction)
def generate_stakeholder_reaction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not payload.get("stakeholder_id"):
        raise ValueError("stakeholder_id is required")
    if not payload.get("event_id"):
        raise ValueError("event_id is required")

    logger.info(
        "Generating reaction for stakeholder %s to event %s",
        payload.get("stakeholder_id"),
        payload.get("event_id"),
    )

    return run_coro(_generate_reaction_async(payload))


@app.task(base=NyxTask, name="nyx.tasks.background.stakeholder_tasks.evaluate_stakeholder_role",
    bind=True,
    max_retries=2,
    acks_late=True,)
@with_retry
@idempotent(key_fn=_idempotency_key_role)
def evaluate_stakeholder_role(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not payload.get("stakeholder_id"):
        raise ValueError("stakeholder_id is required")

    logger.info(
        "Evaluating role adaptation for stakeholder %s",
        payload.get("stakeholder_id"),
    )

    return run_coro(_evaluate_role_async(payload))


__all__ = [
    "llm_json",
    "create_stakeholder_profile",
    "generate_stakeholder_action",
    "generate_stakeholder_reaction",
    "evaluate_stakeholder_role",
]

