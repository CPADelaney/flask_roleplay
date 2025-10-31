"""Celery tasks for background grand conflict generation and evolution."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncpg
from celery import shared_task

from agents import Agent, Runner
from db.connection import get_db_connection_context
from infra.cache import cache_key, redis_client
from logic.conflict_system.background_grand_conflicts import (
    BackgroundIntensity,
    GrandConflictType,
)
from logic.conflict_system.dynamic_conflict_template import extract_runner_response
from logic.conflict_system.background_processor import resolve_intensity_value
from nyx.tasks.utils import run_coro, with_retry
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)


def _redis_list_push(key: str, payload: Dict[str, Any], ttl: int = 3600) -> None:
    """Push a JSON payload onto a Redis list and set expiry."""
    try:
        redis_client.rpush(key, json.dumps(payload))
        if ttl:
            redis_client.expire(key, ttl)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to push background conflict payload to Redis: %s", exc)


async def _get_current_game_day(user_id: int, conversation_id: int) -> int:
    """Fetch the current game day from CurrentRoleplay."""
    try:
        async with get_db_connection_context() as conn:
            raw_value = await conn.fetchval(
                """
                SELECT value FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentDay'
                """,
                user_id,
                conversation_id,
            )
    except (asyncio.TimeoutError, asyncpg.PostgresError) as exc:
        logger.warning(
            "Could not retrieve current game day from database, defaulting to 1. Error: %s",
            exc,
        )
        return 1

    try:
        return int(raw_value) if raw_value is not None else 1
    except (TypeError, ValueError):
        return 1


# Lazy Agent singletons -----------------------------------------------------

_conflict_generator_agent: Optional[Agent] = None
_conflict_evolution_agent: Optional[Agent] = None
_news_agent: Optional[Agent] = None
_ripple_agent: Optional[Agent] = None
_opportunity_agent: Optional[Agent] = None


def _get_conflict_generator() -> Agent:
    global _conflict_generator_agent
    if _conflict_generator_agent is None:
        _conflict_generator_agent = Agent(
            name="Background Conflict Generator",
            instructions="""
            Generate grand-scale conflicts that happen in the background.

            Create conflicts that:
            - Feel massive in scope
            - Have multiple factions
            - Evolve over time
            - Impact daily life indirectly
            - Create atmospheric tension
            - Could involve the player optionally

            Make them feel real and consequential without dominating gameplay.
            """,
            model="gpt-5-nano",
        )
    return _conflict_generator_agent


def _get_evolution_agent() -> Agent:
    global _conflict_evolution_agent
    if _conflict_evolution_agent is None:
        _conflict_evolution_agent = Agent(
            name="Conflict Evolution Agent",
            instructions="""
            Advance background conflicts based on their current state.

            Create developments that:
            - Feel like natural progressions
            - Reflect faction dynamics
            - Create ripple effects
            - Respect current intensity and progress
            - Introduce optional opportunities
            """,
            model="gpt-5-nano",
        )
    return _conflict_evolution_agent


def _get_news_agent() -> Agent:
    global _news_agent
    if _news_agent is None:
        _news_agent = Agent(
            name="News Article Generator",
            instructions="""
            Generate news articles about background conflicts.

            Vary between:
            - Official announcements (formal, careful)
            - Independent reporting (balanced, investigative)
            - Tabloid coverage (sensational, dramatic)
            - Underground news (subversive, revealing)

            Match tone to source. Include bias and spin.
            """,
            model="gpt-5-nano",
        )
    return _news_agent


def _get_ripple_agent() -> Agent:
    global _ripple_agent
    if _ripple_agent is None:
        _ripple_agent = Agent(
            name="Ripple Effect Generator",
            instructions="""
            Generate subtle effects of grand conflicts on daily life.

            Create ripples that:
            - Affect atmosphere and mood
            - Change NPC behaviors subtly
            - Alter available resources
            - Create background tension
            - Suggest larger forces

            Keep effects indirect but noticeable to observant players.
            """,
            model="gpt-5-nano",
        )
    return _ripple_agent


def _get_opportunity_agent() -> Agent:
    global _opportunity_agent
    if _opportunity_agent is None:
        _opportunity_agent = Agent(
            name="Opportunity Creator",
            instructions="""
            Create optional opportunities from background conflicts.

            Generate opportunities that:
            - Are completely optional
            - Offer interesting choices
            - Connect to larger events
            - Have multiple approaches
            - Create memorable moments

            Players should feel these are bonuses, not obligations.
            """,
            model="gpt-5-nano",
        )
    return _opportunity_agent


# Shared helpers -------------------------------------------------------------

def _coerce_conflict_type(raw: Optional[str]) -> GrandConflictType:
    if raw:
        try:
            return GrandConflictType(raw)
        except ValueError:
            try:
                return GrandConflictType[raw.upper()]
            except KeyError:
                pass
    return random.choice(list(GrandConflictType))


def _calculate_new_intensity(current: str, change: str) -> str:
    try:
        current_enum = BackgroundIntensity[current.upper()]
    except KeyError:
        current_enum = BackgroundIntensity.DISTANT_RUMOR

    members = list(BackgroundIntensity)
    idx = members.index(current_enum)

    if change == "increase" and idx < len(members) - 1:
        return members[idx + 1].value
    if change == "decrease" and idx > 0:
        return members[idx - 1].value
    return current_enum.value


def _serialize_conflict_payload(conflict_id: int, conflict_type: GrandConflictType, data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "conflict_id": conflict_id,
        "conflict_type": conflict_type.value,
        "name": data.get("name"),
        "description": data.get("description"),
        "intensity": data.get("intensity_label", "distant_rumor"),
        "progress": data.get("progress", 0.0),
        "factions": data.get("factions", []),
        "current_state": data.get("current_state"),
        "recent_developments": data.get("recent_developments", []),
        "impact_on_daily_life": data.get("impact_on_daily_life", []),
        "player_awareness_level": data.get("player_awareness_level", 0.1),
        "news_count": data.get("news_count", 0),
        "last_news_generation": data.get("last_news_generation"),
    }


# Async implementations ------------------------------------------------------

async def _generate_background_conflict_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    user_id = int(payload["user_id"])
    conversation_id = int(payload["conversation_id"])
    conflict_type = _coerce_conflict_type(payload.get("conflict_type"))

    async with get_db_connection_context() as conn:
        count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM backgroundconflicts
            WHERE user_id = $1 AND conversation_id = $2 AND status = 'active'
            """,
            user_id,
            conversation_id,
        )
        if count and count >= 5:
            return {"status": "skipped", "reason": "max_active", "count": count}

    prompt = f"""
    Generate a {conflict_type.value} background conflict.

    Consider:
    - World setting and established lore
    - Current game state
    - Player's position in the world
    - Other active conflicts

    Create something that feels:
    - Important but distant
    - Complex with multiple sides
    - Slowly evolving
    - Atmospheric

    Return JSON:
    {{
        "name": "Conflict name",
        "description": "2-3 sentence overview",
        "factions": ["Faction 1", "Faction 2", "Faction 3"],
        "initial_state": "Current situation",
        "initial_development": "What just happened to start/escalate this",
        "daily_life_impacts": [
            "How it affects common people",
            "What changes in daily routines",
            "Ambient signs of the conflict"
        ],
        "potential_hooks": ["Optional player involvement 1", "Optional involvement 2"],
        "initial_intensity": "distant_rumor|occasional_news|regular_topic",
        "estimated_duration": 10-50 (game days)
    }}
    """

    agent = _get_conflict_generator()
    response = await Runner.run(agent, prompt)
    data = json.loads(extract_runner_response(response))

    intensity_label = data.get("initial_intensity", "distant_rumor")
    intensity_float = resolve_intensity_value(intensity_label)
    current_day = await _get_current_game_day(user_id, conversation_id)

    async with get_db_connection_context() as conn:
        conflict_id = await conn.fetchval(
            """
            INSERT INTO backgroundconflicts
            (user_id, conversation_id, conflict_type, name, description,
             factions, current_state, intensity, progress, status,
             metadata, news_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING id
            """,
            user_id,
            conversation_id,
            conflict_type.value,
            data["name"],
            data["description"],
            json.dumps(data["factions"]),
            data["initial_state"],
            intensity_float,
            0.0,
            "active",
            json.dumps(
                {
                    "potential_hooks": data.get("potential_hooks", []),
                    "estimated_duration": data.get("estimated_duration", 30),
                    "created_at": datetime.utcnow().isoformat(),
                    "daily_life_impacts": data.get("daily_life_impacts", []),
                }
            ),
            0,
        )

        await conn.execute(
            """
            INSERT INTO backgrounddevelopments
            (conflict_id, development, game_day)
            VALUES ($1, $2, $3)
            """,
            conflict_id,
            data["initial_development"],
            current_day,
        )

    conflict_payload = _serialize_conflict_payload(
        conflict_id,
        conflict_type,
        {
            "name": data["name"],
            "description": data["description"],
            "intensity_label": intensity_label,
            "progress": 0.0,
            "factions": data.get("factions", []),
            "current_state": data.get("initial_state"),
            "recent_developments": [data.get("initial_development")],
            "impact_on_daily_life": data.get("daily_life_impacts", []),
            "player_awareness_level": 0.1,
            "news_count": 0,
            "last_news_generation": None,
        },
    )

    return {"status": "created", "conflict": conflict_payload}


async def _advance_conflict_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    user_id = int(payload["user_id"])
    conversation_id = int(payload["conversation_id"])
    conflict = payload.get("conflict", {})

    conflict_type = _coerce_conflict_type(conflict.get("conflict_type"))
    current_state = conflict.get("current_state", "")
    prompt = f"""
    Advance this background conflict:

    Conflict: {conflict.get("name")}
    Type: {conflict_type.value}
    Current State: {current_state}
    Progress: {conflict.get("progress", 0.0)}%
    Intensity: {conflict.get("intensity", "distant_rumor")}
    Factions: {json.dumps(conflict.get("factions", []))}
    Recent: {conflict.get("recent_developments", ['Beginning'])[-1]}

    Generate the next development:
    - Natural progression from current state
    - Reflects faction dynamics
    - Appropriate to intensity level
    - Moves story forward

    Return JSON:
    {{
        "event_type": "battle|negotiation|escalation|development|revelation",
        "description": "What happened (2-3 sentences)",
        "new_state": "Updated conflict state",
        "progress_change": -10 to +10,
        "intensity_change": "increase|maintain|decrease",
        "faction_impacts": {{"Faction": impact_score}},
        "creates_opportunity": true/false,
        "opportunity_description": "Optional: what player could do",
        "ripple_effects": ["Effect 1", "Effect 2"],
        "news_worthy": true/false
    }}
    """

    agent = _get_evolution_agent()
    response = await Runner.run(agent, prompt)
    data = json.loads(extract_runner_response(response))

    current_intensity = conflict.get("intensity", "distant_rumor")
    new_intensity_label = _calculate_new_intensity(current_intensity, data.get("intensity_change", "maintain"))
    new_intensity_value = resolve_intensity_value(new_intensity_label)
    progress_delta = float(data.get("progress_change", 0.0))
    new_progress = max(0.0, min(100.0, float(conflict.get("progress", 0.0)) + progress_delta))

    current_day = await _get_current_game_day(user_id, conversation_id)

    async with get_db_connection_context() as conn:
        await conn.execute(
            """
            UPDATE backgroundconflicts
            SET current_state = $1,
                progress = $2,
                intensity = $3,
                updated_at = CURRENT_TIMESTAMP,
                last_significant_change = CASE WHEN $4 THEN $5 ELSE last_significant_change END
            WHERE id = $6
            """,
            data.get("new_state", current_state),
            new_progress,
            new_intensity_value,
            data.get("news_worthy", False),
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "magnitude": abs(progress_delta) / 10.0,
                    "description": data.get("description", ""),
                }
            ),
            int(conflict.get("conflict_id")),
        )

        await conn.execute(
            """
            INSERT INTO backgrounddevelopments (conflict_id, development, game_day)
            VALUES ($1, $2, $3)
            """,
            int(conflict.get("conflict_id")),
            data["description"],
            current_day,
        )

        if data.get("creates_opportunity"):
            await conn.execute(
                """
                INSERT INTO conflictopportunities
                (conflict_id, description, expires_on, status, user_id, conversation_id)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                int(conflict.get("conflict_id")),
                data.get("opportunity_description", "An opportunity arises"),
                current_day + 7,
                "available",
                user_id,
                conversation_id,
            )

    event_payload = {
        "conflict_id": int(conflict.get("conflict_id")),
        "event_type": data.get("event_type", "development"),
        "description": data.get("description"),
        "faction_impacts": data.get("faction_impacts", {}),
        "creates_opportunity": data.get("creates_opportunity", False),
        "opportunity_window": 7 if data.get("creates_opportunity") else None,
        "news_worthy": data.get("news_worthy", False),
        "new_state": data.get("new_state"),
        "new_intensity": new_intensity_label,
        "new_progress": new_progress,
    }

    return {"status": "advanced", "event": event_payload}


async def _generate_news_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    user_id = int(payload["user_id"])
    conversation_id = int(payload["conversation_id"])
    conflict = payload.get("conflict", {})

    news_type = payload.get("news_type") or random.choice(["official", "independent", "tabloid", "rumor"])

    prompt = f"""
    Generate {news_type} news about this conflict:

    Conflict: {conflict.get("name")}
    Current State: {conflict.get("current_state")}
    Recent Development: {conflict.get("recent_developments", ['Initial stages'])[-1]}
    Factions: {json.dumps(conflict.get("factions", []))}

    Create news that:
    - Matches {news_type} style perfectly
    - Feels authentic to source
    - Includes appropriate bias
    - Creates atmosphere
    - Could influence opinions

    Return JSON:
    {{
        "headline": "Attention-grabbing headline",
        "source": "News source name",
        "content": "2-3 paragraph article/rumor",
        "reliability": 0.0 to 1.0,
        "bias": "faction or perspective bias",
        "spin": "how truth is distorted",
        "public_reaction": "How people might react",
        "conversation_starter": "How NPCs might discuss this",
        "hidden_truth": "What's really happening"
    }}
    """

    agent = _get_news_agent()
    response = await Runner.run(agent, prompt)
    news_data = json.loads(extract_runner_response(response))

    current_day = await _get_current_game_day(user_id, conversation_id)

    async with get_db_connection_context() as conn:
        await conn.execute(
            """
            INSERT INTO backgroundnews
            (user_id, conversation_id, conflict_id, headline,
             source, content, reliability, game_day)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            user_id,
            conversation_id,
            int(conflict.get("conflict_id")),
            news_data["headline"],
            news_data["source"],
            news_data["content"],
            news_data.get("reliability", 0.5),
            current_day,
        )

        await conn.execute(
            """
            UPDATE backgroundconflicts
            SET news_count = news_count + 1,
                last_news_generation = $1
            WHERE id = $2
            """,
            current_day,
            int(conflict.get("conflict_id")),
        )

    return {"status": "generated", "news": news_data, "conflict_id": int(conflict.get("conflict_id"))}


async def _generate_ripples_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    user_id = int(payload["user_id"])
    conversation_id = int(payload["conversation_id"])
    conflict = payload.get("conflict", {})

    prompt = f"""
    Generate subtle ripple effects from this conflict:

    Conflict: {conflict.get("name")}
    Intensity: {conflict.get("intensity")}
    Current State: {conflict.get("current_state")}

    Create effects that:
    - Match the intensity level
    - Feel atmospheric not intrusive
    - Could be noticed or ignored
    - Add texture to the world

    Return JSON:
    {{
        "ambient_mood": ["Mood descriptor 1", "Mood descriptor 2"],
        "npc_behaviors": ["Behavioral change 1", "Behavioral change 2"],
        "resource_changes": {{"Resource": "change description"}},
        "overheard_snippets": ["Snippet 1", "Snippet 2"],
        "environmental_details": ["Detail 1", "Detail 2"]
    }}
    """

    agent = _get_ripple_agent()
    response = await Runner.run(agent, prompt)
    ripple_data = json.loads(extract_runner_response(response))

    current_day = await _get_current_game_day(user_id, conversation_id)

    async with get_db_connection_context() as conn:
        await conn.execute(
            """
            INSERT INTO conflictripples
            (conflict_id, ripple_data, game_day, user_id, conversation_id)
            VALUES ($1, $2, $3, $4, $5)
            """,
            int(conflict.get("conflict_id")),
            json.dumps(ripple_data),
            current_day,
            user_id,
            conversation_id,
        )

    return {"status": "generated", "ripples": ripple_data}


def _player_could_engage(player_skills: Dict[str, Any], conflict: Dict[str, Any]) -> bool:
    awareness = float(conflict.get("player_awareness_level", 0.0) or 0.0)
    return bool(player_skills) and awareness > 0.3


async def _check_opportunities_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    player_skills = payload.get("player_skills", {})
    conflicts: List[Dict[str, Any]] = payload.get("conflicts", [])

    opportunities: List[Dict[str, Any]] = []

    for conflict in conflicts:
        intensity = str(conflict.get("intensity", ""))
        if intensity not in {
            BackgroundIntensity.AMBIENT_TENSION.value,
            BackgroundIntensity.VISIBLE_EFFECTS.value,
        }:
            continue

        if not _player_could_engage(player_skills, conflict):
            continue

        prompt = f"""
        Create an optional opportunity from this conflict:

        Conflict: {conflict.get("name")}
        State: {conflict.get("current_state")}
        Player Skills: {json.dumps(player_skills)}

        Generate opportunity that:
        - Is completely optional
        - Has multiple approaches
        - Offers meaningful choice
        - Could be ignored without penalty

        Return JSON:
        {{
            "title": "Opportunity title",
            "description": "What's available",
            "approaches": ["Approach 1", "Approach 2"],
            "potential_rewards": ["Reward 1", "Reward 2"],
            "potential_risks": ["Risk 1", "Risk 2"],
            "time_sensitive": true/false,
            "skill_requirements": {{"Skill": level}}
        }}
        """

        agent = _get_opportunity_agent()
        response = await Runner.run(agent, prompt)
        opp_data = json.loads(extract_runner_response(response))
        opp_data["conflict_id"] = int(conflict.get("conflict_id"))
        opportunities.append(opp_data)
        break

    return {"status": "generated", "opportunities": opportunities}


# Celery task definitions ----------------------------------------------------


def _idempotency_key_generate(payload: Dict[str, Any]) -> str:
    return f"grand_conflict:generate:{payload.get('user_id')}:{payload.get('conversation_id')}"


def _idempotency_key_advance(payload: Dict[str, Any]) -> str:
    return f"grand_conflict:advance:{payload.get('conflict', {}).get('conflict_id')}"


def _idempotency_key_news(payload: Dict[str, Any]) -> str:
    return f"grand_conflict:news:{payload.get('conflict', {}).get('conflict_id')}:{payload.get('news_type', 'auto')}"


def _idempotency_key_ripples(payload: Dict[str, Any]) -> str:
    return f"grand_conflict:ripples:{payload.get('user_id')}:{payload.get('conversation_id')}"


def _idempotency_key_opportunities(payload: Dict[str, Any]) -> str:
    return f"grand_conflict:opportunities:{payload.get('user_id')}:{payload.get('conversation_id')}"


@shared_task(
    name="nyx.tasks.background.conflict_grand_conflicts.generate_grand_conflict",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_generate)
def generate_grand_conflict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a new background conflict and persist it."""
    result = run_coro(_generate_background_conflict_async(payload))

    if result.get("status") == "created":
        key = cache_key(
            "background_conflict",
            payload.get("user_id"),
            payload.get("conversation_id"),
            "generated",
        )
        _redis_list_push(key, result["conflict"])

    return result


@shared_task(
    name="nyx.tasks.background.conflict_grand_conflicts.advance_grand_conflict",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_advance)
def advance_grand_conflict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Advance an existing background conflict."""
    result = run_coro(_advance_conflict_async(payload))

    if result.get("status") == "advanced":
        conflict_id = result["event"].get("conflict_id")
        key = cache_key("background_conflict", conflict_id, "events")
        _redis_list_push(key, result["event"])

    return result


@shared_task(
    name="nyx.tasks.background.conflict_grand_conflicts.generate_conflict_news",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_news)
def generate_conflict_news(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a news article about a background conflict."""
    result = run_coro(_generate_news_async(payload))

    if result.get("status") == "generated":
        conflict_id = result.get("conflict_id")
        key = cache_key("background_conflict", conflict_id, "news")
        _redis_list_push(key, result["news"])

    return result


@shared_task(
    name="nyx.tasks.background.conflict_grand_conflicts.generate_conflict_ripples",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_ripples)
def generate_conflict_ripples(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate ripple effects for background conflicts."""
    result = run_coro(_generate_ripples_async(payload))

    if result.get("status") == "generated":
        key = cache_key(
            "background_conflict",
            payload.get("user_id"),
            payload.get("conversation_id"),
            "ripples",
        )
        _redis_list_push(key, result["ripples"])

    return result


@shared_task(
    name="nyx.tasks.background.conflict_grand_conflicts.check_conflict_opportunities",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_opportunities)
def check_conflict_opportunities(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate optional player opportunities arising from conflicts."""
    result = run_coro(_check_opportunities_async(payload))

    if result.get("status") == "generated" and result.get("opportunities"):
        key = cache_key(
            "background_conflict",
            payload.get("user_id"),
            payload.get("conversation_id"),
            "opportunities",
        )
        _redis_list_push(key, result["opportunities"])

    return result


__all__ = [
    "generate_grand_conflict",
    "advance_grand_conflict",
    "generate_conflict_news",
    "generate_conflict_ripples",
    "check_conflict_opportunities",
]
