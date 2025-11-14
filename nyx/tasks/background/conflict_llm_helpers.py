"""LLM-backed helpers for slice-of-life conflict processing."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from agents import Agent
from nyx.config import WARMUP_MODEL
from logic.conflict_system.dynamic_conflict_template import extract_runner_response
from logic.conflict_system.slice_of_life_conflicts import (
    ConflictIntensity,
    DailyConflictEvent,
    ResolutionApproach,
    SliceOfLifeConflictType,
)
from nyx.gateway import llm_gateway
from nyx.gateway.llm_gateway import LLMRequest

logger = logging.getLogger(__name__)


_PATTERN_ANALYZER = Agent(
    name="Pattern Analyzer",
    instructions="""
    Analyze memory patterns and relationship dynamics for emerging slice-of-life conflicts.
    Return structured JSON describing up to three emerging tensions.
    """,
    model="gpt-5-nano",
)

_CONFLICT_AGENT = Agent(
    name="Slice of Life Conflict Director",
    instructions="""
    Generate subtle conflicts embedded in daily routines and surface concrete manifestations
    as compact JSON payloads.
    """,
    model="gpt-5-nano",
)

_RESOLUTION_AGENT = Agent(
    name="Conflict Resolution Analyst",
    instructions="""
    Analyze conflict patterns to determine whether a slice-of-life conflict should resolve on its own
    and describe the resolution pattern if so.
    """,
    model="gpt-5-nano",
)

_TIME_AGENT = Agent(
    name="Daily Conflict Integrator",
    instructions="""
    Determine if a conflict beat is appropriate for a given time of day.
    Respond with a single word: yes or no.
    """,
    model="gpt-5-nano",
)


def _summarize_memories(memories: Sequence[Mapping[str, Any]]) -> str:
    summary = []
    for memory in list(memories)[:10]:
        text = str(memory.get("memory_text", "A fleeting moment."))
        valence = memory.get("emotional_valence", "neutral")
        summary.append(f"- {text} (emotion: {valence})")
    return "\n".join(summary) if summary else "No recent significant memories"


def _summarize_relationships(relationships: Sequence[Mapping[str, Any]]) -> str:
    summary = []
    for rel in list(relationships)[:5]:
        dimension = rel.get("dimension", "dynamic")
        current_value = float(rel.get("current_value") or 0)
        delta = float(rel.get("recent_delta") or 0)
        summary.append(
            f"- {dimension}: {current_value:+.2f} (recent change: {delta:+.2f})"
        )
    return "\n".join(summary) if summary else "No significant relationship dynamics"


async def analyze_patterns_async(
    memories: Sequence[Mapping[str, Any]],
    relationships: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """LLM-backed analysis of emerging tensions."""

    prompt = f"""
Analyze recent patterns for emerging slice-of-life conflicts:

Recent Memories:
{_summarize_memories(memories)}

Relationship Dynamics:
{_summarize_relationships(relationships)}

Return a JSON array (1-3 items). Each item:
{{
  "conflict_type": "permission_patterns|boundary_erosion|subtle_rivalry|...",
  "intensity": "subtext|tension|passive|direct|confrontation",
  "description": "specific, contextual",
  "evidence": ["..."],
  "tension_level": 0.0
}}
"""

    response = await llm_gateway.execute(
        LLMRequest(
            prompt=prompt,
            agent=_PATTERN_ANALYZER,
            model_override=WARMUP_MODEL,
        )
    )
    parsed = extract_runner_response(response)
    try:
        tensions = json.loads(parsed)
    except Exception:
        logger.warning("Failed to parse slice-of-life tension payload: %s", parsed)
        return []

    results: List[Dict[str, Any]] = []
    for item in tensions if isinstance(tensions, list) else []:
        if not isinstance(item, Mapping):
            continue
        try:
            conflict_type = SliceOfLifeConflictType[
                str(item.get("conflict_type", "subtle_rivalry")).upper()
            ]
        except Exception:
            conflict_type = SliceOfLifeConflictType.SUBTLE_RIVALRY
        try:
            intensity = ConflictIntensity[
                str(item.get("intensity", "tension")).upper()
            ]
        except Exception:
            intensity = ConflictIntensity.TENSION
        evidence = item.get("evidence")
        results.append(
            {
                "type": conflict_type.value,
                "intensity": intensity.value,
                "description": item.get(
                    "description", "A subtle tension emerges during daily routines."
                ),
                "evidence": list(evidence) if isinstance(evidence, Iterable) else [],
                "tension_level": float(item.get("tension_level", 0.5)),
            }
        )

    return results


async def generate_manifestation_async(
    conflict: Mapping[str, Any],
    activity_type: str,
    npc_descriptors: Sequence[str],
    participating_npcs: Sequence[int],
) -> DailyConflictEvent:
    """LLM-backed manifestation generation for a conflict activity."""

    prompt = f"""
Generate how this conflict manifests during {activity_type}:

Conflict Type: {conflict.get('conflict_type')}
Intensity: {conflict.get('intensity', 'tension')}
Phase: {conflict.get('phase', 'active')}
NPCs Present: {', '.join(npc_descriptors)}

Return JSON:
{{
  "manifestation": "1-2 sentences",
  "choice_presented": false,
  "impact": 0.1,
  "npc_reactions": ["npc 1 line", "npc 2 line"]
}}
"""

    response = await llm_gateway.execute(
        LLMRequest(
            prompt=prompt,
            agent=_CONFLICT_AGENT,
            model_override=WARMUP_MODEL,
        )
    )
    parsed = extract_runner_response(response)
    result = json.loads(parsed)

    npc_reactions: Dict[int, str] = {}
    reactions = result.get("npc_reactions")
    if isinstance(reactions, Sequence):
        for index, npc_id in enumerate(participating_npcs[: len(reactions)]):
            npc_reactions[int(npc_id)] = str(reactions[index])

    return DailyConflictEvent(
        activity_type=activity_type,
        conflict_manifestation=str(
            result.get(
                "manifestation", "A subtle tension colors the everyday interaction."
            )
        ),
        choice_presented=bool(result.get("choice_presented", False)),
        accumulation_impact=float(result.get("impact", 0.1)),
        npc_reactions=npc_reactions,
    )


async def evaluate_resolution_async(
    conflict: Mapping[str, Any],
    memories: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """LLM-backed resolution evaluation."""

    prompt = f"""
Analyze if this conflict should naturally resolve:

Conflict: {conflict.get('conflict_type')} (Progress: {conflict.get('progress', 0)}%)
Phase: {conflict.get('phase', 'active')}
Recent Pattern:
{_summarize_memories(memories[:10])}

Return JSON:
{{
  "should_resolve": false,
  "type": "time_erosion|subtle_resistance|...",
  "description": "how it resolves",
  "new_patterns": []
}}
"""

    response = await llm_gateway.execute(
        LLMRequest(
            prompt=prompt,
            agent=_RESOLUTION_AGENT,
            model_override=WARMUP_MODEL,
        )
    )
    parsed = extract_runner_response(response)
    data = json.loads(parsed)

    if not bool(data.get("should_resolve", False)):
        return None

    return {
        "resolution_type": str(data.get("type", ResolutionApproach.TIME_EROSION.value)),
        "description": str(
            data.get("description", "The conflict fades quietly into routine.")
        ),
        "new_patterns": list(data.get("new_patterns", [])),
        "final_state": "resolved",
    }


async def evaluate_time_appropriateness_async(
    conflict: Mapping[str, Any], time_of_day: str
) -> bool:
    """LLM-backed time-of-day suitability check."""

    prompt = f"""
Is this conflict appropriate for {time_of_day}?

Conflict Type: {conflict.get('conflict_type')}
Intensity: {conflict.get('intensity', 'tension')}

Answer just yes or no.
"""

    response = await llm_gateway.execute(
        LLMRequest(
            prompt=prompt,
            agent=_TIME_AGENT,
            model_override=WARMUP_MODEL,
        )
    )
    text = extract_runner_response(response).strip().lower()
    return "yes" in text and "no" not in text[:5]


__all__ = [
    "analyze_patterns_async",
    "generate_manifestation_async",
    "evaluate_resolution_async",
    "evaluate_time_appropriateness_async",
]
