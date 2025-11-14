"""Helper utilities for conflict integration background tasks."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List

from agents import Agent
from nyx.config import WARMUP_MODEL
from logic.conflict_system.dynamic_conflict_template import extract_runner_response
from nyx.gateway import llm_gateway
from nyx.gateway.llm_gateway import LLMRequest

logger = logging.getLogger(__name__)


async def run_scene_tension_analysis(
    agent: Agent, context: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute the LLM-powered scene tension analysis."""

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

    try:
        response = await llm_gateway.execute(
            LLMRequest(
                prompt=prompt,
                agent=agent,
                model_override=WARMUP_MODEL,
            )
        )
    except Exception:
        logger.debug("Scene tension analysis runner failed; using fallback", exc_info=True)
        response = None

    try:
        result = json.loads(extract_runner_response(response))
    except json.JSONDecodeError:
        logger.debug("Failed to decode scene tension response; using fallback", exc_info=True)
        result = {
            "tensions": [],
            "should_generate_conflict": False,
        }

    if isinstance(result, dict):
        result.setdefault("tensions", [])
        result.setdefault("manifestation", [])
        result.setdefault("suggested_type", "slice_of_life")
        result.setdefault("context", {})
        result.setdefault("source", "llm")
        result["context"] = context
        result["cached_at"] = datetime.utcnow().isoformat()
    return result if isinstance(result, dict) else {}


async def run_contextual_conflict_generation(
    agent: Agent, tension_data: Dict[str, Any], npcs: List[int]
) -> Dict[str, Any]:
    """Generate a contextual conflict using the LLM helper agent."""

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

    try:
        response = await llm_gateway.execute(
            LLMRequest(
                prompt=prompt,
                agent=agent,
                model_override=WARMUP_MODEL,
            )
        )
    except Exception:
        logger.debug(
            "Contextual conflict generation runner failed; using fallback",
            exc_info=True,
        )
        response = None

    try:
        result = json.loads(extract_runner_response(response))
    except json.JSONDecodeError:
        logger.debug(
            "Failed to decode contextual conflict response; using fallback",
            exc_info=True,
        )
        result = {
            "name": "Emerging Tension",
            "description": "A subtle conflict begins",
            "intensity": "tension",
            "stakes": "Personal dynamics",
            "opening": "Tension fills the air",
        }

    if isinstance(result, dict):
        result.setdefault("intensity", "tension")
        result.setdefault("stakes", "Maintaining daily harmony")
        result.setdefault("opening", "Tension fills the air")
        result.setdefault("source", "llm")
        result["cached_at"] = datetime.utcnow().isoformat()
    return result if isinstance(result, dict) else {}


async def run_activity_integration(
    agent: Agent, activity: str, conflicts: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Integrate conflicts with an activity using the LLM narrator agent."""

    prompt = f"""
    Integrate these conflicts into the activity:

    Activity: {activity}
    Conflicts: {json.dumps(conflicts[:3], indent=2)}

    Return JSON:
    {{
      "manifestations": ["specific details"],
      "environmental_cues": ["atmosphere changes"],
      "npc_behaviors": {{"npc_id": "behavior"}},
      "choices": [{{"text": "choice text","subtext": "hidden meaning"}}]
    }}
    """

    try:
        response = await llm_gateway.execute(
            LLMRequest(
                prompt=prompt,
                agent=agent,
                model_override=WARMUP_MODEL,
            )
        )
    except Exception:
        logger.debug(
            "Activity integration runner failed; using fallback",
            exc_info=True,
        )
        response = None

    try:
        result = json.loads(extract_runner_response(response))
    except Exception:
        logger.debug("Failed to decode activity integration response; using fallback", exc_info=True)
        result = {"manifestations": ["Tension colors the interaction"]}

    if isinstance(result, dict):
        result.setdefault("manifestations", ["Tension colors the interaction"])
        result.setdefault("environmental_cues", [])
        result.setdefault("npc_behaviors", {})
        result.setdefault("choices", [])
        result.setdefault("source", "llm")
        result["conflicts_active"] = True
        result["cached_at"] = datetime.utcnow().isoformat()
    return result if isinstance(result, dict) else {}
