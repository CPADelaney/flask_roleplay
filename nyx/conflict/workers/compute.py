"""Background compute helpers for conflict synthesis."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

from logic.conflict_system.dynamic_conflict_template import extract_runner_response
from nyx.gateway.llm_gateway import LLMOperation, LLMRequest, execute

logger = logging.getLogger(__name__)


def _build_scene_router_prompt(scene_context: Dict[str, Any]) -> str:
    return (
        "Analyze this scene context and determine which conflict subsystems should be active.\n"
        f"{json.dumps(scene_context, indent=2, sort_keys=True)}\n\n"
        "Available subsystems must be returned as a JSON list of subsystem names."
    )


async def llm_route_scene_subsystems(
    synthesizer: Any,
    scene_context: Dict[str, Any],
    *,
    timeout: float,
) -> List[str]:
    """Run the conflict orchestrator agent and decode subsystem names."""

    orchestrator = getattr(synthesizer, "_orchestrator", None)
    if orchestrator is None:
        raise RuntimeError("Conflict synthesizer does not expose an orchestrator agent")

    prompt_builder = getattr(synthesizer, "synthesize_scene_router_prompt", None)
    if callable(prompt_builder):
        prompt = prompt_builder(scene_context)
    else:
        prompt = _build_scene_router_prompt(scene_context)

    request = LLMRequest(
        prompt=prompt,
        agent=orchestrator,
        metadata={
            "operation": LLMOperation.ORCHESTRATION.value,
            "stage": "conflict_route",
        },
    )

    result = await asyncio.wait_for(execute(request), timeout=timeout)
    payload = extract_runner_response(result)

    try:
        parsed = json.loads(payload)
    except Exception as exc:  # pragma: no cover - defensive guard for malformed responses
        logger.exception("Failed to decode conflict routing response")
        raise ValueError("Invalid routing response payload") from exc

    if not isinstance(parsed, list):
        raise ValueError("Routing response must be a list")

    return [name for name in parsed if isinstance(name, str)]


__all__ = ["llm_route_scene_subsystems"]
