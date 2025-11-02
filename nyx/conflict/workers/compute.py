"""LLM-heavy helpers for conflict subsystem routing."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from logic.conflict_system.conflict_synthesizer import LLM_ROUTE_TIMEOUT
from nyx.gateway.llm_gateway import LLMOperation, LLMRequest, execute


def compute_scene_router_prompt(scene_context: Dict[str, Any]) -> str:
    """Build the orchestrator prompt for subsystem routing."""

    return (
        "Analyze this scene context and determine which conflict subsystems should be active:\n"
        f"{json.dumps(scene_context, indent=2, sort_keys=True)}\n\n"
        "Available subsystems must be returned as a JSON list of subsystem names."
    )


async def llm_run_conflict_orchestrator(synthesizer: Any, prompt: str) -> Any:
    """Execute the routing orchestrator via the Nyx LLM gateway."""

    request = LLMRequest(
        prompt=prompt,
        agent=getattr(synthesizer, "_orchestrator", None),
        metadata={
            "operation": LLMOperation.ORCHESTRATION.value,
            "stage": "conflict_route",
        },
    )
    result = await asyncio.wait_for(
        execute(request),
        timeout=LLM_ROUTE_TIMEOUT,
    )
    return result.raw


async def synthesize_scene_route_decisions(synthesizer: Any, scene_context: Dict[str, Any]) -> Any:
    """Convenience helper to build the prompt and execute the orchestrator."""

    prompt = compute_scene_router_prompt(scene_context)
    return await llm_run_conflict_orchestrator(synthesizer, prompt)


__all__ = [
    "compute_scene_router_prompt",
    "llm_run_conflict_orchestrator",
    "synthesize_scene_route_decisions",
]
