"""General mundane action evaluators."""
from typing import Any, Dict

from .trade import evaluate_buy


def evaluate_mundane(intent: Dict[str, Any], world_caps: Dict[str, Any], idx: Dict[str, Any], scene: Dict[str, Any], player: Dict[str, Any]) -> Dict[str, Any]:
    """Handle everyday actions with gentle guidance defaults."""

    categories = {c.lower() for c in intent.get("categories", [])}
    if "trade" in categories:
        return evaluate_buy(intent, world_caps, idx, scene, player)

    prohibitions = {p.lower() for p in world_caps.get("prohibitions", set())}
    if categories & prohibitions:
        return {
            "feasible": False,
            "strategy": "deny",
            "violations": [
                {
                    "rule": "established_impossibility",
                    "reason": "The merged archetypes prohibit this mundane action.",
                }
            ],
        }

    description = intent.get("raw_text") or intent.get("verb") or "action"
    return {
        "feasible": True,
        "strategy": "allow",
        "narrator_guidance": f"You carry out the mundane action: {description}.",
    }
