"""Simple reward evaluator mapping events to scalar values."""
from typing import Dict, List
import os
import json

from nyx.config import get_reward_config

_DEFAULT_EVENT_REWARDS: Dict[str, float] = {
    "unit_test_passed": 0.1,
    "unit_test_failed": -0.1,
    "positive_reinforcement": 0.2,
    "negative_reinforcement": -0.2,
}

def _load_event_rewards() -> Dict[str, float]:
    """Load event rewards from configuration or environment."""
    config = get_reward_config().get("event_values")
    if isinstance(config, dict):
        return {k: float(v) for k, v in config.items()}
    env_cfg = os.getenv("EVENT_REWARDS_JSON")
    if env_cfg:
        try:
            return {k: float(v) for k, v in json.loads(env_cfg).items()}
        except Exception:
            pass
    return _DEFAULT_EVENT_REWARDS

EVENT_REWARDS: Dict[str, float] = _load_event_rewards()

def evaluate(event_type: str) -> float:
    """Return a scalar reward for an event type."""
    return EVENT_REWARDS.get(event_type, 0.0)


def adjust_association_strengths(conditioning_system, batch: List[str]) -> None:
    """Adjust associations in the conditioning system based on a batch of events."""
    from nyx.core.conditioning_models import ConditionedAssociation  # lazy import
    import datetime

    for event in batch:
        reward = evaluate(event)
        key = f"event::{event}"
        for mapping in (
            conditioning_system.classical_associations,
            conditioning_system.operant_associations,
        ):
            if key in mapping:
                assoc = mapping[key]
                old_strength = assoc.association_strength
                assoc.association_strength = max(0.0, min(1.0, old_strength + reward))
                assoc.reinforcement_count += 1
                assoc.last_reinforced = datetime.datetime.now().isoformat()
            else:
                mapping[key] = ConditionedAssociation(
                    stimulus=event,
                    response="internal_reward",
                    association_strength=max(0.0, reward),
                    formation_date=datetime.datetime.now().isoformat(),
                    last_reinforced=datetime.datetime.now().isoformat(),
                    reinforcement_count=1,
                    valence=reward,
                    context_keys=[],
                )
                conditioning_system._weak_epochs[key] = 0

    # extinction logic across both association maps
    for mapping in (
        conditioning_system.classical_associations,
        conditioning_system.operant_associations,
    ):
        to_remove = []
        for key, assoc in mapping.items():
            if assoc.association_strength < 0.05:
                conditioning_system._weak_epochs[key] = conditioning_system._weak_epochs.get(key, 0) + 1
                if conditioning_system._weak_epochs[key] >= 3:
                    to_remove.append(key)
            else:
                conditioning_system._weak_epochs[key] = 0
        for key in to_remove:
            mapping.pop(key, None)
            conditioning_system._weak_epochs.pop(key, None)
