"""Tests for Nyx configuration helpers related to rewards."""

from __future__ import annotations

import copy
import importlib

import nyx.config as config
import nyx.core.reward.evaluator as evaluator


def test_get_reward_config_exposed_from_package() -> None:
    """The configuration package should expose reward helpers."""

    reward_config = config.get_reward_config()
    assert isinstance(reward_config, dict)
    assert "event_values" in reward_config


def test_reward_evaluator_reads_config_values() -> None:
    """Reward evaluator should read settings from the configuration package."""

    original_config = copy.deepcopy(config.CONFIG)
    try:
        config.update_config(
            {"reward": {"event_values": {"custom_event": 2.5, "unit_test_passed": 0.3}}}
        )

        reloaded = importlib.reload(evaluator)
        assert reloaded.evaluate("custom_event") == 2.5
        assert reloaded.evaluate("unit_test_passed") == 0.3
    finally:
        config.update_config(original_config)
        importlib.reload(evaluator)
