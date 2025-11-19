import importlib
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

feasibility = importlib.import_module("nyx.nyx_agent.feasibility")


def test_normalize_scene_location_handles_string_payload():
    assert feasibility._normalize_scene_location_value("Atrium") == {"name": "Atrium"}


def test_normalize_scene_location_handles_token_sequence():
    assert feasibility._normalize_scene_location_value([
        "cafe",
        "town square",
    ]) == {
        "name": "town square",
        "breadcrumbs": ["cafe", "town square"],
    }


def test_normalize_scene_location_handles_pair_sequence():
    assert feasibility._normalize_scene_location_value([
        ("name", "Atrium"),
        ("id", 42),
    ]) == {
        "name": "Atrium",
        "id": 42,
    }
