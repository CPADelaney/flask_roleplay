import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nyx.nyx_agent._feasibility_helpers import extract_defer_details


def test_extract_defer_details_returns_guidance_and_leads():
    feasibility = {
        "overall": {"feasible": False, "strategy": "defer"},
        "per_intent": [
            {
                "narrator_guidance": "You need to locate the key first.",
                "leads": ["Search the study", "Ask the caretaker"],
                "violations": [{"reason": "missing_prerequisite"}],
            }
        ],
    }

    guidance, leads, extra = extract_defer_details(feasibility)

    assert guidance == "You need to locate the key first."
    assert leads == ["Search the study", "Ask the caretaker"]
    assert extra["leads"] == leads
    assert extra["violations"] == [{"reason": "missing_prerequisite"}]


def test_extract_defer_details_empty_for_non_defer():
    feasibility = {
        "overall": {"feasible": True, "strategy": "allow"},
        "per_intent": [],
    }

    guidance, leads, extra = extract_defer_details(feasibility)

    assert guidance == ""
    assert leads == []
    assert extra == {}
