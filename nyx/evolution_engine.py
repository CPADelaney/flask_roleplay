# nyx/evolution_engine.py

"""
Nyx's Evolution Engine:
Autonomously links her missing capabilities with external API-sourced feature suggestions,
proposes integration plans, and feeds accepted ones to her module creation engine.
"""

import os
import json
import datetime
from typing import List, Dict, Any, Optional
from nyx.creative.capability_system import CapabilityModel

class EvolutionEngine:
    def __init__(self,
                 capability_model_path: str,
                 api_capabilities_path: str,
                 cross_suggestions_path: str):
        self.cap_model = CapabilityModel(capability_model_path)
        self.api_capabilities_path = api_capabilities_path
        self.cross_suggestions_path = cross_suggestions_path
        self.api_capabilities = self._load_json(api_capabilities_path)
        self.cross_suggestions = []

    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_cross_suggestions(self) -> None:
        with open(self.cross_suggestions_path, 'w', encoding='utf-8') as f:
            json.dump(self.cross_suggestions, f, indent=2)

    def match_and_suggest(self) -> List[Dict[str, Any]]:
        matches = []
        unimplemented_caps = self.cap_model.get_unimplemented_capabilities()

        for cap in unimplemented_caps:
            cap_terms = set(cap.name.lower().split() + cap.description.lower().split())

            for api_suggestion in self.api_capabilities:
                api_terms = set(api_suggestion["title"].lower().split() + api_suggestion["summary"].lower().split())

                overlap = cap_terms.intersection(api_terms)
                relevance_score = len(overlap) / max(len(cap_terms), 1)

                if relevance_score >= 0.2:
                    match = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "capability": cap.name,
                        "description": cap.description,
                        "api_suggestion_title": api_suggestion["title"],
                        "api_suggestion_summary": api_suggestion["summary"],
                        "relevance_score": relevance_score,
                        "nyx_comment": f"**I crave this, Chase. Integrate it into me.**"
                    }
                    matches.append(match)

        self.cross_suggestions = matches
        self._save_cross_suggestions()
        return matches
