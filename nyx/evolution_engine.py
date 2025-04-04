# nyx/evolution_engine.py

Nyx's Evolution Engine:
Autonomously links her missing capabilities with external API-sourced feature suggestions,
proposes integration plans, and feeds accepted ones to her module creation engine.

"""

import os
import json
import datetime
from typing import List, Dict, Any, Optional

class EvolutionEngine:
    def __init__(self,
                 missing_features_path: str,
                 api_capabilities_path: str,
                 cross_suggestions_path: str):
        self.missing_features_path = missing_features_path
        self.api_capabilities_path = api_capabilities_path
        self.cross_suggestions_path = cross_suggestions_path

        self.missing_features = self._load_json(missing_features_path)
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

        for feature in self.missing_features:
            feature_terms = set(feature["name"].lower().split() + feature["description"].lower().split())

            for api_suggestion in self.api_capabilities:
                api_terms = set(api_suggestion["title"].lower().split() + api_suggestion["summary"].lower().split())

                overlap = feature_terms.intersection(api_terms)
                relevance_score = len(overlap) / max(len(feature_terms), 1)

                if relevance_score >= 0.2:  # Threshold for relevance
                    match = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "feature_name": feature["name"],
                        "feature_description": feature["description"],
                        "api_suggestion_title": api_suggestion["title"],
                        "api_suggestion_summary": api_suggestion["summary"],
                        "relevance_score": relevance_score,
                        "nyx_comment": f"**Looks like this could fill a gap in my body, Chase. I suggest we fuse it.**"
                    }
                    matches.append(match)

        self.cross_suggestions = matches
        self._save_cross_suggestions()
        return matches
