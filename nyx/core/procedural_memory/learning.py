# nyx/core/procedural_memory/learning.py

import datetime
import random
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
from pydantic import BaseModel, Field

class ObservationLearner:
    """System for learning procedures from observation"""
    
    def __init__(self):
        self.observation_history = []
        self.pattern_detection_threshold = 0.7
        self.max_history = 100
    
    async def learn_from_demonstration(
        self, 
        observation_sequence: List[Dict[str, Any]], 
        domain: str
    ) -> Dict[str, Any]:
        """Learn a procedure from a sequence of observed actions"""
        # Store observations in history
        self.observation_history.extend(observation_sequence)
        if len(self.observation_history) > self.max_history:
            self.observation_history = self.observation_history[-self.max_history:]
        
        # Extract action patterns
        action_patterns = self._extract_action_patterns(observation_sequence)
        
        # Identify important state changes
        state_changes = self._identify_significant_state_changes(observation_sequence)
        
        # Generate procedure steps
        steps = self._generate_steps_from_patterns(action_patterns, state_changes)
        
        # Create metadata for the new procedure
        procedure_data = {
            "name": f"learned_procedure_{int(datetime.datetime.now().timestamp())}",
            "steps": steps,
            "description": "Procedure learned from demonstration",
            "domain": domain,
            "created_from_observations": True,
            "observation_count": len(observation_sequence),
            "confidence": self._calculate_learning_confidence(action_patterns)
        }
        
        return procedure_data
    
    def _extract_action_patterns(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract recurring action patterns from observations"""
        # Count action frequencies
        action_counts = Counter()
        action_sequences = []
        
        for i in range(len(observations) - 1):
            current = observations[i]
            next_obs = observations[i + 1]
            
            # Create action pair key
            if "action" in current and "action" in next_obs:
                action_pair = f"{current['action']}→{next_obs['action']}"
                action_counts[action_pair] += 1
        
        # Find common sequences
        common_sequences = [pair for pair, count in action_counts.items() 
                          if count >= len(observations) * 0.3]  # At least 30% of observations
        
        # Convert to structured patterns
        patterns = []
        for seq in common_sequences:
            actions = seq.split("→")
            patterns.append({
                "sequence": actions,
                "frequency": action_counts[seq] / (len(observations) - 1),
                "action_types": actions
            })
        
        return patterns
    
    def _identify_significant_state_changes(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify significant state changes in observations"""
        state_changes = []
        
        for i in range(len(observations) - 1):
            current_state = observations[i].get("state", {})
            next_state = observations[i + 1].get("state", {})
            
            # Find state changes
            changes = {}
            for key in set(current_state.keys()) | set(next_state.keys()):
                if key in current_state and key in next_state:
                    if current_state[key] != next_state[key]:
                        changes[key] = {
                            "from": current_state[key],
                            "to": next_state[key]
                        }
                elif key in next_state:
                    # New state variable
                    changes[key] = {
                        "from": None,
                        "to": next_state[key]
                    }
            
            if changes:
                state_changes.append({
                    "action": observations[i].get("action", "unknown"),
                    "changes": changes,
                    "index": i
                })
        
        return state_changes
    
    def _generate_steps_from_patterns(
        self, 
        patterns: List[Dict[str, Any]],
        state_changes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate procedure steps from detected patterns and state changes"""
        steps = []
        
        # Convert patterns into steps
        for i, pattern in enumerate(patterns):
            # Find related state changes
            related_changes = []
            for change in state_changes:
                if change["action"] in pattern["sequence"]:
                    related_changes.append(change)
            
            # Create parameters from state changes
            parameters = {}
            if related_changes:
                for change in related_changes:
                    for key, value in change["changes"].items():
                        # Only use target state values for parameters
                        if value["to"] is not None:
                            parameters[key] = value["to"]
            
            # Create the step
            steps.append({
                "id": f"step_{i+1}",
                "description": f"Perform action sequence: {', '.join(pattern['sequence'])}",
                "function": pattern["sequence"][0] if pattern["sequence"] else "unknown_action",
                "parameters": parameters
            })
        
        # If no patterns found, create steps directly from observations
        if not steps and state_changes:
            for i, change in enumerate(state_changes):
                # Create the step
                steps.append({
                    "id": f"step_{i+1}",
                    "description": f"Perform action: {change['action']}",
                    "function": change["action"],
                    "parameters": {k: v["to"] for k, v in change["changes"].items() if v["to"] is not None}
                })
        
        return steps
    
    def _calculate_learning_confidence(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the learned procedure"""
        if not patterns:
            return 0.3  # Low confidence if no patterns found
        
        # Average frequency of patterns
        avg_frequency = sum(p["frequency"] for p in patterns) / len(patterns)
        
        # Number of patterns relative to ideal (3-5 patterns is ideal)
        pattern_count_factor = min(1.0, len(patterns) / 5)
        
        # Calculate confidence
        confidence = avg_frequency * 0.7 + pattern_count_factor * 0.3
        
        return min(1.0, confidence)

class ProceduralMemoryConsolidator:
    """Consolidates and optimizes procedural memory"""
    
    def __init__(self, memory_core=None):
        self.memory_core = memory_core
        self.consolidation_history = []
        self.max_history = 20
        self.templates = {}  # Template id -> template
