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

    async def consolidate_procedural_memory(self) -> Dict[str, Any]:
        """Consolidate procedural memory during downtime"""
        # Identify related procedures
        related_procedures = self._find_related_procedures()
        
        # Extract common patterns
        common_patterns = self._extract_common_patterns(related_procedures)
        
        # Create generalized templates
        templates = []
        for pattern in common_patterns:
            template = self._create_template(pattern)
            if template:
                templates.append(template)
                self.templates[template["id"]] = template
        
        # Update existing procedures with references to templates
        updated = await self._update_procedures_with_templates(templates)
        
        # Record consolidation
        self.consolidation_history.append({
            "consolidated_templates": len(templates),
            "procedures_updated": updated,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Trim history
        if len(self.consolidation_history) > self.max_history:
            self.consolidation_history = self.consolidation_history[-self.max_history:]
        
        return {
            "consolidated_templates": len(templates),
            "procedures_updated": updated
        }
    
    def _find_related_procedures(self) -> List[Dict[str, Any]]:
        """Find procedures that might share patterns"""
        # In a real implementation, this would query the memory system
        # For now, return a placeholder list
        return []
    
    def _extract_common_patterns(self, procedures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract common patterns across procedures"""
        # Group steps by function
        step_groups = defaultdict(list)
        
        for procedure in procedures:
            for step in procedure.get("steps", []):
                function = step.get("function")
                if function:
                    step_groups[function].append({
                        "step": step,
                        "procedure_id": procedure.get("id"),
                        "procedure_domain": procedure.get("domain")
                    })
        
        # Find common sequences
        common_patterns = []
        
        # Simple pattern: consecutive steps with same functions
        for i in range(len(procedures)):
            proc1 = procedures[i]
            steps1 = proc1.get("steps", [])
            
            for j in range(i+1, len(procedures)):
                proc2 = procedures[j]
                steps2 = proc2.get("steps", [])
                
                # Find longest common subsequence of steps
                common_seq = self._find_longest_common_subsequence(steps1, steps2)
                
                if len(common_seq) >= 2:  # At least 2 steps to form a pattern
                    common_patterns.append({
                        "steps": common_seq,
                        "procedure_ids": [proc1.get("id"), proc2.get("id")],
                        "domains": [proc1.get("domain"), proc2.get("domain")],
                        "pattern_type": "sequence"
                    })
        
        return common_patterns
    
    def _find_longest_common_subsequence(
        self, 
        steps1: List[Dict[str, Any]], 
        steps2: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find longest common subsequence of steps between two procedures"""
        # Convert steps to function sequences for simpler comparison
        funcs1 = [step.get("function") for step in steps1]
        funcs2 = [step.get("function") for step in steps2]
        
        # DP table
        m, n = len(funcs1), len(funcs2)
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        
        # Fill DP table
        for i in range(1, m+1):
            for j in range(1, n+1):
                if funcs1[i-1] == funcs2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        # Backtrack to find sequence
        common_seq = []
        i, j = m, n
        
        while i > 0 and j > 0:
            if funcs1[i-1] == funcs2[j-1]:
                common_seq.append(steps1[i-1])
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1
        
        # Reverse to get correct order
        common_seq.reverse()
        
        return common_seq
    
    def _create_template(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Create a generalized template from a pattern"""
        if not pattern.get("steps"):
            return None
        
        # Create template ID
        template_id = f"template_{int(datetime.datetime.now().timestamp())}_{random.randint(1000, 9999)}"
        
        # Extract domains
        domains = set(pattern.get("domains", []))
        
        # Create template steps - generalize parameters
        template_steps = []
        for i, step in enumerate(pattern["steps"]):
            # Extract general parameters by comparing across instances
            general_params = {}
            specific_params = {}
            
            for key, value in step.get("parameters", {}).items():
                # Check if this parameter is consistent across domains
                is_general = True
                
                for domain in domains:
                    # Check if domain-specific value exists for this parameter
                    domain_specific = self._get_domain_specific_param(step, key, domain)
                    if domain_specific is not None and domain_specific != value:
                        is_general = False
                        specific_params[domain] = specific_params.get(domain, {})
                        specific_params[domain][key] = domain_specific
                
                if is_general:
                    general_params[key] = value
            
            # Create template step
            template_steps.append({
                "id": f"step_{i+1}",
                "function": step.get("function"),
                "description": step.get("description", f"Step {i+1}"),
                "general_parameters": general_params,
                "domain_specific_parameters": specific_params
            })
        
        # Create the template
        return {
            "id": template_id,
            "name": f"Template for {pattern['pattern_type']}",
            "steps": template_steps,
            "domains": list(domains),
            "created_at": datetime.datetime.now().isoformat()
        }
    
    def _get_domain_specific_param(
        self, 
        step: Dict[str, Any], 
        param_key: str, 
        domain: str
    ) -> Any:
        """Get domain-specific value for a parameter"""
        # This would require domain knowledge about parameter mappings
        # For simplicity, just return the current value
        return step.get("parameters", {}).get(param_key)
    
    async def _update_procedures_with_templates(self, templates: List[Dict[str, Any]]) -> int:
        """Update existing procedures with references to templates"""
        updated_count = 0
        
        # In a real implementation, this would update procedures in memory
        # For now, just return a placeholder count
        return updated_count
