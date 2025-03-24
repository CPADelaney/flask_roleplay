# nyx/core/procedural_memory/learning.py

import datetime
import random
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from collections import Counter, defaultdict
from pydantic import BaseModel, Field
from functools import lru_cache
import numpy as np
from threading import Lock

logger = logging.getLogger(__name__)

class ObservationLearner:
    """System for learning procedures from observation"""
    
    def __init__(self, config=None):
        self.observation_history = []
        self.pattern_detection_threshold = 0.7
        self.max_history = 100
        # Added fields for enhanced functionality
        self.learning_lock = Lock()
        self.model_cache = {}
        self.confidence_thresholds = {
            "pattern_recognition": 0.6,
            "action_sequence": 0.7,
            "state_changes": 0.8
        }
        self.learning_rate = 0.1  # Rate at which new observations affect existing models
        self.min_observations = 3  # Minimum observations needed for learning
        self.timeout = 30  # Timeout in seconds for learning operations
        self.config = config or {}
        # Transformer model references (initialized lazily)
        self.tokenizer = None
        self.model = None
    
    async def learn_from_demonstration(
        self, 
        observation_sequence: List[Dict[str, Any]], 
        domain: str
    ) -> Dict[str, Any]:
        """Learn a procedure from a sequence of observed actions"""
        # Create a timeout for this operation
        try:
            return await asyncio.wait_for(
                self._learn_from_demonstration_impl(observation_sequence, domain),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Learning from demonstration timed out after {self.timeout} seconds")
            # Return partial results if available
            return {
                "name": f"partial_learned_procedure_{int(datetime.datetime.now().timestamp())}",
                "steps": self._generate_steps_from_partial_observations(observation_sequence[:10]), # Use first 10 observations
                "description": "Partially learned procedure (learning timed out)",
                "domain": domain,
                "created_from_observations": True,
                "observation_count": len(observation_sequence),
                "confidence": 0.3,  # Low confidence for partial learning
                "timeout_occurred": True
            }
    
    async def _learn_from_demonstration_impl(
        self, 
        observation_sequence: List[Dict[str, Any]], 
        domain: str
    ) -> Dict[str, Any]:
        """Implementation of learning from demonstration"""
        # Use a lock to prevent concurrent learning operations
        async with self._async_lock():
            # Store observations in history
            self.observation_history.extend(observation_sequence)
            if len(self.observation_history) > self.max_history:
                self.observation_history = self.observation_history[-self.max_history:]
            
            # Try advanced learning with transformers if available
            try:
                if "use_transformers" in self.config and self.config["use_transformers"]:
                    return await self.learn_with_transformers(observation_sequence, domain)
            except Exception as e:
                logger.warning(f"Advanced learning with transformers failed: {str(e)}")
                # Fall back to traditional learning
            
            # Extract action patterns
            action_patterns = self._extract_action_patterns(observation_sequence)
            
            # Identify important state changes
            state_changes = self._identify_significant_state_changes(observation_sequence)
            
            # Generate procedure steps
            steps = self._generate_steps_from_patterns(action_patterns, state_changes)
            
            # Apply confidence thresholds
            confidence = self._calculate_learning_confidence(action_patterns, state_changes)
            
            # Create metadata for the new procedure
            procedure_data = {
                "name": f"learned_procedure_{int(datetime.datetime.now().timestamp())}",
                "steps": steps,
                "description": "Procedure learned from demonstration",
                "domain": domain,
                "created_from_observations": True,
                "observation_count": len(observation_sequence),
                "confidence": confidence,
                "learning_method": "pattern_detection"
            }
            
            return procedure_data
    
    async def learn_with_transformers(
        self, 
        observation_sequence: List[Dict[str, Any]], 
        domain: str
    ) -> Dict[str, Any]:
        """Learn a procedure using transformer models for improved pattern recognition"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            import numpy as np
            
            # Store observations in history
            self.observation_history.extend(observation_sequence)
            if len(self.observation_history) > self.max_history:
                self.observation_history = self.observation_history[-self.max_history:]
            
            # Convert observations to text sequences
            sequences = []
            for obs in observation_sequence:
                # Create a textual representation of the observation
                action = obs.get("action", "unknown")
                state_text = ", ".join([f"{k}={v}" for k, v in obs.get("state", {}).items()])
                sequences.append(f"Action: {action}. State: {state_text}")
            
            # Load models if not already loaded
            if self.tokenizer is None or self.model is None:
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                self.model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
                
            # Generate embeddings for each sequence
            embeddings = []
            for seq in sequences:
                inputs = self.tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = outputs.logits.numpy()
                    embeddings.append(embedding.flatten())
            
            # Cluster embeddings to find patterns
            from sklearn.cluster import KMeans
            
            if len(embeddings) < 3:  # Need at least 3 for meaningful clustering
                # Fall back to traditional pattern extraction
                return await self.learn_from_demonstration(observation_sequence, domain)
            
            embeddings_array = np.array(embeddings)
            num_clusters = min(len(embeddings) // 2, 5)  # At most 5 clusters
            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(embeddings_array)
            
            # Group actions by cluster
            clusters = {}
            for i, label in enumerate(kmeans.labels_):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(observation_sequence[i])
            
            # Generate steps from clusters
            steps = []
            for cluster_id, observations in clusters.items():
                # Find the most common action in this cluster
                actions = [obs.get("action") for obs in observations]
                most_common_action = max(set(actions), key=actions.count)
                
                # Collect parameters from states
                parameters = {}
                for obs in observations:
                    state = obs.get("state", {})
                    for key, value in state.items():
                        if key not in parameters:
                            parameters[key] = []
                        parameters[key].append(value)
                
                # Use most common values for parameters
                final_params = {}
                for key, values in parameters.items():
                    if values:
                        most_common = max(set(values), key=values.count)
                        final_params[key] = most_common
                
                # Create the step
                steps.append({
                    "id": f"step_{len(steps)+1}",
                    "description": f"Perform clustered actions: {most_common_action}",
                    "function": most_common_action,
                    "parameters": final_params
                })
            
            # Create metadata for the new procedure
            procedure_data = {
                "name": f"learned_procedure_{int(datetime.datetime.now().timestamp())}",
                "steps": steps,
                "description": "Procedure learned from demonstration using transformers",
                "domain": domain,
                "created_from_observations": True,
                "observation_count": len(observation_sequence),
                "confidence": 0.8,  # Higher confidence due to transformer-based approach
                "learning_method": "transformer_clustering"
            }
            
            return procedure_data
        
        except (ImportError, Exception) as e:
            # Fall back to traditional method if transformers not available
            logger.warning(f"Transformer learning failed: {str(e)}. Falling back to traditional learning.")
            return await self._learn_from_demonstration_impl(observation_sequence, domain)
    
    async def learn_incrementally(
        self,
        existing_procedure: Dict[str, Any],
        new_observations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Incrementally update an existing procedure with new observations"""
        # Extract existing information
        domain = existing_procedure.get("domain", "general")
        steps = existing_procedure.get("steps", [])
        confidence = existing_procedure.get("confidence", 0.5)
        
        # If no existing steps, treat as new learning
        if not steps:
            return await self.learn_from_demonstration(new_observations, domain)
        
        # Extract action patterns from new observations
        new_patterns = self._extract_action_patterns(new_observations)
        
        # Compare with existing steps to find refinements
        refined_steps = []
        new_step_actions = set(p["sequence"][0] for p in new_patterns if p["sequence"])
        existing_step_actions = set(s.get("function") for s in steps)
        
        # First add refined existing steps
        for step in steps:
            function = step.get("function")
            # Check if this step function is in the new patterns
            if function in new_step_actions:
                # Find matching pattern
                matching_pattern = next((p for p in new_patterns 
                                      if p["sequence"] and p["sequence"][0] == function), None)
                
                if matching_pattern:
                    # Refine parameters based on new observations
                    # Look for state changes related to this function
                    state_changes = []
                    for i, obs in enumerate(new_observations):
                        if obs.get("action") == function and i+1 < len(new_observations):
                            before = obs.get("state", {})
                            after = new_observations[i+1].get("state", {})
                            
                            changes = {}
                            for key in set(before.keys()) | set(after.keys()):
                                if key in before and key in after and before[key] != after[key]:
                                    changes[key] = after[key]
                                elif key not in before and key in after:
                                    changes[key] = after[key]
                            
                            if changes:
                                state_changes.append(changes)
                    
                    # Update parameters if we found state changes
                    if state_changes:
                        # Combine parameters from all state changes
                        new_params = {}
                        for change in state_changes:
                            for key, value in change.items():
                                new_params[key] = value
                        
                        # Update step with new parameters
                        updated_step = step.copy()
                        updated_step["parameters"] = {**step.get("parameters", {}), **new_params}
                        refined_steps.append(updated_step)
                    else:
                        # No changes to parameters
                        refined_steps.append(step)
                else:
                    # No matching pattern, keep as is
                    refined_steps.append(step)
            else:
                # This step wasn't observed in new observations, keep as is
                refined_steps.append(step)
        
        # Then add completely new steps
        new_functions = new_step_actions - existing_step_actions
        for function in new_functions:
            # Find matching pattern
            matching_pattern = next((p for p in new_patterns 
                                  if p["sequence"] and p["sequence"][0] == function), None)
            
            if matching_pattern:
                # Create a new step
                # Find related state changes
                state_changes = []
                for i, obs in enumerate(new_observations):
                    if obs.get("action") == function and i+1 < len(new_observations):
                        before = obs.get("state", {})
                        after = new_observations[i+1].get("state", {})
                        
                        changes = {}
                        for key in set(before.keys()) | set(after.keys()):
                            if key in before and key in after and before[key] != after[key]:
                                changes[key] = after[key]
                            elif key not in before and key in after:
                                changes[key] = after[key]
                        
                        if changes:
                            state_changes.append(changes)
                
                # Create parameters from state changes
                parameters = {}
                for change in state_changes:
                    for key, value in change.items():
                        parameters[key] = value
                
                # Add new step
                refined_steps.append({
                    "id": f"step_{len(refined_steps)+1}",
                    "description": f"Perform action: {function}",
                    "function": function,
                    "parameters": parameters
                })
        
        # Recalculate confidence
        # Confidence increases with more observations, up to a point
        total_observations = existing_procedure.get("observation_count", 0) + len(new_observations)
        new_confidence = min(0.95, confidence + 0.05 * (total_observations / 10))
        
        # Create updated procedure data
        updated_procedure = {
            "name": existing_procedure.get("name"),
            "steps": refined_steps,
            "description": existing_procedure.get("description"),
            "domain": domain,
            "created_from_observations": True,
            "observation_count": total_observations,
            "confidence": new_confidence,
            "incrementally_updated": True,
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        return updated_procedure
    
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
        
        # Find common sequences using adaptive threshold
        threshold = max(0.3, self.confidence_thresholds["action_sequence"] - 0.1 * (len(observations) // 10))
        common_sequences = [pair for pair, count in action_counts.items() 
                          if count >= len(observations) * threshold]
        
        # Convert to structured patterns
        patterns = []
        for seq in common_sequences:
            actions = seq.split("→")
            patterns.append({
                "sequence": actions,
                "frequency": action_counts[seq] / (len(observations) - 1),
                "action_types": actions,
                "confidence": min(1.0, action_counts[seq] / (len(observations) - 1) + 0.2)
            })
        
        return patterns
    
    def _identify_significant_state_changes(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify significant state changes in observations"""
        state_changes = []
        
        for i in range(len(observations) - 1):
            current_state = observations[i].get("state", {})
            next_state = observations[i + 1].get("state", {})
            
            # Find state changes with adaptive filtering
            changes = {}
            change_significance = 0.0
            
            for key in set(current_state.keys()) | set(next_state.keys()):
                if key in current_state and key in next_state:
                    if current_state[key] != next_state[key]:
                        changes[key] = {
                            "from": current_state[key],
                            "to": next_state[key]
                        }
                        # Calculate significance of this change
                        change_significance += 0.2
                elif key in next_state:
                    # New state variable
                    changes[key] = {
                        "from": None,
                        "to": next_state[key]
                    }
                    # New state variables are highly significant
                    change_significance += 0.3
            
            if changes:
                # Only include if we have sufficient changes
                if change_significance >= self.confidence_thresholds["state_changes"]:
                    state_changes.append({
                        "action": observations[i].get("action", "unknown"),
                        "changes": changes,
                        "index": i,
                        "significance": change_significance
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
                "parameters": parameters,
                "confidence": pattern.get("confidence", 0.7)
            })
        
        # If no patterns found, create steps directly from observations
        if not steps and state_changes:
            # Sort state changes by index to preserve order
            state_changes.sort(key=lambda x: x["index"])
            
            for i, change in enumerate(state_changes):
                # Create parameters
                parameters = {}
                for key, value in change["changes"].items():
                    if value["to"] is not None:
                        parameters[key] = value["to"]
                
                # Create the step
                steps.append({
                    "id": f"step_{i+1}",
                    "description": f"Perform action: {change['action']}",
                    "function": change["action"],
                    "parameters": parameters,
                    "confidence": min(0.6, change.get("significance", 0.0)) # Lower confidence for this method
                })
        
        return steps
    
    def _calculate_learning_confidence(self, 
                                    patterns: List[Dict[str, Any]], 
                                    state_changes: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the learned procedure"""
        if not patterns and not state_changes:
            return 0.1  # Very low confidence if nothing found
            
        if not patterns:
            return 0.3  # Low confidence if no patterns found
        
        # Average frequency of patterns
        avg_frequency = sum(p["frequency"] for p in patterns) / len(patterns)
        
        # Pattern coverage (what percentage of possible patterns did we find?)
        pattern_count_factor = min(1.0, len(patterns) / 5)
        
        # State change coverage
        state_change_factor = 0.0
        if state_changes:
            avg_significance = sum(c.get("significance", 0.0) for c in state_changes) / len(state_changes)
            state_change_factor = min(1.0, avg_significance)
        
        # Calculate confidence
        confidence = avg_frequency * 0.5 + pattern_count_factor * 0.3 + state_change_factor * 0.2
        
        return min(1.0, confidence)
    
    def _generate_steps_from_partial_observations(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate basic steps from partial observations when timeout occurs"""
        steps = []
        
        # Simply create a step for each unique action
        seen_actions = set()
        
        for obs in observations:
            action = obs.get("action", "unknown")
            
            # Skip duplicates
            if action in seen_actions:
                continue
                
            seen_actions.add(action)
            
            # Extract parameters from state
            parameters = {}
            state = obs.get("state", {})
            for key, value in state.items():
                parameters[key] = value
            
            # Create basic step
            steps.append({
                "id": f"step_{len(steps)+1}",
                "description": f"Perform action: {action}",
                "function": action,
                "parameters": parameters
            })
        
        return steps
    
    async def _async_lock(self):
        """Create an async context manager for locking"""
        class AsyncLockManager:
            def __init__(self, lock):
                self.lock = lock
                
            async def __aenter__(self):
                self.lock.acquire()
                
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.lock.release()
        
        return AsyncLockManager(self.learning_lock)
    
    # Add to ObservationLearner class
    async def apply_reinforcement_learning(
        self,
        procedure: Dict[str, Any],
        reward_function: Callable[[Dict[str, Any]], float],
        iterations: int = 10
    ) -> Dict[str, Any]:
        """Apply reinforcement learning to optimize a procedure"""
        try:
            # Initialize
            best_procedure = procedure.copy()
            best_reward = await reward_function(best_procedure)
            
            # Store step variations for exploration
            step_variations = {}
            
            # Track iteration history
            history = []
            
            # Run RL iterations
            for i in range(iterations):
                # Create a variation of the procedure
                candidate = best_procedure.copy()
                candidate["steps"] = best_procedure["steps"].copy()
                
                # Choose a random step to modify
                if not candidate["steps"]:
                    break
                    
                step_idx = random.randint(0, len(candidate["steps"]) - 1)
                step = candidate["steps"][step_idx].copy()
                
                # Modify the step (exploration)
                modified_step = await self._explore_step_variations(step, step_variations)
                candidate["steps"][step_idx] = modified_step
                
                # Evaluate the candidate
                reward = await reward_function(candidate)
                
                # Record iteration
                history.append({
                    "iteration": i,
                    "reward": reward,
                    "modified_step_idx": step_idx
                })
                
                # Update best if improvement found
                if reward > best_reward:
                    best_procedure = candidate
                    best_reward = reward
                    
                    # Update step variations with successful modification
                    step_id = step["id"]
                    if step_id not in step_variations:
                        step_variations[step_id] = []
                    
                    step_variations[step_id].append({
                        "step": modified_step,
                        "reward": reward
                    })
            
            # Return optimized procedure
            best_procedure["reinforcement_learning_applied"] = True
            best_procedure["rl_iterations"] = iterations
            best_procedure["rl_reward"] = best_reward
            best_procedure["rl_history"] = history
            
            return best_procedure
            
        except Exception as e:
            logger.error(f"Reinforcement learning failed: {str(e)}")
            # Return original procedure with error info
            procedure["reinforcement_learning_error"] = str(e)
            return procedure
    
    async def _explore_step_variations(
        self,
        step: Dict[str, Any],
        step_variations: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Explore variations of a step for reinforcement learning"""
        # Copy the step
        new_step = step.copy()
        
        # Decide what to modify
        modification_type = random.choice(["parameters", "function", "description"])
        
        if modification_type == "parameters":
            # Modify parameters
            new_step["parameters"] = new_step.get("parameters", {}).copy()
            
            # If we have parameters
            if new_step["parameters"]:
                # Either modify existing or add new
                if random.random() < 0.7 and new_step["parameters"]:
                    # Modify existing
                    param_key = random.choice(list(new_step["parameters"].keys()))
                    param_value = new_step["parameters"][param_key]
                    
                    # Modify based on type
                    if isinstance(param_value, bool):
                        new_step["parameters"][param_key] = not param_value
                    elif isinstance(param_value, (int, float)):
                        # Modify by +/- 10%
                        delta = param_value * 0.1
                        new_step["parameters"][param_key] = param_value + random.uniform(-delta, delta)
                    elif isinstance(param_value, str):
                        # Append or prepend something
                        if random.random() < 0.5:
                            new_step["parameters"][param_key] = param_value + "_modified"
                        else:
                            new_step["parameters"][param_key] = "modified_" + param_value
                else:
                    # Add new parameter
                    new_key = f"param_{random.randint(1, 100)}"
                    new_step["parameters"][new_key] = random.choice([True, False, 0, 1, "value"])
        
        elif modification_type == "function":
            # Modify function name slightly
            if "function" in new_step:
                function = new_step["function"]
                if isinstance(function, str):
                    if random.random() < 0.5:
                        new_step["function"] = function + "_variant"
                    else:
                        new_step["function"] = "variant_" + function
        
        elif modification_type == "description":
            # Modify description
            if "description" in new_step:
                description = new_step["description"]
                if isinstance(description, str):
                    if random.random() < 0.5:
                        new_step["description"] = description + " (modified)"
                    else:
                        new_step["description"] = "Modified: " + description
        
        return new_step
    
    # Enhance the incremental learning method
    async def learn_incrementally(
        self,
        existing_procedure: Dict[str, Any],
        new_observations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Incrementally update an existing procedure with new observations"""
        # Extract existing information
        domain = existing_procedure.get("domain", "general")
        steps = existing_procedure.get("steps", [])
        confidence = existing_procedure.get("confidence", 0.5)
        
        # If no existing steps, treat as new learning
        if not steps:
            return await self.learn_from_demonstration(new_observations, domain)
        
        # Extract action patterns from new observations
        new_patterns = self._extract_action_patterns(new_observations)
        
        # Compare with existing steps to find refinements
        refined_steps = []
        new_step_actions = set(p["sequence"][0] for p in new_patterns if p["sequence"])
        existing_step_actions = set(s.get("function") for s in steps)
        
        # First add refined existing steps
        for step in steps:
            function = step.get("function")
            # Check if this step function is in the new patterns
            if function in new_step_actions:
                # Find matching pattern
                matching_pattern = next((p for p in new_patterns 
                                      if p["sequence"] and p["sequence"][0] == function), None)
                
                if matching_pattern:
                    # Refine parameters based on new observations
                    # Look for state changes related to this function
                    state_changes = []
                    for i, obs in enumerate(new_observations):
                        if obs.get("action") == function and i+1 < len(new_observations):
                            before = obs.get("state", {})
                            after = new_observations[i+1].get("state", {})
                            
                            changes = {}
                            for key in set(before.keys()) | set(after.keys()):
                                if key in before and key in after and before[key] != after[key]:
                                    changes[key] = after[key]
                                elif key not in before and key in after:
                                    changes[key] = after[key]
                            
                            if changes:
                                state_changes.append(changes)
                    
                    # Update parameters if we found state changes
                    if state_changes:
                        # Combine parameters from all state changes
                        new_params = {}
                        for change in state_changes:
                            for key, value in change.items():
                                new_params[key] = value
                        
                        # Update step with new parameters
                        updated_step = step.copy()
                        updated_step["parameters"] = {**step.get("parameters", {}), **new_params}
                        refined_steps.append(updated_step)
                    else:
                        # No changes to parameters
                        refined_steps.append(step)
                else:
                    # No matching pattern, keep as is
                    refined_steps.append(step)
            else:
                # This step wasn't observed in new observations, keep as is
                refined_steps.append(step)
        
        # Then add completely new steps
        new_functions = new_step_actions - existing_step_actions
        for function in new_functions:
            # Find matching pattern
            matching_pattern = next((p for p in new_patterns 
                                  if p["sequence"] and p["sequence"][0] == function), None)
            
            if matching_pattern:
                # Create a new step
                # Find related state changes
                state_changes = []
                for i, obs in enumerate(new_observations):
                    if obs.get("action") == function and i+1 < len(new_observations):
                        before = obs.get("state", {})
                        after = new_observations[i+1].get("state", {})
                        
                        changes = {}
                        for key in set(before.keys()) | set(after.keys()):
                            if key in before and key in after and before[key] != after[key]:
                                changes[key] = after[key]
                            elif key not in before and key in after:
                                changes[key] = after[key]
                        
                        if changes:
                            state_changes.append(changes)
                
                # Create parameters from state changes
                parameters = {}
                for change in state_changes:
                    for key, value in change.items():
                        parameters[key] = value
                
                # Add new step
                refined_steps.append({
                    "id": f"step_{len(refined_steps)+1}",
                    "description": f"Perform action: {function}",
                    "function": function,
                    "parameters": parameters
                })
        
        # Recalculate confidence
        # Confidence increases with more observations, up to a point
        total_observations = existing_procedure.get("observation_count", 0) + len(new_observations)
        new_confidence = min(0.95, confidence + 0.05 * (total_observations / 10))
        
        # Create updated procedure data
        updated_procedure = {
            "name": existing_procedure.get("name"),
            "steps": refined_steps,
            "description": existing_procedure.get("description"),
            "domain": domain,
            "created_from_observations": True,
            "observation_count": total_observations,
            "confidence": new_confidence,
            "incrementally_updated": True,
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        return updated_procedure

class ProceduralMemoryConsolidator:
    """Consolidates and optimizes procedural memory"""
    
    def __init__(self, memory_core=None, config=None):
        self.memory_core = memory_core
        self.consolidation_history = []
        self.max_history = 20
        self.templates = {}  # Template id -> template
        # Added fields for enhanced functionality
        self.consolidation_lock = Lock()
        self.consolidation_interval = 3600  # Seconds between auto-consolidations
        self.last_consolidation = datetime.datetime.now().timestamp()
        self.memory_threshold = 0.8  # Threshold to trigger memory consolidation
        self.config = config or {}
        self.min_procedure_similarity = 0.7  # Minimum similarity for merging
        self.similarity_cache = {}  # Cache for similarity calculations

    async def consolidate_procedural_memory(self) -> Dict[str, Any]:
        """Consolidate procedural memory during downtime"""
        # Create basic concurrency protection
        if not self.consolidation_lock.acquire(blocking=False):
            return {
                "status": "already_running",
                "message": "Consolidation is already running"
            }
            
        try:
            # Check if consolidation is needed
            now = datetime.datetime.now().timestamp()
            if (now - self.last_consolidation < self.consolidation_interval and 
                not self.is_memory_pressure_high()):
                return {
                    "consolidated_templates": 0,
                    "procedures_updated": 0,
                    "status": "skipped",
                    "reason": "Too soon since last consolidation and no memory pressure"
                }
            
            # Update last consolidation time
            self.last_consolidation = now
            
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
            
            # Perform memory optimization if needed
            memory_stats = {}
            if self.is_memory_pressure_high():
                memory_stats = self.optimize_memory_usage()
                
            return {
                "consolidated_templates": len(templates),
                "procedures_updated": updated,
                "status": "success",
                "memory_optimization": memory_stats
            }
        finally:
            # Always release the lock
            self.consolidation_lock.release()
    
    def _find_related_procedures(self) -> List[Dict[str, Any]]:
        """Find procedures that might share patterns"""
        # In a real implementation, this would query the memory system
        if self.memory_core and hasattr(self.memory_core, "get_all_procedures"):
            try:
                # Get procedures from memory core
                procedures = self.memory_core.get_all_procedures()
                
                # Group procedures by domain
                by_domain = defaultdict(list)
                for proc in procedures:
                    by_domain[proc.get("domain", "general")].append(proc)
                
                # Find related procedures within domains
                related = []
                for domain, domain_procs in by_domain.items():
                    # Skip domains with too few procedures
                    if len(domain_procs) < 2:
                        continue
                        
                    # Compare procedures within domain
                    for i in range(len(domain_procs)):
                        for j in range(i+1, len(domain_procs)):
                            similarity = self._calculate_procedure_similarity(
                                domain_procs[i], domain_procs[j]
                            )
                            
                            if similarity >= self.min_procedure_similarity:
                                related.append(domain_procs[i])
                                related.append(domain_procs[j])
                
                return list(set(related))  # Remove duplicates
            except Exception as e:
                logger.error(f"Error finding related procedures: {str(e)}")
                
        # Return empty list if no memory core or error
        return []
    
    @lru_cache(maxsize=100)
    def _calculate_procedure_similarity(self, proc1: Dict[str, Any], proc2: Dict[str, Any]) -> float:
        """Calculate similarity between two procedures"""
        # Check cache
        cache_key = (proc1.get("id", ""), proc2.get("id", ""))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
            
        # Get steps from both procedures
        steps1 = proc1.get("steps", [])
        steps2 = proc2.get("steps", [])
        
        # If either has no steps, similarity is 0
        if not steps1 or not steps2:
            return 0.0
            
        # Calculate step function similarity
        funcs1 = [s.get("function") for s in steps1 if "function" in s]
        funcs2 = [s.get("function") for s in steps2 if "function" in s]
        
        if not funcs1 or not funcs2:
            return 0.0
            
        # Calculate Jaccard similarity
        set1 = set(funcs1)
        set2 = set(funcs2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            jaccard = 0.0
        else:
            jaccard = intersection / union
            
        # Calculate sequence similarity
        sequence_sim = 0.0
        
        # Find longest common subsequence
        lcs_length = self._longest_common_subsequence_length(funcs1, funcs2)
        
        # Normalize by average length
        avg_length = (len(funcs1) + len(funcs2)) / 2
        if avg_length > 0:
            sequence_sim = lcs_length / avg_length
            
        # Combine similarities
        similarity = (jaccard * 0.6) + (sequence_sim * 0.4)
        
        # Cache result
        self.similarity_cache[cache_key] = similarity
        
        return similarity
    
    def _longest_common_subsequence_length(self, seq1: List[Any], seq2: List[Any]) -> int:
        """Find length of longest common subsequence between two lists"""
        # Create DP table
        m, n = len(seq1), len(seq2)
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        
        # Fill DP table
        for i in range(1, m+1):
            for j in range(1, n+1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
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
                
                # Skip if procedures have different domains
                if proc1.get("domain") != proc2.get("domain"):
                    continue
                
                # Find longest common subsequence of steps
                common_seq = self._find_longest_common_subsequence(steps1, steps2)
                
                if len(common_seq) >= 2:  # At least 2 steps to form a pattern
                    common_patterns.append({
                        "steps": common_seq,
                        "procedure_ids": [proc1.get("id"), proc2.get("id")],
                        "domains": [proc1.get("domain"), proc2.get("domain")],
                        "pattern_type": "sequence",
                        "confidence": 0.5 + (0.1 * len(common_seq))  # Higher confidence for longer sequences
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
        
        # Generate semantic tags for better searchability
        semantic_tags = set()
        for step in pattern["steps"]:
            # Add function name as tag
            function = step.get("function")
            if function:
                semantic_tags.add(function)
            
            # Add words from description
            description = step.get("description", "")
            if description:
                # Add key words from description
                for word in description.lower().split():
                    if len(word) > 3 and word not in ["with", "from", "this", "that", "step"]:
                        semantic_tags.add(word)
        
        # Create the template
        return {
            "id": template_id,
            "name": f"Template for {pattern['pattern_type']}",
            "steps": template_steps,
            "domains": list(domains),
            "semantic_tags": list(semantic_tags),
            "confidence": pattern.get("confidence", 0.7),
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
        if self.memory_core and hasattr(self.memory_core, "get_all_procedures"):
            try:
                procedures = self.memory_core.get_all_procedures()
                
                for procedure in procedures:
                    # Check each template
                    for template in templates:
                        # Check if procedure steps match template
                        match = self._check_procedure_template_match(procedure, template)
                        
                        if match["matches"]:
                            # Reference template in procedure
                            self._apply_template_to_procedure(procedure, template, match["match_indices"])
                            updated_count += 1
                            
                            # Update in memory core
                            if hasattr(self.memory_core, "update_procedure"):
                                await self.memory_core.update_procedure(procedure)
            except Exception as e:
                logger.error(f"Error updating procedures with templates: {str(e)}")
                
        return updated_count
    
    def _check_procedure_template_match(
        self, 
        procedure: Dict[str, Any], 
        template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if a procedure matches a template's pattern"""
        proc_steps = procedure.get("steps", [])
        template_steps = template.get("steps", [])
        
        # Quick check - need at least as many steps as template
        if len(proc_steps) < len(template_steps):
            return {"matches": False}
            
        # Check if procedure domain is compatible
        proc_domain = procedure.get("domain")
        if proc_domain not in template.get("domains", []):
            return {"matches": False}
            
        # Look for template pattern in procedure steps
        for i in range(len(proc_steps) - len(template_steps) + 1):
            match = True
            match_indices = []
            
            for j, template_step in enumerate(template_steps):
                proc_step = proc_steps[i + j]
                
                # Check if function matches
                if proc_step.get("function") != template_step.get("function"):
                    match = False
                    break
                    
                match_indices.append(i + j)
            
            if match:
                return {
                    "matches": True, 
                    "match_indices": match_indices,
                    "start_index": i
                }
                
        return {"matches": False}
    
    def _apply_template_to_procedure(
        self, 
        procedure: Dict[str, Any], 
        template: Dict[str, Any],
        match_indices: List[int]
    ) -> None:
        """Apply a template to a procedure by adding references"""
        # Add template reference if not exists
        if "templates" not in procedure:
            procedure["templates"] = []
            
        # Check if template already referenced
        template_id = template["id"]
        for existing in procedure.get("templates", []):
            if existing.get("template_id") == template_id:
                # Already referenced
                return
                
        # Add template reference
        procedure["templates"].append({
            "template_id": template_id,
            "match_indices": match_indices,
            "applied_at": datetime.datetime.now().isoformat()
        })
        
        # Mark procedure as updated
        procedure["last_updated"] = datetime.datetime.now().isoformat()
        procedure["has_templates"] = True
    
    def is_memory_pressure_high(self) -> bool:
        """Check if memory pressure is high enough to need consolidation"""
        # Check memory core if available
        if self.memory_core and hasattr(self.memory_core, "get_memory_stats"):
            try:
                stats = self.memory_core.get_memory_stats()
                
                # Check memory usage
                memory_usage = stats.get("memory_usage_percent", 0.0)
                return memory_usage >= (self.memory_threshold * 100)
            except Exception:
                pass
                
        # Default - check if there are many procedures
        if self.memory_core and hasattr(self.memory_core, "get_all_procedures"):
            try:
                procedures = self.memory_core.get_all_procedures()
                return len(procedures) > 100  # Arbitrary threshold
            except Exception:
                pass
                
        return False
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage by cleaning up old data and merging similar procedures"""
        start_time = datetime.datetime.now()
        stats = {
            "templates_removed": 0,
            "procedures_merged": 0,
            "memory_freed": 0
        }
        
        # Clean up old templates
        old_templates = [
            template_id for template_id, template in self.templates.items()
            if "created_at" in template and 
            (datetime.datetime.now() - datetime.datetime.fromisoformat(template["created_at"])).days > 30
        ]
        
        for template_id in old_templates:
            if template_id in self.templates:
                del self.templates[template_id]
                stats["templates_removed"] += 1
                stats["memory_freed"] += 1000  # Rough estimate
        
        # Try to merge similar procedures
        if self.memory_core and hasattr(self.memory_core, "get_all_procedures"):
            try:
                procedures = self.memory_core.get_all_procedures()
                
                # Group by domain
                by_domain = defaultdict(list)
                for proc in procedures:
                    by_domain[proc.get("domain", "general")].append(proc)
                
                # Check for similar procedures to merge
                for domain, domain_procs in by_domain.items():
                    merged = set()  # Track already merged procedures
                    
                    for i in range(len(domain_procs)):
                        if i in merged:
                            continue
                            
                        for j in range(i+1, len(domain_procs)):
                            if j in merged:
                                continue
                                
                            # Calculate similarity
                            similarity = self._calculate_procedure_similarity(
                                domain_procs[i], domain_procs[j]
                            )
                            
                            # If very similar, merge them
                            if similarity >= 0.9:  # High threshold for merging
                                merged_proc = self._merge_procedures(
                                    domain_procs[i], domain_procs[j]
                                )
                                
                                # Update the first procedure with merged result
                                self._update_procedure(domain_procs[i], merged_proc)
                                
                                # Delete the second procedure
                                if hasattr(self.memory_core, "delete_procedure"):
                                    self.memory_core.delete_procedure(domain_procs[j]["id"])
                                    
                                merged.add(j)
                                stats["procedures_merged"] += 1
                                stats["memory_freed"] += 5000  # Rough estimate
            except Exception as e:
                logger.error(f"Error optimizing memory usage: {str(e)}")
        
        # Calculate time taken
        stats["execution_time"] = (datetime.datetime.now() - start_time).total_seconds()
        
        return stats
    
    def _merge_procedures(
        self, 
        proc1: Dict[str, Any], 
        proc2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge two similar procedures"""
        # Create new procedure with combined info
        merged = proc1.copy()
        
        # Use more descriptive name if available
        if len(proc2.get("description", "")) > len(proc1.get("description", "")):
            merged["description"] = proc2["description"]
            
        # Combine steps if they differ
        if proc1.get("steps") != proc2.get("steps"):
            # Use steps from procedure with higher confidence or execution count
            if proc2.get("confidence", 0.0) > proc1.get("confidence", 0.0):
                merged["steps"] = proc2.get("steps", [])
            elif proc2.get("execution_count", 0) > proc1.get("execution_count", 0) * 2:
                # Only use steps from proc2 if it has been executed much more
                merged["steps"] = proc2.get("steps", [])
        
        # Combine execution statistics
        merged["execution_count"] = (
            proc1.get("execution_count", 0) + proc2.get("execution_count", 0)
        )
        merged["successful_executions"] = (
            proc1.get("successful_executions", 0) + proc2.get("successful_executions", 0)
        )
        
        # Use higher confidence
        merged["confidence"] = max(
            proc1.get("confidence", 0.0),
            proc2.get("confidence", 0.0)
        )
        
        # Mark as merged
        merged["merged"] = True
        merged["merged_from"] = [proc1.get("id"), proc2.get("id")]
        merged["last_updated"] = datetime.datetime.now().isoformat()
        
        return merged
    
    def _update_procedure(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Update a procedure in place with data from another"""
        # Update fields in target
        for key, value in source.items():
            if key != "id":  # Preserve original ID
                target[key] = value
