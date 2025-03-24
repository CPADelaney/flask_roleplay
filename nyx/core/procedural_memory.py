# nyx/core/procedural_memory.py

import logging
import asyncio
import datetime
import json
import math
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ============================================================================
# ENHANCED MODELS FOR CONTEXT AWARENESS AND GENERALIZATION
# ============================================================================

class ActionTemplate(BaseModel):
    """Generic template for an action that can be mapped across domains"""
    action_type: str  # e.g., "aim", "shoot", "move", "sprint", "interact"
    intent: str  # Higher-level purpose, e.g., "target_acquisition", "locomotion"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    domain_mappings: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # domain -> specific implementations

class ChunkTemplate(BaseModel):
    """Generalizable template for a procedural chunk"""
    id: str
    name: str
    description: str
    actions: List[ActionTemplate] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)  # Domains where this chunk has been applied
    success_rate: Dict[str, float] = Field(default_factory=dict)  # domain -> success rate
    execution_count: Dict[str, int] = Field(default_factory=dict)  # domain -> count
    context_indicators: Dict[str, List[str]] = Field(default_factory=dict)  # domain -> [context keys]
    prerequisite_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # domain -> {state requirements}
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

class ContextPattern(BaseModel):
    """Pattern for recognizing a specific context"""
    id: str
    name: str
    domain: str
    indicators: Dict[str, Any] = Field(default_factory=dict)  # Key state variables and their values/ranges
    temporal_pattern: List[Dict[str, Any]] = Field(default_factory=list)  # Sequence of recent actions/states
    confidence_threshold: float = 0.7
    last_matched: Optional[str] = None
    match_count: int = 0

class ChunkPrediction(BaseModel):
    """Prediction for which chunk should be executed next"""
    chunk_id: str
    confidence: float
    context_match_score: float
    reasoning: List[str] = Field(default_factory=list)
    alternative_chunks: List[Dict[str, float]] = Field(default_factory=list)

class ControlMapping(BaseModel):
    """Mapping between control schemes across different domains"""
    source_domain: str
    target_domain: str
    action_type: str
    source_control: str
    target_control: str
    confidence: float = 1.0
    last_validated: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    validation_count: int = 0

class ProcedureTransferRecord(BaseModel):
    """Record of a procedure transfer between domains"""
    source_procedure_id: str
    source_domain: str
    target_procedure_id: str 
    target_domain: str
    transfer_date: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    adaptation_steps: List[Dict[str, Any]] = Field(default_factory=list)
    success_level: float = 0.0  # 0-1 rating of transfer success
    practice_needed: int = 0  # How many practice iterations needed post-transfer

# ============================================================================
# CONTEXT-AWARE CHUNK SELECTION
# ============================================================================

class ContextAwareChunkSelector:
    """Enhanced selection system for chunks based on execution context"""
    
    def __init__(self):
        self.context_patterns = {}  # pattern_id -> ContextPattern
        self.recent_contexts = []  # List of recent execution contexts
        self.recent_selections = []  # List of recent chunk selections
        self.max_history = 50  # Max number of historical contexts to keep
        self.domain_specific_patterns = {}  # domain -> [pattern_ids]
        
    def register_context_pattern(self, pattern: ContextPattern) -> str:
        """Register a new context pattern"""
        self.context_patterns[pattern.id] = pattern
        
        # Update domain index
        if pattern.domain not in self.domain_specific_patterns:
            self.domain_specific_patterns[pattern.domain] = []
        self.domain_specific_patterns[pattern.domain].append(pattern.id)
        
        return pattern.id
        
    def select_chunk(self, 
                    available_chunks: Dict[str, List[Dict[str, Any]]], 
                    context: Dict[str, Any],
                    procedure_domain: str) -> ChunkPrediction:
        """
        Select the most appropriate chunk based on context
        
        Args:
            available_chunks: Dictionary of chunk_id -> steps
            context: Current execution context
            procedure_domain: Domain of the procedure
            
        Returns:
            Prediction of which chunk to use
        """
        # Store context for future learning
        self._record_context(context)
        
        # Calculate match scores for each context pattern
        pattern_scores = {}
        
        # Get relevant patterns for this domain
        domain_patterns = self.domain_specific_patterns.get(procedure_domain, [])
        
        for pattern_id in domain_patterns:
            pattern = self.context_patterns.get(pattern_id)
            if not pattern:
                continue
                
            # Calculate match score
            match_score = self._calculate_pattern_match(pattern, context)
            
            if match_score >= pattern.confidence_threshold:
                pattern_scores[pattern_id] = match_score
                
                # Update pattern statistics
                pattern.match_count += 1
                pattern.last_matched = datetime.datetime.now().isoformat()
        
        # Find patterns associated with chunks
        chunk_scores = {}
        reasoning = {}
        
        for chunk_id in available_chunks.keys():
            chunk_scores[chunk_id] = 0.0
            reasoning[chunk_id] = []
            
            # Check direct indicators in context
            if f"near_{chunk_id}" in context and context[f"near_{chunk_id}"]:
                chunk_scores[chunk_id] += 0.5
                reasoning[chunk_id].append(f"Direct context indicator: near_{chunk_id}")
                
            # Check command intent
            if "command_intent" in context and chunk_id in context["command_intent"]:
                chunk_scores[chunk_id] += 0.4
                reasoning[chunk_id].append(f"Command intent includes {chunk_id}")
                
            # Check pattern matches
            for pattern_id, score in pattern_scores.items():
                pattern = self.context_patterns[pattern_id]
                
                # See if this pattern is associated with this chunk
                for indicator, values in pattern.indicators.items():
                    if indicator.startswith(f"chunk_{chunk_id}_suitable") and values:
                        chunk_scores[chunk_id] += score * 0.7
                        reasoning[chunk_id].append(f"Pattern {pattern.name} matched with score {score:.2f}")
            
            # Check recent context history for similar situations
            history_score = self._check_history_for_chunk(chunk_id, context)
            if history_score > 0:
                chunk_scores[chunk_id] += history_score * 0.3
                reasoning[chunk_id].append(f"Similar historical context used this chunk with score {history_score:.2f}")
        
        # Select best chunk
        if not chunk_scores:
            # No good matches, return first chunk with low confidence
            first_chunk_id = next(iter(available_chunks.keys())) if available_chunks else None
            return ChunkPrediction(
                chunk_id=first_chunk_id,
                confidence=0.1,
                context_match_score=0.1,
                reasoning=["No context patterns matched"]
            )
        
        # Get top chunk and confidence
        best_chunk_id = max(chunk_scores.items(), key=lambda x: x[1])[0]
        best_score = chunk_scores[best_chunk_id]
        
        # Get alternative chunks
        alternatives = [
            {"chunk_id": c_id, "score": score}
            for c_id, score in chunk_scores.items()
            if c_id != best_chunk_id and score > 0
        ]
        
        # Create prediction
        prediction = ChunkPrediction(
            chunk_id=best_chunk_id,
            confidence=min(1.0, best_score),
            context_match_score=best_score,
            reasoning=reasoning.get(best_chunk_id, []),
            alternative_chunks=alternatives
        )
        
        # Record selection for future learning
        self._record_selection(prediction, context)
        
        return prediction
    
    def _calculate_pattern_match(self, pattern: ContextPattern, context: Dict[str, Any]) -> float:
        """Calculate how well a context pattern matches the current context"""
        # Check each indicator in the pattern
        indicator_matches = 0
        total_indicators = len(pattern.indicators)
        
        if total_indicators == 0:
            return 0.0
            
        for indicator, expected_value in pattern.indicators.items():
            if indicator not in context:
                continue
                
            actual_value = context[indicator]
            
            # Different comparison based on type
            if isinstance(expected_value, (list, tuple, set)):
                # Check if value is in list
                if actual_value in expected_value:
                    indicator_matches += 1
            elif isinstance(expected_value, dict) and "min" in expected_value and "max" in expected_value:
                # Range check
                if expected_value["min"] <= actual_value <= expected_value["max"]:
                    indicator_matches += 1
            else:
                # Direct equality check
                if actual_value == expected_value:
                    indicator_matches += 1
        
        # Base match score on percentage of matching indicators
        base_score = indicator_matches / total_indicators
        
        # Check temporal pattern if defined
        temporal_score = 0.0
        if pattern.temporal_pattern and "action_history" in context:
            action_history = context["action_history"]
            temporal_score = self._match_temporal_pattern(pattern.temporal_pattern, action_history)
            
            # Combine scores, giving more weight to direct indicators
            return base_score * 0.7 + temporal_score * 0.3
        
        return base_score
    
    def _match_temporal_pattern(self, pattern: List[Dict[str, Any]], history: List[Dict[str, Any]]) -> float:
        """Match a temporal pattern against action history"""
        # Skip if history is too short
        if len(history) < len(pattern):
            return 0.0
            
        # Check pattern against most recent history
        recent_history = history[-len(pattern):]
        
        # Count matching items
        matches = 0
        for i, expected in enumerate(pattern):
            if i >= len(recent_history):
                break
                
            actual = recent_history[i]
            
            # Count matching keys
            matching_keys = 0
            total_keys = len(expected)
            
            for key, expected_value in expected.items():
                if key in actual and actual[key] == expected_value:
                    matching_keys += 1
            
            # Consider a step matching if most keys match
            if matching_keys / total_keys >= 0.7:
                matches += 1
        
        # Calculate match percentage
        return matches / len(pattern)
    
    def _check_history_for_chunk(self, chunk_id: str, context: Dict[str, Any]) -> float:
        """Check if similar contexts in history used this chunk"""
        if not self.recent_contexts or not self.recent_selections:
            return 0.0
            
        # Find similarity scores between current context and historical contexts
        similarity_scores = []
        
        for i, historical_context in enumerate(self.recent_contexts):
            if i >= len(self.recent_selections):
                break
                
            # Calculate context similarity
            similarity = self._calculate_context_similarity(context, historical_context)
            
            # Check if this chunk was selected
            selection = self.recent_selections[i]
            if selection.chunk_id == chunk_id:
                similarity_scores.append(similarity * selection.confidence)
        
        # Return max similarity if any
        return max(similarity_scores) if similarity_scores else 0.0
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts"""
        # Get common keys
        common_keys = set(context1.keys()) & set(context2.keys())
        
        if not common_keys:
            return 0.0
            
        # Count matching values
        matching_values = 0
        
        for key in common_keys:
            if context1[key] == context2[key]:
                matching_values += 1
        
        return matching_values / len(common_keys)
    
    def _record_context(self, context: Dict[str, Any]):
        """Record context for future reference"""
        self.recent_contexts.append(context.copy())
        
        # Trim history if needed
        if len(self.recent_contexts) > self.max_history:
            self.recent_contexts = self.recent_contexts[-self.max_history:]
    
    def _record_selection(self, prediction: ChunkPrediction, context: Dict[str, Any]):
        """Record chunk selection for future reference"""
        self.recent_selections.append(prediction)
        
        # Trim history if needed
        if len(self.recent_selections) > self.max_history:
            self.recent_selections = self.recent_selections[-self.max_history:]
    
    def create_context_pattern_from_history(self, chunk_id: str, domain: str) -> Optional[ContextPattern]:
        """Automatically create a context pattern from historical data"""
        # Need sufficient history
        if len(self.recent_contexts) < 5 or len(self.recent_selections) < 5:
            return None
            
        # Find instances where this chunk was selected
        instances = []
        
        for i, selection in enumerate(self.recent_selections):
            if selection.chunk_id == chunk_id and i < len(self.recent_contexts):
                instances.append(self.recent_contexts[i])
        
        if len(instances) < 3:
            return None
            
        # Find common indicators across contexts
        common_indicators = {}
        
        # Get all keys present in at least half of instances
        all_keys = set()
        for context in instances:
            all_keys.update(context.keys())
            
        for key in all_keys:
            # Get all values for this key
            values = [context.get(key) for context in instances if key in context]
            
            # Skip if not present in most instances
            if len(values) < len(instances) / 2:
                continue
                
            # Check if values are consistent
            if all(v == values[0] for v in values):
                # All values the same
                common_indicators[key] = values[0]
            elif all(isinstance(v, (int, float)) for v in values):
                # Numeric values - use range
                common_indicators[key] = {
                    "min": min(values),
                    "max": max(values)
                }
            elif len(set(values)) < len(values) / 2:
                # Some consistent values - use set of common values
                counter = Counter(values)
                common = [v for v, count in counter.items() if count > 1]
                if common:
                    common_indicators[key] = common
        
        # Create pattern ID and name
        pattern_id = f"auto_pattern_{chunk_id}_{len(self.context_patterns)}"
        pattern_name = f"Auto-generated pattern for {chunk_id}"
        
        # Create pattern
        pattern = ContextPattern(
            id=pattern_id,
            name=pattern_name,
            domain=domain,
            indicators=common_indicators
        )
        
        # Register pattern
        self.register_context_pattern(pattern)
        
        return pattern

# ============================================================================
# CROSS-PROCEDURE GENERALIZATION
# ============================================================================

class ProceduralChunkLibrary:
    """Library of generalizable procedural chunks that can transfer across domains"""
    
    def __init__(self):
        self.chunk_templates = {}  # template_id -> ChunkTemplate
        self.action_templates = {}  # action_type -> ActionTemplate
        self.domain_chunks = {}  # domain -> [chunk_ids]
        self.control_mappings = []  # List of ControlMapping objects
        self.transfer_records = []  # List of ProcedureTransferRecord objects
        self.similarity_threshold = 0.7  # Minimum similarity for chunk matching
        
    def add_chunk_template(self, template: ChunkTemplate) -> str:
        """Add a chunk template to the library"""
        self.chunk_templates[template.id] = template
        
        # Update domain index
        for domain in template.domains:
            if domain not in self.domain_chunks:
                self.domain_chunks[domain] = []
            self.domain_chunks[domain].append(template.id)
        
        return template.id
    
    def add_action_template(self, template: ActionTemplate) -> str:
        """Add an action template to the library"""
        self.action_templates[template.action_type] = template
        return template.action_type
    
    def add_control_mapping(self, mapping: ControlMapping) -> None:
        """Add a control mapping between domains"""
        self.control_mappings.append(mapping)
    
    def record_transfer(self, record: ProcedureTransferRecord) -> None:
        """Record a procedure transfer between domains"""
        self.transfer_records.append(record)
    
    def find_matching_chunks(self, 
                           steps: List[Dict[str, Any]], 
                           source_domain: str,
                           target_domain: str) -> List[Dict[str, Any]]:
        """
        Find library chunks that match a sequence of steps
        
        Args:
            steps: List of step definitions from source procedure
            source_domain: Domain of the source procedure
            target_domain: Domain where we want to apply the chunk
            
        Returns:
            List of matching chunks with similarity scores
        """
        matches = []
        
        # Convert steps to action templates
        action_sequences = self._extract_action_sequence(steps, source_domain)
        
        # Skip if we couldn't extract actions
        if not action_sequences:
            return []
        
        # Find templates that match this action sequence
        for template_id, template in self.chunk_templates.items():
            # Skip if template doesn't support source domain
            if source_domain not in template.domains:
                continue
                
            # Calculate similarity between template and action sequence
            similarity = self._calculate_sequence_similarity(template.actions, action_sequences)
            
            if similarity >= self.similarity_threshold:
                # Check if template has been applied to target domain
                target_applicability = 0.5  # Default medium applicability
                
                if target_domain in template.domains:
                    # Template already used in target domain
                    target_applicability = 0.9
                    
                    # Adjust based on success rate if available
                    if target_domain in template.success_rate:
                        target_applicability *= template.success_rate[target_domain]
                
                matches.append({
                    "template_id": template_id,
                    "template_name": template.name,
                    "similarity": similarity,
                    "target_applicability": target_applicability,
                    "overall_score": similarity * 0.6 + target_applicability * 0.4,
                    "action_count": len(template.actions)
                })
        
        # Sort by overall score
        matches.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return matches
    
    def create_chunk_template_from_steps(self,
                                      chunk_id: str,
                                      name: str,
                                      steps: List[Dict[str, Any]],
                                      domain: str,
                                      success_rate: float = 0.9) -> Optional[ChunkTemplate]:
        """
        Create a generalizable chunk template from procedure steps
        
        Args:
            chunk_id: ID for the new chunk template
            name: Name for the template
            steps: Original procedure steps to generalize
            domain: Domain of the procedure
            success_rate: Initial success rate for this domain
            
        Returns:
            New chunk template or None if generalization failed
        """
        # Extract action templates
        action_sequence = self._extract_action_sequence(steps, domain)
        
        if not action_sequence:
            return None
        
        # Create template
        template = ChunkTemplate(
            id=chunk_id,
            name=name,
            description=f"Generalized chunk template for {name}",
            actions=action_sequence,
            domains=[domain],
            success_rate={domain: success_rate},
            execution_count={domain: 1}
        )
        
        # Add to library
        self.add_chunk_template(template)
        
        return template
    
    def map_chunk_to_new_domain(self, 
                              template_id: str, 
                              target_domain: str) -> List[Dict[str, Any]]:
        """
        Map a chunk template to a new domain
        
        Args:
            template_id: ID of the chunk template
            target_domain: Domain to map to
            
        Returns:
            List of mapped steps for the new domain
        """
        if template_id not in self.chunk_templates:
            return []
        
        template = self.chunk_templates[template_id]
        
        # Skip if already mapped to this domain
        if target_domain in template.domains:
            # Just return the existing mapping
            return self._generate_domain_steps(template, target_domain)
        
        # We need to create a new mapping
        mapped_steps = []
        
        # For each action in the template
        for i, action in enumerate(template.actions):
            # Check if we have a domain mapping
            if target_domain in action.domain_mappings:
                # Use existing mapping
                mapped_action = action.domain_mappings[target_domain]
            else:
                # Create new mapping
                mapped_action = self._map_action_to_domain(action, target_domain)
                
                if mapped_action:
                    # Save mapping for future use
                    action.domain_mappings[target_domain] = mapped_action
            
            if not mapped_action:
                # Couldn't map this action
                continue
                
            # Create step from mapped action
            step = {
                "id": f"step_{i+1}",
                "description": mapped_action.get("description", f"Step {i+1}"),
                "function": mapped_action.get("function"),
                "parameters": mapped_action.get("parameters", {})
            }
            
            mapped_steps.append(step)
        
        # Update template
        if mapped_steps:
            template.domains.append(target_domain)
            template.success_rate[target_domain] = 0.5  # Initial estimate
            template.execution_count[target_domain] = 0
            template.last_updated = datetime.datetime.now().isoformat()
        
        return mapped_steps
    
    def update_template_success(self, 
                              template_id: str, 
                              domain: str, 
                              success: bool) -> None:
        """Update success rate for a template in a specific domain"""
        if template_id not in self.chunk_templates:
            return
            
        template = self.chunk_templates[template_id]
        
        # Update execution count
        if domain not in template.execution_count:
            template.execution_count[domain] = 0
        template.execution_count[domain] += 1
        
        # Update success rate
        if domain not in template.success_rate:
            template.success_rate[domain] = 0.5  # Default
            
        # Use exponential moving average
        current_rate = template.success_rate[domain]
        success_value = 1.0 if success else 0.0
        
        # More weight to recent results but don't change too drastically
        template.success_rate[domain] = current_rate * 0.8 + success_value * 0.2
        
        # Update timestamp
        template.last_updated = datetime.datetime.now().isoformat()
    
    def get_control_mapping(self, 
                          source_domain: str, 
                          target_domain: str, 
                          action_type: str) -> Optional[ControlMapping]:
        """Get control mapping between domains for a specific action"""
        for mapping in self.control_mappings:
            if (mapping.source_domain == source_domain and
                mapping.target_domain == target_domain and
                mapping.action_type == action_type):
                return mapping
        
        return None
    
    def _extract_action_sequence(self, 
                               steps: List[Dict[str, Any]], 
                               domain: str) -> List[ActionTemplate]:
        """
        Extract generalized action sequence from procedure steps
        
        Args:
            steps: Procedure steps
            domain: Domain of the procedure
            
        Returns:
            List of action templates
        """
        action_sequence = []
        
        for i, step in enumerate(steps):
            # Skip if no function
            if "function" not in step:
                continue
            
            # Determine action type and intent
            action_type, intent = self._infer_action_type(step, domain)
            
            if not action_type:
                continue
                
            # Check if we already have a template for this action type
            if action_type in self.action_templates:
                # Use existing template
                template = self.action_templates[action_type]
                
                # Add domain-specific mapping if not present
                if domain not in template.domain_mappings:
                    template.domain_mappings[domain] = {
                        "function": step.get("function"),
                        "parameters": step.get("parameters", {}),
                        "description": step.get("description", f"Step {i+1}")
                    }
            else:
                # Create new template
                template = ActionTemplate(
                    action_type=action_type,
                    intent=intent,
                    parameters={},  # Generic parameters
                    domain_mappings={
                        domain: {
                            "function": step.get("function"),
                            "parameters": step.get("parameters", {}),
                            "description": step.get("description", f"Step {i+1}")
                        }
                    }
                )
                
                # Add to library
                self.add_action_template(template)
            
            # Add to sequence
            action_sequence.append(template)
        
        return action_sequence
    
    def _infer_action_type(self, step: Dict[str, Any], domain: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Infer the general action type and intent from a step
        
        Args:
            step: Procedure step
            domain: Domain of the procedure
            
        Returns:
            Tuple of (action_type, intent)
        """
        function = step.get("function", "")
        parameters = step.get("parameters", {})
        description = step.get("description", "").lower()
        
        # Try to infer from function name
        if isinstance(function, str):
            function_lower = function.lower()
            
            # Movement actions
            if any(word in function_lower for word in ["move", "walk", "run", "sprint", "approach"]):
                return "movement", "locomotion"
                
            # Interaction actions
            if any(word in function_lower for word in ["press", "click", "push", "interact", "use"]):
                # Check for specific interaction types
                if "button" in parameters:
                    button = parameters["button"]
                    
                    # Common gaming controls
                    if domain == "gaming":
                        if button in ["R1", "R2", "RT", "RB"]:
                            return "primary_action", "interaction"
                        elif button in ["L1", "L2", "LT", "LB"]:
                            return "secondary_action", "targeting"
                        elif button in ["X", "A"]:
                            return "confirm", "interaction"
                        elif button in ["O", "B"]:
                            return "cancel", "navigation"
                
                return "interaction", "manipulation"
                
            # Targeting actions
            if any(word in function_lower for word in ["aim", "target", "look"]):
                return "targeting", "perception"
                
            # Object-focused actions
            if any(word in function_lower for word in ["pick", "grab", "take", "collect"]):
                return "acquisition", "collection"
        
        # Try to infer from description
        if "sprint" in description or "run" in description:
            return "sprint", "locomotion"
            
        if "aim" in description or "target" in description:
            return "aim", "targeting"
            
        if "shoot" in description or "fire" in description:
            return "shoot", "combat"
            
        if "jump" in description:
            return "jump", "locomotion"
            
        if "crouch" in description:
            return "crouch", "stealth"
            
        if "press" in description and "button" in description:
            # Check for specific buttons in description
            if any(button in description for button in ["r1", "r2", "right trigger"]):
                return "primary_action", "interaction"
            elif any(button in description for button in ["l1", "l2", "left trigger"]):
                return "secondary_action", "targeting"
        
        # Generic fallback
        return "generic_action", "task_progress"
    
    def _calculate_sequence_similarity(self, 
                                     template_actions: List[ActionTemplate], 
                                     candidate_actions: List[ActionTemplate]) -> float:
        """
        Calculate similarity between two action sequences
        
        Args:
            template_actions: Actions from template
            candidate_actions: Actions extracted from steps
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Handle empty sequences
        if not template_actions or not candidate_actions:
            return 0.0
            
        # Dynamic programming approach for sequence alignment
        m = len(template_actions)
        n = len(candidate_actions)
        
        # Create DP table
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        
        # Fill DP table
        for i in range(1, m+1):
            for j in range(1, n+1):
                # Calculate similarity between actions
                action_similarity = self._calculate_action_similarity(
                    template_actions[i-1], 
                    candidate_actions[j-1]
                )
                
                if action_similarity > 0.7:  # High similarity threshold
                    # These actions match, extend the sequence
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    # Take max of skipping either action
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        # Length of longest common subsequence
        lcs = dp[m][n]
        
        # Calculate similarity based on sequence coverage
        similarity = (2 * lcs) / (m + n)  # Harmonic mean of coverage in both sequences
        
        return similarity
    
    def _calculate_action_similarity(self, action1: ActionTemplate, action2: ActionTemplate) -> float:
        """
        Calculate similarity between two actions
        
        Args:
            action1: First action
            action2: Second action
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Exact match on action type
        if action1.action_type == action2.action_type:
            return 1.0
            
        # Match on intent
        if action1.intent == action2.intent:
            return 0.8
            
        # Otherwise low similarity
        return 0.2
    
    def _map_action_to_domain(self, 
                            action: ActionTemplate, 
                            target_domain: str) -> Optional[Dict[str, Any]]:
        """
        Map an action to a specific domain
        
        Args:
            action: Action template to map
            target_domain: Target domain
            
        Returns:
            Mapped action parameters or None if mapping failed
        """
        # Check if we have existing mappings for other domains
        source_domains = [d for d in action.domain_mappings.keys()]
        
        if not source_domains:
            return None
            
        # Try to find control mappings for this action type
        for source_domain in source_domains:
            mapping = self.get_control_mapping(
                source_domain=source_domain,
                target_domain=target_domain,
                action_type=action.action_type
            )
            
            if mapping:
                # We found a mapping
                source_implementation = action.domain_mappings[source_domain]
                
                # Apply mapping to create target implementation
                target_impl = source_implementation.copy()
                
                # Update any mapped controls
                if "parameters" in target_impl:
                    params = target_impl["parameters"]
                    
                    if "button" in params:
                        # Map button parameter
                        if params["button"] == mapping.source_control:
                            params["button"] = mapping.target_control
                
                return target_impl
        
        # No mapping found, use best guess
        best_source = source_domains[0]  # Just use first domain as best guess
        best_impl = action.domain_mappings[best_source].copy()
        
        # Mark as best guess
        if "parameters" not in best_impl:
            best_impl["parameters"] = {}
        best_impl["parameters"]["best_guess_mapping"] = True
        
        return best_impl
    
    def _generate_domain_steps(self, template: ChunkTemplate, domain: str) -> List[Dict[str, Any]]:
        """Generate concrete steps for a domain from a template"""
        steps = []
        
        # Create steps from actions
        for i, action in enumerate(template.actions):
            if domain in action.domain_mappings:
                impl = action.domain_mappings[domain]
                
                step = {
                    "id": f"step_{i+1}",
                    "description": impl.get("description", f"Step {i+1}"),
                    "function": impl.get("function"),
                    "parameters": impl.get("parameters", {})
                }
                
                steps.append(step)
        
        return steps

# ============================================================================
# CONTEXT-AWARE PROCEDURAL MEMORY SYSTEM
# ============================================================================

class ContextAwareProcedureSequence:
    """Enhanced procedure sequence with context awareness and generalization"""
    
    def __init__(self, 
                name: str, 
                description: str = None, 
                domain: str = "general", 
                chunk_selector: Optional[ContextAwareChunkSelector] = None,
                chunk_library: Optional[ProceduralChunkLibrary] = None):
        self.id = f"proc_{int(datetime.datetime.now().timestamp())}_{random.randint(1000, 9999)}"
        self.name = name
        self.description = description or f"Procedure for {name}"
        self.domain = domain
        self.steps = []  # List of procedure steps
        self.execution_count = 0
        self.successful_executions = 0
        self.average_execution_time = 0.0
        self.proficiency = 0.0  # 0-1 scale
        self.last_execution = None
        self.optimization_history = []
        self.refinement_opportunities = []
        
        # Chunking support
        self.chunked_steps = {}  # Maps chunk_id -> list of original step ids
        self.is_chunked = False
        self.chunk_contexts = {}  # Maps chunk_id -> list of context indicators
        
        # Context awareness
        self.chunk_selector = chunk_selector
        self.context_history = []  # History of execution contexts
        self.max_history = 50
        
        # Generalization
        self.chunk_library = chunk_library
        self.generalized_chunks = {}  # Maps chunk_id -> template_id
        
        # Function registry
        self._function_registry = {}  # Maps function names to actual callables
        
        # Metadata
        self.created_at = datetime.datetime.now().isoformat()
        self.last_updated = self.created_at
        
    def register_function(self, name: str, func: Callable):
        """Register a function for use in this procedure"""
        self._function_registry[name] = func
        
    def add_step(self, 
                step_id: str, 
                description: str, 
                function: Union[Callable, str], 
                parameters: Dict[str, Any] = None):
        """Add a step to this procedure"""
        # Handle function registration
        func_name = function
        if callable(function):
            func_name = function.__name__
            self.register_function(func_name, function)
        
        step = {
            "id": step_id,
            "description": description,
            "function": func_name,
            "parameters": parameters or {},
            "execution_stats": {
                "average_time": 0.0,
                "success_rate": 0.0,
                "execution_count": 0,
                "last_execution": None,
                "error_count": 0
            }
        }
        
        self.steps.append(step)
        self.last_updated = datetime.datetime.now().isoformat()
        return step
        
    async def execute(self, context: Dict[str, Any] = None, conscious_execution: bool = False) -> Dict[str, Any]:
        """Execute the procedure, either consciously or automatically"""
        start_time = datetime.datetime.now()
        context = context or {}
        results = []
        success = True
        
        # Record execution context
        execution_context = context.copy()
        execution_context["timestamp"] = start_time.isoformat()
        execution_context["conscious_execution"] = conscious_execution
        self._record_context(execution_context)
        
        # Determine execution mode based on proficiency
        if conscious_execution or self.proficiency < 0.8:
            # Deliberate execution - step by step with full monitoring
            for i, step in enumerate(self.steps):
                step_result = await self._execute_step(step, context)
                results.append(step_result)
                
                # Update context with step result
                context[f"step_{step['id']}_result"] = step_result
                
                # Add to action history
                if "action_history" not in context:
                    context["action_history"] = []
                context["action_history"].append({
                    "step_id": step["id"],
                    "function": step["function"],
                    "success": step_result["success"]
                })
                
                # Stop execution if a step fails and we're in conscious mode
                if not step_result["success"] and conscious_execution:
                    success = False
                    break
        else:
            # Automatic execution - optimized chunked execution
            if self.is_chunked:
                # Get available chunks
                chunks = self._get_chunks()
                
                if self.chunk_selector:
                    # Context-aware chunk selection
                    prediction = self.chunk_selector.select_chunk(
                        available_chunks=chunks,
                        context=context,
                        procedure_domain=self.domain
                    )
                    
                    # Execute chunks based on prediction
                    executed_chunks = []
                    
                    # First execute most likely chunk
                    main_chunk_id = prediction.chunk_id
                    
                    if main_chunk_id in chunks:
                        chunk_steps = chunks[main_chunk_id]
                        chunk_result = await self._execute_chunk(
                            chunk_steps, 
                            context, 
                            minimal_monitoring=True,
                            chunk_id=main_chunk_id
                        )
                        results.extend(chunk_result["results"])
                        executed_chunks.append(main_chunk_id)
                        
                        if not chunk_result["success"]:
                            success = False
                    
                    # Execute remaining steps that weren't in chunks
                    remaining_steps = self._get_steps_not_in_chunks(executed_chunks)
                    
                    for step in remaining_steps:
                        step_result = await self._execute_step(step, context, minimal_monitoring=True)
                        results.append(step_result)
                        
                        if not step_result["success"]:
                            success = False
                            break
                else:
                    # Simple chunk execution without context awareness
                    for chunk_id, chunk_steps in chunks.items():
                        chunk_result = await self._execute_chunk(
                            chunk_steps, 
                            context, 
                            minimal_monitoring=True,
                            chunk_id=chunk_id
                        )
                        results.extend(chunk_result["results"])
                        
                        if not chunk_result["success"]:
                            success = False
                            break
            else:
                # No chunks yet, but still in automatic mode - execute with minimal monitoring
                for step in self.steps:
                    step_result = await self._execute_step(step, context, minimal_monitoring=True)
                    results.append(step_result)
                    
                    if not step_result["success"]:
                        success = False
                        break
        
        # Calculate overall execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Update overall statistics
        self._update_execution_stats(execution_time, success)
        
        # Check for opportunities to improve
        if self.execution_count % 5 == 0:  # Every 5 executions
            self._identify_refinement_opportunities(results)
        
        # Check for chunking opportunities
        if not self.is_chunked and self.proficiency > 0.7 and self.execution_count >= 10:
            self._identify_chunking_opportunities(results)
            
            # After chunking, try to generalize chunks
            if self.is_chunked and self.chunk_library:
                self._generalize_chunks()
        
        return {
            "success": success,
            "results": results,
            "execution_time": execution_time,
            "proficiency": self.proficiency,
            "automatic": not conscious_execution and self.proficiency >= 0.8,
            "chunked": self.is_chunked
        }
    
    async def _execute_step(self, 
                          step: Dict[str, Any], 
                          context: Dict[str, Any], 
                          minimal_monitoring: bool = False) -> Dict[str, Any]:
        """Execute a single step and update its statistics"""
        # Get the actual function to call
        func = self._function_registry.get(step["function"])
        if not func:
            return {
                "success": False,
                "error": f"Function {step['function']} not registered",
                "execution_time": 0.0
            }
        
        # Execute with timing
        step_start = datetime.datetime.now()
        try:
            # Prepare parameters with context
            params = step["parameters"].copy()
            if isinstance(func, Callable) and "context" in func.__code__.co_varnames:
                params["context"] = context
                
            # Execute the function
            result = await func(**params)
            
            # Check result format and standardize
            if isinstance(result, dict):
                success = result.get("error") is None
                step_result = {
                    "success": success,
                    "data": result,
                    "execution_time": 0.0
                }
                
                if not success:
                    step_result["error"] = result.get("error")
            else:
                step_result = {
                    "success": True,
                    "data": {"result": result},
                    "execution_time": 0.0
                }
        except Exception as e:
            logger.error(f"Error executing step {step['id']}: {str(e)}")
            step_result = {
                "success": False,
                "error": str(e),
                "execution_time": 0.0
            }
        
        # Calculate execution time
        step_time = (datetime.datetime.now() - step_start).total_seconds()
        step_result["execution_time"] = step_time
        
        # Update step statistics unless we're in minimal monitoring mode
        if not minimal_monitoring:
            self._update_step_stats(step, step_time, step_result["success"])
        
        return step_result
    
    async def _execute_chunk(self, 
                           steps: List[Dict[str, Any]], 
                           context: Dict[str, Any], 
                           minimal_monitoring: bool = False,
                           chunk_id: str = None) -> Dict[str, Any]:
        """Execute a chunk of steps as a unit"""
        results = []
        success = True
        start_time = datetime.datetime.now()
        
        # Create chunk-specific context
        chunk_context = context.copy()
        if chunk_id:
            chunk_context["current_chunk"] = chunk_id
        
        # Execute steps
        for step in steps:
            step_result = await self._execute_step(step, chunk_context, minimal_monitoring)
            results.append(step_result)
            
            if not step_result["success"]:
                success = False
                break
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Update chunk template if using library
        if self.chunk_library and chunk_id in self.generalized_chunks:
            template_id = self.generalized_chunks[chunk_id]
            self.chunk_library.update_template_success(
                template_id=template_id,
                domain=self.domain,
                success=success
            )
        
        return {
            "success": success,
            "results": results,
            "execution_time": execution_time
        }
    
    def _update_step_stats(self, step: Dict[str, Any], execution_time: float, success: bool):
        """Update statistics for a single step"""
        stats = step["execution_stats"]
        count = stats.get("execution_count", 0)
        
        # Update average time with exponential moving average
        if count == 0:
            stats["average_time"] = execution_time
        else:
            stats["average_time"] = (stats["average_time"] * 0.8) + (execution_time * 0.2)
        
        # Update success rate
        success_value = 1.0 if success else 0.0
        if count == 0:
            stats["success_rate"] = success_value
        else:
            stats["success_rate"] = (stats["success_rate"] * 0.9) + (success_value * 0.1)
        
        # Update counts
        stats["execution_count"] = count + 1
        if not success:
            stats["error_count"] = stats.get("error_count", 0) + 1
            
        stats["last_execution"] = datetime.datetime.now().isoformat()
    
    def _update_execution_stats(self, execution_time: float, success: bool):
        """Update overall execution statistics"""
        # Update average time
        if self.execution_count == 0:
            self.average_execution_time = execution_time
        else:
            self.average_execution_time = (self.average_execution_time * 0.8) + (execution_time * 0.2)
        
        # Update counts
        self.execution_count += 1
        if success:
            self.successful_executions += 1
        
        # Update proficiency based on multiple factors:
        # 1. Execution count (more practice = higher proficiency)
        # 2. Success rate
        # 3. Execution time stability
        
        # Calculate execution count factor (saturates at 100 executions)
        count_factor = min(self.execution_count / 100, 1.0)
        
        # Calculate success rate
        success_rate = self.successful_executions / max(1, self.execution_count)
        
        # Calculate time factor (consistency in execution time indicates proficiency)
        # This is simplified - real implementation would track time variance
        time_factor = 0.5  # Default middle value
        
        # Combine factors with weights
        self.proficiency = (count_factor * 0.3) + (success_rate * 0.5) + (time_factor * 0.2)
        
        # Update last execution timestamp
        self.last_execution = datetime.datetime.now().isoformat()
        self.last_updated = datetime.datetime.now().isoformat()
    
    def _identify_chunking_opportunities(self, recent_results: List[Dict[str, Any]]):
        """Look for opportunities to chunk steps together"""
        # Need at least 3 steps to consider chunking
        if len(self.steps) < 3:
            return
        
        # Find sequences of steps that always succeed together
        chunks = []
        current_chunk = []
        
        for i in range(len(self.steps) - 1):
            # Start a new potential chunk
            if not current_chunk:
                current_chunk = [self.steps[i]["id"]]
            
            # Check if next step is consistently executed after this one
            co_occurrence = self._calculate_co_occurrence(self.steps[i]["id"], self.steps[i+1]["id"])
            
            if co_occurrence > 0.9:  # High co-occurrence threshold
                # Add to current chunk
                current_chunk.append(self.steps[i+1]["id"])
            else:
                # End current chunk if it has multiple steps
                if len(current_chunk) > 1:
                    chunks.append(current_chunk)
                current_chunk = []
        
        # Add the last chunk if it exists
        if len(current_chunk) > 1:
            chunks.append(current_chunk)
        
        # Apply chunking if we found opportunities
        if chunks:
            self._apply_chunking(chunks)
    
    def _calculate_co_occurrence(self, step1_id: str, step2_id: str) -> float:
        """Calculate how often step2 follows step1 in successful executions"""
        # Get the steps
        step1 = next((s for s in self.steps if s["id"] == step1_id), None)
        step2 = next((s for s in self.steps if s["id"] == step2_id), None)
        
        if not step1 or not step2:
            return 0.0
            
        # Check step statistics
        step1_stats = step1.get("execution_stats", {})
        step2_stats = step2.get("execution_stats", {})
        
        # If both steps have high success rates, assume they co-occur
        step1_success = step1_stats.get("success_rate", 0)
        step2_success = step2_stats.get("success_rate", 0)
        
        # Check historical co-occurrence in context history
        actual_co_occurrences = 0
        possible_co_occurrences = 0
        
        for context in self.context_history:
            action_history = context.get("action_history", [])
            
            # Look for sequential occurrences
            for i in range(len(action_history) - 1):
                if action_history[i].get("step_id") == step1_id:
                    possible_co_occurrences += 1
                    
                    if i+1 < len(action_history) and action_history[i+1].get("step_id") == step2_id:
                        actual_co_occurrences += 1
        
        if possible_co_occurrences > 0:
            return actual_co_occurrences / possible_co_occurrences
        
        # Fallback: use success rates if no history
        if step1_success > 0.9 and step2_success > 0.9:
            return 0.95
            
        return 0.5  # Default moderate co-occurrence
    
    def _apply_chunking(self, chunks: List[List[str]]):
        """Apply identified chunks to the procedure"""
        # Record original step order
        original_step_ids = [step["id"] for step in self.steps]
        
        # Create chunks
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i+1}"
            self.chunked_steps[chunk_id] = chunk
            
            # Look for context patterns in history
            if self.chunk_selector:
                context_pattern = self.chunk_selector.create_context_pattern_from_history(
                    chunk_id=chunk_id,
                    domain=self.domain
                )
                
                if context_pattern:
                    # Store reference to context pattern
                    self.chunk_contexts[chunk_id] = context_pattern.id
        
        # Mark as chunked
        self.is_chunked = True
        
        logger.info(f"Applied chunking to procedure {self.name}: {self.chunked_steps}")
    
    def _get_chunks(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get the current chunks as step dictionaries"""
        chunks = {}
        
        for chunk_id, step_ids in self.chunked_steps.items():
            # Convert step IDs to actual step dictionaries
            steps = [next((s for s in self.steps if s["id"] == step_id), None) for step_id in step_ids]
            steps = [s for s in steps if s is not None]  # Remove None values
            
            chunks[chunk_id] = steps
            
        return chunks
    
    def _get_steps_not_in_chunks(self, executed_chunks: List[str]) -> List[Dict[str, Any]]:
        """Get steps that aren't in the specified chunks"""
        # Get all step IDs in executed chunks
        chunked_step_ids = set()
        for chunk_id in executed_chunks:
            if chunk_id in self.chunked_steps:
                chunked_step_ids.update(self.chunked_steps[chunk_id])
        
        # Return steps not in chunks
        return [step for step in self.steps if step["id"] not in chunked_step_ids]
    
    def _identify_refinement_opportunities(self, recent_results: List[Dict[str, Any]]):
        """Look for opportunities to refine the procedure"""
        # Skip if too few executions
        if self.execution_count < 5:
            return
        
        # Check for steps with low success rates
        for step in self.steps:
            stats = step.get("execution_stats", {})
            success_rate = stats.get("success_rate", 1.0)
            
            if success_rate < 0.8 and stats.get("execution_count", 0) >= 3:
                # This step needs improvement
                self.refinement_opportunities.append({
                    "step_id": step["id"],
                    "type": "improve_reliability",
                    "current_success_rate": success_rate,
                    "identified_at": datetime.datetime.now().isoformat(),
                    "description": f"Step '{step['description']}' has a low success rate of {success_rate:.2f}"
                })
        
        # Check for consistently slow steps
        step_times = [step.get("execution_stats", {}).get("average_time", 0) for step in self.steps]
        if step_times:
            avg_step_time = sum(step_times) / len(step_times)
            
            for step in self.steps:
                step_time = step.get("execution_stats", {}).get("average_time", 0)
                
                if step_time > avg_step_time * 2:
                    # This step is much slower than average
                    self.refinement_opportunities.append({
                        "step_id": step["id"],
                        "type": "optimize_performance",
                        "current_time": step_time,
                        "average_time": avg_step_time,
                        "identified_at": datetime.datetime.now().isoformat(),
                        "description": f"Step '{step['description']}' is significantly slower than average"
                    })
    
    def refine_step(self, 
                   step_id: str, 
                   new_function: Optional[Union[Callable, str]] = None, 
                   new_parameters: Optional[Dict[str, Any]] = None,
                   new_description: Optional[str] = None) -> bool:
        """Refine a specific step in the procedure"""
        # Find the step
        step = next((s for s in self.steps if s["id"] == step_id), None)
        if not step:
            return False
            
        # Update function if provided
        if new_function:
            if callable(new_function):
                func_name = new_function.__name__
                self.register_function(func_name, new_function)
                step["function"] = func_name
            else:
                step["function"] = new_function
            
        # Update parameters if provided
        if new_parameters:
            step["parameters"] = new_parameters
            
        # Update description if provided
        if new_description:
            step["description"] = new_description
            
        # Record refinement
        self.last_updated = datetime.datetime.now().isoformat()
        
        # If this step is part of a chunk, temporarily de-chunk
        self._unchunk_step(step_id)
        
        # Remove this step from refinement opportunities
        self.refinement_opportunities = [
            r for r in self.refinement_opportunities if r.get("step_id") != step_id
        ]
        
        logger.info(f"Refined step {step_id} in procedure {self.name}")
        return True
    
    def _unchunk_step(self, step_id: str):
        """Temporarily remove a step from any chunks it's in"""
        affected_chunks = []
        
        # Find chunks containing this step
        for chunk_id, step_ids in self.chunked_steps.items():
            if step_id in step_ids:
                affected_chunks.append(chunk_id)
                
        # If any chunks are affected, we need to reset chunking
        if affected_chunks:
            logger.info(f"Resetting chunks in procedure {self.name} due to refinement of step {step_id}")
            self.is_chunked = False
            self.chunked_steps = {}
            self.chunk_contexts = {}
            self.generalized_chunks = {}
    
    def add_step_after(self, 
                      existing_step_id: str, 
                      new_step_id: str, 
                      description: str, 
                      function: Union[Callable, str], 
                      parameters: Dict[str, Any] = None) -> bool:
        """Add a new step after an existing step"""
        # Find the index of the existing step
        existing_index = -1
        for i, step in enumerate(self.steps):
            if step["id"] == existing_step_id:
                existing_index = i
                break
                
        if existing_index == -1:
            return False
            
        # Create the new step
        if callable(function):
            func_name = function.__name__
            self.register_function(func_name, function)
        else:
            func_name = function
        
        new_step = {
            "id": new_step_id,
            "description": description,
            "function": func_name,
            "parameters": parameters or {},
            "execution_stats": {
                "average_time": 0.0,
                "success_rate": 0.0,
                "execution_count": 0,
                "last_execution": None,
                "error_count": 0
            }
        }
        
        # Insert the new step after the existing one
        self.steps.insert(existing_index + 1, new_step)
        
        # Reset chunking since we modified the step sequence
        self.is_chunked = False
        self.chunked_steps = {}
        self.chunk_contexts = {}
        self.generalized_chunks = {}
        
        # Update timestamp
        self.last_updated = datetime.datetime.now().isoformat()
        
        logger.info(f"Added new step {new_step_id} after {existing_step_id} in procedure {self.name}")
        return True
    
    def remove_step(self, step_id: str) -> bool:
        """Remove a step from the procedure"""
        # Find the step
        step_index = -1
        for i, step in enumerate(self.steps):
            if step["id"] == step_id:
                step_index = i
                break
                
        if step_index == -1:
            return False
            
        # Remove the step
        self.steps.pop(step_index)
        
        # Reset chunking since we modified the step sequence
        self.is_chunked = False
        self.chunked_steps = {}
        self.chunk_contexts = {}
        self.generalized_chunks = {}
        
        # Update timestamp
        self.last_updated = datetime.datetime.now().isoformat()
        
        logger.info(f"Removed step {step_id} from procedure {self.name}")
        return True
    
    def _record_context(self, context: Dict[str, Any]):
        """Record context for future reference"""
        self.context_history.append(context.copy())
        
        # Trim history if needed
        if len(self.context_history) > self.max_history:
            self.context_history = self.context_history[-self.max_history:]
    
    def _generalize_chunks(self):
        """Try to create generalizable templates from chunks"""
        if not self.chunk_library:
            return
            
        # Get chunks as steps
        chunks = self._get_chunks()
        
        for chunk_id, chunk_steps in chunks.items():
            # Skip if already generalized
            if chunk_id in self.generalized_chunks:
                continue
                
            # Try to create a template
            template = self.chunk_library.create_chunk_template_from_steps(
                chunk_id=f"template_{chunk_id}_{self.name}",
                name=f"{self.name} - {chunk_id}",
                steps=chunk_steps,
                domain=self.domain,
                success_rate=0.9  # High initial success rate in source domain
            )
            
            if template:
                # Store reference to template
                self.generalized_chunks[chunk_id] = template.id
                logger.info(f"Created generalized template {template.id} from chunk {chunk_id}")

# ============================================================================
# ENHANCED PROCEDURAL MEMORY MANAGER
# ============================================================================

class EnhancedProceduralMemoryManager:
    """Enhanced manager for procedural memory with context awareness and generalization"""
    
    def __init__(self, memory_core=None, knowledge_core=None):
        self.procedures = {}  # name -> ContextAwareProcedureSequence
        self.memory_core = memory_core
        self.knowledge_core = knowledge_core
        
        # Context awareness
        self.chunk_selector = ContextAwareChunkSelector()
        
        # Generalization
        self.chunk_library = ProceduralChunkLibrary()
        
        # Control mappings
        self._initialize_control_mappings()
        
        # Function registry
        self.function_registry = {}  # Global function registry
        
        # Transfer stats
        self.transfer_stats = {
            "total_transfers": 0,
            "successful_transfers": 0,
            "avg_success_level": 0.0,
            "avg_practice_needed": 0
        }
    
    def register_function(self, name: str, func: Callable):
        """Register a function for use in procedures"""
        self.function_registry[name] = func
    
    def _initialize_control_mappings(self):
        """Initialize common control mappings between domains"""
        # PlayStation to Xbox mappings
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="playstation",
            target_domain="xbox",
            action_type="primary_action",
            source_control="R1",
            target_control="RB"
        ))
        
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="playstation",
            target_domain="xbox",
            action_type="secondary_action",
            source_control="L1",
            target_control="LB"
        ))
        
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="playstation",
            target_domain="xbox",
            action_type="aim",
            source_control="L2",
            target_control="LT"
        ))
        
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="playstation",
            target_domain="xbox",
            action_type="shoot",
            source_control="R2",
            target_control="RT"
        ))
        
        # FPS game genre mappings
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="call_of_duty",
            target_domain="battlefield",
            action_type="aim",
            source_control="L2",
            target_control="L2"
        ))
        
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="call_of_duty",
            target_domain="battlefield",
            action_type="shoot",
            source_control="R2",
            target_control="R2"
        ))
        
        # Different genre mappings
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="dbd",  # Dead by Daylight
            target_domain="dbd",  # Default same-game mapping
            action_type="sprint",
            source_control="L1",
            target_control="L1"
        ))
        
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="dbd",
            target_domain="dbd",
            action_type="interaction",
            source_control="R1",
            target_control="R1"
        ))
    
    async def learn_procedure(self, 
                             name: str, 
                             steps: List[Dict[str, Any]], 
                             description: str = None,
                             domain: str = "general") -> ContextAwareProcedureSequence:
        """
        Learn a new procedure from a sequence of steps
        
        Args:
            name: Name of the procedure
            steps: List of step definitions
            description: Optional description
            domain: Domain for the procedure
            
        Returns:
            New procedure sequence
        """
        procedure = ContextAwareProcedureSequence(
            name=name, 
            description=description or f"Procedure for {name}",
            domain=domain,
            chunk_selector=self.chunk_selector,
            chunk_library=self.chunk_library
        )
        
        # Add the steps
        for i, step in enumerate(steps):
            step_id = step.get("id", f"step_{i+1}")
            step_desc = step.get("description", f"Step {i+1}")
            step_func = step.get("function")
            step_params = step.get("parameters", {})
            
            # Register function if it's callable
            if callable(step_func):
                func_name = step_func.__name__
                self.register_function(func_name, step_func)
                procedure.register_function(func_name, step_func)
                step_func = func_name
            elif isinstance(step_func, str) and step_func in self.function_registry:
                # If function name is provided and exists in registry, register it with procedure
                procedure.register_function(step_func, self.function_registry[step_func])
            
            procedure.add_step(
                step_id=step_id,
                description=step_desc,
                function=step_func,
                parameters=step_params
            )
        
        # Store the procedure
        self.procedures[name] = procedure
        
        # Store procedural knowledge in knowledge core if available
        if self.knowledge_core:
            await self.knowledge_core.add_knowledge(
                type="procedural",
                content={
                    "procedure_name": name,
                    "description": description or f"Procedure for {name}",
                    "steps_count": len(steps),
                    "steps_summary": [s.get("description", f"Step {i+1}") for i, s in enumerate(steps)],
                    "domain": domain
                },
                source="procedural_learning",
                confidence=0.7
            )
        
        # Create memory of procedure creation if memory core is available
        if self.memory_core:
            await self.memory_core.add_memory(
                memory_text=f"Learned a new procedural skill: {name} with {len(steps)} steps in the {domain} domain.",
                memory_type="procedural",
                memory_scope="system",
                significance=6,  # Significant event
                tags=["procedural", "learning", name, domain],
                metadata={
                    "procedure_name": name,
                    "procedure_id": procedure.id,
                    "domain": domain,
                    "steps_count": len(steps)
                }
            )
        
        logger.info(f"Learned new procedure '{name}' with {len(steps)} steps in {domain} domain")
        return procedure
    
    async def execute_procedure(self, 
                              name: str, 
                              context: Dict[str, Any] = None, 
                              force_conscious: bool = False) -> Dict[str, Any]:
        """
        Execute a procedure by name
        
        Args:
            name: Name of the procedure to execute
            context: Context data for execution
            force_conscious: Force conscious (deliberate) execution
            
        Returns:
            Execution results
        """
        if name not in self.procedures:
            return {"error": f"Procedure '{name}' not found"}
        
        procedure = self.procedures[name]
        
        # Add domain to context
        context = context or {}
        context["domain"] = procedure.domain
        
        # Determine execution mode
        automatic = procedure.proficiency > 0.8 and not force_conscious
        
        # Execute the procedure
        start_time = datetime.datetime.now()
        try:
            result = await procedure.execute(context, conscious_execution=not automatic)
        except Exception as e:
            logger.error(f"Error executing procedure {name}: {str(e)}")
            return {
                "error": str(e),
                "success": False,
                "execution_time": (datetime.datetime.now() - start_time).total_seconds()
            }
        
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Record execution in memory if available
        if self.memory_core and (not automatic or procedure.execution_count % 10 == 0):
            # Only log automatic executions occasionally
            try:
                await self.memory_core.add_memory(
                    memory_text=(
                        f"Executed procedure '{name}' with{' ' if result['success'] else ' not '}successful outcome. "
                        f"Current proficiency: {procedure.proficiency:.2f}"
                    ),
                    memory_type="procedural",
                    memory_scope="system",
                    significance=4 if result["success"] else 5,  # Failed executions are more significant
                    tags=["procedural", "execution", name, procedure.domain, 
                          "automatic" if automatic else "conscious"],
                    metadata={
                        "procedure_name": name,
                        "procedure_id": procedure.id,
                        "execution_time": execution_time,
                        "success": result["success"],
                        "proficiency": procedure.proficiency,
                        "automatic": automatic,
                        "execution_count": procedure.execution_count,
                        "domain": procedure.domain,
                        "chunked": procedure.is_chunked
                    }
                )
            except Exception as e:
                logger.error(f"Error storing execution memory: {str(e)}")
        
        # Update result with procedure details
        result["procedure_name"] = name
        result["procedure_id"] = procedure.id
        result["domain"] = procedure.domain
        result["execution_time"] = execution_time
        
        return result
    
    async def transfer_procedure(self, 
                               source_name: str, 
                               target_name: str, 
                               target_domain: str) -> Dict[str, Any]:
        """
        Transfer a procedure from one domain to another
        
        Args:
            source_name: Name of the source procedure
            target_name: Name for the new procedure
            target_domain: Domain for the new procedure
            
        Returns:
            Transfer results
        """
        if source_name not in self.procedures:
            return {"error": f"Source procedure '{source_name}' not found"}
        
        source = self.procedures[source_name]
        
        # Get chunks from source
        chunks = source._get_chunks() if source.is_chunked else {}
        
        # Find generalizable chunks
        transferable_chunks = {}
        steps_from_chunks = set()
        
        for chunk_id, chunk_steps in chunks.items():
            # Skip if not generalized
            if chunk_id not in source.generalized_chunks:
                continue
                
            template_id = source.generalized_chunks[chunk_id]
            
            # Try to map chunk to target domain
            mapped_steps = self.chunk_library.map_chunk_to_new_domain(
                template_id=template_id,
                target_domain=target_domain
            )
            
            if mapped_steps:
                transferable_chunks[chunk_id] = mapped_steps
                # Track which source steps are covered by chunks
                for step in chunk_steps:
                    steps_from_chunks.add(step["id"])
        
        # Get steps not in chunks
        steps_to_transfer = []
        
        for step in source.steps:
            if step["id"] not in steps_from_chunks:
                # Try to map individual step
                mapped_step = self._map_step_to_domain(
                    step=step,
                    source_domain=source.domain,
                    target_domain=target_domain
                )
                
                if mapped_step:
                    steps_to_transfer.append(mapped_step)
        
        # Combine steps from chunks and individual steps
        all_steps = []
        
        # First add any steps from the beginning that weren't in chunks
        for step in source.steps:
            if step["id"] not in steps_from_chunks:
                # Find mapped version
                mapped = next((s for s in steps_to_transfer if s.get("original_id") == step["id"]), None)
                if mapped:
                    all_steps.append(mapped)
        
        # Now add chunk steps in the right order
        if chunks:
            # Track which steps we've added
            added_steps = set(s.get("original_id") for s in all_steps)
            
            # Follow original step order
            for orig_step in source.steps:
                if orig_step["id"] in steps_from_chunks and orig_step["id"] not in added_steps:
                    # Find which chunk this step belongs to
                    for chunk_id, chunk_steps in chunks.items():
                        chunk_step_ids = [s["id"] for s in chunk_steps]
                        if orig_step["id"] in chunk_step_ids and chunk_id in transferable_chunks:
                            # Add all steps from this chunk in order
                            for mapped_step in transferable_chunks[chunk_id]:
                                all_steps.append(mapped_step)
                            
                            # Mark all steps in this chunk as added
                            added_steps.update(chunk_step_ids)
                            break
        
        # Create new procedure
        procedure = None
        if all_steps:
            try:
                procedure = await self.learn_procedure(
                    name=target_name,
                    steps=all_steps,
                    description=f"Transferred from {source_name} ({source.domain} to {target_domain})",
                    domain=target_domain
                )
                
                # Record transfer
                transfer_record = ProcedureTransferRecord(
                    source_procedure_id=source.id,
                    source_domain=source.domain,
                    target_procedure_id=procedure.id,
                    target_domain=target_domain,
                    transfer_date=datetime.datetime.now().isoformat(),
                    adaptation_steps=[{
                        "chunk_id": chunk_id,
                        "step_count": len(steps)
                    } for chunk_id, steps in transferable_chunks.items()],
                    success_level=0.8,  # Initial estimate
                    practice_needed=5  # Initial estimate
                )
                
                self.chunk_library.record_transfer(transfer_record)
                
                # Update transfer stats
                self.transfer_stats["total_transfers"] += 1
                self.transfer_stats["successful_transfers"] += 1
                
            except Exception as e:
                logger.error(f"Error creating transferred procedure: {str(e)}")
                return {
                    "error": str(e),
                    "success": False
                }
            
        # Create memory of transfer if memory core is available
        if self.memory_core and procedure:
            try:
                await self.memory_core.add_memory(
                    memory_text=(
                        f"Transferred procedural skill from '{source_name}' in {source.domain} domain "
                        f"to '{target_name}' in {target_domain} domain with {len(all_steps)} steps."
                    ),
                    memory_type="procedural",
                    memory_scope="system",
                    significance=7,  # Very significant learning event
                    tags=["procedural", "transfer", "learning", target_name, target_domain],
                    metadata={
                        "source_procedure": source_name,
                        "source_domain": source.domain,
                        "target_procedure": target_name,
                        "target_domain": target_domain,
                        "steps_count": len(all_steps),
                        "chunks_transferred": len(transferable_chunks)
                    }
                )
            except Exception as e:
                logger.error(f"Error storing transfer memory: {str(e)}")
        
        return {
            "success": procedure is not None,
            "source_name": source_name,
            "target_name": target_name,
            "source_domain": source.domain,
            "target_domain": target_domain,
            "steps_count": len(all_steps),
            "chunks_transferred": len(transferable_chunks),
            "procedure_id": procedure.id if procedure else None
        }
    
    def _map_step_to_domain(self, 
                           step: Dict[str, Any], 
                           source_domain: str, 
                           target_domain: str) -> Optional[Dict[str, Any]]:
        """Map a procedure step from one domain to another"""
        # Get original function and parameters
        function = step.get("function")
        params = step.get("parameters", {})
        
        if not function:
            return None
        
        # Try to find a control mapping
        mapped_params = params.copy()
        
        if "button" in params:
            button = params["button"]
            
            # Look for control mappings for common actions
            if "aim" in step.get("description", "").lower() or "look" in step.get("description", "").lower():
                mapping = self.chunk_library.get_control_mapping(
                    source_domain=source_domain,
                    target_domain=target_domain,
                    action_type="aim"
                )
                
                if mapping and mapping.source_control == button:
                    mapped_params["button"] = mapping.target_control
            
            elif "shoot" in step.get("description", "").lower() or "fire" in step.get("description", "").lower():
                mapping = self.chunk_library.get_control_mapping(
                    source_domain=source_domain,
                    target_domain=target_domain,
                    action_type="shoot"
                )
                
                if mapping and mapping.source_control == button:
                    mapped_params["button"] = mapping.target_control
            
            elif "sprint" in step.get("description", "").lower() or "run" in step.get("description", "").lower():
                mapping = self.chunk_library.get_control_mapping(
                    source_domain=source_domain,
                    target_domain=target_domain,
                    action_type="sprint"
                )
                
                if mapping and mapping.source_control == button:
                    mapped_params["button"] = mapping.target_control
            
            elif "interact" in step.get("description", "").lower() or "use" in step.get("description", "").lower():
                mapping = self.chunk_library.get_control_mapping(
                    source_domain=source_domain,
                    target_domain=target_domain,
                    action_type="interaction"
                )
                
                if mapping and mapping.source_control == button:
                    mapped_params["button"] = mapping.target_control
        
        # Create mapped step
        mapped_step = {
            "id": step["id"],
            "description": step["description"],
            "function": function,
            "parameters": mapped_params,
            "original_id": step["id"]
        }
        
        return mapped_step
    
    def get_procedure_proficiency(self, name: str) -> Dict[str, Any]:
        """
        Get the current proficiency level for a procedure
        
        Args:
            name: Name of the procedure
            
        Returns:
            Proficiency information
        """
        if name not in self.procedures:
            return {"error": f"Procedure '{name}' not found"}
        
        procedure = self.procedures[name]
        
        # Determine proficiency level
        proficiency_level = "novice"
        if procedure.proficiency >= 0.95:
            proficiency_level = "automatic"
        elif procedure.proficiency >= 0.8:
            proficiency_level = "expert"
        elif procedure.proficiency >= 0.5:
            proficiency_level = "competent"
        
        return {
            "procedure_name": name,
            "procedure_id": procedure.id,
            "proficiency": procedure.proficiency,
            "level": proficiency_level,
            "execution_count": procedure.execution_count,
            "success_rate": procedure.successful_executions / max(1, procedure.execution_count),
            "average_execution_time": procedure.average_execution_time,
            "is_chunked": procedure.is_chunked,
            "chunks_count": len(procedure.chunked_steps) if procedure.is_chunked else 0,
            "steps_count": len(procedure.steps),
            "last_execution": procedure.last_execution,
            "refinement_opportunities": len(procedure.refinement_opportunities),
            "domain": procedure.domain,
            "generalized_chunks": len(procedure.generalized_chunks) if hasattr(procedure, "generalized_chunks") else 0
        }
    
    async def list_procedures(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all procedures, optionally filtered by domain
        
        Args:
            domain: Optional domain to filter by
            
        Returns:
            List of procedure summaries
        """
        procedure_list = []
        
        for name, procedure in self.procedures.items():
            # Filter by domain if specified
            if domain and procedure.domain != domain:
                continue
                
            # Create summary
            procedure_list.append({
                "name": name,
                "id": procedure.id,
                "description": procedure.description,
                "domain": procedure.domain,
                "proficiency": procedure.proficiency,
                "proficiency_level": self.get_procedure_proficiency(name).get("level", "novice"),
                "steps_count": len(procedure.steps),
                "execution_count": procedure.execution_count,
                "is_chunked": procedure.is_chunked,
                "created_at": procedure.created_at,
                "last_execution": procedure.last_execution
            })
            
        # Sort by domain and then name
        procedure_list.sort(key=lambda x: (x["domain"], x["name"]))
        
        return procedure_list
    
    async def get_transfer_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about procedure transfers
        
        Returns:
            Transfer statistics
        """
        stats = self.transfer_stats.copy()
        
        # Get chunks by domain
        chunks_by_domain = {}
        for domain, chunk_ids in self.chunk_library.domain_chunks.items():
            chunks_by_domain[domain] = len(chunk_ids)
        
        # Get recent transfers
        recent_transfers = []
        for record in self.chunk_library.transfer_records[-5:]:
            recent_transfers.append({
                "source_domain": record.source_domain,
                "target_domain": record.target_domain,
                "transfer_date": record.transfer_date,
                "success_level": record.success_level
            })
        
        # Add additional stats
        stats["chunks_by_domain"] = chunks_by_domain
        stats["recent_transfers"] = recent_transfers
        stats["templates_count"] = len(self.chunk_library.chunk_templates)
        stats["actions_count"] = len(self.chunk_library.action_templates)
        
        return stats
    
    async def find_similar_procedures(self, name: str, target_domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find procedures similar to the specified one
        
        Args:
            name: Name of the procedure to compare
            target_domain: Optional domain to filter by
            
        Returns:
            List of similar procedures with similarity scores
        """
        if name not in self.procedures:
            return []
        
        source = self.procedures[name]
        source_chunks = source._get_chunks() if source.is_chunked else {}
        
        similar_procedures = []
        for proc_name, procedure in self.procedures.items():
            # Skip self
            if proc_name == name:
                continue
                
            # Filter by domain if specified
            if target_domain and procedure.domain != target_domain:
                continue
                
            # Calculate similarity
            similarity = self._calculate_procedure_similarity(source, procedure)
            
            if similarity > 0.3:  # Minimum similarity threshold
                similar_procedures.append({
                    "name": proc_name,
                    "id": procedure.id,
                    "domain": procedure.domain,
                    "similarity": similarity,
                    "steps_count": len(procedure.steps),
                    "proficiency": procedure.proficiency
                })
        
        # Sort by similarity
        similar_procedures.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similar_procedures
    
    def _calculate_procedure_similarity(self, proc1, proc2) -> float:
        """Calculate similarity between two procedures"""
        # If either doesn't have steps, return 0
        if not proc1.steps or not proc2.steps:
            return 0.0
        
        # If they have the same domain, higher base similarity
        domain_similarity = 0.3 if proc1.domain == proc2.domain else 0.0
        
        # Compare steps
        steps1 = [(s["function"], s.get("description", "")) for s in proc1.steps]
        steps2 = [(s["function"], s.get("description", "")) for s in proc2.steps]
        
        # Calculate Jaccard similarity on functions
        funcs1 = set(f for f, _ in steps1)
        funcs2 = set(f for f, _ in steps2)
        
        if not funcs1 or not funcs2:
            func_similarity = 0.0
        else:
            intersection = len(funcs1.intersection(funcs2))
            union = len(funcs1.union(funcs2))
            func_similarity = intersection / union
        
        # Calculate approximate sequence similarity
        step_similarity = 0.0
        if len(steps1) > 0 and len(steps2) > 0:
            # Simplified sequence comparison
            matched_steps = 0
            for i in range(min(len(steps1), len(steps2))):
                if steps1[i][0] == steps2[i][0]:
                    matched_steps += 1
            
            step_similarity = matched_steps / min(len(steps1), len(steps2))
        
        # Calculate final similarity
        return 0.3 * domain_similarity + 0.4 * func_similarity + 0.3 * step_similarity

# ============================================================================
# DEMONSTRATION
# ============================================================================

async def demonstrate_cross_game_transfer():
    """Demonstrate procedural memory with cross-game transfer"""
    
    # Create an enhanced procedural memory manager
    manager = EnhancedProceduralMemoryManager()
    
    # Define step functions for our Dead by Daylight example
    async def press_button(button: str, context: Dict[str, Any] = None):
        print(f"Pressing {button}")
        # Update context
        if context and button == "L1":
            context["sprinting"] = True
        return {"button": button, "pressed": True}
        
    async def approach_object(object_type: str, context: Dict[str, Any] = None):
        print(f"Approaching {object_type}")
        # Update context
        if context:
            context[f"near_{object_type}"] = True
        return {"object": object_type, "approached": True}
        
    async def check_surroundings(context: Dict[str, Any] = None):
        print(f"Checking surroundings")
        return {"surroundings_checked": True, "clear": True}
        
    async def vault_window(context: Dict[str, Any] = None):
        print(f"Vaulting through window")
        # Use context to see if we're sprinting
        sprinting = context.get("sprinting", False) if context else False
        return {"vaulted": True, "fast_vault": sprinting}
        
    async def work_on_generator(context: Dict[str, Any] = None):
        print(f"Working on generator")
        # Simulate a skill check
        skill_check_success = random.random() > 0.3  # 70% success rate
        return {"working_on_gen": True, "skill_check": skill_check_success}
    
    # Register functions
    manager.register_function("press_button", press_button)
    manager.register_function("approach_object", approach_object)
    manager.register_function("check_surroundings", check_surroundings)
    manager.register_function("vault_window", vault_window)
    manager.register_function("work_on_generator", work_on_generator)
    
    # Define steps for DBD window-generator procedure
    window_gen_steps = [
        {
            "id": "start_sprint",
            "description": "Start sprinting",
            "function": "press_button",
            "parameters": {"button": "L1"}
        },
        {
            "id": "approach_window",
            "description": "Approach the window",
            "function": "approach_object",
            "parameters": {"object_type": "window"}
        },
        {
            "id": "vault",
            "description": "Vault through the window",
            "function": "press_button",
            "parameters": {"button": "R1"}
        },
        {
            "id": "resume_sprint",
            "description": "Resume sprinting",
            "function": "press_button", 
            "parameters": {"button": "L1"}
        },
        {
            "id": "approach_gen",
            "description": "Approach the generator",
            "function": "approach_object",
            "parameters": {"object_type": "generator"}
        },
        {
            "id": "repair_gen",
            "description": "Work on the generator",
            "function": "work_on_generator",
            "parameters": {}
        }
    ]
    
    # Learn the procedure
    dbd_procedure = await manager.learn_procedure(
        name="window_to_generator",
        steps=window_gen_steps,
        description="Navigate through a window and start working on a generator",
        domain="dbd"  # Dead by Daylight
    )
    
    print("\nInitial Procedure:")
    print(f"Proficiency: {dbd_procedure.proficiency:.2f}")
    
    # Execute procedure multiple times
    print("\nPracticing procedure...")
    for i in range(15):
        print(f"\nExecution {i+1}:")
        context = {"sprinting": False}
        result = await manager.execute_procedure("window_to_generator", context)
        print(f"Success: {result['success']}, Time: {result['execution_time']:.4f}s, Proficiency: {dbd_procedure.proficiency:.2f}")
    
    # Check if chunking occurred
    print("\nAfter practice:")
    print(f"Proficiency: {dbd_procedure.proficiency:.2f}")
    print(f"Is chunked: {dbd_procedure.is_chunked}")
    if dbd_procedure.is_chunked:
        print("Chunks:")
        for chunk_id, step_ids in dbd_procedure.chunked_steps.items():
            print(f"  {chunk_id}: {step_ids}")
    
    # Now refine by adding a missing step
    print
