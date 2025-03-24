# nyx/core/procedural_memory.py

import logging
import asyncio
import datetime
import json
import math
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from collections import Counter
from pydantic import BaseModel, Field

# OpenAI Agents SDK imports
from agents import Agent, Runner, trace, function_tool, handoff, RunContextWrapper
from agents.exceptions import ModelBehaviorError, UserError

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

class Procedure(BaseModel):
    """A procedure to be executed"""
    id: str
    name: str
    description: str
    domain: str
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    execution_count: int = 0
    successful_executions: int = 0
    average_execution_time: float = 0.0
    proficiency: float = 0.0
    chunked_steps: Dict[str, List[str]] = Field(default_factory=dict)
    is_chunked: bool = False
    chunk_contexts: Dict[str, str] = Field(default_factory=dict)  # Maps chunk_id -> context pattern id
    generalized_chunks: Dict[str, str] = Field(default_factory=dict)  # Maps chunk_id -> template_id
    context_history: List[Dict[str, Any]] = Field(default_factory=list)
    refinement_opportunities: List[Dict[str, Any]] = Field(default_factory=list)
    optimization_history: List[Dict[str, Any]] = Field(default_factory=list)
    max_history: int = 50
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_execution: Optional[str] = None
    
class StepResult(BaseModel):
    """Result from executing a step"""
    success: bool
    error: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float = 0.0

class ProcedureStats(BaseModel):
    """Statistics about a procedure"""
    procedure_name: str
    procedure_id: str
    proficiency: float
    level: str
    execution_count: int
    success_rate: float
    average_execution_time: float
    is_chunked: bool
    chunks_count: int
    steps_count: int
    last_execution: Optional[str] = None
    domain: str
    generalized_chunks: int = 0
    refinement_opportunities: int = 0

class TransferStats(BaseModel):
    """Statistics about procedure transfers"""
    total_transfers: int = 0
    successful_transfers: int = 0
    avg_success_level: float = 0.0
    avg_practice_needed: int = 0
    chunks_by_domain: Dict[str, int] = Field(default_factory=dict)
    recent_transfers: List[Dict[str, Any]] = Field(default_factory=list)
    templates_count: int = 0
    actions_count: int = 0

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
        
        # Try to infer from function name and description
        if isinstance(function, str):
            function_lower = function.lower()
            
            # Locomotion actions
            if any(word in function_lower or word in description for word in 
                ["move", "walk", "run", "navigate", "approach", "go", "sprint"]):
                return "locomotion", "navigation"
                
            # Interaction actions
            if any(word in function_lower or word in description for word in 
                ["press", "click", "push", "interact", "use", "activate"]):
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
                
            # Target/select actions
            if any(word in function_lower or word in description for word in 
                ["select", "choose", "target", "aim", "focus", "look"]):
                return "selection", "targeting"
                
            # Acquisition actions
            if any(word in function_lower or word in description for word in 
                ["get", "pick", "grab", "take", "collect", "acquire"]):
                return "acquisition", "collection"
                
            # Communication actions
            if any(word in function_lower or word in description for word in 
                ["speak", "say", "ask", "tell", "communicate"]):
                return "communication", "information_exchange"
                
            # Cognitive actions
            if any(word in function_lower or word in description for word in 
                ["think", "decide", "analyze", "evaluate", "assess"]):
                return "cognition", "decision_making"
                
            # Creation actions
            if any(word in function_lower or word in description for word in 
                ["make", "build", "create", "construct", "compose"]):
                return "creation", "production"
        
        # Domain-specific inference
        if domain == "dbd":  # Dead by Daylight specific actions
            if "vault" in description:
                return "vault", "locomotion"
            if "generator" in description or "gen" in description:
                return "repair", "objective"
        
        # Default fallback - use a generic action type based on domain
        domain_action_types = {
            "cooking": "preparation",
            "driving": "operation",
            "gaming": "gameplay",
            "writing": "composition",
            "music": "performance",
            "programming": "coding",
            "sports": "movement"
        }
        
        return domain_action_types.get(domain, "generic_action"), "task_progress"
    
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
                    
                    for param_key, param_value in params.items():
                        # Check if this parameter is a control that needs mapping
                        if param_key in ["control", "button", "input_method"]:
                            if param_value == mapping.source_control:
                                params[param_key] = mapping.target_control
                
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
# FUNCTION TOOLS FOR PROCEDURAL MEMORY
# ============================================================================

@function_tool
async def add_procedure(
    ctx,
    name: str, 
    steps: List[Dict[str, Any]], 
    description: str = None,
    domain: str = "general"
) -> Dict[str, Any]:
    """
    Add a new procedure to the procedural memory system.
    
    Args:
        name: Name of the procedure
        steps: List of step definitions with function, description and parameters
        description: Optional description of what the procedure accomplishes
        domain: Domain/context where this procedure applies
        
    Returns:
        Information about the created procedure
    """
    manager = ctx.context
    
    # Generate a unique ID for the procedure
    procedure_id = f"proc_{int(datetime.datetime.now().timestamp())}_{random.randint(1000, 9999)}"
    
    # Create procedure object
    procedure = Procedure(
        id=procedure_id,
        name=name,
        description=description or f"Procedure for {name}",
        domain=domain,
        steps=steps,
        created_at=datetime.datetime.now().isoformat(),
        last_updated=datetime.datetime.now().isoformat()
    )
    
    # Register function names if needed
    for step in steps:
        function_name = step.get("function")
        if function_name and callable(function_name):
            # It's a callable, register it by name
            func_name = function_name.__name__
            manager.register_function(func_name, function_name)
            step["function"] = func_name
    
    # Store the procedure
    manager.procedures[name] = procedure
    
    # Create a trace for analytics
    with trace(workflow_name="add_procedure"):
        logger.info(f"Added new procedure '{name}' with {len(steps)} steps in {domain} domain")
    
    return {
        "procedure_id": procedure_id,
        "name": name,
        "domain": domain,
        "steps_count": len(steps)
    }

@function_tool
async def execute_procedure(
    ctx,
    name: str,
    context: Dict[str, Any] = None,
    force_conscious: bool = False
) -> Dict[str, Any]:
    """
    Execute a procedure by name
    
    Args:
        name: Name of the procedure to execute
        context: Context data for execution
        force_conscious: Force deliberate execution even if proficient
        
    Returns:
        Execution results including success and execution time
    """
    manager = ctx.context
    
    if name not in manager.procedures:
        return {"error": f"Procedure '{name}' not found"}
    
    procedure = manager.procedures[name]
    
    # Create execution trace
    with trace(workflow_name="execute_procedure"):
        # Execute the procedure
        result = await manager.execute_procedure_steps(procedure, context or {}, force_conscious)
        
        # Record execution context
        if hasattr(procedure, "context_history"):
            execution_context = (context or {}).copy()
            execution_context["timestamp"] = datetime.datetime.now().isoformat()
            execution_context["result"] = result["success"]
            execution_context["execution_time"] = result["execution_time"]
            
            if len(procedure.context_history) >= procedure.max_history:
                procedure.context_history = procedure.context_history[-(procedure.max_history-1):]
            procedure.context_history.append(execution_context)
    
    return result

@function_tool
async def transfer_procedure(
    ctx,
    source_name: str,
    target_name: str,
    target_domain: str
) -> Dict[str, Any]:
    """
    Transfer a procedure from one domain to another
    
    Args:
        source_name: Name of the source procedure
        target_name: Name for the new procedure
        target_domain: Domain for the new procedure
        
    Returns:
        Transfer results with adapted procedure details
    """
    manager = ctx.context
    
    if source_name not in manager.procedures:
        return {"error": f"Source procedure '{source_name}' not found"}
    
    source = manager.procedures[source_name]
    
    # Create a trace for the transfer operation
    with trace(workflow_name="transfer_procedure"):
        # Use the chunk library to map steps to the new domain
        mapped_steps = []
        
        for step in source.steps:
            # Try to map step to new domain
            mapped_step = manager.map_step_to_domain(
                step=step,
                source_domain=source.domain,
                target_domain=target_domain
            )
            
            if mapped_step:
                mapped_steps.append(mapped_step)
        
        if not mapped_steps:
            return {
                "success": False,
                "error": "Could not map any steps to the target domain"
            }
        
        # Create new procedure
        new_procedure = await add_procedure(
            ctx,
            name=target_name,
            steps=mapped_steps,
            description=f"Transferred from {source_name} ({source.domain} to {target_domain})",
            domain=target_domain
        )
        
        # Record transfer
        transfer_record = ProcedureTransferRecord(
            source_procedure_id=source.id,
            source_domain=source.domain,
            target_procedure_id=new_procedure["procedure_id"],
            target_domain=target_domain,
            transfer_date=datetime.datetime.now().isoformat(),
            success_level=0.8,  # Initial estimate
            practice_needed=5  # Initial estimate
        )
        
        manager.chunk_library.record_transfer(transfer_record)
        
        # Update transfer stats
        manager.transfer_stats["total_transfers"] += 1
        manager.transfer_stats["successful_transfers"] += 1
    
    return {
        "success": True,
        "source_name": source_name,
        "target_name": target_name,
        "source_domain": source.domain,
        "target_domain": target_domain,
        "steps_count": len(mapped_steps),
        "procedure_id": new_procedure["procedure_id"]
    }

@function_tool
async def get_procedure_proficiency(ctx, name: str) -> ProcedureStats:
    """
    Get the current proficiency level for a procedure
    
    Args:
        name: Name of the procedure
        
    Returns:
        Proficiency information including level and execution statistics
    """
    manager = ctx.context
    
    if name not in manager.procedures:
        raise UserError(f"Procedure '{name}' not found")
    
    procedure = manager.procedures[name]
    
    # Determine proficiency level
    proficiency_level = "novice"
    if procedure.proficiency >= 0.95:
        proficiency_level = "automatic"
    elif procedure.proficiency >= 0.8:
        proficiency_level = "expert"
    elif procedure.proficiency >= 0.5:
        proficiency_level = "competent"
    
    # Calculate success rate
    success_rate = procedure.successful_executions / max(1, procedure.execution_count)
    
    return ProcedureStats(
        procedure_name=name,
        procedure_id=procedure.id,
        proficiency=procedure.proficiency,
        level=proficiency_level,
        execution_count=procedure.execution_count,
        success_rate=success_rate,
        average_execution_time=procedure.average_execution_time,
        is_chunked=procedure.is_chunked,
        chunks_count=len(procedure.chunked_steps) if procedure.is_chunked else 0,
        steps_count=len(procedure.steps),
        last_execution=procedure.last_execution,
        domain=procedure.domain,
        generalized_chunks=len(procedure.generalized_chunks) if hasattr(procedure, "generalized_chunks") else 0,
        refinement_opportunities=len(procedure.refinement_opportunities) if hasattr(procedure, "refinement_opportunities") else 0
    )

@function_tool
async def list_procedures(ctx, domain: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all procedures, optionally filtered by domain
    
    Args:
        domain: Optional domain to filter by
        
    Returns:
        List of procedure summaries
    """
    manager = ctx.context
    
    procedure_list = []
    
    for name, procedure in manager.procedures.items():
        # Filter by domain if specified
        if domain and procedure.domain != domain:
            continue
            
        # Get proficiency level
        proficiency_level = "novice"
        if procedure.proficiency >= 0.95:
            proficiency_level = "automatic"
        elif procedure.proficiency >= 0.8:
            proficiency_level = "expert"
        elif procedure.proficiency >= 0.5:
            proficiency_level = "competent"
            
        # Create summary
        procedure_list.append({
            "name": name,
            "id": procedure.id,
            "description": procedure.description,
            "domain": procedure.domain,
            "proficiency": procedure.proficiency,
            "proficiency_level": proficiency_level,
            "steps_count": len(procedure.steps),
            "execution_count": procedure.execution_count,
            "is_chunked": procedure.is_chunked,
            "created_at": procedure.created_at,
            "last_updated": procedure.last_updated,
            "last_execution": procedure.last_execution
        })
        
    # Sort by domain and then name
    procedure_list.sort(key=lambda x: (x["domain"], x["name"]))
    
    return procedure_list

@function_tool
async def get_transfer_statistics(ctx) -> TransferStats:
    """
    Get statistics about procedure transfers
    
    Returns:
        Transfer statistics including success rates and domain coverage
    """
    manager = ctx.context
    
    # Get chunks by domain
    chunks_by_domain = {}
    for domain, chunk_ids in manager.chunk_library.domain_chunks.items():
        chunks_by_domain[domain] = len(chunk_ids)
    
    # Get recent transfers
    recent_transfers = []
    for record in manager.chunk_library.transfer_records[-5:]:
        recent_transfers.append({
            "source_domain": record.source_domain,
            "target_domain": record.target_domain,
            "transfer_date": record.transfer_date,
            "success_level": record.success_level
        })
    
    return TransferStats(
        total_transfers=manager.transfer_stats["total_transfers"],
        successful_transfers=manager.transfer_stats["successful_transfers"],
        avg_success_level=manager.transfer_stats.get("avg_success_level", 0.0),
        avg_practice_needed=manager.transfer_stats.get("avg_practice_needed", 0),
        chunks_by_domain=chunks_by_domain,
        recent_transfers=recent_transfers,
        templates_count=len(manager.chunk_library.chunk_templates),
        actions_count=len(manager.chunk_library.action_templates)
    )

@function_tool
async def identify_chunking_opportunities(ctx, procedure_name: str) -> Dict[str, Any]:
    """
    Identify opportunities to chunk steps in a procedure
    
    Args:
        procedure_name: Name of the procedure to analyze
        
    Returns:
        Potential chunks that could be formed
    """
    manager = ctx.context
    
    if procedure_name not in manager.procedures:
        raise UserError(f"Procedure '{procedure_name}' not found")
    
    procedure = manager.procedures[procedure_name]
    
    # Need at least 3 steps and some executions to consider chunking
    if len(procedure.steps) < 3 or procedure.execution_count < 5:
        return {
            "can_chunk": False,
            "reason": f"Need at least 3 steps and 5 executions (has {len(procedure.steps)} steps and {procedure.execution_count} executions)"
        }
    
    # Find sequences of steps that could be chunked
    chunks = []
    current_chunk = []
    
    for i in range(len(procedure.steps) - 1):
        # Start a new potential chunk
        if not current_chunk:
            current_chunk = [procedure.steps[i]["id"]]
        
        # Check if next step is consistently executed after this one
        co_occurrence = manager.calculate_step_co_occurrence(
            procedure,
            procedure.steps[i]["id"], 
            procedure.steps[i+1]["id"]
        )
        
        if co_occurrence > 0.8:  # High co-occurrence threshold
            # Add to current chunk
            current_chunk.append(procedure.steps[i+1]["id"])
        else:
            # End current chunk if it has multiple steps
            if len(current_chunk) > 1:
                chunks.append(current_chunk)
            current_chunk = []
    
    # Add the last chunk if it exists
    if len(current_chunk) > 1:
        chunks.append(current_chunk)
    
    return {
        "can_chunk": len(chunks) > 0,
        "potential_chunks": chunks,
        "chunk_count": len(chunks),
        "procedure_name": procedure_name
    }

@function_tool
async def apply_chunking(ctx, procedure_name: str) -> Dict[str, Any]:
    """
    Apply chunking to a procedure based on execution patterns
    
    Args:
        procedure_name: Name of the procedure to chunk
        
    Returns:
        Results of the chunking process
    """
    manager = ctx.context
    
    if procedure_name not in manager.procedures:
        raise UserError(f"Procedure '{procedure_name}' not found")
    
    procedure = manager.procedures[procedure_name]
    
    # Find chunking opportunities
    chunking_result = await identify_chunking_opportunities(ctx, procedure_name)
    
    if not chunking_result["can_chunk"]:
        return chunking_result
    
    # Apply chunks
    chunks = chunking_result["potential_chunks"]
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i+1}"
        procedure.chunked_steps[chunk_id] = chunk
        
        # Look for context patterns if chunk selector is available
        if hasattr(manager, "chunk_selector") and manager.chunk_selector:
            context_pattern = manager.chunk_selector.create_context_pattern_from_history(
                chunk_id=chunk_id,
                domain=procedure.domain
            )
            
            if context_pattern:
                # Store reference to context pattern
                procedure.chunk_contexts[chunk_id] = context_pattern.id
    
    # Mark as chunked
    procedure.is_chunked = True
    procedure.last_updated = datetime.datetime.now().isoformat()
    
    return {
        "success": True,
        "chunks_applied": len(chunks),
        "chunk_ids": list(procedure.chunked_steps.keys()),
        "procedure_name": procedure_name
    }

@function_tool
async def generalize_chunk_from_steps(
    ctx,
    chunk_name: str,
    procedure_name: str,
    step_ids: List[str],
    domain: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a generalizable chunk template from specific procedure steps
    
    Args:
        chunk_name: Name for the new chunk template
        procedure_name: Name of the procedure containing the steps
        step_ids: IDs of the steps to include in the chunk
        domain: Optional domain override (defaults to procedure's domain)
        
    Returns:
        Information about the created chunk template
    """
    manager = ctx.context
    
    if procedure_name not in manager.procedures:
        return {"error": f"Procedure '{procedure_name}' not found"}
    
    procedure = manager.procedures[procedure_name]
    
    # Get the steps by ID
    steps = []
    for step_id in step_ids:
        step = next((s for s in procedure.steps if s["id"] == step_id), None)
        if step:
            steps.append(step)
    
    if not steps:
        return {"error": "No valid steps found"}
    
    # Use the procedure's domain if not specified
    chunk_domain = domain or procedure.domain
    
    # Generate a unique template ID
    template_id = f"template_{chunk_name}_{int(datetime.datetime.now().timestamp())}"
    
    # Create the chunk template
    template = manager.chunk_library.create_chunk_template_from_steps(
        chunk_id=template_id,
        name=chunk_name,
        steps=steps,
        domain=chunk_domain
    )
    
    if not template:
        return {"error": "Failed to create chunk template"}
    
    # If these steps were already part of a chunk, store the template reference
    for chunk_id, step_list in procedure.chunked_steps.items():
        if all(step_id in step_list for step_id in step_ids):
            # This existing chunk contains all our steps
            procedure.generalized_chunks[chunk_id] = template.id
            break
    
    return {
        "template_id": template.id,
        "name": template.name,
        "domain": chunk_domain,
        "actions_count": len(template.actions),
        "can_transfer": True
    }

@function_tool
async def find_matching_chunks(
    ctx,
    procedure_name: str,
    target_domain: str
) -> Dict[str, Any]:
    """
    Find library chunks that match a procedure's steps for transfer
    
    Args:
        procedure_name: Name of the procedure to find chunks for
        target_domain: Target domain to transfer to
        
    Returns:
        List of matching chunks with similarity scores
    """
    manager = ctx.context
    
    if procedure_name not in manager.procedures:
        return {"error": f"Procedure '{procedure_name}' not found"}
    
    procedure = manager.procedures[procedure_name]
    
    # Find matching chunks
    matches = manager.chunk_library.find_matching_chunks(
        steps=procedure.steps,
        source_domain=procedure.domain,
        target_domain=target_domain
    )
    
    if not matches:
        return {
            "chunks_found": False,
            "message": "No matching chunks found for transfer"
        }
    
    return {
        "chunks_found": True,
        "matches": matches,
        "count": len(matches),
        "source_domain": procedure.domain,
        "target_domain": target_domain
    }

@function_tool
async def transfer_chunk(
    ctx,
    template_id: str,
    target_domain: str
) -> Dict[str, Any]:
    """
    Transfer a chunk template to a new domain
    
    Args:
        template_id: ID of the chunk template to transfer
        target_domain: Domain to transfer to
        
    Returns:
        Mapped steps for the new domain
    """
    manager = ctx.context
    
    # Map the chunk to the new domain
    mapped_steps = manager.chunk_library.map_chunk_to_new_domain(
        template_id=template_id,
        target_domain=target_domain
    )
    
    if not mapped_steps:
        return {
            "success": False,
            "error": "Failed to map chunk to new domain"
        }
    
    return {
        "success": True,
        "steps": mapped_steps,
        "steps_count": len(mapped_steps),
        "target_domain": target_domain
    }

@function_tool
async def transfer_with_chunking(
    ctx,
    source_name: str,
    target_name: str,
    target_domain: str
) -> Dict[str, Any]:
    """
    Transfer a procedure from one domain to another using chunk-level transfer
    
    Args:
        source_name: Name of the source procedure
        target_name: Name for the new procedure
        target_domain: Domain for the new procedure
        
    Returns:
        Transfer results
    """
    manager = ctx.context
    
    if source_name not in manager.procedures:
        return {"error": f"Source procedure '{source_name}' not found"}
    
    source = manager.procedures[source_name]
    
    # Get chunks from source if chunked
    steps_from_chunks = set()
    transferred_chunks = []
    all_steps = []
    
    if source.is_chunked:
        # Try to transfer each chunk
        for chunk_id, step_ids in source.chunked_steps.items():
            # Get chunk steps
            chunk_steps = [s for s in source.steps if s["id"] in step_ids]
            
            # Check if this chunk has a template
            if hasattr(source, "generalized_chunks") and chunk_id in source.generalized_chunks:
                template_id = source.generalized_chunks[chunk_id]
                
                # Map template to new domain
                mapped_steps = manager.chunk_library.map_chunk_to_new_domain(
                    template_id=template_id,
                    target_domain=target_domain
                )
                
                if mapped_steps:
                    # Add steps from this chunk
                    all_steps.extend(mapped_steps)
                    transferred_chunks.append({
                        "chunk_id": chunk_id,
                        "template_id": template_id,
                        "steps_count": len(mapped_steps)
                    })
                    
                    # Track which source steps were covered
                    steps_from_chunks.update(step_ids)
                    continue
            
            # Find matching templates
            matches = manager.chunk_library.find_matching_chunks(
                steps=chunk_steps,
                source_domain=source.domain,
                target_domain=target_domain
            )
            
            if matches:
                # Use best match
                best_match = matches[0]
                template_id = best_match["template_id"]
                
                # Map template to new domain
                mapped_steps = manager.chunk_library.map_chunk_to_new_domain(
                    template_id=template_id,
                    target_domain=target_domain
                )
                
                if mapped_steps:
                    # Add steps from this chunk
                    all_steps.extend(mapped_steps)
                    transferred_chunks.append({
                        "chunk_id": chunk_id,
                        "template_id": template_id,
                        "steps_count": len(mapped_steps)
                    })
                    
                    # Track which source steps were covered
                    steps_from_chunks.update(step_ids)
            
            else:
                # No matching templates, try to create one
                template = manager.chunk_library.create_chunk_template_from_steps(
                    chunk_id=f"template_{chunk_id}_{int(datetime.datetime.now().timestamp())}",
                    name=f"{source_name}_{chunk_id}",
                    steps=chunk_steps,
                    domain=source.domain
                )
                
                if template:
                    # Map to new domain
                    mapped_steps = manager.chunk_library.map_chunk_to_new_domain(
                        template_id=template.id,
                        target_domain=target_domain
                    )
                    
                    if mapped_steps:
                        all_steps.extend(mapped_steps)
                        transferred_chunks.append({
                            "chunk_id": chunk_id,
                            "template_id": template.id,
                            "steps_count": len(mapped_steps),
                            "newly_created": True
                        })
                        
                        # Track which source steps were covered
                        steps_from_chunks.update(step_ids)
    
    # Get remaining steps not covered by chunks
    remaining_steps = [s for s in source.steps if s["id"] not in steps_from_chunks]
    
    # Map remaining steps individually
    for step in remaining_steps:
        mapped_step = manager.map_step_to_domain(
            step=step,
            source_domain=source.domain,
            target_domain=target_domain
        )
        
        if mapped_step:
            all_steps.append(mapped_step)
    
    if not all_steps:
        return {
            "success": False,
            "error": "Could not map any steps or chunks to the target domain"
        }
    
    # Create new procedure
    new_procedure = await add_procedure(
        ctx,
        name=target_name,
        steps=all_steps,
        description=f"Transferred from {source_name} ({source.domain} to {target_domain})",
        domain=target_domain
    )
    
    # Record transfer
    transfer_record = ProcedureTransferRecord(
        source_procedure_id=source.id,
        source_domain=source.domain,
        target_procedure_id=new_procedure["procedure_id"],
        target_domain=target_domain,
        transfer_date=datetime.datetime.now().isoformat(),
        adaptation_steps=[{
            "chunk_id": info["chunk_id"],
            "template_id": info["template_id"],
            "steps_count": info["steps_count"]
        } for info in transferred_chunks],
        success_level=0.8,  # Initial estimate
        practice_needed=5  # Initial estimate
    )
    
    manager.chunk_library.record_transfer(transfer_record)
    
    # Update transfer stats
    manager.transfer_stats["total_transfers"] += 1
    manager.transfer_stats["successful_transfers"] += 1
    
    return {
        "success": True,
        "source_name": source_name,
        "target_name": target_name,
        "source_domain": source.domain,
        "target_domain": target_domain,
        "steps_count": len(all_steps),
        "chunks_transferred": len(transferred_chunks),
        "procedure_id": new_procedure["procedure_id"],
        "chunk_transfer_details": transferred_chunks
    }

@function_tool
async def find_similar_procedures(
    ctx,
    name: str, 
    target_domain: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Find procedures similar to the specified one
    
    Args:
        name: Name of the procedure to compare
        target_domain: Optional domain to filter by
        
    Returns:
        List of similar procedures with similarity scores
    """
    manager = ctx.context
    
    if name not in manager.procedures:
        return []
    
    source = manager.procedures[name]
    
    similar_procedures = []
    for proc_name, procedure in manager.procedures.items():
        # Skip self
        if proc_name == name:
            continue
            
        # Filter by domain if specified
        if target_domain and procedure.domain != target_domain:
            continue
            
        # Calculate similarity
        similarity = manager.calculate_procedure_similarity(source, procedure)
        
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

@function_tool
async def refine_step(
    ctx,
    procedure_name: str,
    step_id: str,
    new_function: Optional[str] = None,
    new_parameters: Optional[Dict[str, Any]] = None,
    new_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Refine a specific step in a procedure
    
    Args:
        procedure_name: Name of the procedure
        step_id: ID of the step to refine
        new_function: Optional new function name
        new_parameters: Optional new parameters
        new_description: Optional new description
        
    Returns:
        Result of the refinement
    """
    manager = ctx.context
    
    if procedure_name not in manager.procedures:
        return {"error": f"Procedure '{procedure_name}' not found"}
    
    procedure = manager.procedures[procedure_name]
    
    # Find the step
    step = None
    for s in procedure.steps:
        if s["id"] == step_id:
            step = s
            break
            
    if not step:
        return {"error": f"Step '{step_id}' not found in procedure '{procedure_name}'"}
        
    # Update function if provided
    if new_function:
        if callable(new_function):
            func_name = new_function.__name__
            manager.register_function(func_name, new_function)
            step["function"] = func_name
        else:
            step["function"] = new_function
        
    # Update parameters if provided
    if new_parameters:
        step["parameters"] = new_parameters
        
    # Update description if provided
    if new_description:
        step["description"] = new_description
        
    # Update procedure
    procedure.last_updated = datetime.datetime.now().isoformat()
    
    # If this step is part of a chunk, unchunk
    affected_chunks = []
    if procedure.is_chunked:
        # Find chunks containing this step
        for chunk_id, step_ids in procedure.chunked_steps.items():
            if step_id in step_ids:
                affected_chunks.append(chunk_id)
                
        # If any chunks are affected, reset chunking
        if affected_chunks:
            procedure.is_chunked = False
            procedure.chunked_steps = {}
            procedure.chunk_contexts = {}
            procedure.generalized_chunks = {}
    
    # Remove this step from refinement opportunities if it was there
    if hasattr(procedure, "refinement_opportunities"):
        procedure.refinement_opportunities = [
            r for r in procedure.refinement_opportunities if r.get("step_id") != step_id
        ]
    
    return {
        "success": True,
        "procedure_name": procedure_name,
        "step_id": step_id,
        "function_updated": new_function is not None,
        "parameters_updated": new_parameters is not None,
        "description_updated": new_description is not None,
        "chunking_reset": len(affected_chunks) > 0
    }

# ============================================================================
# PROCEDURAL MEMORY MANAGER WITH AGENTS INTEGRATION
# ============================================================================

class ProceduralMemoryManager:
    """
    Procedural memory system integrated with Agents SDK
    
    Manages procedural knowledge including learning, execution,
    chunking, and cross-domain transfer through agent-based architecture.
    """
    
    def __init__(self, memory_core=None, knowledge_core=None):
        self.procedures = {}  # name -> Procedure
        self.memory_core = memory_core
        self.knowledge_core = knowledge_core
        
        # Context awareness
        self.chunk_selector = ContextAwareChunkSelector()
        
        # Generalization
        self.chunk_library = ProceduralChunkLibrary()
        
        # Function registry
        self.function_registry = {}  # Global function registry
        
        # Transfer stats
        self.transfer_stats = {
            "total_transfers": 0,
            "successful_transfers": 0,
            "avg_success_level": 0.0,
            "avg_practice_needed": 0
        }
        
        # Initialize agents
        self._proc_manager_agent = self._create_manager_agent()
        self._proc_execution_agent = self._create_execution_agent()
        self._proc_analysis_agent = self._create_analysis_agent()
        
        # Initialize common control mappings
        self._initialize_control_mappings()
    
    def _create_manager_agent(self) -> Agent:
        """Create the main procedural memory manager agent"""
        return Agent(
            name="Procedural Memory Manager",
            instructions="""
            You are a procedural memory manager agent that handles the storage, retrieval,
            and execution of procedural knowledge. You help the system learn, optimize, and
            transfer procedural skills across domains.
            
            Your responsibilities include:
            - Managing procedural memory entries
            - Facilitating procedural skill transfer between domains
            - Tracking procedural memory statistics
            - Optimizing procedures through chunking and refinement
            
            You have enhanced capabilities for cross-domain transfer:
            - You can identify and transfer chunks of procedural knowledge
            - You can generalize procedures across different domains
            - You can find similar patterns across diverse procedural skills
            - You can optimize transfer through specialized chunk mapping
            """,
            tools=[
                add_procedure,
                execute_procedure,
                transfer_procedure,
                get_procedure_proficiency,
                list_procedures,
                get_transfer_statistics,
                apply_chunking,
                find_similar_procedures,
                refine_step,
                # Enhanced chunk-based transfer tools
                generalize_chunk_from_steps,
                find_matching_chunks,
                transfer_chunk,
                transfer_with_chunking
            ]
        )
    
    def _create_execution_agent(self) -> Agent:
        """Create the agent responsible for procedure execution"""
        return Agent(
            name="Procedure Execution Agent",
            instructions="""
            You are a procedure execution agent that carries out procedural skills.
            Your job is to execute procedures efficiently, adapting to context and
            making appropriate decisions during execution.
            
            Your responsibilities include:
            - Executing procedure steps in the correct order
            - Adapting to different execution contexts
            - Monitoring execution success and performance
            - Providing feedback on execution quality
            """,
            tools=[
                execute_procedure,
                function_tool(self.execute_step)
            ]
        )
    
    def _create_analysis_agent(self) -> Agent:
        """Create the agent responsible for procedure analysis"""
        return Agent(
            name="Procedure Analysis Agent",
            instructions="""
            You are a procedure analysis agent that examines procedural knowledge.
            Your job is to identify patterns, optimization opportunities, and
            potential transfers between domains.
            
            Your responsibilities include:
            - Identifying chunking opportunities
            - Finding similarities between procedures
            - Recommending procedure refinements
            - Analyzing procedure performance
            - Identifying generalizable chunks across domains
            - Evaluating transfer potential between domains
            - Finding optimal chunking strategies for transfer
            """,
            tools=[
                identify_chunking_opportunities,
                find_similar_procedures,
                get_procedure_proficiency,
                function_tool(self.analyze_execution_history),
                # Enhanced chunk analysis tools
                generalize_chunk_from_steps,
                find_matching_chunks
            ]
        )
    
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
        
        # Input method mappings (touch to mouse/keyboard)
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="touch_interface",
            target_domain="mouse_interface",
            action_type="select",
            source_control="tap",
            target_control="click"
        ))
        
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="voice_interface",
            target_domain="touch_interface",
            action_type="activate",
            source_control="speak_command",
            target_control="tap"
        ))
        
        # Cross-domain action mappings (driving to flying)
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="driving",
            target_domain="flying",
            action_type="accelerate",
            source_control="pedal_press",
            target_control="throttle_forward"
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
    
    def register_function(self, name: str, func: Callable):
        """Register a function for use in procedures"""
        self.function_registry[name] = func
    
    async def execute_procedure_steps(self, 
                                    procedure: Procedure, 
                                    context: Dict[str, Any], 
                                    conscious_execution: bool) -> Dict[str, Any]:
        """Execute the steps of a procedure"""
        start_time = datetime.datetime.now()
        results = []
        success = True
        
        # Record execution context
        execution_context = context.copy()
        execution_context["timestamp"] = start_time.isoformat()
        execution_context["conscious_execution"] = conscious_execution
        
        if hasattr(procedure, "context_history"):
            if len(procedure.context_history) >= procedure.max_history:
                procedure.context_history = procedure.context_history[-(procedure.max_history-1):]
            procedure.context_history.append(execution_context)
        
        # Execute in different modes based on proficiency and settings
        if conscious_execution or procedure.proficiency < 0.8:
            # Deliberate step-by-step execution
            for step in procedure.steps:
                step_result = await self.execute_step(step, context)
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
            # Automatic chunked execution if available
            if procedure.is_chunked:
                # Get available chunks
                chunks = self._get_chunks(procedure)
                
                if hasattr(self, "chunk_selector") and self.chunk_selector:
                    # Context-aware chunk selection
                    prediction = self.chunk_selector.select_chunk(
                        available_chunks=chunks,
                        context=context,
                        procedure_domain=procedure.domain
                    )
                    
                    # Execute chunks based on prediction
                    executed_chunks = []
                    
                    # First execute most likely chunk
                    main_chunk_id = prediction.chunk_id
                    
                    if main_chunk_id in chunks:
                        chunk_steps = chunks[main_chunk_id]
                        chunk_result = await self._execute_chunk(
                            chunk_steps=chunk_steps, 
                            context=context, 
                            minimal_monitoring=True,
                            chunk_id=main_chunk_id,
                            procedure=procedure
                        )
                        results.extend(chunk_result["results"])
                        executed_chunks.append(main_chunk_id)
                        
                        if not chunk_result["success"]:
                            success = False
                    
                    # Execute remaining steps that weren't in chunks
                    remaining_steps = self._get_steps_not_in_chunks(procedure, executed_chunks)
                    
                    for step in remaining_steps:
                        step_result = await self.execute_step(step, context, minimal_monitoring=True)
                        results.append(step_result)
                        
                        if not step_result["success"]:
                            success = False
                            break
                else:
                    # Simple chunk execution without context awareness
                    for chunk_id, chunk_steps in chunks.items():
                        chunk_result = await self._execute_chunk(
                            chunk_steps=chunk_steps, 
                            context=context, 
                            minimal_monitoring=True,
                            chunk_id=chunk_id,
                            procedure=procedure
                        )
                        results.extend(chunk_result["results"])
                        
                        if not chunk_result["success"]:
                            success = False
                            break
            else:
                # No chunks yet, but still in automatic mode - execute with minimal monitoring
                for step in procedure.steps:
                    step_result = await self.execute_step(step, context, minimal_monitoring=True)
                    results.append(step_result)
                    
                    if not step_result["success"]:
                        success = False
                        break
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Update overall statistics
        self.update_procedure_stats(procedure, execution_time, success)
        
        # Check for opportunities to improve
        if procedure.execution_count % 5 == 0:  # Every 5 executions
            self._identify_refinement_opportunities(procedure, results)
        
        # Check for chunking opportunities
        if not procedure.is_chunked and procedure.proficiency > 0.7 and procedure.execution_count >= 10:
            self._identify_chunking_opportunities(procedure, results)
            
            # After chunking, try to generalize chunks
            if procedure.is_chunked and hasattr(self, "chunk_library"):
                self._generalize_chunks(procedure)
        
        # Return execution results
        return {
            "success": success,
            "results": results,
            "execution_time": execution_time,
            "proficiency": procedure.proficiency,
            "automatic": not conscious_execution and procedure.proficiency >= 0.8,
            "chunked": procedure.is_chunked
        }
    
    async def execute_step(self, 
                         step: Dict[str, Any], 
                         context: Dict[str, Any], 
                         minimal_monitoring: bool = False) -> Dict[str, Any]:
        """Execute a single step of a procedure"""
        # Get the actual function to call
        func_name = step["function"]
        func = self.function_registry.get(func_name)
        
        if not func:
            return StepResult(
                success=False,
                error=f"Function {func_name} not registered",
                execution_time=0.0
            ).dict()
        
        # Execute with timing
        step_start = datetime.datetime.now()
        try:
            # Prepare parameters with context
            params = step.get("parameters", {}).copy()
            
            # Check if function accepts context parameter
            if callable(func) and hasattr(func, "__code__") and "context" in func.__code__.co_varnames:
                params["context"] = context
                
            # Execute the function
            result = await func(**params)
            
            # Check result format and standardize
            if isinstance(result, dict):
                success = "error" not in result
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
        
        return step_result
    
    async def _execute_chunk(self, 
                           chunk_steps: List[Dict[str, Any]], 
                           context: Dict[str, Any], 
                           minimal_monitoring: bool = False,
                           chunk_id: str = None,
                           procedure: Procedure = None) -> Dict[str, Any]:
        """Execute a chunk of steps as a unit"""
        results = []
        success = True
        start_time = datetime.datetime.now()
        
        # Create chunk-specific context
        chunk_context = context.copy()
        if chunk_id:
            chunk_context["current_chunk"] = chunk_id
        
        # Execute steps
        for step in chunk_steps:
            step_result = await self.execute_step(step, chunk_context, minimal_monitoring)
            results.append(step_result)
            
            if not step_result["success"]:
                success = False
                break
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Update chunk template if using library
        if hasattr(self, "chunk_library") and procedure and hasattr(procedure, "generalized_chunks") and chunk_id in procedure.generalized_chunks:
            template_id = procedure.generalized_chunks[chunk_id]
            self.chunk_library.update_template_success(
                template_id=template_id,
                domain=procedure.domain,
                success=success
            )
        
        return {
            "success": success,
            "results": results,
            "execution_time": execution_time
        }
    
    def update_procedure_stats(self, procedure: Procedure, execution_time: float, success: bool):
        """Update statistics for a procedure after execution"""
        # Update average time
        if procedure.execution_count == 0:
            procedure.average_execution_time = execution_time
        else:
            procedure.average_execution_time = (procedure.average_execution_time * 0.8) + (execution_time * 0.2)
        
        # Update counts
        procedure.execution_count += 1
        if success:
            procedure.successful_executions += 1
        
        # Update proficiency based on multiple factors
        count_factor = min(procedure.execution_count / 100, 1.0)
        success_rate = procedure.successful_executions / max(1, procedure.execution_count)
        
        # Calculate time factor (simplified)
        time_factor = 0.5
        
        # Combine factors with weights
        procedure.proficiency = (count_factor * 0.3) + (success_rate * 0.5) + (time_factor * 0.2)
        
        # Update last execution timestamp
        procedure.last_execution = datetime.datetime.now().isoformat()
        procedure.last_updated = datetime.datetime.now().isoformat()
    
    def _get_chunks(self, procedure: Procedure) -> Dict[str, List[Dict[str, Any]]]:
        """Get the current chunks as step dictionaries"""
        chunks = {}
        
        for chunk_id, step_ids in procedure.chunked_steps.items():
            # Convert step IDs to actual step dictionaries
            steps = [next((s for s in procedure.steps if s["id"] == step_id), None) for step_id in step_ids]
            steps = [s for s in steps if s is not None]  # Remove None values
            
            chunks[chunk_id] = steps
            
        return chunks
    
    def _get_steps_not_in_chunks(self, procedure: Procedure, executed_chunks: List[str]) -> List[Dict[str, Any]]:
        """Get steps that aren't in the specified chunks"""
        # Get all step IDs in executed chunks
        chunked_step_ids = set()
        for chunk_id in executed_chunks:
            if chunk_id in procedure.chunked_steps:
                chunked_step_ids.update(procedure.chunked_steps[chunk_id])
        
        # Return steps not in chunks
        return [step for step in procedure.steps if step["id"] not in chunked_step_ids]
    
    def _identify_chunking_opportunities(self, procedure: Procedure, recent_results: List[Dict[str, Any]]):
        """Look for opportunities to chunk steps together"""
        # Need at least 3 steps to consider chunking
        if len(procedure.steps) < 3:
            return
        
        # Find sequences of steps that always succeed together
        chunks = []
        current_chunk = []
        
        for i in range(len(procedure.steps) - 1):
            # Start a new potential chunk
            if not current_chunk:
                current_chunk = [procedure.steps[i]["id"]]
            
            # Check if next step is consistently executed after this one
            co_occurrence = self.calculate_step_co_occurrence(
                procedure,
                procedure.steps[i]["id"], 
                procedure.steps[i+1]["id"]
            )
            
            if co_occurrence > 0.9:  # High co-occurrence threshold
                # Add to current chunk
                current_chunk.append(procedure.steps[i+1]["id"])
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
            self._apply_chunking(procedure, chunks)
    
    def calculate_step_co_occurrence(self, procedure: Procedure, step1_id: str, step2_id: str) -> float:
        """Calculate how often step2 follows step1 in successful executions"""
        # Check if we have sufficient context history
        if hasattr(procedure, "context_history") and len(procedure.context_history) >= 5:
            # Check historical co-occurrence in context history
            actual_co_occurrences = 0
            possible_co_occurrences = 0
            
            for context in procedure.context_history:
                action_history = context.get("action_history", [])
                
                # Look for sequential occurrences
                for i in range(len(action_history) - 1):
                    if action_history[i].get("step_id") == step1_id:
                        possible_co_occurrences += 1
                        
                        if i+1 < len(action_history) and action_history[i+1].get("step_id") == step2_id:
                            actual_co_occurrences += 1
            
            if possible_co_occurrences > 0:
                return actual_co_occurrences / possible_co_occurrences
        
        # Fallback: use execution count as a proxy for co-occurrence
        # Higher execution count = more likely the steps have been executed together
        if procedure.execution_count > 5:
            return 0.8
        
        return 0.5  # Default moderate co-occurrence
    
    def _apply_chunking(self, procedure: Procedure, chunks: List[List[str]]):
        """Apply identified chunks to the procedure"""
        # Create chunks
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i+1}"
            procedure.chunked_steps[chunk_id] = chunk
            
            # Look for context patterns in history
            if hasattr(self, "chunk_selector") and self.chunk_selector:
                context_pattern = self.chunk_selector.create_context_pattern_from_history(
                    chunk_id=chunk_id,
                    domain=procedure.domain
                )
                
                if context_pattern:
                    # Store reference to context pattern
                    procedure.chunk_contexts[chunk_id] = context_pattern.id
        
        # Mark as chunked
        procedure.is_chunked = True
        
        logger.info(f"Applied chunking to procedure {procedure.name}: {procedure.chunked_steps}")
    
    def _identify_refinement_opportunities(self, procedure: Procedure, recent_results: List[Dict[str, Any]]):
        """Look for opportunities to refine the procedure"""
        # Skip if too few executions
        if procedure.execution_count < 5:
            return
        
        # Check for steps with low success rates
        step_success_rates = {}
        for i, step in enumerate(procedure.steps):
            # Try to find success rate in recent results
            if i < len(recent_results):
                success = recent_results[i].get("success", False)
                
                # Initialize if not exists
                if step["id"] not in step_success_rates:
                    step_success_rates[step["id"]] = {"successes": 0, "total": 0}
                
                # Update counts
                step_success_rates[step["id"]]["total"] += 1
                if success:
                    step_success_rates[step["id"]]["successes"] += 1
        
        # Check for low success rates
        for step_id, stats in step_success_rates.items():
            if stats["total"] >= 3:  # Only consider steps executed at least 3 times
                success_rate = stats["successes"] / stats["total"]
                
                if success_rate < 0.8:
                    # This step needs improvement
                    step = next((s for s in procedure.steps if s["id"] == step_id), None)
                    if step:
                        # Create refinement opportunity
                        new_opportunity = {
                            "step_id": step_id,
                            "type": "improve_reliability",
                            "current_success_rate": success_rate,
                            "identified_at": datetime.datetime.now().isoformat(),
                            "description": f"Step '{step.get('description', step_id)}' has a low success rate of {success_rate:.2f}"
                        }
                        
                        # Add to opportunities if not already present
                        if not any(r.get("step_id") == step_id and r.get("type") == "improve_reliability" 
                                for r in procedure.refinement_opportunities):
                            procedure.refinement_opportunities.append(new_opportunity)
    
    def _generalize_chunks(self, procedure: Procedure):
        """Try to create generalizable templates from chunks"""
        if not hasattr(self, "chunk_library") or not self.chunk_library:
            return
            
        # Skip if not chunked
        if not procedure.is_chunked:
            return
            
        # Get chunks as steps
        chunks = self._get_chunks(procedure)
        
        for chunk_id, chunk_steps in chunks.items():
            # Skip if already generalized
            if hasattr(procedure, "generalized_chunks") and chunk_id in procedure.generalized_chunks:
                continue
                
            # Try to create a template
            template = self.chunk_library.create_chunk_template_from_steps(
                chunk_id=f"template_{chunk_id}_{procedure.name}",
                name=f"{procedure.name} - {chunk_id}",
                steps=chunk_steps,
                domain=procedure.domain,
                success_rate=0.9  # High initial success rate in source domain
            )
            
            if template:
                # Store reference to template
                procedure.generalized_chunks[chunk_id] = template.id
                logger.info(f"Created generalized template {template.id} from chunk {chunk_id}")
    
    def calculate_procedure_similarity(self, proc1: Procedure, proc2: Procedure) -> float:
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
    
    def map_step_to_domain(self, step: Dict[str, Any], source_domain: str, target_domain: str) -> Optional[Dict[str, Any]]:
        """Map a procedure step from one domain to another"""
        # Get original function and parameters
        function = step.get("function")
        params = step.get("parameters", {})
        
        if not function:
            return None
        
        # Try to find a control mapping
        mapped_params = params.copy()
        
        # Check for control-like parameters that might need mapping
        param_keys = ["button", "control", "input_method", "key"]
        
        for param_key in param_keys:
            if param_key in params:
                control_value = params[param_key]
                
                # Look for control mappings for this action type
                for mapping in self.chunk_library.control_mappings:
                    if (mapping.source_domain == source_domain and 
                        mapping.target_domain == target_domain and 
                        mapping.source_control == control_value):
                        # Found a mapping, apply it
                        mapped_params[param_key] = mapping.target_control
                        break
        
        # Create mapped step
        mapped_step = {
            "id": step["id"],
            "description": step["description"],
            "function": function,
            "parameters": mapped_params,
            "original_id": step["id"]
        }
        
        return mapped_step
    
    async def analyze_execution_history(self, procedure_name: str) -> Dict[str, Any]:
        """Analyze execution history of a procedure for patterns"""
        if procedure_name not in self.procedures:
            return {"error": f"Procedure '{procedure_name}' not found"}
        
        procedure = self.procedures[procedure_name]
        
        # Skip if insufficient execution history
        if procedure.execution_count < 3:
            return {
                "procedure_name": procedure_name,
                "executions": procedure.execution_count,
                "analysis": "Insufficient execution history for analysis"
            }
        
        # Analyze context history if available
        context_patterns = []
        if hasattr(procedure, "context_history") and len(procedure.context_history) >= 3:
            # Look for common context indicators
            context_keys = set()
            for context in procedure.context_history:
                context_keys.update(context.keys())
            
            # Filter out standard keys
            standard_keys = {"timestamp", "conscious_execution", "result", "execution_time", "action_history"}
            context_keys = context_keys - standard_keys
            
            # Analyze values for each key
            for key in context_keys:
                values = [context.get(key) for context in procedure.context_history if key in context]
                if len(values) >= 3:  # Need at least 3 occurrences
                    # Check consistency
                    unique_values = set(str(v) for v in values)
                    if len(unique_values) == 1:
                        # Consistent value
                        context_patterns.append({
                            "key": key,
                            "value": values[0],
                            "occurrences": len(values),
                            "pattern_type": "consistent_value"
                        })
                    elif len(unique_values) <= len(values) / 2:
                        # Semi-consistent values
                        value_counts = {}
                        for v in values:
                            v_str = str(v)
                            if v_str not in value_counts:
                                value_counts[v_str] = 0
                            value_counts[v_str] += 1
                        
                        # Find most common value
                        most_common = max(value_counts.items(), key=lambda x: x[1])
                        
                        context_patterns.append({
                            "key": key,
                            "most_common_value": most_common[0],
                            "occurrence_rate": most_common[1] / len(values),
                            "pattern_type": "common_value"
                        })
        
        # Analyze successful vs. unsuccessful executions
        success_patterns = []
        if hasattr(procedure, "context_history") and len(procedure.context_history) >= 3:
            successful_contexts = [ctx for ctx in procedure.context_history if ctx.get("result", False)]
            unsuccessful_contexts = [ctx for ctx in procedure.context_history if not ctx.get("result", True)]
            
            if successful_contexts and unsuccessful_contexts:
                # Find keys that differ between successful and unsuccessful executions
                for key in context_keys:
                    # Get values for successful executions
                    success_values = [context.get(key) for context in successful_contexts if key in context]
                    if not success_values:
                        continue
                        
                    # Get values for unsuccessful executions
                    failure_values = [context.get(key) for context in unsuccessful_contexts if key in context]
                    if not failure_values:
                        continue
                    
                    # Check if values are consistently different
                    success_unique = set(str(v) for v in success_values)
                    failure_unique = set(str(v) for v in failure_values)
                    
                    # If no overlap, this might be a discriminating factor
                    if not success_unique.intersection(failure_unique):
                        success_patterns.append({
                            "key": key,
                            "success_values": list(success_unique),
                            "failure_values": list(failure_unique),
                            "pattern_type": "success_factor"
                        })
        
        # Analyze chunks if available
        chunk_patterns = []
        if procedure.is_chunked:
            for chunk_id, step_ids in procedure.chunked_steps.items():
                chunk_patterns.append({
                    "chunk_id": chunk_id,
                    "step_count": len(step_ids),
                    "has_template": chunk_id in procedure.generalized_chunks if hasattr(procedure, "generalized_chunks") else False,
                    "has_context_pattern": chunk_id in procedure.chunk_contexts if hasattr(procedure, "chunk_contexts") else False
                })
        
        return {
            "procedure_name": procedure_name,
            "executions": procedure.execution_count,
            "success_rate": procedure.successful_executions / max(1, procedure.execution_count),
            "avg_execution_time": procedure.average_execution_time,
            "proficiency": procedure.proficiency,
            "is_chunked": procedure.is_chunked,
            "chunks_count": len(procedure.chunked_steps) if procedure.is_chunked else 0,
            "context_patterns": context_patterns,
            "success_patterns": success_patterns,
            "chunk_patterns": chunk_patterns,
            "refinement_opportunities": len(procedure.refinement_opportunities) if hasattr(procedure, "refinement_opportunities") else 0
        }
    
    async def get_manager_agent(self) -> Agent:
        """Get the procedural memory manager agent"""
        return self._proc_manager_agent
    
    async def get_execution_agent(self) -> Agent:
        """Get the procedure execution agent"""
        return self._proc_execution_agent
    
    async def get_analysis_agent(self) -> Agent:
        """Get the procedure analysis agent"""
        return self._proc_analysis_agent

    # Add these methods to the ProceduralMemoryManager class
    
    async def adaptive_execute_step(self, 
                                  step: Dict[str, Any], 
                                  context: Dict[str, Any], 
                                  alternatives: int = 3) -> Dict[str, Any]:
        """
        Execute a step with automatic adaptation if it fails
        
        Args:
            step: Step definition to execute
            context: Execution context
            alternatives: Number of alternative implementations to try
            
        Returns:
            Execution result
        """
        # Try the original mapping first
        result = await self.execute_step(step, context)
        
        # If successful, return the result
        if result["success"]:
            return result
        
        # Record the original failure for debugging
        original_failure = {
            "step_id": step["id"],
            "function": step["function"],
            "parameters": step.get("parameters", {}),
            "error": result.get("error")
        }
        
        # If failed, try alternative mappings
        for i in range(alternatives):
            # Generate alternative mapping
            alt_step = self._generate_alternative_mapping(step, context, i)
            
            # Skip if couldn't generate alternative
            if not alt_step:
                continue
                
            # Execute the alternative
            alt_result = await self.execute_step(alt_step, context)
            
            # Add information about the alternative
            alt_result["alternative_used"] = i + 1
            alt_result["original_failure"] = original_failure
            alt_result["alternative_mapping"] = {
                "original_function": step["function"],
                "alternative_function": alt_step["function"],
                "original_parameters": step.get("parameters", {}),
                "alternative_parameters": alt_step.get("parameters", {})
            }
            
            if alt_result["success"]:
                # Record successful alternative for future use
                self._record_successful_alternative(step, alt_step)
                
                # Annotate with adaptation information
                alt_result["adaptation"] = {
                    "original_step": step,
                    "adapted_step": alt_step,
                    "adaptation_method": f"alternative_{i+1}"
                }
                
                return alt_result
        
        # All alternatives failed, return the original failure
        result["alternatives_tried"] = alternatives
        return result
    
    def _generate_alternative_mapping(self, step: Dict[str, Any], context: Dict[str, Any], attempt: int) -> Optional[Dict[str, Any]]:
        """
        Generate an alternative mapping for a failed step
        
        Args:
            step: Original step that failed
            context: Execution context
            attempt: Which attempt this is (0, 1, 2, etc.)
            
        Returns:
            Alternative step implementation or None if no alternative available
        """
        # Get the original function and parameters
        original_function = step["function"]
        original_params = step.get("parameters", {}).copy()
        
        # Make a copy of the step
        alt_step = step.copy()
        alt_step["parameters"] = original_params.copy()
        
        # Different alternatives based on attempt number
        if attempt == 0:
            # First attempt: Try equivalent function if available
            function_equivalents = self._get_function_equivalents(original_function)
            if function_equivalents:
                # Use the first equivalent function
                alt_step["function"] = function_equivalents[0]
                alt_step["description"] = f"Alternative to {step.get('description', step['id'])} using {function_equivalents[0]}"
                return alt_step
        
        elif attempt == 1:
            # Second attempt: Try parameter variation
            alt_params = self._generate_parameter_variants(original_params, context)
            if alt_params:
                alt_step["parameters"] = alt_params
                alt_step["description"] = f"Alternative to {step.get('description', step['id'])} with modified parameters"
                return alt_step
        
        elif attempt == 2:
            # Third attempt: Try decomposition into simpler steps
            # For simplicity, just trying parameters + function substitution
            function_equivalents = self._get_function_equivalents(original_function)
            alt_params = self._generate_parameter_variants(original_params, context)
            
            if function_equivalents and alt_params:
                alt_step["function"] = function_equivalents[0]
                alt_step["parameters"] = alt_params
                alt_step["description"] = f"Alternative to {step.get('description', step['id'])} with substitute function and parameters"
                return alt_step
        
        # No viable alternative for this attempt
        return None
    
    def _get_function_equivalents(self, function_name: str) -> List[str]:
        """
        Get equivalent functions for a given function
        
        Args:
            function_name: Original function name
            
        Returns:
            List of equivalent function names
        """
        # Define equivalence groups
        equivalence_groups = {
            "navigate_to": ["move_to", "goto", "approach", "travel_to"],
            "select_item": ["choose_item", "click_item", "pick_item", "select_object"],
            "press_button": ["click_button", "push_button", "activate_button", "trigger_button"],
            "perform_action": ["do_action", "execute_action", "trigger_action", "run_action"],
            "check_condition": ["test_condition", "evaluate_condition", "verify_condition"]
        }
        
        # Add reverse mappings
        all_equivalents = {}
        for primary, equivalents in equivalence_groups.items():
            all_equivalents[primary] = equivalents
            for equiv in equivalents:
                if equiv not in all_equivalents:
                    all_equivalents[equiv] = [primary]
        
        # Return equivalents if found, empty list otherwise
        return all_equivalents.get(function_name, [])
    
    def _generate_parameter_variants(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate parameter variations for a step
        
        Args:
            params: Original parameters
            context: Execution context
            
        Returns:
            Modified parameter dictionary
        """
        # Make a copy of the parameters
        variant_params = params.copy()
        
        # Check for common parameter types and generate variants
        parameter_variants = {
            "button": self._get_button_variants,
            "control": self._get_control_variants,
            "object_type": self._get_object_variants,
            "location": self._get_location_variants,
            "method": self._get_method_variants
        }
        
        # Apply variants for each parameter type
        for param_key, variant_func in parameter_variants.items():
            if param_key in params:
                # Get variants for this parameter
                variants = variant_func(params[param_key], context)
                
                # Use the first variant
                if variants:
                    variant_params[param_key] = variants[0]
        
        # Only return if different from original
        if variant_params != params:
            return variant_params
        
        return params  # No variants generated
    
    def _get_button_variants(self, button: str, context: Dict[str, Any]) -> List[str]:
        """Get variant button mappings"""
        button_variants = {
            # Playstation to Xbox mappings
            "X": ["A", "Cross"],
            "O": ["B", "Circle"],
            "Square": ["X", "Box"],
            "Triangle": ["Y", "Pyramid"],
            "R1": ["RB", "R", "R Bumper"],
            "R2": ["RT", "R Trigger"],
            "L1": ["LB", "L", "L Bumper"],
            "L2": ["LT", "L Trigger"],
            # Touch to mouse mappings
            "tap": ["click", "press", "select"],
            "swipe": ["drag", "slide", "move"],
            "pinch": ["zoom", "scale"],
            # Generic variations
            "confirm": ["accept", "yes", "ok"],
            "cancel": ["back", "no", "escape"]
        }
        
        # Add reverse mappings
        all_variants = {}
        for primary, variants in button_variants.items():
            all_variants[primary] = variants
            for variant in variants:
                if variant not in all_variants:
                    all_variants[variant] = [primary]
        
        return all_variants.get(button, [])
    
    def _get_control_variants(self, control: str, context: Dict[str, Any]) -> List[str]:
        """Get variant control method mappings"""
        # Similar to button variants but for general control methods
        control_variants = {
            "touch": ["click", "tap", "press"],
            "mouse": ["cursor", "pointer"],
            "keyboard": ["keys", "typing"],
            "gamepad": ["controller", "joypad", "joystick"],
            "voice": ["speech", "audio"],
            "gesture": ["motion", "movement"]
        }
        
        # Add reverse mappings
        all_variants = {}
        for primary, variants in control_variants.items():
            all_variants[primary] = variants
            for variant in variants:
                if variant not in all_variants:
                    all_variants[variant] = [primary]
        
        return all_variants.get(control, [])
    
    def _get_object_variants(self, object_type: str, context: Dict[str, Any]) -> List[str]:
        """Get variant object type mappings"""
        object_variants = {
            "window": ["opening", "gap", "passage"],
            "generator": ["gen", "machine", "power source"],
            "door": ["entrance", "exit", "gate"],
            "item": ["object", "thing", "element"],
            "button": ["control", "switch", "trigger"]
        }
        
        # Add reverse mappings
        all_variants = {}
        for primary, variants in object_variants.items():
            all_variants[primary] = variants
            for variant in variants:
                if variant not in all_variants:
                    all_variants[variant] = [primary]
        
        return all_variants.get(object_type, [])
    
    def _get_location_variants(self, location: str, context: Dict[str, Any]) -> List[str]:
        """Get variant location mappings"""
        location_variants = {
            "main_menu": ["menu", "home screen", "start screen"],
            "settings": ["options", "preferences", "configuration"],
            "inventory": ["items", "backpack", "storage"],
            "map": ["world map", "overview", "terrain view"]
        }
        
        # Add reverse mappings
        all_variants = {}
        for primary, variants in location_variants.items():
            all_variants[primary] = variants
            for variant in variants:
                if variant not in all_variants:
                    all_variants[variant] = [primary]
        
        return all_variants.get(location, [])
    
    def _get_method_variants(self, method: str, context: Dict[str, Any]) -> List[str]:
        """Get variant method mappings"""
        method_variants = {
            "swipe": ["drag", "slide", "flick"],
            "click": ["tap", "press", "select"],
            "type": ["enter", "input", "key in"],
            "speak": ["say", "voice", "dictate"]
        }
        
        # Add reverse mappings
        all_variants = {}
        for primary, variants in method_variants.items():
            all_variants[primary] = variants
            for variant in variants:
                if variant not in all_variants:
                    all_variants[variant] = [primary]
        
        return all_variants.get(method, [])
    
    def _record_successful_alternative(self, original_step: Dict[str, Any], successful_alt: Dict[str, Any]) -> None:
        """
        Record a successful alternative for future reference
        
        Args:
            original_step: Original step that failed
            successful_alt: Alternative step that succeeded
        """
        # Initialize successful alternatives storage if not exists
        if not hasattr(self, "_successful_alternatives"):
            self._successful_alternatives = {}
        
        # Create a key for this step
        step_key = f"{original_step['function']}:{original_step.get('id', 'unknown')}"
        
        # Store the successful alternative
        if step_key not in self._successful_alternatives:
            self._successful_alternatives[step_key] = []
        
        # Add this alternative if not already present
        alt_info = {
            "original": {
                "function": original_step["function"],
                "parameters": original_step.get("parameters", {})
            },
            "alternative": {
                "function": successful_alt["function"],
                "parameters": successful_alt.get("parameters", {})
            },
            "success_count": 1,
            "last_success": datetime.datetime.now().isoformat()
        }
        
        # Check if this alternative already exists
        exists = False
        for existing in self._successful_alternatives[step_key]:
            if (existing["alternative"]["function"] == alt_info["alternative"]["function"] and
                existing["alternative"]["parameters"] == alt_info["alternative"]["parameters"]):
                # Update existing record
                existing["success_count"] += 1
                existing["last_success"] = alt_info["last_success"]
                exists = True
                break
        
        # Add if new
        if not exists:
            self._successful_alternatives[step_key].append(alt_info)
    
    # Reinforcement Learning Approach
    async def update_from_execution_feedback(self, 
                                           procedure_id: str, 
                                           step_id: str, 
                                           success: bool, 
                                           context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update transfer mappings based on execution feedback using simple RL
        
        Args:
            procedure_id: ID of the procedure
            step_id: ID of the step
            success: Whether execution was successful
            context: Execution context
            
        Returns:
            Updated mapping information
        """
        # Initialize RL parameters storage if not exists
        if not hasattr(self, "_mapping_q_values"):
            self._mapping_q_values = {}
        
        # Get procedure and step
        procedure = None
        for p in self.procedures.values():
            if p.id == procedure_id:
                procedure = p
                break
        
        if not procedure:
            return {"error": f"Procedure with ID {procedure_id} not found"}
        
        # Find the step
        step = None
        for s in procedure.steps:
            if s["id"] == step_id:
                step = s
                break
        
        if not step:
            return {"error": f"Step {step_id} not found in procedure {procedure_id}"}
        
        # Identify which mapping was used
        mapping = self._get_step_mapping(procedure, step)
        
        # Create mapping key
        mapping_key = f"{mapping['source_domain']}:{mapping['target_domain']}:{mapping['function']}:{mapping['parameter_key']}"
        
        # Initialize Q-value if not exists
        if mapping_key not in self._mapping_q_values:
            self._mapping_q_values[mapping_key] = {
                "q_value": 0.5,  # Initial neutral value
                "attempts": 0,
                "successes": 0,
                "alternatives": {}
            }
        
        # Update stats
        self._mapping_q_values[mapping_key]["attempts"] += 1
        if success:
            self._mapping_q_values[mapping_key]["successes"] += 1
        
        # Calculate success rate
        success_rate = self._mapping_q_values[mapping_key]["successes"] / self._mapping_q_values[mapping_key]["attempts"]
        
        # Update Q-value with simple update rule
        reward = 1.0 if success else -0.5
        alpha = 0.1  # Learning rate
        current_q = self._mapping_q_values[mapping_key]["q_value"]
        new_q = current_q + alpha * (reward - current_q)
        self._mapping_q_values[mapping_key]["q_value"] = new_q
        
        # If failed, generate alternative mappings to try next time
        alternatives = []
        if not success:
            alternatives = self._generate_mapping_alternatives(mapping, context)
            
            # Store alternatives in the Q-value table
            for alt in alternatives:
                alt_key = f"{alt['function']}:{alt['parameter_value']}"
                if alt_key not in self._mapping_q_values[mapping_key]["alternatives"]:
                    self._mapping_q_values[mapping_key]["alternatives"][alt_key] = {
                        "function": alt["function"],
                        "parameter_key": mapping["parameter_key"],
                        "parameter_value": alt["parameter_value"],
                        "q_value": 0.3,  # Initial lower value for untested alternatives
                        "attempts": 0,
                        "successes": 0
                    }
        
        return {
            "mapping_key": mapping_key,
            "new_q_value": new_q,
            "previous_q_value": current_q,
            "success_rate": success_rate,
            "alternatives_generated": len(alternatives),
            "alternatives": alternatives
        }
    
    def _get_step_mapping(self, procedure: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the mapping information for a step
        
        Args:
            procedure: Procedure containing the step
            step: Step to get mapping for
            
        Returns:
            Mapping information
        """
        # Extract parameter that looks like a control mapping
        parameter_key = None
        parameter_value = None
        
        for key in ["button", "control", "input_method", "key"]:
            if key in step.get("parameters", {}):
                parameter_key = key
                parameter_value = step["parameters"][key]
                break
        
        return {
            "source_domain": procedure.domain,
            "target_domain": procedure.domain,  # Same for non-transferred procedures
            "function": step["function"],
            "parameter_key": parameter_key,
            "parameter_value": parameter_value
        }
    
    def _generate_mapping_alternatives(self, mapping: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate alternative mappings to try
        
        Args:
            mapping: Current mapping information
            context: Execution context
            
        Returns:
            List of alternative mappings
        """
        alternatives = []
        
        # Only generate alternatives if we have a parameter
        if not mapping["parameter_key"] or not mapping["parameter_value"]:
            return alternatives
        
        # Get parameter variants
        parameter_variants = []
        
        if mapping["parameter_key"] == "button":
            parameter_variants = self._get_button_variants(mapping["parameter_value"], context)
        elif mapping["parameter_key"] == "control":
            parameter_variants = self._get_control_variants(mapping["parameter_value"], context)
        elif mapping["parameter_key"] == "object_type":
            parameter_variants = self._get_object_variants(mapping["parameter_value"], context)
        elif mapping["parameter_key"] == "location":
            parameter_variants = self._get_location_variants(mapping["parameter_value"], context)
        elif mapping["parameter_key"] == "method":
            parameter_variants = self._get_method_variants(mapping["parameter_value"], context)
        
        # Get function equivalents
        function_equivalents = self._get_function_equivalents(mapping["function"])
        
        # Create alternatives from parameter variants
        for variant in parameter_variants:
            alternatives.append({
                "function": mapping["function"],
                "parameter_key": mapping["parameter_key"],
                "parameter_value": variant
            })
        
        # Create alternatives from function equivalents
        for func in function_equivalents:
            alternatives.append({
                "function": func,
                "parameter_key": mapping["parameter_key"],
                "parameter_value": mapping["parameter_value"]
            })
        
        # Create combined alternatives (both function and parameter)
        for func in function_equivalents:
            for variant in parameter_variants:
                alternatives.append({
                    "function": func,
                    "parameter_key": mapping["parameter_key"],
                    "parameter_value": variant
                })
        
        return alternatives
    
    def _prioritize_alternatives(self, alternatives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prioritize alternative mappings based on Q-values
        
        Args:
            alternatives: List of alternative mappings
            
        Returns:
            Prioritized list of alternatives
        """
        # If no Q-values storage, return as is
        if not hasattr(self, "_mapping_q_values"):
            return alternatives
        
        # Calculate priority for each alternative
        prioritized = []
        for alt in alternatives:
            # Try to find Q-value for this alternative
            q_value = 0.3  # Default for unknown alternatives
            
            # Check if we have this alternative in our Q-values
            mapping_key = None
            for key in self._mapping_q_values:
                for alt_key, alt_data in self._mapping_q_values[key]["alternatives"].items():
                    if (alt_data["function"] == alt["function"] and
                        alt_data["parameter_key"] == alt["parameter_key"] and
                        alt_data["parameter_value"] == alt["parameter_value"]):
                        # Found a match
                        q_value = alt_data["q_value"]
                        break
            
            # Add to prioritized list with Q-value
            prioritized.append({
                "alternative": alt,
                "q_value": q_value
            })
        
        # Sort by Q-value, highest first
        prioritized.sort(key=lambda x: x["q_value"], reverse=True)
        
        # Return just the alternatives in prioritized order
        return [item["alternative"] for item in prioritized]
    
    # Incremental Transfer Testing
    async def incremental_transfer(self, 
                                 source_name: str, 
                                 target_name: str, 
                                 target_domain: str,
                                 max_adaptation_attempts: int = 3) -> Dict[str, Any]:
        """
        Transfer and test procedure step by step
        
        Args:
            source_name: Name of the source procedure
            target_name: Name for the new procedure
            target_domain: Domain for the new procedure
            max_adaptation_attempts: Maximum number of adaptation attempts per step
            
        Returns:
            Transfer results
        """
        if source_name not in self.procedures:
            return {"error": f"Source procedure '{source_name}' not found"}
        
        procedure = self.procedures[source_name]
        
        # Create trace for the operation
        with trace(workflow_name="incremental_transfer"):
            # Prepare transfer results
            transfer_stats = {
                "total_steps": len(procedure.steps),
                "successful_steps": 0,
                "steps_requiring_adaptation": 0,
                "adaptation_attempts": 0,
                "steps_failed": 0,
                "adaptation_success_rate": 0.0
            }
            
            # Transfer and test each step individually
            transferred_steps = []
            failed_steps = []
            
            for step in procedure.steps:
                # Try to transfer the step
                mapped_step = self.map_step_to_domain(
                    step=step,
                    source_domain=procedure.domain,
                    target_domain=target_domain
                )
                
                if not mapped_step:
                    # Couldn't map this step
                    failed_steps.append({
                        "step_id": step["id"],
                        "error": "Could not map step to target domain"
                    })
                    transfer_stats["steps_failed"] += 1
                    continue
                
                # Test execution of just this step
                test_context = {"domain": target_domain, "test_mode": True}
                step_result = await self.execute_step(mapped_step, test_context)
                
                if step_result["success"]:
                    # Step executed successfully
                    transferred_steps.append(mapped_step)
                    transfer_stats["successful_steps"] += 1
                else:
                    # Step failed, try adaptation
                    transfer_stats["steps_requiring_adaptation"] += 1
                    adapted = False
                    
                    for attempt in range(max_adaptation_attempts):
                        transfer_stats["adaptation_attempts"] += 1
                        
                        # Try to adapt the step
                        adapted_step = await self._adapt_until_successful(
                            original_step=step,
                            mapped_step=mapped_step,
                            target_domain=target_domain,
                            attempt=attempt
                        )
                        
                        if adapted_step:
                            # Adaptation successful
                            transferred_steps.append(adapted_step)
                            adapted = True
                            break
                    
                    if not adapted:
                        # All adaptation attempts failed
                        failed_steps.append({
                            "step_id": step["id"],
                            "error": "Failed to adapt step after maximum attempts",
                            "last_error": step_result.get("error")
                        })
                        transfer_stats["steps_failed"] += 1
            
            # Calculate adaptation success rate
            if transfer_stats["steps_requiring_adaptation"] > 0:
                adaptation_successes = transfer_stats["steps_requiring_adaptation"] - transfer_stats["steps_failed"]
                transfer_stats["adaptation_success_rate"] = adaptation_successes / transfer_stats["steps_requiring_adaptation"]
            
            # Create new procedure if we have any transferred steps
            if not transferred_steps:
                return {
                    "success": False,
                    "error": "No steps could be transferred successfully",
                    "transfer_stats": transfer_stats,
                    "failed_steps": failed_steps
                }
            
            # Create the new procedure
            ctx = RunContextWrapper(context=self)
            new_procedure = await add_procedure(
                ctx,
                name=target_name,
                steps=transferred_steps,
                description=f"Incrementally transferred from {source_name} ({procedure.domain} to {target_domain})",
                domain=target_domain
            )
            
            # Record transfer
            transfer_record = ProcedureTransferRecord(
                source_procedure_id=procedure.id,
                source_domain=procedure.domain,
                target_procedure_id=new_procedure["procedure_id"],
                target_domain=target_domain,
                transfer_date=datetime.datetime.now().isoformat(),
                adaptation_steps=[{
                    "step_id": step["id"],
                    "adapted": True
                } for step in transferred_steps if "original_id" in step],
                success_level=transfer_stats["successful_steps"] / transfer_stats["total_steps"],
                practice_needed=5  # Initial estimate
            )
            
            self.chunk_library.record_transfer(transfer_record)
            
            # Update transfer stats
            self.transfer_stats["total_transfers"] += 1
            if transfer_stats["steps_failed"] == 0:
                self.transfer_stats["successful_transfers"] += 1
            
            return {
                "success": transfer_stats["steps_failed"] == 0,
                "source_name": source_name,
                "target_name": target_name,
                "source_domain": procedure.domain,
                "target_domain": target_domain,
                "steps_count": len(transferred_steps),
                "procedure_id": new_procedure["procedure_id"],
                "transfer_stats": transfer_stats,
                "failed_steps": failed_steps
            }
    
    async def _adapt_until_successful(self, 
                                    original_step: Dict[str, Any], 
                                    mapped_step: Dict[str, Any],
                                    target_domain: str,
                                    attempt: int) -> Optional[Dict[str, Any]]:
        """
        Adapt a step until it executes successfully or max attempts reached
        
        Args:
            original_step: Original step from source procedure
            mapped_step: Step mapped to target domain that failed
            target_domain: Target domain
            attempt: Current adaptation attempt
            
        Returns:
            Successfully adapted step or None if adaptation failed
        """
        # Try different adaptation strategies based on attempt number
        adapted_step = None
        
        if attempt == 0:
            # First attempt: Try parameter variants
            adapted_step = self._adapt_step_parameters(mapped_step, target_domain)
        elif attempt == 1:
            # Second attempt: Try function substitution
            adapted_step = self._adapt_step_function(mapped_step, target_domain)
        elif attempt == 2:
            # Third attempt: Try contextual adaptation
            adapted_step = await self._adapt_step_to_context(mapped_step, target_domain)
        
        if not adapted_step:
            return None
            
        # Test execution of adapted step
        test_context = {"domain": target_domain, "test_mode": True}
        step_result = await self.execute_step(adapted_step, test_context)
        
        if step_result["success"]:
            # Adaptation successful
            return adapted_step
        
        # Adaptation failed
        return None
    
    def _adapt_step_parameters(self, step: Dict[str, Any], target_domain: str) -> Optional[Dict[str, Any]]:
        """Adapt a step by trying different parameter values"""
        # Create a copy of the step
        adapted_step = step.copy()
        adapted_step["parameters"] = step.get("parameters", {}).copy()
        
        # Try parameter variants
        modified = False
        for key, value in step.get("parameters", {}).items():
            # Get variants based on parameter type
            variants = []
            
            if key == "button":
                variants = self._get_button_variants(value, {})
            elif key == "control":
                variants = self._get_control_variants(value, {})
            elif key == "object_type":
                variants = self._get_object_variants(value, {})
            elif key == "location":
                variants = self._get_location_variants(value, {})
            elif key == "method":
                variants = self._get_method_variants(value, {})
            
            # Use first variant if available
            if variants:
                adapted_step["parameters"][key] = variants[0]
                modified = True
        
        # If no modifications were made, return None
        if not modified:
            return None
            
        # Update description to indicate adaptation
        adapted_step["description"] = f"{step.get('description', step['id'])} (parameter-adapted)"
        
        # Indicate this was adapted
        adapted_step["adapted_from"] = step["id"]
        adapted_step["adaptation_type"] = "parameter"
        
        return adapted_step
    
    def _adapt_step_function(self, step: Dict[str, Any], target_domain: str) -> Optional[Dict[str, Any]]:
        """Adapt a step by trying a different function"""
        # Get function equivalents
        function_equivalents = self._get_function_equivalents(step["function"])
        
        if not function_equivalents:
            return None
            
        # Create a copy of the step with the first equivalent function
        adapted_step = step.copy()
        adapted_step["function"] = function_equivalents[0]
        
        # Update description to indicate adaptation
        adapted_step["description"] = f"{step.get('description', step['id'])} (function-adapted)"
        
        # Indicate this was adapted
        adapted_step["adapted_from"] = step["id"]
        adapted_step["adaptation_type"] = "function"
        
        return adapted_step
    
    async def _adapt_step_to_context(self, step: Dict[str, Any], target_domain: str) -> Optional[Dict[str, Any]]:
        """
        Adapt a step based on context from successful executions in similar domains
        
        Args:
            step: Step to adapt
            target_domain: Target domain
            
        Returns:
            Adapted step or None if no adaptation found
        """
        # Find procedures in similar domains
        similar_domains = self._find_similar_domains(target_domain)
        
        # Find similar steps from successful procedures
        similar_steps = []
        
        for proc_name, procedure in self.procedures.items():
            # Skip procedures with different domains
            if procedure.domain not in similar_domains and procedure.domain != target_domain:
                continue
                
            # Skip procedures with low proficiency
            if procedure.proficiency < 0.7:
                continue
                
            # Find steps with similar function or description
            for proc_step in procedure.steps:
                similarity = self._calculate_step_similarity(step, proc_step)
                
                if similarity > 0.6:  # Threshold for similarity
                    similar_steps.append({
                        "step": proc_step,
                        "similarity": similarity,
                        "domain": procedure.domain
                    })
        
        # Sort by similarity (highest first)
        similar_steps.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Take the most similar step
        if not similar_steps:
            return None
            
        best_match = similar_steps[0]
        
        # Adapt using the best match
        adapted_step = step.copy()
        
        # Adapt function if different
        if best_match["step"]["function"] != step["function"]:
            adapted_step["function"] = best_match["step"]["function"]
            
        # Adapt parameters
        adapted_step["parameters"] = {}
        
        # Keep original parameters that likely don't need adaptation
        for key, value in step.get("parameters", {}).items():
            if key not in ["button", "control", "input_method", "method"]:
                adapted_step["parameters"][key] = value
        
        # Copy parameters from best match that might need adaptation
        for key, value in best_match["step"].get("parameters", {}).items():
            if key in ["button", "control", "input_method", "method"]:
                adapted_step["parameters"][key] = value
        
        # Update description to indicate adaptation
        adapted_step["description"] = f"{step.get('description', step['id'])} (context-adapted)"
        
        # Indicate this was adapted
        adapted_step["adapted_from"] = step["id"]
        adapted_step["adaptation_type"] = "context"
        adapted_step["adaptation_source"] = {
            "domain": best_match["domain"],
            "step_id": best_match["step"]["id"],
            "similarity": best_match["similarity"]
        }
        
        return adapted_step
    
    def _find_similar_domains(self, domain: str) -> List[str]:
        """
        Find domains similar to the given domain
        
        Args:
            domain: Domain to find similar domains for
            
        Returns:
            List of similar domain names
        """
        # Define domain similarities
        domain_groups = [
            ["playstation", "xbox", "nintendo", "gaming"],
            ["touch_interface", "mouse_interface", "touchscreen"],
            ["dbd", "horror_game", "survival_game"],
            ["fps", "shooter", "call_of_duty", "battlefield"],
            ["cooking", "baking", "food_preparation"],
            ["driving", "racing", "vehicle_operation"],
            ["programming", "coding", "software_development"]
        ]
        
        # Find group containing the domain
        for group in domain_groups:
            if domain in group:
                return [d for d in group if d != domain]
        
        # No similar domains found
        return []
    
    def _calculate_step_similarity(self, step1: Dict[str, Any], step2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two steps
        
        Args:
            step1: First step
            step2: Second step
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Initialize similarity components
        function_similarity = 0.0
        description_similarity = 0.0
        parameter_similarity = 0.0
        
        # Function similarity
        if step1["function"] == step2["function"]:
            function_similarity = 1.0
        else:
            # Check for equivalent functions
            equivalents1 = self._get_function_equivalents(step1["function"])
            if step2["function"] in equivalents1:
                function_similarity = 0.8
        
        # Description similarity (simple word overlap)
        desc1 = step1.get("description", "").lower().split()
        desc2 = step2.get("description", "").lower().split()
        
        if desc1 and desc2:
            common_words = set(desc1) & set(desc2)
            description_similarity = len(common_words) / max(len(desc1), len(desc2))
        
        # Parameter similarity
        params1 = set(step1.get("parameters", {}).keys())
        params2 = set(step2.get("parameters", {}).keys())
        
        if params1 and params2:
            common_params = params1 & params2
            
            # Check for common parameter values
            common_values = 0
            for param in common_params:
                value1 = step1["parameters"][param]
                value2 = step2["parameters"][param]
                
                if value1 == value2:
                    common_values += 1
                    continue
                    
                # Check for equivalent values
                variant_func = None
                if param == "button":
                    variant_func = self._get_button_variants
                elif param == "control":
                    variant_func = self._get_control_variants
                elif param == "object_type":
                    variant_func = self._get_object_variants
                elif param == "location":
                    variant_func = self._get_location_variants
                elif param == "method":
                    variant_func = self._get_method_variants
                
                if variant_func:
                    variants = variant_func(value1, {})
                    if value2 in variants:
                        common_values += 0.8  # Partial credit for equivalent values
            
            parameter_similarity = common_values / len(common_params) if common_params else 0.0
        
        # Weighted combination
        return (function_similarity * 0.5) + (description_similarity * 0.3) + (parameter_similarity * 0.2)
    
    # Add a function tool for the new incremental transfer method
    @function_tool
    async def transfer_procedure_incrementally(
        ctx,
        source_name: str,
        target_name: str,
        target_domain: str,
        max_adaptation_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Transfer a procedure step by step with testing and adaptation
        
        Args:
            source_name: Name of the source procedure
            target_name: Name for the new procedure
            target_domain: Domain for the new procedure
            max_adaptation_attempts: Maximum adaptation attempts per step
            
        Returns:
            Incremental transfer results with detailed statistics
        """
        manager = ctx.context
        
        return await manager.incremental_transfer(
            source_name=source_name,
            target_name=target_name,
            target_domain=target_domain,
            max_adaptation_attempts=max_adaptation_attempts
        )
    
    # Add a function tool for updating from execution feedback
    @function_tool
    async def update_transfer_from_feedback(
        ctx,
        procedure_name: str,
        step_id: str,
        success: bool,
        execution_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Update transfer mappings based on execution feedback
        
        Args:
            procedure_name: Name of the procedure
            step_id: ID of the step
            success: Whether execution was successful
            execution_context: Execution context data
            
        Returns:
            Update results with mapping information
        """
        manager = ctx.context
        
        # Get procedure ID
        if procedure_name not in manager.procedures:
            return {"error": f"Procedure '{procedure_name}' not found"}
        
        procedure_id = manager.procedures[procedure_name].id
        
        return await manager.update_from_execution_feedback(
            procedure_id=procedure_id,
            step_id=step_id,
            success=success,
            context=execution_context or {}
        )
    
    # Add a function tool for adaptive execution
    @function_tool
    async def execute_step_adaptively(
        ctx,
        procedure_name: str,
        step_id: str,
        context: Dict[str, Any] = None,
        alternatives: int = 3
    ) -> Dict[str, Any]:
        """
        Execute a single step with automatic adaptation if it fails
        
        Args:
            procedure_name: Name of the procedure
            step_id: ID of the step to execute
            context: Execution context
            alternatives: Number of alternative implementations to try
            
        Returns:
            Execution result
        """
        manager = ctx.context
        
        # Get the procedure
        if procedure_name not in manager.procedures:
            return {"error": f"Procedure '{procedure_name}' not found"}
        
        procedure = manager.procedures[procedure_name]
        
        # Find the step
        step = None
        for s in procedure.steps:
            if s["id"] == step_id:
                step = s
                break
                
        if not step:
            return {"error": f"Step '{step_id}' not found in procedure '{procedure_name}'"}
        
        # Execute the step adaptively
        return await manager.adaptive_execute_step(
            step=step,
            context=context or {},
            alternatives=alternatives
        )
    
    # Add a function tool for executing a procedure with adaptation
    @function_tool
    async def execute_procedure_adaptively(
        ctx,
        name: str,
        context: Dict[str, Any] = None,
        max_alternatives: int = 3
    ) -> Dict[str, Any]:
        """
        Execute a procedure with automatic adaptation for failing steps
        
        Args:
            name: Name of the procedure to execute
            context: Execution context
            max_alternatives: Maximum number of alternatives to try per step
            
        Returns:
            Execution results with adaptation details
        """
        manager = ctx.context
        
        if name not in manager.procedures:
            return {"error": f"Procedure '{name}' not found"}
        
        procedure = manager.procedures[name]
        
        # Create execution trace
        with trace(workflow_name="execute_procedure_adaptively"):
            start_time = datetime.datetime.now()
            results = []
            success = True
            adapted_steps = []
            
            # Execute each step
            for step in procedure.steps:
                # Try to execute with adaptation
                step_result = await manager.adaptive_execute_step(
                    step=step,
                    context=context or {},
                    alternatives=max_alternatives
                )
                
                results.append(step_result)
                
                # Track adaptation
                if "alternative_used" in step_result:
                    adapted_steps.append({
                        "step_id": step["id"],
                        "alternative": step_result["alternative_used"],
                        "adaptation": step_result.get("adaptation", {})
                    })
                
                # Update context with step result
                if context is None:
                    context = {}
                context[f"step_{step['id']}_result"] = step_result
                
                # Add to action history
                if "action_history" not in context:
                    context["action_history"] = []
                context["action_history"].append({
                    "step_id": step["id"],
                    "function": step["function"],
                    "success": step_result["success"]
                })
                
                # Check for failure
                if not step_result["success"]:
                    success = False
                    break
            
            # Calculate execution time
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Update procedure statistics
            manager.update_procedure_stats(procedure, execution_time, success)
            
            # Return execution results
            return {
                "success": success,
                "results": results,
                "execution_time": execution_time,
                "proficiency": procedure.proficiency,
                "adapted_steps": adapted_steps,
                "adaptation_count": len(adapted_steps),
                "procedure_name": name
            }

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

async def demonstrate_cross_game_transfer():
    """Demonstrate procedural memory with cross-game transfer"""
    
    # Create an enhanced procedural memory manager
    manager = ProceduralMemoryManager()
    
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
            "function": "vault_window",
            "parameters": {}
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
    
    # Create RunContextWrapper for agent tools
    ctx = RunContextWrapper(context=manager)
    
    # Learn the procedure
    print("\nLearning procedure:")
    dbd_result = await add_procedure(
        ctx,
        name="window_to_generator",
        steps=window_gen_steps,
        description="Navigate through a window and start working on a generator",
        domain="dbd"  # Dead by Daylight
    )
    
    print(f"Created procedure: {dbd_result}")
    
    # Execute procedure multiple times
    print("\nPracticing procedure...")
    for i in range(10):
        print(f"\nExecution {i+1}:")
        context = {"sprinting": False}
        result = await execute_procedure(ctx, "window_to_generator", context)
        
        dbd_procedure = manager.procedures["window_to_generator"]
        print(f"Success: {result['success']}, " 
              f"Time: {result['execution_time']:.4f}s, "
              f"Proficiency: {dbd_procedure.proficiency:.2f}")
    
    # Check for chunking opportunities
    print("\nIdentifying chunking opportunities:")
    chunking_result = await identify_chunking_opportunities(ctx, "window_to_generator")
    print(f"Chunking analysis: {chunking_result}")
    
    # Apply chunking
    if chunking_result.get("can_chunk", False):
        print("\nApplying chunking:")
        chunk_result = await apply_chunking(ctx, "window_to_generator")
        print(f"Chunking result: {chunk_result}")
    
    # Create a template from the main chunk
    if manager.procedures["window_to_generator"].is_chunked:
        print("\nGeneralizing chunk template:")
        template_result = await generalize_chunk_from_steps(
            ctx,
            chunk_name="window_to_generator_combo",
            procedure_name="window_to_generator",
            step_ids=["start_sprint", "approach_window", "vault"],
            domain="dbd"
        )
        print(f"Template created: {template_result}")
    
    # Transfer to another domain
    print("\nTransferring procedure to new domain:")
    transfer_result = await transfer_procedure(
        ctx,
        source_name="window_to_generator",
        target_name="xbox_window_to_generator",
        target_domain="xbox"
    )
    print(f"Transfer result: {transfer_result}")
    
    # Execute the transferred procedure
    print("\nExecuting transferred procedure:")
    xbox_result = await execute_procedure(ctx, "xbox_window_to_generator")
    print(f"Xbox execution result: {xbox_result}")
    
    # Get procedure statistics
    print("\nProcedure statistics:")
    stats = await get_procedure_proficiency(ctx, "window_to_generator")
    print(f"Original procedure: {stats}")
    
    # Find similar procedures
    print("\nFinding similar procedures:")
    similar = await find_similar_procedures(ctx, "window_to_generator")
    print(f"Similar procedures: {similar}")
    
    return manager

async def demonstrate_procedural_memory():
    """Demonstrate the procedural memory system with Agents SDK"""
    
    # Create the procedural memory manager
    manager = ProceduralMemoryManager()
    
    # Register example functions
    async def perform_action(action_name: str, action_target: str, context: Dict[str, Any] = None):
        print(f"Performing action: {action_name} on {action_target}")
        return {"action": action_name, "target": action_target, "performed": True}
    
    async def check_condition(condition_name: str, context: Dict[str, Any] = None):
        print(f"Checking condition: {condition_name}")
        # Simulate condition check
        result = True  # In real usage, this would evaluate something
        return {"condition": condition_name, "result": result}
        
    async def select_item(item_id: str, control: str, context: Dict[str, Any] = None):
        print(f"Selecting item {item_id} using {control}")
        return {"item": item_id, "selected": True}
        
    async def navigate_to(location: str, method: str, context: Dict[str, Any] = None):
        print(f"Navigating to {location} using {method}")
        return {"location": location, "arrived": True}
    
    # Register functions
    manager.register_function("perform_action", perform_action)
    manager.register_function("check_condition", check_condition)
    manager.register_function("select_item", select_item)
    manager.register_function("navigate_to", navigate_to)
    
    # Create RunContextWrapper for agent tools
    ctx = RunContextWrapper(context=manager)
    
    # Create a procedure in the "touch_interface" domain
    touch_procedure_steps = [
        {
            "id": "step_1",
            "description": "Navigate to menu",
            "function": "navigate_to",
            "parameters": {"location": "main_menu", "method": "swipe"}
        },
        {
            "id": "step_2",
            "description": "Select item from menu",
            "function": "select_item",
            "parameters": {"item_id": "settings", "control": "tap"}
        },
        {
            "id": "step_3",
            "description": "Adjust settings",
            "function": "perform_action",
            "parameters": {"action_name": "adjust", "action_target": "brightness"}
        }
    ]
    
    print("Creating touch interface procedure...")
    
    touch_result = await add_procedure(
        ctx,
        name="touch_settings_procedure",
        steps=touch_procedure_steps,
        description="Adjust settings using touch interface",
        domain="touch_interface"
    )
    
    print(f"Created procedure: {touch_result}")
    
    # Execute the procedure multiple times to develop proficiency
    print("\nPracticing touch procedure...")
    for i in range(5):
        await execute_procedure(ctx, "touch_settings_procedure")
    
    # Apply chunking to identify patterns
    print("\nApplying chunking to the procedure...")
    
    chunking_result = await apply_chunking(ctx, "touch_settings_procedure")
    print(f"Chunking result: {chunking_result}")
    
    # Generalize a chunk for navigation and selection
    print("\nGeneralizing a chunk for navigation and selection...")
    
    chunk_result = await generalize_chunk_from_steps(
        ctx,
        chunk_name="navigate_and_select",
        procedure_name="touch_settings_procedure",
        step_ids=["step_1", "step_2"]
    )
    
    print(f"Chunk generalization result: {chunk_result}")
    
    # Transfer the procedure to mouse_interface domain
    print("\nTransferring procedure to mouse interface domain...")
    
    transfer_result = await transfer_with_chunking(
        ctx,
        source_name="touch_settings_procedure",
        target_name="mouse_settings_procedure",
        target_domain="mouse_interface"
    )
    
    print(f"Transfer result: {transfer_result}")
    
    # Execute the transferred procedure
    print("\nExecuting transferred procedure...")
    
    transfer_execution = await execute_procedure(
        ctx,
        name="mouse_settings_procedure"
    )
    
    print(f"Transferred procedure execution: {transfer_execution}")
    
    # Compare the two procedures
    print("\nFinding similar procedures to our touch procedure...")
    
    similar = await find_similar_procedures(ctx, "touch_settings_procedure")
    
    print(f"Similar procedures: {similar}")
    
    # Get transfer statistics
    print("\nGetting transfer statistics...")
    
    stats = await get_transfer_statistics(ctx)
    
    print(f"Transfer statistics: {stats}")
    
    # List all procedures
    print("\nListing all procedures:")
    
    procedures = await list_procedures(ctx)
    
    for proc in procedures:
        print(f"- {proc['name']} ({proc['domain']}) - Proficiency: {proc['proficiency']:.2f}")
    
    return manager

if __name__ == "__main__":
    # Run both demonstration functions in sequence
    asyncio.run(demonstrate_cross_game_transfer())
    asyncio.run(demonstrate_procedural_memory())
