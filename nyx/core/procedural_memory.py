# nyx/core/procedural_memory.py

import logging
import asyncio
import datetime
import json
import math
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from collections import Counter, defaultdict
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

# enhanced pieces

# ============================================================================
# 1. HIERARCHICAL PROCEDURE REPRESENTATION
# ============================================================================

class HierarchicalProcedure(BaseModel):
    """Hierarchical representation of procedures with sub-procedures and goals"""
    id: str
    name: str
    description: str
    parent_id: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    goal_state: Dict[str, Any] = Field(default_factory=dict)
    preconditions: Dict[str, Any] = Field(default_factory=dict)
    postconditions: Dict[str, Any] = Field(default_factory=dict)
    is_abstract: bool = False
    domain: str
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    execution_count: int = 0
    successful_executions: int = 0
    average_execution_time: float = 0.0
    proficiency: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_execution: Optional[str] = None
    
    def add_child(self, child_id: str) -> None:
        """Add a child procedure to this procedure"""
        if child_id not in self.children:
            self.children.append(child_id)
            self.last_updated = datetime.datetime.now().isoformat()
    
    def remove_child(self, child_id: str) -> None:
        """Remove a child procedure from this procedure"""
        if child_id in self.children:
            self.children.remove(child_id)
            self.last_updated = datetime.datetime.now().isoformat()
    
    def meets_preconditions(self, context: Dict[str, Any]) -> bool:
        """Check if context meets all preconditions"""
        for key, value in self.preconditions.items():
            if key not in context:
                return False
            
            # Handle different value types
            if isinstance(value, (list, tuple, set)):
                if context[key] not in value:
                    return False
            elif isinstance(value, dict) and "min" in value and "max" in value:
                if not (value["min"] <= context[key] <= value["max"]):
                    return False
            elif context[key] != value:
                return False
        
        return True
    
    def update_goal_state(self, goal: Dict[str, Any]) -> None:
        """Update the goal state"""
        self.goal_state.update(goal)
        self.last_updated = datetime.datetime.now().isoformat()

# ============================================================================
# 2. LEARNING FROM OBSERVATION
# ============================================================================

class ObservationLearner(BaseModel):
    """System for learning procedures from observation"""
    observation_history: List[Dict[str, Any]] = Field(default_factory=list)
    pattern_detection_threshold: float = 0.7
    max_history: int = 100
    
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

# ============================================================================
# 3. ENHANCED ERROR RECOVERY WITH CAUSAL MODELS
# ============================================================================

class CausalModel(BaseModel):
    """Causal model for reasoning about procedure failures"""
    causes: Dict[str, List[Dict[str, float]]] = Field(default_factory=dict)
    interventions: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    error_history: List[Dict[str, Any]] = Field(default_factory=list)
    max_history: int = 50
    
    def identify_likely_causes(self, error: Dict[str, Any]) -> List[Dict[str, float]]:
        """Identify likely causes of an error"""
        # Add to error history
        self.error_history.append({
            "error": error,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Trim history if needed
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
        
        # Extract error type and context
        error_type = error.get("type", "unknown_error")
        context = error.get("context", {})
        
        # Find matching causes in our model
        if error_type in self.causes:
            likely_causes = self.causes[error_type].copy()
            
            # Adjust probabilities based on context
            for cause in likely_causes:
                # Check for context matches
                if "context_factors" in cause:
                    match_score = self._calculate_context_match(cause["context_factors"], context)
                    cause["probability"] *= match_score
            
            # Sort by probability and return
            likely_causes.sort(key=lambda x: x["probability"], reverse=True)
            return likely_causes
        
        # No known causes, generate basic hypothesis
        return [{
            "cause": "unknown",
            "description": "Unknown cause for this error type",
            "probability": 0.5
        }]
    
    def suggest_interventions(self, causes: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Suggest interventions based on identified causes"""
        suggested_interventions = []
        
        for cause in causes:
            cause_id = cause.get("cause", "unknown")
            
            # Look for interventions for this cause
            if cause_id in self.interventions:
                for intervention in self.interventions[cause_id]:
                    # Copy intervention and add confidence based on cause probability
                    intervention_copy = intervention.copy()
                    intervention_copy["confidence"] = cause["probability"] * intervention.get("effectiveness", 0.7)
                    suggested_interventions.append(intervention_copy)
        
        # Sort by confidence
        suggested_interventions.sort(key=lambda x: x["confidence"], reverse=True)
        
        # If no specific interventions found, suggest general ones
        if not suggested_interventions:
            suggested_interventions = [
                {
                    "type": "retry",
                    "description": "Retry the failed operation",
                    "confidence": 0.5
                },
                {
                    "type": "alternative_approach",
                    "description": "Try an alternative approach to accomplish the same goal",
                    "confidence": 0.4
                }
            ]
        
        return suggested_interventions
    
    def update_from_outcome(self, 
                          error: Dict[str, Any], 
                          cause: str, 
                          intervention: Dict[str, Any], 
                          success: bool) -> None:
        """Update the causal model based on intervention outcome"""
        error_type = error.get("type", "unknown_error")
        
        # Update cause probability
        if error_type in self.causes:
            for cause_entry in self.causes[error_type]:
                if cause_entry["cause"] == cause:
                    # Update probability based on success
                    cause_entry["probability"] = (cause_entry["probability"] * 0.8) + (0.2 if success else 0.0)
                    break
        else:
            # New error type
            self.causes[error_type] = [{
                "cause": cause,
                "description": f"Cause for {error_type}",
                "probability": 0.7 if success else 0.3,
                "context_factors": error.get("context", {})
            }]
        
        # Update intervention effectiveness
        if cause in self.interventions:
            # Look for matching intervention
            found = False
            for int_entry in self.interventions[cause]:
                if int_entry["type"] == intervention["type"]:
                    # Update effectiveness based on success
                    int_entry["effectiveness"] = (int_entry["effectiveness"] * 0.8) + (0.2 if success else 0.0)
                    found = True
                    break
            
            if not found:
                # Add new intervention
                self.interventions[cause].append({
                    "type": intervention["type"],
                    "description": intervention["description"],
                    "effectiveness": 0.7 if success else 0.3
                })
        else:
            # New cause
            self.interventions[cause] = [{
                "type": intervention["type"],
                "description": intervention["description"],
                "effectiveness": 0.7 if success else 0.3
            }]
    
    def _calculate_context_match(self, factors: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate how well context matches the factors"""
        if not factors:
            return 1.0  # No factors means everything matches
        
        matches = 0
        total_factors = len(factors)
        
        for key, value in factors.items():
            if key in context:
                if isinstance(value, (list, tuple, set)):
                    # Check if value is in list
                    if context[key] in value:
                        matches += 1
                elif isinstance(value, dict) and "min" in value and "max" in value:
                    # Range check
                    if value["min"] <= context[key] <= value["max"]:
                        matches += 1
                elif context[key] == value:
                    # Direct equality
                    matches += 1
        
        # Return match percentage
        return matches / total_factors if total_factors > 0 else 1.0

# ============================================================================
# 4. TEMPORAL ABSTRACTION AND SEQUENCE MODELING
# ============================================================================

class TemporalNode(BaseModel):
    """Node in a temporal procedure graph"""
    id: str
    action: Dict[str, Any]
    temporal_constraints: List[Dict[str, Any]] = Field(default_factory=list)
    duration: Optional[Tuple[float, float]] = None  # (min, max) duration
    next_nodes: List[str] = Field(default_factory=list)
    prev_nodes: List[str] = Field(default_factory=list)
    
    def add_constraint(self, constraint: Dict[str, Any]) -> None:
        """Add a temporal constraint to this node"""
        self.temporal_constraints.append(constraint)
    
    def is_valid(self, execution_history: List[Dict[str, Any]]) -> bool:
        """Check if this node's temporal constraints are valid"""
        for constraint in self.temporal_constraints:
            constraint_type = constraint.get("type")
            
            if constraint_type == "after":
                # Must occur after another action
                ref_action = constraint.get("action")
                if not any(h["action"] == ref_action for h in execution_history):
                    return False
            elif constraint_type == "before":
                # Must occur before another action
                ref_action = constraint.get("action")
                if any(h["action"] == ref_action for h in execution_history):
                    return False
            elif constraint_type == "delay":
                # Must wait minimum time from last action
                if execution_history:
                    last_time = execution_history[-1].get("timestamp")
                    min_delay = constraint.get("min_delay", 0)
                    if last_time:
                        last_time = datetime.datetime.fromisoformat(last_time)
                        elapsed = (datetime.datetime.now() - last_time).total_seconds()
                        if elapsed < min_delay:
                            return False
        
        return True

class TemporalProcedureGraph(BaseModel):
    """Graph representation of a temporal procedure"""
    id: str
    name: str
    nodes: Dict[str, TemporalNode] = Field(default_factory=dict)
    edges: List[Tuple[str, str, Dict[str, Any]]] = Field(default_factory=list)
    start_nodes: List[str] = Field(default_factory=list)
    end_nodes: List[str] = Field(default_factory=list)
    domain: str
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    def add_node(self, node: TemporalNode) -> None:
        """Add a node to the graph"""
        self.nodes[node.id] = node
        self.last_updated = datetime.datetime.now().isoformat()
    
    def add_edge(self, from_id: str, to_id: str, properties: Dict[str, Any] = None) -> None:
        """Add an edge between nodes"""
        if from_id in self.nodes and to_id in self.nodes:
            self.edges.append((from_id, to_id, properties or {}))
            
            # Update node connections
            self.nodes[from_id].next_nodes.append(to_id)
            self.nodes[to_id].prev_nodes.append(from_id)
            
            self.last_updated = datetime.datetime.now().isoformat()
    
    def get_next_executable_nodes(self, execution_history: List[Dict[str, Any]]) -> List[str]:
        """Get nodes that can be executed next based on history"""
        # Start with nodes that have no predecessors if no history
        if not execution_history:
            return self.start_nodes
        
        # Get last executed node
        last_action = execution_history[-1].get("node_id")
        if not last_action or last_action not in self.nodes:
            # Can't determine next actions
            return []
        
        # Get possible next nodes
        next_nodes = self.nodes[last_action].next_nodes
        
        # Filter by temporal constraints
        valid_nodes = []
        for node_id in next_nodes:
            if node_id in self.nodes and self.nodes[node_id].is_valid(execution_history):
                valid_nodes.append(node_id)
        
        return valid_nodes
    
    def validate_temporal_constraints(self) -> bool:
        """Validate that temporal constraints are consistent"""
        # Check for cycles with minimum durations
        visited = set()
        path = set()
        
        # Check each start node
        for start in self.start_nodes:
            if not self._check_for_negative_cycles(start, visited, path, 0):
                return False
        
        return True
    
    def _check_for_negative_cycles(self, 
                                 node_id: str, 
                                 visited: Set[str], 
                                 path: Set[str], 
                                 current_duration: float) -> bool:
        """Check for negative cycles in the graph (would make it impossible to satisfy)"""
        if node_id in path:
            # Found a cycle, check if the total duration is negative
            return current_duration >= 0
        
        if node_id in visited:
            return True
        
        visited.add(node_id)
        path.add(node_id)
        
        # Check outgoing edges
        for source, target, props in self.edges:
            if source == node_id:
                # Get edge duration
                min_duration = props.get("min_duration", 0)
                
                # Recurse
                if not self._check_for_negative_cycles(target, visited, path, 
                                                      current_duration + min_duration):
                    return False
        
        path.remove(node_id)
        return True
    
    @classmethod
    def from_procedure(cls, procedure: Procedure) -> 'TemporalProcedureGraph':
        """Convert a standard procedure to a temporal procedure graph"""
        graph = cls(
            id=f"temporal_{procedure.id}",
            name=f"Temporal graph for {procedure.name}",
            domain=procedure.domain
        )
        
        # Create nodes for each step
        for i, step in enumerate(procedure.steps):
            node = TemporalNode(
                id=f"node_{step['id']}",
                action={
                    "function": step["function"],
                    "parameters": step.get("parameters", {}),
                    "description": step.get("description", f"Step {i+1}")
                }
            )
            
            graph.add_node(node)
            
            # First step is a start node
            if i == 0:
                graph.start_nodes.append(node.id)
            
            # Last step is an end node
            if i == len(procedure.steps) - 1:
                graph.end_nodes.append(node.id)
        
        # Create edges for sequential execution
        for i in range(len(procedure.steps) - 1):
            current_id = f"node_{procedure.steps[i]['id']}"
            next_id = f"node_{procedure.steps[i+1]['id']}"
            
            graph.add_edge(current_id, next_id)
        
        return graph

# ============================================================================
# 5. INTEGRATION WITH WORKING MEMORY AND ATTENTION
# ============================================================================

class WorkingMemoryController:
    """Controls working memory during procedure execution"""
    
    def __init__(self, capacity: int = 5):
        self.items = []
        self.capacity = capacity
        self.focus_history = []
        self.max_history = 20
    
    def update(self, context: Dict[str, Any], procedure: Procedure) -> None:
        """Update working memory based on context and current procedure"""
        # Clear items that are no longer relevant
        self.items = [item for item in self.items if self._is_still_relevant(item, context)]
        
        # Add new items from context
        for key, value in context.items():
            # Only consider simple types for working memory
            if isinstance(value, (str, int, float, bool)):
                # Prioritize items explicitly mentioned in procedure steps
                priority = self._calculate_item_priority(key, value, procedure)
                
                # Create new working memory item
                new_item = {
                    "key": key,
                    "value": value,
                    "priority": priority,
                    "added": datetime.datetime.now().isoformat()
                }
                
                # Check if already in working memory
                existing = next((i for i in self.items if i["key"] == key), None)
                if existing:
                    # Update existing item
                    existing["value"] = value
                    existing["priority"] = max(existing["priority"], priority)
                else:
                    # Add new item
                    self.items.append(new_item)
        
        # Sort by priority and trim to capacity
        self.items.sort(key=lambda x: x["priority"], reverse=True)
        self.items = self.items[:self.capacity]
    
    def get_attention_focus(self) -> Dict[str, Any]:
        """Get current focus of attention"""
        if not self.items:
            return {}
        
        # Choose the highest priority item as focus
        focus_item = self.items[0]
        
        # Record focus for history
        self.focus_history.append({
            "key": focus_item["key"],
            "value": focus_item["value"],
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Trim history
        if len(self.focus_history) > self.max_history:
            self.focus_history = self.focus_history[-self.max_history:]
        
        return {
            "focus_key": focus_item["key"],
            "focus_value": focus_item["value"],
            "working_memory": {item["key"]: item["value"] for item in self.items},
            "memory_usage": f"{len(self.items)}/{self.capacity}"
        }
    
    def _is_still_relevant(self, item: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if a working memory item is still relevant"""
        # Item mentioned in current context
        if item["key"] in context:
            return True
        
        # Recently added items stay relevant
        added_time = datetime.datetime.fromisoformat(item["added"])
        time_in_memory = (datetime.datetime.now() - added_time).total_seconds()
        if time_in_memory < 60:  # Items stay relevant for at least 60 seconds
            return True
        
        # High priority items stay relevant longer
        if item["priority"] > 0.8:
            return True
        
        return False
    
    def _calculate_item_priority(self, key: str, value: Any, procedure: Procedure) -> float:
        """Calculate priority for an item"""
        base_priority = 0.5  # Default priority
        
        # Check if mentioned in procedure steps
        for step in procedure.steps:
            # Check function name
            if step["function"] == key:
                base_priority = max(base_priority, 0.9)
            
            # Check parameters
            params = step.get("parameters", {})
            if key in params:
                base_priority = max(base_priority, 0.8)
            
            # Check if value is used in parameters
            if value in params.values():
                base_priority = max(base_priority, 0.7)
        
        # Recency effect - recent focus gets higher priority
        for i, focus in enumerate(reversed(self.focus_history)):
            if focus["key"] == key:
                # Calculate recency factor (higher for more recent focus)
                recency = max(0.0, 1.0 - (i / 10))
                base_priority = max(base_priority, 0.6 * recency)
                break
        
        return base_priority

# ============================================================================
# 6. PARAMETER OPTIMIZATION WITH BAYESIAN METHODS
# ============================================================================

class ParameterOptimizer:
    """Optimizes procedure parameters using Bayesian optimization"""
    
    def __init__(self):
        self.parameter_models = {}
        self.optimization_history = {}
        self.bounds = {}  # Parameter bounds
    
    async def optimize_parameters(
        self, 
        procedure: Procedure, 
        objective_function: Callable,
        iterations: int = 10
    ) -> Dict[str, Any]:
        """Optimize parameters for a procedure"""
        # Collect optimizable parameters
        parameters = self._get_optimizable_parameters(procedure)
        
        if not parameters:
            return {
                "status": "no_parameters",
                "message": "No optimizable parameters found"
            }
        
        # Initialize history for this procedure
        if procedure.id not in self.optimization_history:
            self.optimization_history[procedure.id] = []
        
        # Prepare parameter space
        param_space = {}
        for param_info in parameters:
            param_id = f"{param_info['step_id']}.{param_info['param_key']}"
            
            # Get or create bounds
            if param_id not in self.bounds:
                # Auto-detect bounds based on parameter type
                self.bounds[param_id] = self._auto_detect_bounds(param_info["param_value"])
            
            param_space[param_id] = self.bounds[param_id]
        
        # Run optimization iterations
        results = []
        best_params = None
        best_score = float('-inf')
        
        for i in range(iterations):
            # Generate next parameters to try
            if i == 0:
                # First iteration: use current parameters
                test_params = {param_id: self.bounds[param_id][0] for param_id in param_space}
            else:
                # Use Bayesian optimization to suggest next parameters
                test_params = self._suggest_next_parameters(
                    procedure.id, 
                    param_space, 
                    results
                )
            
            # Apply parameters to procedure
            procedure_copy = procedure.model_copy(deep=True)
            self._apply_parameters(procedure_copy, test_params)
            
            # Evaluate objective function
            score = await objective_function(procedure_copy)
            
            # Record result
            result = {
                "parameters": test_params,
                "score": score,
                "iteration": i
            }
            results.append(result)
            self.optimization_history[procedure.id].append(result)
            
            # Track best parameters
            if score > best_score:
                best_score = score
                best_params = test_params
            
            # Update models
            self._update_parameter_models(procedure.id, results)
        
        # Return best parameters
        return {
            "status": "success",
            "best_parameters": best_params,
            "best_score": best_score,
            "iterations": iterations,
            "history": results
        }
    
    def _get_optimizable_parameters(self, procedure: Procedure) -> List[Dict[str, Any]]:
        """Get parameters that can be optimized"""
        optimizable_params = []
        
        for step in procedure.steps:
            for key, value in step.get("parameters", {}).items():
                # Check if parameter is optimizable (numeric or boolean)
                if isinstance(value, (int, float, bool)):
                    optimizable_params.append({
                        "step_id": step["id"],
                        "param_key": key,
                        "param_value": value,
                        "param_type": type(value).__name__
                    })
        
        return optimizable_params
    
    def _auto_detect_bounds(self, value: Any) -> Tuple[float, float]:
        """Auto-detect reasonable bounds for a parameter"""
        if isinstance(value, bool):
            return (0, 1)  # Boolean as 0/1
        elif isinstance(value, int):
            # Integer bounds: go 5x below and above, with minimum of 0
            lower = max(0, value // 5)
            upper = value * 5
            return (lower, upper)
        elif isinstance(value, float):
            # Float bounds: go 5x below and above, with minimum of 0
            lower = max(0.0, value / 5)
            upper = value * 5
            return (lower, upper)
        else:
            # Default bounds
            return (0, 10)
    
    def _suggest_next_parameters(
        self, 
        procedure_id: str, 
        param_space: Dict[str, Tuple[float, float]], 
        results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Suggest next parameters to try using Bayesian optimization"""
        # If not enough results yet, use random sampling
        if len(results) < 3:
            return {
                param_id: random.uniform(bounds[0], bounds[1])
                for param_id, bounds in param_space.items()
            }
        
        # Simple exploitation-exploration strategy
        explore = random.random() < 0.3  # 30% chance to explore
        
        if explore:
            # Random exploration
            return {
                param_id: random.uniform(bounds[0], bounds[1])
                for param_id, bounds in param_space.items()
            }
        else:
            # Exploitation: use parameters from best result with small perturbations
            best_result = max(results, key=lambda x: x["score"])
            best_params = best_result["parameters"]
            
            # Add small random perturbations
            return {
                param_id: self._perturb_parameter(param_id, value, param_space[param_id])
                for param_id, value in best_params.items()
            }
    
    def _perturb_parameter(
        self, 
        param_id: str, 
        value: float, 
        bounds: Tuple[float, float]
    ) -> float:
        """Add a small perturbation to a parameter value"""
        min_val, max_val = bounds
        range_val = max_val - min_val
        
        # Perturbation size: 5-15% of parameter range
        perturbation_size = range_val * random.uniform(0.05, 0.15)
        
        # Add/subtract perturbation
        if random.random() < 0.5:
            new_value = value + perturbation_size
        else:
            new_value = value - perturbation_size
        
        # Ensure value stays within bounds
        return max(min_val, min(max_val, new_value))
    
    def _apply_parameters(self, procedure: Procedure, parameters: Dict[str, float]) -> None:
        """Apply parameters to a procedure"""
        for param_id, value in parameters.items():
            step_id, param_key = param_id.split(".")
            
            # Find the step
            for step in procedure.steps:
                if step["id"] == step_id and "parameters" in step:
                    # Update parameter if it exists
                    if param_key in step["parameters"]:
                        # Convert type if needed
                        original_type = type(step["parameters"][param_key])
                        if original_type == bool:
                            step["parameters"][param_key] = value > 0.5
                        elif original_type == int:
                            step["parameters"][param_key] = int(value)
                        else:
                            step["parameters"][param_key] = value
    
    def _update_parameter_models(self, procedure_id: str, results: List[Dict[str, Any]]) -> None:
        """Update internal parameter models based on results"""
        # This would normally update a Gaussian Process or other Bayesian model
        # For simplicity, we're just storing the results
        
        # In a real implementation, this would use libraries like scikit-learn
        # or GPyTorch to update a surrogate model of the objective function
        pass

# ============================================================================
# 7. DYNAMIC EXECUTION STRATEGIES
# ============================================================================

class ExecutionStrategy(BaseModel):
    """Strategy for executing a procedure"""
    id: str
    name: str
    description: str
    selection_criteria: Dict[str, Any] = Field(default_factory=dict)
    
    async def execute(
        self, 
        procedure: Procedure, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the procedure according to this strategy"""
        # Base implementation - must be overridden
        raise NotImplementedError("This method must be implemented by subclasses")
    
    def should_select(self, context: Dict[str, Any], procedure: Procedure) -> float:
        """Calculate how well this strategy matches the current context"""
        score = 0.5  # Default score
        
        # Check each selection criterion
        for key, value in self.selection_criteria.items():
            if key in context:
                if isinstance(value, (list, tuple, set)):
                    # Check if context value is in list
                    if context[key] in value:
                        score += 0.1
                elif isinstance(value, dict) and "min" in value and "max" in value:
                    # Range check
                    if value["min"] <= context[key] <= value["max"]:
                        score += 0.1
                elif context[key] == value:
                    # Exact match
                    score += 0.2
        
        return min(1.0, score)

class DeliberateExecutionStrategy(ExecutionStrategy):
    """Executes procedure carefully with validation between steps"""
    
    async def execute(
        self, 
        procedure: Procedure, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute procedure deliberately with checks between steps"""
        start_time = datetime.datetime.now()
        results = []
        success = True
        
        # Initialize execution state
        execution_state = context.copy()
        execution_state["strategy"] = "deliberate"
        execution_state["execution_history"] = []
        
        # Execute steps sequentially with validation
        for i, step in enumerate(procedure.steps):
            # Validate preconditions
            if not self._validate_preconditions(step, execution_state):
                results.append({
                    "step_id": step["id"],
                    "success": False,
                    "error": "Preconditions not met",
                    "execution_time": 0.0
                })
                success = False
                break
            
            # Execute the step
            step_result = await self._execute_step(step, execution_state)
            results.append(step_result)
            
            # Update execution state
            execution_state[f"step_{step['id']}_result"] = step_result
            execution_state["execution_history"].append({
                "step_id": step["id"],
                "function": step["function"],
                "success": step_result["success"],
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Check for failure
            if not step_result["success"]:
                success = False
                break
            
            # Validate postconditions
            if not self._validate_postconditions(step, execution_state):
                results.append({
                    "step_id": step["id"],
                    "success": False,
                    "error": "Postconditions not met",
                    "execution_time": 0.0
                })
                success = False
                break
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        return {
            "success": success,
            "results": results,
            "execution_time": execution_time,
            "strategy": "deliberate"
        }
    
    def _validate_preconditions(self, step: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """Validate preconditions for a step"""
        preconditions = step.get("preconditions", {})
        
        for key, value in preconditions.items():
            if key not in state:
                return False
            
            # Compare values
            if isinstance(value, (list, tuple, set)):
                if state[key] not in value:
                    return False
            elif isinstance(value, dict) and "min" in value and "max" in value:
                if not (value["min"] <= state[key] <= value["max"]):
                    return False
            elif state[key] != value:
                return False
        
        return True
    
    def _validate_postconditions(self, step: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """Validate postconditions for a step"""
        postconditions = step.get("postconditions", {})
        
        for key, value in postconditions.items():
            if key not in state:
                return False
            
            # Compare values
            if isinstance(value, (list, tuple, set)):
                if state[key] not in value:
                    return False
            elif isinstance(value, dict) and "min" in value and "max" in value:
                if not (value["min"] <= state[key] <= value["max"]):
                    return False
            elif state[key] != value:
                return False
        
        return True
    
    async def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step"""
        # This would normally call the actual function
        # For now, just return a success result
        return {
            "step_id": step["id"],
            "success": True,
            "execution_time": 0.1,
            "data": {}
        }

class AutomaticExecutionStrategy(ExecutionStrategy):
    """Fast execution without validation between steps"""
    
    async def execute(
        self, 
        procedure: Procedure, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute procedure automatically without validation"""
        start_time = datetime.datetime.now()
        results = []
        success = True
        
        # Initialize execution state
        execution_state = context.copy()
        execution_state["strategy"] = "automatic"
        execution_state["execution_history"] = []
        
        # Check if procedure is chunked
        if procedure.is_chunked:
            # Execute chunks
            chunks = self._get_chunks(procedure)
            
            for chunk_id, chunk_steps in chunks.items():
                # Execute chunk as a unit
                chunk_result = await self._execute_chunk(
                    chunk_steps, 
                    execution_state, 
                    chunk_id
                )
                
                results.extend(chunk_result["results"])
                
                # Update execution state
                execution_state[f"chunk_{chunk_id}_result"] = chunk_result
                for step_result in chunk_result["results"]:
                    step_id = step_result["step_id"]
                    execution_state[f"step_{step_id}_result"] = step_result
                    execution_state["execution_history"].append({
                        "step_id": step_id,
                        "chunk_id": chunk_id,
                        "success": step_result["success"],
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                
                # Check for failure
                if not chunk_result["success"]:
                    success = False
                    break
        else:
            # Execute steps sequentially without validation
            for step in procedure.steps:
                # Execute the step
                step_result = await self._execute_step(step, execution_state)
                results.append(step_result)
                
                # Update execution state
                execution_state[f"step_{step['id']}_result"] = step_result
                execution_state["execution_history"].append({
                    "step_id": step["id"],
                    "function": step["function"],
                    "success": step_result["success"],
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                # Check for failure
                if not step_result["success"]:
                    success = False
                    break
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        return {
            "success": success,
            "results": results,
            "execution_time": execution_time,
            "strategy": "automatic"
        }
    
    def _get_chunks(self, procedure: Procedure) -> Dict[str, List[Dict[str, Any]]]:
        """Get chunks from a procedure"""
        chunks = {}
        
        for chunk_id, step_ids in procedure.chunked_steps.items():
            # Get steps for this chunk
            chunk_steps = [step for step in procedure.steps if step["id"] in step_ids]
            chunks[chunk_id] = chunk_steps
        
        return chunks
    
    async def _execute_chunk(
        self, 
        steps: List[Dict[str, Any]], 
        context: Dict[str, Any], 
        chunk_id: str
    ) -> Dict[str, Any]:
        """Execute a chunk of steps as a unit"""
        results = []
        success = True
        start_time = datetime.datetime.now()
        
        for step in steps:
            # Execute the step
            step_result = await self._execute_step(step, context)
            results.append(step_result)
            
            # Check for failure
            if not step_result["success"]:
                success = False
                break
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        return {
            "success": success,
            "results": results,
            "execution_time": execution_time,
            "chunk_id": chunk_id
        }
    
    async def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step"""
        # This would normally call the actual function
        # For now, just return a success result
        return {
            "step_id": step["id"],
            "success": True,
            "execution_time": 0.05,  # Faster than deliberate execution
            "data": {}
        }

class AdaptiveExecutionStrategy(ExecutionStrategy):
    """Adapts execution based on context and feedback"""
    
    async def execute(
        self, 
        procedure: Procedure, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute procedure with adaptive strategy selection"""
        start_time = datetime.datetime.now()
        results = []
        success = True
        
        # Initialize execution state
        execution_state = context.copy()
        execution_state["strategy"] = "adaptive"
        execution_state["execution_history"] = []
        
        # Determine initial execution mode based on proficiency
        deliberate_execution = procedure.proficiency < 0.8
        
        # Execute steps with adaptive strategy
        for i, step in enumerate(procedure.steps):
            # Decide execution strategy for this step
            step_strategy = self._select_step_strategy(step, execution_state, deliberate_execution)
            
            # Execute step with selected strategy
            if step_strategy == "deliberate":
                # Careful execution with validation
                if not self._validate_preconditions(step, execution_state):
                    results.append({
                        "step_id": step["id"],
                        "success": False,
                        "error": "Preconditions not met",
                        "execution_time": 0.0,
                        "strategy": "deliberate"
                    })
                    success = False
                    break
                
                step_result = await self._execute_step(step, execution_state)
                step_result["strategy"] = "deliberate"
                
                if not self._validate_postconditions(step, execution_state, step_result):
                    step_result["success"] = False
                    step_result["error"] = "Postconditions not met"
                    success = False
                    results.append(step_result)
                    break
            else:
                # Fast execution without validation
                step_result = await self._execute_step(step, execution_state)
                step_result["strategy"] = "automatic"
            
            # Add to results
            results.append(step_result)
            
            # Update execution state
            execution_state[f"step_{step['id']}_result"] = step_result
            execution_state["execution_history"].append({
                "step_id": step["id"],
                "function": step["function"],
                "success": step_result["success"],
                "strategy": step_strategy,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Check for failure and adapt
            if not step_result["success"]:
                # Adapt strategy on failure
                deliberate_execution = True
                
                # Try to recover if possible
                if i < len(procedure.steps) - 1:
                    recovery_successful = await self._attempt_recovery(
                        step, 
                        procedure.steps[i+1:], 
                        execution_state
                    )
                    
                    if recovery_successful:
                        # Continue execution
                        continue
                
                success = False
                break
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        return {
            "success": success,
            "results": results,
            "execution_time": execution_time,
            "strategy": "adaptive",
            "adaptations": execution_state.get("adaptations", [])
        }
    
    def _select_step_strategy(
        self, 
        step: Dict[str, Any], 
        state: Dict[str, Any], 
        default_deliberate: bool
    ) -> str:
        """Select execution strategy for a step"""
        # Check if step has explicit strategy preference
        if "preferred_strategy" in step:
            return step["preferred_strategy"]
        
        # Check if step is high-risk
        if "risk_level" in step and step["risk_level"] > 0.7:
            return "deliberate"
        
        # Check execution history for this step
        history = state.get("execution_history", [])
        step_history = [h for h in history if h.get("step_id") == step["id"]]
        
        if step_history:
            # Check success rate
            success_rate = sum(1 for h in step_history if h.get("success", False)) / len(step_history)
            
            if success_rate < 0.8:
                # Low success rate, use deliberate execution
                return "deliberate"
        
        # Use default (based on overall procedure proficiency)
        return "deliberate" if default_deliberate else "automatic"
    
    def _validate_preconditions(self, step: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """Validate preconditions for a step"""
        preconditions = step.get("preconditions", {})
        
        for key, value in preconditions.items():
            if key not in state:
                return False
            
            # Compare values
            if isinstance(value, (list, tuple, set)):
                if state[key] not in value:
                    return False
            elif isinstance(value, dict) and "min" in value and "max" in value:
                if not (value["min"] <= state[key] <= value["max"]):
                    return False
            elif state[key] != value:
                return False
        
        return True
    
    def _validate_postconditions(
        self, 
        step: Dict[str, Any], 
        state: Dict[str, Any], 
        result: Dict[str, Any]
    ) -> bool:
        """Validate postconditions for a step"""
        postconditions = step.get("postconditions", {})
        
        # First check result success
        if not result.get("success", False):
            return False
        
        # Check explicit postconditions
        for key, value in postconditions.items():
            if key not in state:
                return False
            
            # Compare values
            if isinstance(value, (list, tuple, set)):
                if state[key] not in value:
                    return False
            elif isinstance(value, dict) and "min" in value and "max" in value:
                if not (value["min"] <= state[key] <= value["max"]):
                    return False
            elif state[key] != value:
                return False
        
        return True
    
    async def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step"""
        # This would normally call the actual function
        # For now, just return a success result
        return {
            "step_id": step["id"],
            "success": True,
            "execution_time": 0.07,  # Between deliberate and automatic
            "data": {}
        }
    
    async def _attempt_recovery(
        self, 
        failed_step: Dict[str, Any], 
        remaining_steps: List[Dict[str, Any]], 
        state: Dict[str, Any]
    ) -> bool:
        """Attempt to recover from a step failure"""
        # Track adaptation
        if "adaptations" not in state:
            state["adaptations"] = []
        
        state["adaptations"].append({
            "type": "recovery_attempt",
            "step_id": failed_step["id"],
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Try a retry with modified parameters
        modified_params = self._modify_parameters(failed_step.get("parameters", {}))
        
        retry_step = failed_step.copy()
        retry_step["parameters"] = modified_params
        retry_step["is_recovery"] = True
        
        # Execute with modified parameters
        retry_result = await self._execute_step(retry_step, state)
        
        if retry_result.get("success", False):
            # Recovery successful
            state["adaptations"][-1]["result"] = "success"
            state["adaptations"][-1]["method"] = "parameter_modification"
            
            # Update execution state
            state[f"step_{failed_step['id']}_result"] = retry_result
            state["execution_history"].append({
                "step_id": failed_step["id"],
                "function": failed_step["function"],
                "success": True,
                "strategy": "recovery",
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            return True
        
        # Recovery failed
        state["adaptations"][-1]["result"] = "failure"
        return False
    
    def _modify_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Modify parameters for recovery attempt"""
        modified = parameters.copy()
        
        # Modify numeric parameters slightly
        for key, value in parameters.items():
            if isinstance(value, (int, float)):
                # Adjust by small percentage
                if random.random() < 0.5:
                    modified[key] = value * 1.1  # Increase by 10%
                else:
                    modified[key] = value * 0.9  # Decrease by 10%
        
        return modified

class StrategySelector:
    """Selects appropriate execution strategy based on context"""
    
    def __init__(self):
        self.strategies = {}  # id -> ExecutionStrategy
        self.execution_history = []
        self.max_history = 50
    
    def register_strategy(self, strategy: ExecutionStrategy) -> None:
        """Register an execution strategy"""
        self.strategies[strategy.id] = strategy
    
    def select_strategy(self, context: Dict[str, Any], procedure: Procedure) -> ExecutionStrategy:
        """Select the most appropriate execution strategy"""
        if not self.strategies:
            # No strategies registered, return a default
            return ExecutionStrategy(
                id="default", 
                name="Default Strategy", 
                description="Default execution strategy"
            )
        
        # Calculate scores for each strategy
        scores = []
        for strategy_id, strategy in self.strategies.items():
            score = strategy.should_select(context, procedure)
            scores.append((strategy_id, score))
        
        # Get highest scoring strategy
        best_strategy_id, best_score = max(scores, key=lambda x: x[1])
        
        # Record selection
        self.execution_history.append({
            "strategy_id": best_strategy_id,
            "score": best_score,
            "context": {k: v for k, v in context.items() if not isinstance(v, (dict, list))},
            "procedure_id": procedure.id,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Trim history
        if len(self.execution_history) > self.max_history:
            self.execution_history = self.execution_history[-self.max_history:]
        
        return self.strategies[best_strategy_id]

# ============================================================================
# 8. MEMORY CONSOLIDATION FOR PROCEDURAL KNOWLEDGE
# ============================================================================

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

# ============================================================================
# 9. GRAPH-BASED REPRESENTATION FOR FLEXIBLE EXECUTION
# ============================================================================

class ProcedureGraph(BaseModel):
    """Graph representation of a procedure for flexible execution"""
    nodes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    entry_points: List[str] = Field(default_factory=list)
    exit_points: List[str] = Field(default_factory=list)
    
    def add_node(self, node_id: str, data: Dict[str, Any]) -> None:
        """Add a node to the graph"""
        self.nodes[node_id] = data
    
    def add_edge(self, from_id: str, to_id: str, properties: Dict[str, Any] = None) -> None:
        """Add an edge to the graph"""
        self.edges.append({
            "from": from_id,
            "to": to_id,
            "properties": properties or {}
        })
    
    def find_execution_path(
        self, 
        context: Dict[str, Any],
        goal: Dict[str, Any]
    ) -> List[str]:
        """Find execution path through the graph given context and goal"""
        if not self.entry_points:
            return []
        
        # Find all paths from entry to exit points
        all_paths = []
        
        for entry in self.entry_points:
            for exit_point in self.exit_points:
                paths = self._find_all_paths(entry, exit_point)
                all_paths.extend(paths)
        
        if not all_paths:
            return []
        
        # Score each path based on context and goal
        scored_paths = []
        
        for path in all_paths:
            score = self._score_path(path, context, goal)
            scored_paths.append((path, score))
        
        # Return highest scoring path
        best_path, _ = max(scored_paths, key=lambda x: x[1])
        return best_path
    
    def _find_all_paths(self, start: str, end: str, path: List[str] = None) -> List[List[str]]:
        """Find all paths between two nodes"""
        if path is None:
            path = []
        
        path = path + [start]
        
        if start == end:
            return [path]
        
        if start not in self.nodes:
            return []
        
        paths = []
        
        # Find outgoing edges
        for edge in self.edges:
            if edge["from"] == start and edge["to"] not in path:
                new_paths = self._find_all_paths(edge["to"], end, path)
                for new_path in new_paths:
                    paths.append(new_path)
        
        return paths
    
    def _score_path(self, path: List[str], context: Dict[str, Any], goal: Dict[str, Any]) -> float:
        """Score a path based on context and goal"""
        score = 0.5  # Base score
        
        # Check context match for each node
        for node_id in path:
            node_data = self.nodes.get(node_id, {})
            preconditions = node_data.get("preconditions", {})
            
            # Check if preconditions match context
            matches = 0
            total = len(preconditions)
            
            for key, value in preconditions.items():
                if key in context:
                    if isinstance(value, (list, tuple, set)):
                        if context[key] in value:
                            matches += 1
                    elif isinstance(value, dict) and "min" in value and "max" in value:
                        if value["min"] <= context[key] <= value["max"]:
                            matches += 1
                    elif context[key] == value:
                        matches += 1
            
            # Add to score based on precondition match percentage
            if total > 0:
                score += 0.1 * (matches / total)
        
        # Check if path achieves goal
        last_node = self.nodes.get(path[-1], {})
        postconditions = last_node.get("postconditions", {})
        
        goal_matches = 0
        goal_total = len(goal)
        
        for key, value in goal.items():
            if key in postconditions:
                if isinstance(value, (list, tuple, set)):
                    if postconditions[key] in value:
                        goal_matches += 1
                elif isinstance(value, dict) and "min" in value and "max" in value:
                    if value["min"] <= postconditions[key] <= value["max"]:
                        goal_matches += 1
                elif postconditions[key] == value:
                    goal_matches += 1
        
        # Add to score based on goal match percentage
        if goal_total > 0:
            goal_score = goal_matches / goal_total
            score += 0.4 * goal_score  # Goal achievement is important
        
        return score
    
    @classmethod
    def from_procedure(cls, procedure: Procedure) -> 'ProcedureGraph':
        """Convert a standard procedure to a graph representation"""
        graph = cls()
        
        # Create nodes for each step
        for i, step in enumerate(procedure.steps):
            node_id = f"node_{step['id']}"
            
            # Extract preconditions and postconditions
            preconditions = step.get("preconditions", {})
            postconditions = step.get("postconditions", {})
            
            # Create node
            graph.add_node(node_id, {
                "step_id": step["id"],
                "function": step["function"],
                "parameters": step.get("parameters", {}),
                "description": step.get("description", f"Step {i+1}"),
                "preconditions": preconditions,
                "postconditions": postconditions
            })
            
            # First step is an entry point
            if i == 0:
                graph.entry_points.append(node_id)
            
            # Last step is an exit point
            if i == len(procedure.steps) - 1:
                graph.exit_points.append(node_id)
        
        # Create edges for sequential execution
        for i in range(len(procedure.steps) - 1):
            from_id = f"node_{procedure.steps[i]['id']}"
            to_id = f"node_{procedure.steps[i+1]['id']}"
            
            graph.add_edge(from_id, to_id)
        
        return graph

# ============================================================================
# 10. META-LEARNING FOR TRANSFER OPTIMIZATION
# ============================================================================

class TransferLearningOptimizer:
    """Optimizes transfer learning between domains using meta-learning"""
    
    def __init__(self):
        self.domain_embeddings = {}
        self.transfer_success_history = []
        self.domain_similarities = {}  # pair_key -> similarity
        self.max_history = 50
    
    async def optimize_transfer(
        self,
        source_procedure: Procedure,
        target_domain: str
    ) -> Dict[str, Any]:
        """Optimize transfer from source procedure to target domain"""
        # Get domain embeddings
        source_embedding = await self._get_domain_embedding(source_procedure.domain)
        target_embedding = await self._get_domain_embedding(target_domain)
        
        # Calculate similarity
        similarity = self._calculate_domain_similarity(source_procedure.domain, target_domain)
        
        # Determine transfer strategy based on similarity
        if similarity > 0.8:
            # High similarity - direct transfer with minimal adaptation
            strategy = "direct_transfer"
            adaptation_level = "minimal"
        elif similarity > 0.5:
            # Medium similarity - transfer with parameter adaptation
            strategy = "parameter_adaptation"
            adaptation_level = "moderate"
        else:
            # Low similarity - transfer with structural adaptation
            strategy = "structural_adaptation"
            adaptation_level = "extensive"
        
        # Identify optimal mappings for transfer
        mappings = await self._identify_optimal_mappings(
            source_procedure, 
            target_domain,
            strategy
        )
        
        # Estimate success probability
        success_probability = self._estimate_transfer_success(
            source_procedure.domain,
            target_domain,
            strategy
        )
        
        # Create transfer plan
        transfer_plan = {
            "source_domain": source_procedure.domain,
            "target_domain": target_domain,
            "domain_similarity": similarity,
            "transfer_strategy": strategy,
            "adaptation_level": adaptation_level,
            "mappings": mappings,
            "estimated_success": success_probability
        }
        
        return transfer_plan
    
    async def _get_domain_embedding(self, domain: str) -> List[float]:
        """Get embedding vector for a domain"""
        # Check if embedding already exists
        if domain in self.domain_embeddings:
            return self.domain_embeddings[domain]
        
        # In a real implementation, this would be a learned embedding
        # For now, generate a random embedding
        embedding = [random.uniform(-1, 1) for _ in range(10)]
        self.domain_embeddings[domain] = embedding
        
        return embedding
    
    def _calculate_domain_similarity(self, domain1: str, domain2: str) -> float:
        """Calculate similarity between domains"""
        # Check if already calculated
        pair_key = f"{domain1}:{domain2}"
        reverse_key = f"{domain2}:{domain1}"
        
        if pair_key in self.domain_similarities:
            return self.domain_similarities[pair_key]
        elif reverse_key in self.domain_similarities:
            return self.domain_similarities[reverse_key]
        
        # Calculate similarity from embeddings if available
        if domain1 in self.domain_embeddings and domain2 in self.domain_embeddings:
            embedding1 = self.domain_embeddings[domain1]
            embedding2 = self.domain_embeddings[domain2]
            
            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            norm1 = sum(a * a for a in embedding1) ** 0.5
            norm2 = sum(b * b for b in embedding2) ** 0.5
            
            if norm1 * norm2 == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm1 * norm2)
        else:
            # Default similarity based on domain name similarity
            common_substring = self._longest_common_substring(domain1, domain2)
            similarity = len(common_substring) / max(len(domain1), len(domain2))
        
        # Store for future reference
        self.domain_similarities[pair_key] = similarity
        
        return similarity
    
    def _longest_common_substring(self, str1: str, str2: str) -> str:
        """Find longest common substring between two strings"""
        if not str1 or not str2:
            return ""
            
        m = len(str1)
        n = len(str2)
        
        # Create DP table
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        
        # Variables to store longest substring info
        max_length = 0
        end_pos = 0
        
        # Fill DP table
        for i in range(1, m+1):
            for j in range(1, n+1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    
                    if dp[i][j] > max_length:
                        max_length = dp[i][j]
                        end_pos = i
        
        # Extract substring
        return str1[end_pos - max_length:end_pos]
    
    async def _identify_optimal_mappings(
        self,
        procedure: Procedure,
        target_domain: str,
        strategy: str
    ) -> List[Dict[str, Any]]:
        """Identify optimal function and parameter mappings"""
        mappings = []
        
        if strategy == "direct_transfer":
            # Simple 1:1 mappings
            for step in procedure.steps:
                mappings.append({
                    "source_function": step["function"],
                    "target_function": step["function"],
                    "parameters": step.get("parameters", {}),
                    "confidence": 0.9
                })
        elif strategy == "parameter_adaptation":
            # Map functions directly but adapt parameters
            for step in procedure.steps:
                # Get adapted parameters
                adapted_params = await self._adapt_parameters(
                    step.get("parameters", {}),
                    procedure.domain,
                    target_domain
                )
                
                mappings.append({
                    "source_function": step["function"],
                    "target_function": step["function"],
                    "source_parameters": step.get("parameters", {}),
                    "target_parameters": adapted_params,
                    "confidence": 0.7
                })
        else:  # structural_adaptation
            # Look for equivalent functions in target domain
            for step in procedure.steps:
                # Look for equivalent function
                equivalent = await self._find_equivalent_function(
                    step["function"],
                    target_domain
                )
                
                # Get adapted parameters
                adapted_params = await self._adapt_parameters(
                    step.get("parameters", {}),
                    procedure.domain,
                    target_domain
                )
                
                mappings.append({
                    "source_function": step["function"],
                    "target_function": equivalent or step["function"],
                    "source_parameters": step.get("parameters", {}),
                    "target_parameters": adapted_params,
                    "confidence": 0.5 if equivalent else 0.3
                })
        
        return mappings
    
    async def _adapt_parameters(
        self,
        parameters: Dict[str, Any],
        source_domain: str,
        target_domain: str
    ) -> Dict[str, Any]:
        """Adapt parameters from source to target domain"""
        adapted = {}
        
        for key, value in parameters.items():
            # Check past transfers to find typical mappings
            mapping = self._find_parameter_mapping(key, value, source_domain, target_domain)
            
            if mapping:
                adapted[key] = mapping
            else:
                # Default: keep original value
                adapted[key] = value
        
        return adapted
    
    async def _find_equivalent_function(self, function: str, target_domain: str) -> Optional[str]:
        """Find equivalent function in target domain"""
        # Check past transfer history for this function
        for history in self.transfer_success_history:
            if (history["source_domain"] == target_domain and
                function in history.get("function_mappings", {})):
                return history["function_mappings"][function]
        
        # No known mapping
        return None
    
    def _find_parameter_mapping(
        self, 
        param_key: str, 
        param_value: Any, 
        source_domain: str, 
        target_domain: str
    ) -> Any:
        """Find mapping for a parameter based on past transfers"""
        for history in self.transfer_success_history:
            if (history["source_domain"] == source_domain and
                history["target_domain"] == target_domain and
                param_key in history.get("parameter_mappings", {}) and
                str(param_value) in history["parameter_mappings"][param_key]):
                return history["parameter_mappings"][param_key][str(param_value)]
        
        # No known mapping
        return param_value
    
    def _estimate_transfer_success(
        self,
        source_domain: str,
        target_domain: str,
        strategy: str
    ) -> float:
        """Estimate probability of successful transfer"""
        # Check for similar past transfers
        similar_transfers = [h for h in self.transfer_success_history
                          if h["source_domain"] == source_domain and 
                             h["target_domain"] == target_domain]
        
        if similar_transfers:
            # Calculate average success rate
            success_rate = sum(h["success_rate"] for h in similar_transfers) / len(similar_transfers)
            return success_rate
        
        # Base on domain similarity
        similarity = self._calculate_domain_similarity(source_domain, target_domain)
        
        # Adjust based on strategy
        if strategy == "direct_transfer":
            return similarity * 0.9  # High confidence if using direct transfer
        elif strategy == "parameter_adaptation":
            return similarity * 0.7  # Medium confidence with parameter adaptation
        else:  # structural_adaptation
            return similarity * 0.5  # Lower confidence with structural changes
    
    def update_from_transfer_result(
        self,
        source_domain: str,
        target_domain: str,
        success_rate: float,
        mappings: Dict[str, Any]
    ) -> None:
        """Update optimizer based on transfer results"""
        # Record transfer result
        transfer_record = {
            "source_domain": source_domain,
            "target_domain": target_domain,
            "success_rate": success_rate,
            "function_mappings": {},
            "parameter_mappings": {},
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Extract mappings
        for mapping in mappings:
            source_func = mapping.get("source_function")
            target_func = mapping.get("target_function")
            
            if source_func and target_func:
                transfer_record["function_mappings"][source_func] = target_func
            
            # Extract parameter mappings
            source_params = mapping.get("source_parameters", {})
            target_params = mapping.get("target_parameters", {})
            
            for key in source_params:
                if key in target_params:
                    if key not in transfer_record["parameter_mappings"]:
                        transfer_record["parameter_mappings"][key] = {}
                    
                    transfer_record["parameter_mappings"][key][str(source_params[key])] = target_params[key]
        
        # Add to history
        self.transfer_success_history.append(transfer_record)
        
        # Trim history if needed
        if len(self.transfer_success_history) > self.max_history:
            self.transfer_success_history = self.transfer_success_history[-self.max_history:]
        
        # Update domain similarity based on transfer success
        pair_key = f"{source_domain}:{target_domain}"
        
        # Adjust similarity based on success
        if pair_key in self.domain_similarities:
            current = self.domain_similarities[pair_key]
            # Move similarity closer to success rate
            self.domain_similarities[pair_key] = current * 0.7 + success_rate * 0.3

# ============================================================================
# ENHANCED PROCEDURAL MEMORY MANAGER
# ============================================================================

class EnhancedProceduralMemoryManager(ProceduralMemoryManager):
    """Enhanced version of ProceduralMemoryManager with new capabilities"""
    
    def __init__(self, memory_core=None, knowledge_core=None):
        # Initialize base class
        super().__init__(memory_core, knowledge_core)
        
        # Initialize new components
        self.observation_learner = ObservationLearner()
        self.causal_model = CausalModel()
        self.working_memory = WorkingMemoryController()
        self.parameter_optimizer = ParameterOptimizer()
        self.strategy_selector = StrategySelector()
        self.memory_consolidator = ProceduralMemoryConsolidator(memory_core)
        self.transfer_optimizer = TransferLearningOptimizer()
        
        # Initialize execution strategies
        self._init_execution_strategies()
        
        # Add hierarchical procedure storage
        self.hierarchical_procedures = {}  # name -> HierarchicalProcedure
        
        # Add temporal procedure graph storage
        self.temporal_graphs = {}  # id -> TemporalProcedureGraph
        
        # Add procedure graph storage
        self.procedure_graphs = {}  # id -> ProcedureGraph
        
        # Initialization flag
        self.enhanced_initialized = False
    
    async def initialize_enhanced_components(self):
        """Initialize enhanced components and integrations"""
        if self.enhanced_initialized:
            return
        
        # Initialize base components first
        if not self.initialized:
            await self.initialize()
        
        # Set up error recovery patterns
        self._initialize_causal_model()
        
        # Integrate with memory core if available
        if self.memory_core:
            await self.integrate_with_memory_core()
        
        # Integrate with knowledge core if available
        if self.knowledge_core:
            await self.integrate_with_knowledge_core()
        
        # Initialize pre-built templates for common patterns
        self._initialize_common_templates()
        
        self.enhanced_initialized = True
        logger.info("Enhanced procedural memory components initialized")
    
    def _initialize_causal_model(self):
        """Initialize causal model with common error patterns"""
        # Define common error causes for different error types
        self.causal_model.causes = {
            "execution_failure": [
                {
                    "cause": "invalid_parameters",
                    "description": "Invalid parameters provided to function",
                    "probability": 0.6,
                    "context_factors": {}
                },
                {
                    "cause": "missing_precondition",
                    "description": "Required precondition not met",
                    "probability": 0.4,
                    "context_factors": {}
                }
            ],
            "timeout": [
                {
                    "cause": "slow_execution",
                    "description": "Operation taking too long to complete",
                    "probability": 0.5,
                    "context_factors": {}
                },
                {
                    "cause": "resource_contention",
                    "description": "Resources needed are being used by another process",
                    "probability": 0.3,
                    "context_factors": {}
                }
            ],
            "parameter_error": [
                {
                    "cause": "type_mismatch",
                    "description": "Parameter type does not match expected type",
                    "probability": 0.7,
                    "context_factors": {}
                },
                {
                    "cause": "out_of_range",
                    "description": "Parameter value outside of valid range",
                    "probability": 0.5,
                    "context_factors": {}
                }
            ]
        }
        
        # Define common interventions for each cause
        self.causal_model.interventions = {
            "invalid_parameters": [
                {
                    "type": "modify_parameters",
                    "description": "Modify parameters to valid values",
                    "effectiveness": 0.8
                },
                {
                    "type": "check_documentation",
                    "description": "Check documentation for correct parameter format",
                    "effectiveness": 0.6
                }
            ],
            "missing_precondition": [
                {
                    "type": "establish_precondition",
                    "description": "Ensure required precondition is met before execution",
                    "effectiveness": 0.9
                },
                {
                    "type": "alternative_approach",
                    "description": "Use an alternative approach that doesn't require this precondition",
                    "effectiveness": 0.5
                }
            ],
            "slow_execution": [
                {
                    "type": "optimization",
                    "description": "Optimize the operation for faster execution",
                    "effectiveness": 0.7
                },
                {
                    "type": "incremental_execution",
                    "description": "Break operation into smaller steps",
                    "effectiveness": 0.6
                }
            ],
            "resource_contention": [
                {
                    "type": "retry_later",
                    "description": "Retry operation after a delay",
                    "effectiveness": 0.8
                },
                {
                    "type": "release_resources",
                    "description": "Release unused resources before execution",
                    "effectiveness": 0.7
                }
            ],
            "type_mismatch": [
                {
                    "type": "convert_type",
                    "description": "Convert parameter to required type",
                    "effectiveness": 0.9
                }
            ],
            "out_of_range": [
                {
                    "type": "clamp_value",
                    "description": "Clamp parameter value to valid range",
                    "effectiveness": 0.8
                }
            ]
        }
    
    def _initialize_common_templates(self):
        """Initialize common procedure templates"""
        # Define common templates for navigation
        navigation_template = ChunkTemplate(
            id="template_navigation",
            name="Navigation Template",
            description="Template for navigation operations",
            actions=[
                ActionTemplate(
                    action_type="move",
                    intent="navigation",
                    parameters={"destination": "target_location"},
                    domain_mappings={
                        "gaming": {
                            "function": "move_character",
                            "parameters": {"location": "target_location"},
                            "description": "Move character to location"
                        },
                        "ui": {
                            "function": "navigate_to",
                            "parameters": {"page": "target_location"},
                            "description": "Navigate to page"
                        }
                    }
                )
            ],
            domains=["gaming", "ui"],
            success_rate={"gaming": 0.9, "ui": 0.9},
            execution_count={"gaming": 10, "ui": 10}
        )
        
        # Define template for interaction
        interaction_template = ChunkTemplate(
            id="template_interaction",
            name="Interaction Template",
            description="Template for interaction operations",
            actions=[
                ActionTemplate(
                    action_type="select",
                    intent="interaction",
                    parameters={"target": "interaction_target"},
                    domain_mappings={
                        "gaming": {
                            "function": "select_object",
                            "parameters": {"object": "interaction_target"},
                            "description": "Select object in game"
                        },
                        "ui": {
                            "function": "click_element",
                            "parameters": {"element": "interaction_target"},
                            "description": "Click UI element"
                        }
                    }
                ),
                ActionTemplate(
                    action_type="activate",
                    intent="interaction",
                    parameters={"action": "interaction_action"},
                    domain_mappings={
                        "gaming": {
                            "function": "use_object",
                            "parameters": {"action": "interaction_action"},
                            "description": "Use selected object"
                        },
                        "ui": {
                            "function": "submit_form",
                            "parameters": {"action": "interaction_action"},
                            "description": "Submit form with action"
                        }
                    }
                )
            ],
            domains=["gaming", "ui"],
            success_rate={"gaming": 0.85, "ui": 0.9},
            execution_count={"gaming": 8, "ui": 12}
        )
        
        # Add templates to library
        self.chunk_library.add_chunk_template(navigation_template)
        self.chunk_library.add_chunk_template(interaction_template)
    
    def _init_execution_strategies(self):
        """Initialize execution strategies"""
        # Create deliberate execution strategy
        deliberate = DeliberateExecutionStrategy(
            id="deliberate",
            name="Deliberate Execution",
            description="Careful step-by-step execution with validation",
            selection_criteria={
                "proficiency": {"min": 0.0, "max": 0.7},
                "importance": {"min": 0.7, "max": 1.0},
                "risk_level": {"min": 0.7, "max": 1.0}
            }
        )
        
        # Create automatic execution strategy
        automatic = AutomaticExecutionStrategy(
            id="automatic",
            name="Automatic Execution",
            description="Fast execution with minimal monitoring",
            selection_criteria={
                "proficiency": {"min": 0.8, "max": 1.0},
                "importance": {"min": 0.0, "max": 0.6},
                "risk_level": {"min": 0.0, "max": 0.3}
            }
        )
        
        # Create adaptive execution strategy
        adaptive = AdaptiveExecutionStrategy(
            id="adaptive",
            name="Adaptive Execution",
            description="Execution that adapts based on context and feedback",
            selection_criteria={
                "proficiency": {"min": 0.4, "max": 0.9},
                "importance": {"min": 0.3, "max": 0.8},
                "risk_level": {"min": 0.3, "max": 0.7},
                "adaptivity_required": True
            }
        )
        
        # Register strategies
        self.strategy_selector.register_strategy(deliberate)
        self.strategy_selector.register_strategy(automatic)
        self.strategy_selector.register_strategy(adaptive)
    
    # -------------------------------------------------------------------------
    # New Function Tools
    # -------------------------------------------------------------------------
    
    @function_tool
    async def learn_from_demonstration(
        self, 
        observation_sequence: List[Dict[str, Any]], 
        domain: str,
        name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Learn a procedure from a sequence of observed actions
        
        Args:
            observation_sequence: Sequence of observed actions with state
            domain: Domain for the new procedure
            name: Optional name for the new procedure
            
        Returns:
            Information about the learned procedure
        """
        # Learn from observations
        procedure_data = await self.observation_learner.learn_from_demonstration(
            observation_sequence=observation_sequence,
            domain=domain
        )
        
        # Use provided name if available
        if name:
            procedure_data["name"] = name
        
        # Create the procedure
        ctx = RunContextWrapper(context=self)
        procedure_result = await add_procedure(
            ctx,
            name=procedure_data["name"],
            steps=procedure_data["steps"],
            description=procedure_data["description"],
            domain=domain
        )
        
        # Add confidence information
        procedure_result["confidence"] = procedure_data["confidence"]
        procedure_result["learned_from_observations"] = True
        procedure_result["observation_count"] = procedure_data["observation_count"]
        
        return procedure_result
    
    @function_tool
    async def create_hierarchical_procedure(
        self,
        name: str,
        description: str,
        domain: str,
        steps: List[Dict[str, Any]],
        goal_state: Dict[str, Any] = None,
        preconditions: Dict[str, Any] = None,
        postconditions: Dict[str, Any] = None,
        parent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a hierarchical procedure
        
        Args:
            name: Name of the procedure
            description: Description of what the procedure does
            domain: Domain for the procedure
            steps: List of step definitions
            goal_state: Optional goal state for the procedure
            preconditions: Optional preconditions
            postconditions: Optional postconditions
            parent_id: Optional parent procedure ID
            
        Returns:
            Information about the created procedure
        """
        # Generate ID
        proc_id = f"hierproc_{int(datetime.datetime.now().timestamp())}_{random.randint(1000, 9999)}"
        
        # Create the hierarchical procedure
        procedure = HierarchicalProcedure(
            id=proc_id,
            name=name,
            description=description,
            domain=domain,
            steps=steps,
            goal_state=goal_state or {},
            preconditions=preconditions or {},
            postconditions=postconditions or {},
            parent_id=parent_id
        )
        
        # Store the procedure
        self.hierarchical_procedures[name] = procedure
        
        # Create standard procedure as well
        ctx = RunContextWrapper(context=self)
        standard_proc = await add_procedure(
            ctx,
            name=name,
            steps=steps,
            description=description,
            domain=domain
        )
        
        # If has parent, update parent's children list
        if parent_id:
            for parent in self.hierarchical_procedures.values():
                if parent.id == parent_id:
                    parent.add_child(proc_id)
                    break
        
        # Return information
        return {
            "id": proc_id,
            "name": name,
            "domain": domain,
            "steps_count": len(steps),
            "standard_procedure_id": standard_proc["procedure_id"],
            "hierarchical": True,
            "parent_id": parent_id
        }
    
    @function_tool
    async def execute_hierarchical_procedure(
        self,
        name: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a hierarchical procedure
        
        Args:
            name: Name of the procedure
            context: Execution context
            
        Returns:
            Execution results
        """
        if name not in self.hierarchical_procedures:
            return {"error": f"Hierarchical procedure '{name}' not found"}
        
        procedure = self.hierarchical_procedures[name]
        
        # Create trace for execution
        with trace(workflow_name="execute_hierarchical_procedure"):
            # Check preconditions
            if not procedure.meets_preconditions(context or {}):
                return {
                    "success": False,
                    "error": "Preconditions not met",
                    "procedure_name": name
                }
            
            # Initialize context if needed
            execution_context = context.copy() if context else {}
            
            # Set procedure context
            execution_context["current_procedure"] = name
            execution_context["hierarchical"] = True
            
            # Update working memory
            self.working_memory.update(execution_context, procedure)
            
            # Select execution strategy
            strategy = self.strategy_selector.select_strategy(execution_context, procedure)
            
            # Execute with selected strategy
            start_time = datetime.datetime.now()
            execution_result = await strategy.execute(procedure, execution_context)
            
            # Calculate execution time
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Update procedure statistics
            self._update_hierarchical_stats(procedure, execution_time, execution_result["success"])
            
            # Verify goal state was achieved
            goal_achieved = True
            if procedure.goal_state:
                for key, value in procedure.goal_state.items():
                    if key not in execution_context or execution_context[key] != value:
                        goal_achieved = False
                        break
            
            # Add information to result
            execution_result["procedure_name"] = name
            execution_result["hierarchical"] = True
            execution_result["goal_achieved"] = goal_achieved
            execution_result["strategy_id"] = strategy.id
            execution_result["working_memory"] = self.working_memory.get_attention_focus()
            
            return execution_result
    
    def _update_hierarchical_stats(self, procedure: HierarchicalProcedure, execution_time: float, success: bool):
        """Update statistics for a hierarchical procedure"""
        # Update count
        procedure.execution_count += 1
        if success:
            procedure.successful_executions += 1
        
        # Update average time
        if procedure.execution_count == 1:
            procedure.average_execution_time = execution_time
        else:
            procedure.average_execution_time = (
                (procedure.average_execution_time * (procedure.execution_count - 1) + execution_time) / 
                procedure.execution_count
            )
        
        # Update proficiency based on multiple factors
        count_factor = min(procedure.execution_count / 50, 1.0)
        success_rate = procedure.successful_executions / max(1, procedure.execution_count)
        
        # Calculate time factor (lower times = higher proficiency)
        if procedure.execution_count < 2:
            time_factor = 0.5  # Default for first execution
        else:
            # Normalize time - lower is better
            time_factor = max(0.0, 1.0 - (procedure.average_execution_time / 10.0))
            time_factor = min(1.0, time_factor)
        
        # Combine factors with weights
        procedure.proficiency = (count_factor * 0.3) + (success_rate * 0.5) + (time_factor * 0.2)
        
        # Update timestamps
        procedure.last_execution = datetime.datetime.now().isoformat()
        procedure.last_updated = datetime.datetime.now().isoformat()
    
    @function_tool
    async def optimize_procedure_parameters(
        self,
        procedure_name: str,
        iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Optimize parameters for a procedure using Bayesian optimization
        
        Args:
            procedure_name: Name of the procedure to optimize
            iterations: Number of optimization iterations
            
        Returns:
            Optimization results
        """
        if procedure_name not in self.procedures:
            return {"error": f"Procedure '{procedure_name}' not found"}
        
        procedure = self.procedures[procedure_name]
        
        # Define objective function (success rate and execution time)
        async def objective_function(test_procedure: Procedure) -> float:
            # Create simulated context
            test_context = {"optimization_run": True}
            
            # Execute procedure
            ctx = RunContextWrapper(context=self)
            result = await execute_procedure(ctx, test_procedure.name, test_context)
            
            # Calculate objective score (combination of success and speed)
            success_score = 1.0 if result["success"] else 0.0
            time_score = max(0.0, 1.0 - (result["execution_time"] / 10.0))  # Lower time is better
            
            # Combined score (success is more important)
            return success_score * 0.7 + time_score * 0.3
        
        # Run optimization
        optimization_result = await self.parameter_optimizer.optimize_parameters(
            procedure=procedure,
            objective_function=objective_function,
            iterations=iterations
        )
        
        # Apply best parameters if optimization succeeded
        if optimization_result["status"] == "success" and optimization_result["best_parameters"]:
            # Create modified procedure
            modified_procedure = procedure.model_copy(deep=True)
            self.parameter_optimizer._apply_parameters(modified_procedure, optimization_result["best_parameters"])
            
            # Update original procedure
            for step in procedure.steps:
                for modified_step in modified_procedure.steps:
                    if step["id"] == modified_step["id"]:
                        step["parameters"] = modified_step["parameters"]
            
            # Update timestamp
            procedure.last_updated = datetime.datetime.now().isoformat()
            
            # Add update information
            optimization_result["procedure_updated"] = True
        else:
            optimization_result["procedure_updated"] = False
        
        return optimization_result
    
    @function_tool
    async def handle_execution_error(
        self,
        error: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Handle an execution error using the causal model
        
        Args:
            error: Error details
            context: Execution context
            
        Returns:
            Recovery suggestions
        """
        # Identify likely causes
        likely_causes = self.causal_model.identify_likely_causes(error)
        
        # Get recovery suggestions
        interventions = self.causal_model.suggest_interventions(likely_causes)
        
        # Return results
        return {
            "likely_causes": likely_causes,
            "interventions": interventions,
            "context": context
        }
    
    @function_tool
    async def create_temporal_procedure(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        temporal_constraints: List[Dict[str, Any]],
        domain: str,
        description: str = None
    ) -> Dict[str, Any]:
        """
        Create a procedure with temporal constraints
        
        Args:
            name: Name of the procedure
            steps: List of step definitions
            temporal_constraints: List of temporal constraints between steps
            domain: Domain for the procedure
            description: Optional description
            
        Returns:
            Information about the created procedure
        """
        # Create normal procedure first
        ctx = RunContextWrapper(context=self)
        normal_proc = await add_procedure(
            ctx,
            name=name,
            steps=steps,
            description=description or f"Temporal procedure: {name}",
            domain=domain
        )
        
        # Create temporal graph
        procedure = self.procedures[name]
        graph = TemporalProcedureGraph.from_procedure(procedure)
        
        # Add temporal constraints
        for constraint in temporal_constraints:
            from_id = constraint.get("from_step")
            to_id = constraint.get("to_step")
            constraint_type = constraint.get("type")
            
            if from_id and to_id and constraint_type:
                # Find nodes
                from_node_id = f"node_{from_id}"
                to_node_id = f"node_{to_id}"
                
                if from_node_id in graph.nodes and to_node_id in graph.nodes:
                    # Add constraint based on type
                    if constraint_type == "min_delay":
                        # Minimum delay between steps
                        min_delay = constraint.get("delay", 0)
                        
                        # Add to edge
                        for i, edge in enumerate(graph.edges):
                            if edge["from"] == from_node_id and edge["to"] == to_node_id:
                                if "properties" not in edge:
                                    edge["properties"] = {}
                                edge["properties"]["min_duration"] = min_delay
                                break
                    elif constraint_type == "must_follow":
                        # Must follow constraint
                        if "properties" not in constraint:
                            constraint["properties"] = {}
                        constraint["properties"]["must_follow"] = True
                        
                        # Add constraint to node
                        graph.nodes[to_node_id].add_constraint({
                            "type": "after",
                            "action": graph.nodes[from_node_id].action["function"]
                        })
        
        # Validate constraints
        if not graph.validate_temporal_constraints():
            return {
                "error": "Invalid temporal constraints - contains negative cycles",
                "procedure_id": normal_proc["procedure_id"]
            }
        
        # Store the temporal graph
        self.temporal_graphs[graph.id] = graph
        
        # Link procedure to graph
        procedure.temporal_graph_id = graph.id
        
        return {
            "procedure_id": normal_proc["procedure_id"],
            "temporal_graph_id": graph.id,
            "name": name,
            "domain": domain,
            "steps_count": len(steps),
            "constraints_count": len(temporal_constraints),
            "is_temporal": True
        }
    
    @function_tool
    async def execute_temporal_procedure(
        self,
        name: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a procedure with temporal constraints
        
        Args:
            name: Name of the procedure
            context: Execution context
            
        Returns:
            Execution results
        """
        if name not in self.procedures:
            return {"error": f"Procedure '{name}' not found"}
        
        procedure = self.procedures[name]
        
        # Check if procedure has temporal graph
        if not hasattr(procedure, "temporal_graph_id") or procedure.temporal_graph_id not in self.temporal_graphs:
            # Fall back to normal execution
            ctx = RunContextWrapper(context=self)
            return await execute_procedure(ctx, name, context)
        
        # Get temporal graph
        graph = self.temporal_graphs[procedure.temporal_graph_id]
        
        # Create execution trace
        with trace(workflow_name="execute_temporal_procedure"):
            start_time = datetime.datetime.now()
            results = []
            success = True
            
            # Initialize execution context
            execution_context = context.copy() if context else {}
            execution_context["temporal_execution"] = True
            execution_context["execution_history"] = []
            
            # Execute in temporal order
            while True:
                # Get next executable nodes
                next_nodes = graph.get_next_executable_nodes(execution_context["execution_history"])
                
                if not next_nodes:
                    # Check if we've executed all exit nodes
                    executed_nodes = set(hist["node_id"] for hist in execution_context["execution_history"])
                    if all(exit_node in executed_nodes for exit_node in graph.exit_points):
                        # Successfully completed all nodes
                        break
                    else:
                        # No executable nodes but haven't reached all exits
                        success = False
                        break
                
                # Execute first valid node
                node_id = next_nodes[0]
                node = graph.nodes[node_id]
                
                # Extract step information
                step = {
                    "id": node.action.get("step_id", node_id),
                    "function": node.action["function"],
                    "parameters": node.action.get("parameters", {}),
                    "description": node.action.get("description", f"Step {node_id}")
                }
                
                # Execute the step
                step_result = await self.execute_step(step, execution_context)
                results.append(step_result)
                
                # Update execution history
                execution_context["execution_history"].append({
                    "node_id": node_id,
                    "step_id": step["id"],
                    "function": step["function"],
                    "success": step_result["success"],
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                # Check for failure
                if not step_result["success"]:
                    success = False
                    break
            
            # Calculate execution time
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Update procedure statistics
            self.update_procedure_stats(procedure, execution_time, success)
            
            return {
                "success": success,
                "results": results,
                "execution_time": execution_time,
                "is_temporal": True,
                "nodes_executed": len(execution_context["execution_history"])
            }
    
    @function_tool
    async def create_procedure_graph(
        self,
        procedure_name: str
    ) -> Dict[str, Any]:
        """
        Create a graph representation of a procedure for flexible execution
        
        Args:
            procedure_name: Name of the existing procedure
            
        Returns:
            Information about the created graph
        """
        if procedure_name not in self.procedures:
            return {"error": f"Procedure '{procedure_name}' not found"}
        
        procedure = self.procedures[procedure_name]
        
        # Create graph representation
        graph = ProcedureGraph.from_procedure(procedure)
        
        # Generate graph ID
        graph_id = f"graph_{int(datetime.datetime.now().timestamp())}_{random.randint(1000, 9999)}"
        
        # Store graph
        self.procedure_graphs[graph_id] = graph
        
        # Link procedure to graph
        procedure.graph_id = graph_id
        
        return {
            "graph_id": graph_id,
            "procedure_name": procedure_name,
            "nodes_count": len(graph.nodes),
            "edges_count": len(graph.edges),
            "entry_points": len(graph.entry_points),
            "exit_points": len(graph.exit_points)
        }
    
    @function_tool
    async def execute_graph_procedure(
        self,
        procedure_name: str,
        context: Dict[str, Any] = None,
        goal: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a procedure using its graph representation
        
        Args:
            procedure_name: Name of the procedure
            context: Execution context
            goal: Optional goal state to achieve
            
        Returns:
            Execution results
        """
        if procedure_name not in self.procedures:
            return {"error": f"Procedure '{procedure_name}' not found"}
        
        procedure = self.procedures[procedure_name]
        
        # Check if procedure has graph
        if not hasattr(procedure, "graph_id") or procedure.graph_id not in self.procedure_graphs:
            # Fall back to normal execution
            ctx = RunContextWrapper(context=self)
            return await execute_procedure(ctx, procedure_name, context)
        
        # Get graph
        graph = self.procedure_graphs[procedure.graph_id]
        
        # Create execution trace
        with trace(workflow_name="execute_graph_procedure"):
            start_time = datetime.datetime.now()
            results = []
            success = True
            
            # Initialize execution context
            execution_context = context.copy() if context else {}
            execution_context["graph_execution"] = True
            
            # Find execution path to goal
            path = graph.find_execution_path(execution_context, goal or {})
            
            if not path:
                return {
                    "success": False,
                    "error": "Could not find a valid execution path",
                    "procedure_name": procedure_name
                }
            
            # Execute nodes in path
            for node_id in path:
                # Get node data
                node_data = graph.nodes[node_id]
                
                # Create step from node data
                step = {
                    "id": node_data.get("step_id", node_id),
                    "function": node_data["function"],
                    "parameters": node_data.get("parameters", {}),
                    "description": node_data.get("description", f"Step {node_id}")
                }
                
                # Execute the step
                step_result = await self.execute_step(step, execution_context)
                results.append(step_result)
                
                # Update execution context
                execution_context[f"step_{step['id']}_result"] = step_result
                
                # Check for failure
                if not step_result["success"]:
                    success = False
                    break
            
            # Calculate execution time
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Update procedure statistics
            self.update_procedure_stats(procedure, execution_time, success)
            
            # Check if goal achieved
            goal_achieved = True
            if goal:
                for key, value in goal.items():
                    if key not in execution_context or execution_context[key] != value:
                        goal_achieved = False
                        break
            
            return {
                "success": success,
                "results": results,
                "execution_time": execution_time,
                "is_graph": True,
                "path_length": len(path),
                "goal_achieved": goal_achieved
            }
    
    @function_tool
    async def consolidate_procedural_memory(self) -> Dict[str, Any]:
        """
        Consolidate procedural memory to optimize storage and execution
        
        Returns:
            Consolidation results
        """
        # Run memory consolidation
        return await self.memory_consolidator.consolidate_procedural_memory()
    
    @function_tool
    async def optimize_procedure_transfer(
        self,
        source_procedure: str,
        target_domain: str
    ) -> Dict[str, Any]:
        """
        Optimize transfer of a procedure to a new domain
        
        Args:
            source_procedure: Name of the source procedure
            target_domain: Target domain
            
        Returns:
            Transfer optimization plan
        """
        if source_procedure not in self.procedures:
            return {"error": f"Procedure '{source_procedure}' not found"}
        
        # Get source procedure
        procedure = self.procedures[source_procedure]
        
        # Optimize transfer
        transfer_plan = await self.transfer_optimizer.optimize_transfer(
            source_procedure=procedure,
            target_domain=target_domain
        )
        
        return transfer_plan
        
    @function_tool
    async def execute_transfer_plan(
        self,
        transfer_plan: Dict[str, Any],
        target_name: str
    ) -> Dict[str, Any]:
        """
        Execute a transfer plan to create a new procedure
        
        Args:
            transfer_plan: Transfer plan from optimize_procedure_transfer
            target_name: Name for the new procedure
            
        Returns:
            Results of transfer execution
        """
        source_domain = transfer_plan.get("source_domain")
        target_domain = transfer_plan.get("target_domain")
        mappings = transfer_plan.get("mappings", [])
        
        if not source_domain or not target_domain or not mappings:
            return {
                "success": False,
                "error": "Invalid transfer plan"
            }
        
        # Find source procedure
        source_procedure = None
        for name, proc in self.procedures.items():
            if proc.domain == source_domain:
                source_procedure = proc
                break
        
        if not source_procedure:
            return {
                "success": False,
                "error": f"Could not find procedure in domain {source_domain}"
            }
        
        # Create new steps based on mappings
        new_steps = []
        
        for i, mapping in enumerate(mappings):
            source_func = mapping.get("source_function")
            target_func = mapping.get("target_function")
            target_params = mapping.get("target_parameters", {})
            
            if not source_func or not target_func:
                continue
            
            # Find corresponding step in source procedure
            source_step = None
            for step in source_procedure.steps:
                if step["function"] == source_func:
                    source_step = step
                    break
            
            if not source_step:
                continue
            
            # Create new step
            new_step = {
                "id": f"step_{i+1}",
                "function": target_func,
                "parameters": target_params,
                "description": f"Transferred from {source_step.get('description', source_func)}"
            }
            
            new_steps.append(new_step)
        
        if not new_steps:
            return {
                "success": False,
                "error": "No steps could be transferred"
            }
        
        # Create new procedure
        ctx = RunContextWrapper(context=self)
        new_procedure = await add_procedure(
            ctx,
            name=target_name,
            steps=new_steps,
            description=f"Transferred from {source_procedure.name} ({source_domain} to {target_domain})",
            domain=target_domain
        )
        
        # Update transfer history
        self.transfer_optimizer.update_from_transfer_result(
            source_domain=source_domain,
            target_domain=target_domain,
            success_rate=0.8,  # Initial estimate
            mappings=mappings
        )
        
        return {
            "success": True,
            "procedure_id": new_procedure["procedure_id"],
            "name": target_name,
            "domain": target_domain,
            "steps_count": len(new_steps),
            "transfer_strategy": transfer_plan.get("transfer_strategy")
        }
    
    # -------------------------------------------------------------------------
    # Integration with Memory Core
    # -------------------------------------------------------------------------
    
    async def integrate_with_memory_core(self) -> bool:
        """Integrate procedural memory with main memory core"""
        if not self.memory_core:
            logger.warning("No memory core available for integration")
            return False
        
        try:
            # Register handlers for procedural memory operations
            self.memory_core.register_procedural_handler(self)
            
            # Set up memory event listeners
            self._setup_memory_listeners()
            
            logger.info("Procedural memory integrated with memory core")
            return True
        except Exception as e:
            logger.error(f"Error integrating with memory core: {e}")
            return False
    
    def _setup_memory_listeners(self):
        """Set up listeners for memory core events"""
        if not self.memory_core:
            return
        
        # Listen for new procedural observations
        self.memory_core.add_event_listener(
            "new_procedural_observation",
            self._handle_procedural_observation
        )
        
        # Listen for memory decay events
        self.memory_core.add_event_listener(
            "memory_decay",
            self._handle_memory_decay
        )
    
    async def _handle_procedural_observation(self, data: Dict[str, Any]):
        """Handle new procedural observation from memory core"""
        # Check if observation has steps
        if "steps" not in data:
            return
        
        # Create sequence of observations
        observation_sequence = [{
            "action": step.get("action"),
            "state": step.get("state", {}),
            "timestamp": step.get("timestamp", datetime.datetime.now().isoformat())
        } for step in data["steps"]]
        
        # Learn from demonstration
        if len(observation_sequence) >= 3:  # Need at least 3 steps
            await self.learn_from_demonstration(
                observation_sequence=observation_sequence,
                domain=data.get("domain", "general"),
                name=data.get("name")
            )
    
    async def _handle_memory_decay(self, data: Dict[str, Any]):
        """Handle memory decay events"""
        # Check if affecting procedural memory
        if data.get("memory_type") != "procedural":
            return
        
        # Run consolidation to optimize storage
        await self.consolidate_procedural_memory()
    
    # -------------------------------------------------------------------------
    # Store/Retrieve Procedures in Memory Core
    # -------------------------------------------------------------------------
    
    async def store_procedure_in_memory(self, procedure_name: str) -> Dict[str, Any]:
        """Store a procedure in the memory core for long-term storage"""
        if not self.memory_core:
            return {"error": "No memory core available"}
        
        if procedure_name not in self.procedures:
            return {"error": f"Procedure '{procedure_name}' not found"}
        
        procedure = self.procedures[procedure_name]
        
        # Convert to memory format
        procedure_data = {
            "id": procedure.id,
            "name": procedure_name,
            "description": procedure.description,
            "domain": procedure.domain,
            "steps": procedure.steps,
            "proficiency": procedure.proficiency,
            "execution_count": procedure.execution_count,
            "type": "procedure"
        }
        
        # Store in memory core
        memory_id = await self.memory_core.add_memory(
            memory_text=f"Procedure: {procedure_name} - {procedure.description}",
            memory_type="procedural",
            memory_scope="system",
            significance=int(procedure.proficiency * 10),  # Convert to 0-10 scale
            tags=["procedure", procedure.domain],
            metadata={
                "procedure_data": procedure_data,
                "timestamp": datetime.datetime.now().isoformat()
            }
        )
        
        return {
            "memory_id": memory_id,
            "procedure_name": procedure_name,
            "stored": True
        }
    
    async def retrieve_procedure_from_memory(self, procedure_name: str) -> Dict[str, Any]:
        """Retrieve a procedure from memory core"""
        if not self.memory_core:
            return {"error": "No memory core available"}
        
        # Query memory core for the procedure
        memories = await self.memory_core.retrieve_memories(
            query=f"procedure {procedure_name}",
            memory_types=["procedural"],
            limit=1
        )
        
        if not memories:
            return {"error": f"Procedure '{procedure_name}' not found in memory core"}
        
        memory = memories[0]
        procedure_data = memory.get("metadata", {}).get("procedure_data", {})
        
        if not procedure_data or procedure_data.get("type") != "procedure":
            return {"error": "Retrieved memory does not contain valid procedure data"}
        
        # Create procedure from memory data
        ctx = RunContextWrapper(context=self)
        result = await add_procedure(
            ctx,
            name=procedure_data["name"],
            steps=procedure_data["steps"],
            description=procedure_data["description"],
            domain=procedure_data["domain"]
        )
        
        # Update statistics
        if procedure_data["name"] in self.procedures:
            procedure = self.procedures[procedure_data["name"]]
            procedure.proficiency = procedure_data["proficiency"]
            procedure.execution_count = procedure_data["execution_count"]
        
        return {
            "memory_id": memory["id"],
            "procedure_name": procedure_data["name"],
            "loaded": True,
            "procedure_id": result["procedure_id"]
        }
    
    # -------------------------------------------------------------------------
    # Integration with Knowledge Core
    # -------------------------------------------------------------------------
    
    async def integrate_with_knowledge_core(self) -> bool:
        """Integrate procedural memory with knowledge core"""
        if not self.knowledge_core:
            logger.warning("No knowledge core available for integration")
            return False
        
        try:
            # Register handlers for knowledge queries
            self.knowledge_core.register_procedural_handler(self)
            
            # Set up knowledge listeners
            self._setup_knowledge_listeners()
            
            logger.info("Procedural memory integrated with knowledge core")
            return True
        except Exception as e:
            logger.error(f"Error integrating with knowledge core: {e}")
            return False
    
    def _setup_knowledge_listeners(self):
        """Set up listeners for knowledge core events"""
        if not self.knowledge_core:
            return
        
        # Listen for new domain knowledge
        self.knowledge_core.add_event_listener(
            "new_domain_knowledge",
            self._handle_domain_knowledge
        )
    
    async def _handle_domain_knowledge(self, data: Dict[str, Any]):
        """Handle new domain knowledge from knowledge core"""
        domain = data.get("domain")
        if not domain:
            return
        
        # Update domain similarities in transfer optimizer
        if hasattr(self, "transfer_optimizer"):
            # Create or update domain embedding
            await self.transfer_optimizer._get_domain_embedding(domain)
    
    async def share_domain_knowledge(self, domain: str) -> Dict[str, Any]:
        """Share procedural knowledge about a domain with knowledge core"""
        if not self.knowledge_core:
            return {"error": "No knowledge core available"}
        
        # Find procedures in this domain
        domain_procedures = [p for p in self.procedures.values() if p.domain == domain]
        
        if not domain_procedures:
            return {"error": f"No procedures found for domain {domain}"}
        
        # Extract knowledge from procedures
        knowledge_items = []
        
        for procedure in domain_procedures:
            # Create knowledge about procedure purpose
            knowledge_items.append({
                "content": f"In the {domain} domain, '{procedure.name}' is a procedure for {procedure.description}",
                "confidence": procedure.proficiency,
                "source": "procedural_memory"
            })
            
            # Create knowledge about specific steps
            for i, step in enumerate(procedure.steps):
                knowledge_items.append({
                    "content": f"In the {domain} domain, the '{step['function']}' function is used for {step.get('description', f'step {i+1}')}",
                    "confidence": procedure.proficiency * 0.9,  # Slightly lower confidence
                    "source": "procedural_memory"
                })
        
        # Add knowledge to knowledge core
        added_count = 0
        for item in knowledge_items:
            try:
                await self.knowledge_core.add_knowledge_item(
                    domain=domain,
                    content=item["content"],
                    source=item["source"],
                    confidence=item["confidence"]
                )
                added_count += 1
            except Exception as e:
                logger.error(f"Error adding knowledge item: {e}")
        
        return {
            "domain": domain,
            "knowledge_items_added": added_count,
            "procedures_analyzed": len(domain_procedures)
        }
    
    # -------------------------------------------------------------------------
    # Additional Helper Methods
    # -------------------------------------------------------------------------
    
    async def record_procedure_execution(self, procedure_name: str, success: bool, execution_time: float, context: Dict[str, Any] = None) -> None:
        """Record procedure execution in memory"""
        if not self.memory_core:
            return
        
        if procedure_name not in self.procedures:
            return
        
        procedure = self.procedures[procedure_name]
        
        # Create memory text
        result_text = "successfully" if success else "unsuccessfully"
        memory_text = f"Executed procedure '{procedure_name}' {result_text} in {execution_time:.2f} seconds"
        
        # Store in memory core
        await self.memory_core.add_memory(
            memory_text=memory_text,
            memory_type="procedural_execution",
            memory_scope="system",
            significance=5 + (3 if success else 0),  # Higher significance for successful executions
            tags=["procedure_execution", procedure.domain, procedure_name],
            metadata={
                "procedure_name": procedure_name,
                "success": success,
                "execution_time": execution_time,
                "domain": procedure.domain,
                "timestamp": datetime.datetime.now().isoformat(),
                "context": {k: v for k, v in (context or {}).items() if isinstance(v, (str, int, float, bool))}
            }
        )

if __name__ == "__main__":
    # Run both demonstration functions in sequence
    asyncio.run(demonstrate_cross_game_transfer())
    asyncio.run(demonstrate_procedural_memory())
