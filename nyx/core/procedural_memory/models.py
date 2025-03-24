# nyx/core/procedural_memory/models.py

import datetime
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from pydantic import BaseModel, Field
import random
from collections import Counter
from functools import lru_cache

logger = logging.getLogger(__name__)

class ActionTemplate(BaseModel):
    """Generic template for an action that can be mapped across domains"""
    action_type: str  # e.g., "aim", "shoot", "move", "sprint", "interact"
    intent: str  # Higher-level purpose, e.g., "target_acquisition", "locomotion"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    domain_mappings: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # domain -> specific implementations
    # Added metadata for semantic search and matching
    semantic_tags: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_used: Optional[str] = None

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
    # Added fields for enhanced functionality
    semantic_embedding: Optional[List[float]] = None  # For semantic similarity calculations
    transfer_compatibility: Dict[str, float] = Field(default_factory=dict)  # Target domain -> transfer compatibility 
    variations: List[Dict[str, Any]] = Field(default_factory=list)  # Variations of this template for different contexts
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_used: Optional[str] = None
    usage_frequency: int = 0

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
    # Added fields for improved context matching
    weighted_indicators: Dict[str, float] = Field(default_factory=dict)  # Indicator -> weight
    negative_indicators: Dict[str, Any] = Field(default_factory=dict)  # Indicators that should NOT be present
    context_examples: List[Dict[str, Any]] = Field(default_factory=list)  # Example contexts for training
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    learned: bool = False  # Whether this pattern was automatically learned

class ChunkPrediction(BaseModel):
    """Prediction for which chunk should be executed next"""
    chunk_id: str
    confidence: float
    context_match_score: float
    reasoning: List[str] = Field(default_factory=list)
    alternative_chunks: List[Dict[str, float]] = Field(default_factory=list)
    # Added fields for prediction explanation and feedback
    context_indicators_matched: Dict[str, float] = Field(default_factory=dict)  # Indicator -> match strength
    prediction_time: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    execution_feedback: Optional[Dict[str, Any]] = None  # Feedback after execution

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
    # Added fields for transfer success tracking
    success_count: int = 0
    failure_count: int = 0
    context_specificity: Dict[str, Any] = Field(default_factory=dict)  # Context constraints for this mapping
    bidirectional: bool = False  # Whether mapping works both ways

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
    # Added fields for improved transfer learning
    transfer_method: str = "default"  # Method used for transfer
    troublesome_steps: List[str] = Field(default_factory=list)  # Steps that were difficult to transfer
    successful_steps: List[str] = Field(default_factory=list)  # Steps that transferred well
    feedback: Optional[str] = None  # User feedback on transfer quality
    manual_corrections: Dict[str, Any] = Field(default_factory=dict)  # Corrections made after transfer
    execution_results: List[Dict[str, Any]] = Field(default_factory=list)  # Results from executing transferred procedure

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
    # Added fields for enhanced functionality
    semantic_embedding: Optional[List[float]] = None  # For semantic similarity search
    tags: List[str] = Field(default_factory=list)  # Categorization tags
    version: int = 1  # Version tracking
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_execution: Optional[str] = None
    # Added fields for error handling
    error_history: List[Dict[str, Any]] = Field(default_factory=list)
    recovery_strategies: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)  # Error type -> recovery strategies
    
    # Added methods for resource management
    def estimate_memory_usage(self) -> int:
        """Estimate memory usage of this procedure in bytes"""
        # Base size
        memory = 1000  # Base object overhead
        
        # Add size for steps
        memory += len(self.steps) * 500  # Approximate size per step
        
        # Add size for context history
        memory += len(self.context_history) * 200  # Approximate size per context entry
        
        # Add size for other lists and dicts
        memory += len(self.chunked_steps) * 100
        memory += len(self.generalized_chunks) * 100
        memory += len(self.refinement_opportunities) * 200
        memory += len(self.optimization_history) * 200
        
        # Add size for semantic embedding if present
        if self.semantic_embedding:
            memory += len(self.semantic_embedding) * 8  # 8 bytes per float
        
        return memory
    
    def cleanup_history(self, keep_count: int = 10) -> int:
        """Clean up history to reduce memory usage, returns bytes saved"""
        original_size = len(self.context_history)
        if len(self.context_history) > keep_count:
            saved_contexts = len(self.context_history) - keep_count
            self.context_history = self.context_history[-keep_count:]
            return saved_contexts * 200  # Approximate bytes saved
        return 0

    def estimate_memory_usage(self) -> int:
        """Estimate memory usage of this procedure in bytes"""
        # Base size
        memory = 1000  # Base object overhead
        
        # Add size for steps
        memory += len(self.steps) * 500  # Approximate size per step
        
        # Add size for context history
        memory += len(self.context_history) * 200  # Approximate size per context entry
        
        # Add size for other lists and dicts
        memory += len(self.chunked_steps) * 100
        memory += len(self.generalized_chunks) * 100
        memory += len(self.refinement_opportunities) * 200
        memory += len(self.optimization_history) * 200
        
        # Add size for semantic embedding if present
        if self.semantic_embedding:
            memory += len(self.semantic_embedding) * 8  # 8 bytes per float
        
        return memory
    
    def cleanup_history(self, keep_count: int = 10) -> int:
        """Clean up history to reduce memory usage, returns bytes saved"""
        original_size = len(self.context_history)
        if len(self.context_history) > keep_count:
            saved_contexts = len(self.context_history) - keep_count
            self.context_history = self.context_history[-keep_count:]
            return saved_contexts * 200  # Approximate bytes saved
        return 0
    
class StepResult(BaseModel):
    """Result from executing a step"""
    success: bool
    error: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float = 0.0
    # Added fields for enhanced execution tracking
    resources_used: Dict[str, float] = Field(default_factory=dict)  # Resource usage during execution
    execution_path: List[str] = Field(default_factory=list)  # Path taken during execution
    state_changes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # State variables changed
    output_artifacts: List[Dict[str, Any]] = Field(default_factory=list)  # Any artifacts produced

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
    # Added fields for enhanced analytics
    execution_time_trend: List[float] = Field(default_factory=list)  # Recent execution times
    success_rate_trend: List[float] = Field(default_factory=list)  # Recent success rates
    common_errors: Dict[str, int] = Field(default_factory=dict)  # Error type -> count
    performance_percentile: Optional[float] = None  # Performance compared to similar procedures
    optimization_potential: float = 0.0  # Estimated potential for improvement

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
    # Added fields for transfer analytics
    domain_transfer_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict)  # Source -> Target -> Success rate
    most_transferable_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    transfer_difficulty_factors: Dict[str, float] = Field(default_factory=dict)
    transfer_success_predictors: Dict[str, float] = Field(default_factory=dict)

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
    # Added fields for hierarchical analysis
    dependency_graph: Dict[str, List[str]] = Field(default_factory=dict)  # Step dependencies
    critical_path: List[str] = Field(default_factory=list)  # Critical execution path
    execution_strategies: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # Strategy configurations
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
        
    # Added method for hierarchical optimization
    def optimize_hierarchy(self) -> Dict[str, Any]:
        """Optimize hierarchical structure based on execution history"""
        if self.execution_count < 5:
            return {"optimized": False, "reason": "Insufficient execution data"}
        
        optimizations = []
        
        # Check for steps that could be parallelized
        parallel_candidates = []
        dependencies = set()
        
        # Build dependency set
        for step_id, deps in self.dependency_graph.items():
            for dep in deps:
                dependencies.add((dep, step_id))  # dep must execute before step_id
        
        # Find independent steps that could run in parallel
        for i, step1 in enumerate(self.steps[:-1]):
            for step2 in self.steps[i+1:]:
                step1_id = step1["id"]
                step2_id = step2["id"]
                
                # Check if neither depends on the other
                if (step1_id, step2_id) not in dependencies and (step2_id, step1_id) not in dependencies:
                    parallel_candidates.append((step1_id, step2_id))
        
        if parallel_candidates:
            optimizations.append({
                "type": "parallelization",
                "candidates": parallel_candidates,
                "potential_speedup": 0.3  # Estimated speedup
            })
        
        # Check for steps that could be reordered for efficiency
        execution_times = {}
        for step in self.steps:
            step_id = step["id"]
            # Calculate average execution time for this step
            times = [result.get("execution_time", 0) for result in self.execution_history 
                    if result.get("step_id") == step_id]
            if times:
                execution_times[step_id] = sum(times) / len(times)
        
        # Find slow steps that could be moved earlier if no dependencies
        slow_steps = sorted(execution_times.items(), key=lambda x: x[1], reverse=True)
        for step_id, time in slow_steps[:3]:  # Consider top 3 slowest
            # Find current position
            current_pos = next(i for i, step in enumerate(self.steps) if step["id"] == step_id)
            
            # Check if it could be moved earlier without violating dependencies
            for new_pos in range(current_pos):
                intervening_steps = [s["id"] for s in self.steps[new_pos:current_pos]]
                
                # Check if move would violate dependencies
                violation = False
                for intervening in intervening_steps:
                    if (step_id, intervening) in dependencies:
                        violation = True
                        break
                
                if not violation:
                    optimizations.append({
                        "type": "reordering",
                        "step_id": step_id,
                        "current_position": current_pos,
                        "proposed_position": new_pos,
                        "potential_speedup": time * 0.1  # Estimated benefit
                    })
                    break
        
        return {
            "optimized": len(optimizations) > 0,
            "optimizations": optimizations
        }

class CausalModel(BaseModel):
    """Causal model for reasoning about procedure failures"""
    causes: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    interventions: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    error_history: List[Dict[str, Any]] = Field(default_factory=list)
    max_history: int = 50
    
    def identify_likely_causes(self, error: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify likely causes of an error based on causal model"""
        error_type = error.get("type", "execution_failure")
        error_message = error.get("message", "Unknown error")
        error_context = error.get("context", {})
        
        # Get candidate causes for this error type
        candidate_causes = self.causes.get(error_type, [])
        if not candidate_causes:
            # Try to match with similar error types
            for key, causes in self.causes.items():
                if key in error_message.lower():
                    candidate_causes = causes
                    break
        
        # Still no candidates - use generic ones
        if not candidate_causes and "execution_failure" in self.causes:
            candidate_causes = self.causes["execution_failure"]
        
        # Score each cause based on context
        scored_causes = []
        for cause in candidate_causes:
            base_probability = cause.get("probability", 0.5)
            
            # Adjust probability based on context factors
            context_factors = cause.get("context_factors", {})
            context_multiplier = 1.0
            
            for factor_key, factor_value in context_factors.items():
                if factor_key in error_context:
                    # Check if context value matches factor value
                    if error_context[factor_key] == factor_value:
                        context_multiplier *= 1.2  # Increase probability by 20%
                    else:
                        context_multiplier *= 0.8  # Decrease probability by 20%
            
            # Adjust based on error message
            if "description" in cause and cause["description"].lower() in error_message.lower():
                context_multiplier *= 1.5  # Strong match with error message
            
            # Check error history for similar errors
            for past_error in self.error_history:
                if past_error.get("type") == error_type and past_error.get("cause") == cause.get("cause"):
                    context_multiplier *= 1.2  # Similar error happened before
                    break
            
            # Calculate final probability
            final_probability = min(1.0, base_probability * context_multiplier)
            
            scored_causes.append({
                "cause": cause.get("cause"),
                "description": cause.get("description"),
                "probability": final_probability
            })
        
        # Sort by probability
        scored_causes.sort(key=lambda x: x["probability"], reverse=True)
        
        # Record this error for future reference
        # Record top cause
        top_cause = scored_causes[0]["cause"] if scored_causes else "unknown"
        self.error_history.append({
            "type": error_type,
            "message": error_message,
            "cause": top_cause,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Trim history
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
        
        return scored_causes

    def suggest_interventions(self, causes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest interventions for the likely causes"""
        suggested_interventions = []
        
        for cause in causes:
            cause_id = cause.get("cause")
            
            # Skip if no cause identified
            if not cause_id:
                continue
            
            # Get interventions for this cause
            cause_interventions = self.interventions.get(cause_id, [])
            
            for intervention in cause_interventions:
                # Calculate intervention score based on cause probability
                cause_probability = cause.get("probability", 0.5)
                effectiveness = intervention.get("effectiveness", 0.5)
                
                # Calculate overall score
                score = cause_probability * effectiveness
                
                # Add to suggestions
                suggested_interventions.append({
                    "intervention_type": intervention.get("type"),
                    "description": intervention.get("description"),
                    "for_cause": cause.get("description"),
                    "expected_effectiveness": effectiveness,
                    "overall_score": score
                })
        
        # Sort by overall score
        suggested_interventions.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return suggested_interventions
    
    def learn_from_intervention(self, error: Dict[str, Any], intervention: Dict[str, Any], success: bool) -> None:
        """Learn from intervention attempts to improve future suggestions"""
        cause_id = intervention.get("for_cause")
        intervention_type = intervention.get("intervention_type")
        
        if not cause_id or not intervention_type:
            return
        
        # Find the cause and intervention
        for cause in self.causes.get(error.get("type", "execution_failure"), []):
            if cause.get("cause") == cause_id:
                for interv in self.interventions.get(cause_id, []):
                    if interv.get("type") == intervention_type:
                        # Update effectiveness based on result
                        current_effectiveness = interv.get("effectiveness", 0.5)
                        
                        if success:
                            # Intervention worked - increase effectiveness
                            interv["effectiveness"] = min(1.0, current_effectiveness * 1.1)
                        else:
                            # Intervention failed - decrease effectiveness
                            interv["effectiveness"] = max(0.1, current_effectiveness * 0.9)
                        
                        # Update learning count
                        if "learning_count" not in interv:
                            interv["learning_count"] = 0
                        interv["learning_count"] += 1
                        
                        break
                break

class ProcedureGraph(BaseModel):
    """Graph representation of a procedure for flexible execution"""
    nodes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    entry_points: List[str] = Field(default_factory=list)
    exit_points: List[str] = Field(default_factory=list)
    # Added fields for graph analysis
    critical_paths: List[List[str]] = Field(default_factory=list)
    bottlenecks: List[str] = Field(default_factory=list)
    parallelizable_paths: List[List[str]] = Field(default_factory=list)
    execution_statistics: Dict[str, Dict[str, float]] = Field(default_factory=dict)  # Node -> stats

class WorkingMemoryController:
    """Controls working memory during procedure execution"""
    
    def __init__(self, capacity: int = 5):
        self.items = []
        self.capacity = capacity
        self.focus_history = []
        self.max_history = 20
        # Added fields for enhanced working memory
        self.decay_rate = 0.9  # Memory decay rate per update
        self.capacity_warning_threshold = 0.9  # When to warn about capacity
        self.prioritization_strategy = "recency"  # Default strategy
        self.context_switching_penalty = 0.2  # Penalty when switching contexts

    def update(self, context: Dict[str, Any], procedure: Procedure) -> None:
        """Update working memory based on context and current procedure"""
        # Apply decay to existing items
        for item in self.items:
            item["priority"] *= self.decay_rate
        
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
        
        # Check memory capacity 
        capacity_usage = len(self.items) / self.capacity
        capacity_warning = None
        if capacity_usage >= self.capacity_warning_threshold:
            capacity_warning = f"Working memory at {capacity_usage*100:.0f}% capacity"
        
        return {
            "focus_key": focus_item["key"],
            "focus_value": focus_item["value"],
            "working_memory": {item["key"]: item["value"] for item in self.items},
            "memory_usage": f"{len(self.items)}/{self.capacity}",
            "capacity_warning": capacity_warning
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
    
    # Added method for memory diagnostics
    def get_memory_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics about the working memory state"""
        if not self.items:
            return {
                "status": "empty",
                "memory_load": 0,
                "focus_stability": 1.0
            }
        
        # Calculate metrics
        memory_load = len(self.items) / self.capacity
        
        # Calculate focus stability (how consistent the focus has been)
        focus_stability = 1.0
        if len(self.focus_history) > 1:
            focus_changes = 0
            for i in range(1, len(self.focus_history)):
                if self.focus_history[i]["key"] != self.focus_history[i-1]["key"]:
                    focus_changes += 1
            
            focus_stability = 1.0 - (focus_changes / len(self.focus_history))
        
        # Calculate priority distribution
        priority_sum = sum(item["priority"] for item in self.items)
        priority_distribution = {
            item["key"]: item["priority"] / priority_sum 
            for item in self.items
        }
        
        # Calculate memory efficiency (higher when important items are remembered)
        memory_efficiency = sum(item["priority"] for item in self.items) / len(self.items)
        
        return {
            "status": "active",
            "memory_load": memory_load,
            "focus_stability": focus_stability,
            "priority_distribution": priority_distribution,
            "memory_efficiency": memory_efficiency
        }

class ParameterOptimizer:
    """Optimizes procedure parameters using Bayesian optimization"""
    
    def __init__(self):
        self.parameter_models = {}
        self.optimization_history = {}
        self.bounds = {}  # Parameter bounds
        # Added fields for enhanced optimization
        self.exploration_rate = 0.3  # Rate of exploration vs exploitation
        self.fitness_weights = {"success": 0.6, "speed": 0.4}  # Default weights
        self.max_iterations = 20
        self.early_stopping_patience = 3
        self.cross_validate = True

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
        
        # Early stopping variables
        best_iteration = 0
        patience_counter = 0
        
        for i in range(min(iterations, self.max_iterations)):
            # Generate next parameters to try
            if i == 0:
                # First iteration: use current parameters
                test_params = {param_id: self.bounds[param_id][0] for param_id in param_space}
            else:
                # Use optimization to suggest next parameters
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
                best_iteration = i
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Check for early stopping
            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping optimization after {i+1} iterations")
                break
            
            # Update models
            self._update_parameter_models(procedure.id, results)
        
        # Return best parameters
        return {
            "status": "success",
            "best_parameters": best_params,
            "best_score": best_score,
            "iterations_performed": len(results),
            "best_iteration": best_iteration,
            "early_stopped": patience_counter >= self.early_stopping_patience,
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
        
        # Decide whether to explore or exploit
        explore = random.random() < self.exploration_rate
        
        if explore:
            # Random exploration with smart sampling
            explore_params = {}
            for param_id, bounds in param_space.items():
                min_val, max_val = bounds
                
                # Check if we've explored this dimension well
                param_values = [r["parameters"].get(param_id, 0) for r in results]
                
                # If we have a good spread of values, use Latin Hypercube sampling
                # to fill in gaps, otherwise use uniform random
                if len(set(param_values)) > 3:
                    # Find the least explored regions
                    sorted_values = sorted(param_values)
                    gaps = [(sorted_values[i+1] - sorted_values[i], i) 
                          for i in range(len(sorted_values)-1)]
                    
                    if gaps:
                        # Find largest gap
                        largest_gap = max(gaps, key=lambda x: x[0])
                        gap_size, gap_idx = largest_gap
                        
                        # Sample from the largest gap
                        min_gap = sorted_values[gap_idx]
                        max_gap = sorted_values[gap_idx + 1]
                        explore_params[param_id] = min_gap + random.random() * gap_size
                    else:
                        explore_params[param_id] = random.uniform(min_val, max_val)
                else:
                    explore_params[param_id] = random.uniform(min_val, max_val)
            
            return explore_params
        else:
            # Exploitation: use Gaussian process regression to suggest next point
            try:
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
                import numpy as np
                
                # Extract parameters and scores from results
                X = np.array([[r["parameters"].get(param_id, 0) for param_id in param_space.keys()] 
                            for r in results])
                y = np.array([r["score"] for r in results])
                
                # Normalize data
                X_mean = X.mean(axis=0)
                X_std = X.std(axis=0) + 1e-8  # Avoid division by zero
                X_norm = (X - X_mean) / X_std
                
                # Train Gaussian Process model
                kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
                gp.fit(X_norm, y)
                
                # Generate candidates and predict their values
                n_candidates = 1000
                X_candidates = np.random.uniform(size=(n_candidates, len(param_space)))
                
                # Scale candidates back to original bounds
                X_candidates_scaled = np.zeros_like(X_candidates)
                for j, param_id in enumerate(param_space.keys()):
                    min_val, max_val = param_space[param_id]
                    X_candidates_scaled[:, j] = X_candidates[:, j] * (max_val - min_val) + min_val
                
                # Normalize candidates
                X_candidates_norm = (X_candidates_scaled - X_mean) / X_std
                
                # Predict means and standard deviations
                y_mean, y_std = gp.predict(X_candidates_norm, return_std=True)
                
                # Calculate acquisition function (Upper Confidence Bound)
                acquisition = y_mean + 1.96 * y_std
                
                # Get best candidate
                best_idx = np.argmax(acquisition)
                best_candidate = X_candidates_scaled[best_idx]
                
                # Convert to parameters dictionary
                best_params = {
                    param_id: best_candidate[i]
                    for i, param_id in enumerate(param_space.keys())
                }
                
                return best_params
                
            except ImportError:
                # Fall back to simpler approach if scikit-learn not available
                logger.warning("scikit-learn not available, falling back to simpler optimization")
                # Use parameters from best result with small perturbations
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
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
            import numpy as np
            
            # Need at least 3 results to build a model
            if len(results) < 3:
                return
            
            # Get all parameter IDs
            param_ids = set()
            for result in results:
                param_ids.update(result["parameters"].keys())
            
            # Build dataset
            X = []
            y = []
            
            for result in results:
                # Parameter vector
                params = [result["parameters"].get(pid, 0) for pid in sorted(param_ids)]
                X.append(params)
                y.append(result["score"])
            
            X = np.array(X)
            y = np.array(y)
            
            # Create and fit model
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
            gp.fit(X, y)
            
            # Store the model and parameter IDs
            self.parameter_models[procedure_id] = {
                "model": gp,
                "param_ids": sorted(param_ids),
                "X_data": X,
                "y_data": y,
                "updated_at": datetime.datetime.now().isoformat()
            }
            
        except ImportError:
            # sklearn not available, just store raw data
            logger.warning("scikit-learn not available, storing raw optimization data only")
            self.parameter_models[procedure_id] = {
                "model": None,
                "X_data": None,
                "y_data": None,
                "results": results,
                "updated_at": datetime.datetime.now().isoformat()
            }

class TransferLearningOptimizer:
    """Optimizes transfer learning between domains using meta-learning"""
    
    def __init__(self):
        self.domain_embeddings = {}
        self.transfer_success_history = []
        self.domain_similarities = {}  # pair_key -> similarity
        self.max_history = 50
        # Added fields for enhanced transfer learning
        self.domain_knowledge = {}  # domain -> domain-specific knowledge
        self.transfer_strategies = ["direct", "parameter_adaptation", "structural_adaptation", "chunking"]
        self.transfer_models = {}  # (source_domain, target_domain) -> model
        self.embedding_model = None
        self.embedding_tokenizer = None

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
        
        # Get statistical model for transfer
        transfer_model = self._build_statistical_mapping_model(
            source_procedure.domain,
            target_domain
        )
        
        # Create transfer plan
        transfer_plan = {
            "source_domain": source_procedure.domain,
            "target_domain": target_domain,
            "domain_similarity": similarity,
            "transfer_strategy": strategy,
            "adaptation_level": adaptation_level,
            "mappings": mappings,
            "estimated_success": success_probability,
            "model_confidence": transfer_model.get("confidence", 0.3),
            "recommended_validation_steps": self._recommend_validation_steps(
                source_procedure, target_domain, similarity
            )
        }
        
        return transfer_plan

    async def _generate_domain_embeddings_batch(self, domains: List[str]) -> Dict[str, List[float]]:
        """Generate embeddings for multiple domains in batch"""
        # Check which domains need new embeddings
        domains_to_generate = [d for d in domains if d not in self.domain_embeddings]
        
        if not domains_to_generate:
            return {d: self.domain_embeddings[d] for d in domains}
        
        # Try to use a pre-trained model via HuggingFace
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            # Load model (only once)
            if self.embedding_tokenizer is None or self.embedding_model is None:
                self.embedding_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                self.embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            
            # Tokenize domains (or domain descriptions)
            domain_descriptions = [f"Domain for {d} related procedures and actions" for d in domains_to_generate]
            inputs = self.embedding_tokenizer(domain_descriptions, padding=True, truncation=True, return_tensors="pt")
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0].numpy()  # Use [CLS] token embedding
            
            # Store embeddings
            for i, domain in enumerate(domains_to_generate):
                self.domain_embeddings[domain] = embeddings[i].tolist()
            
            return {d: self.domain_embeddings[d] for d in domains}
            
        except ImportError:
            # Fallback if transformers not available
            for domain in domains_to_generate:
                self.domain_embeddings[domain] = [random.uniform(-1, 1) for _ in range(10)]
            
            return {d: self.domain_embeddings[d] for d in domains}
    
    async def _get_domain_embedding(self, domain: str) -> List[float]:
        """Get embedding vector for a domain using sophisticated embedding techniques"""
        # Check if embedding already exists
        if domain in self.domain_embeddings:
            return self.domain_embeddings[domain]
        
        try:
            # Try to use a pre-trained model for better embeddings
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            # Load model (only once)
            if self.embedding_tokenizer is None or self.embedding_model is None:
                # Use a sentence transformer model
                self.embedding_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                self.embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            
            # Create a rich description of the domain for better embedding
            domain_description = f"Domain related to {domain} activities, procedures, and actions."
            
            # Add domain-specific context if available
            domain_contexts = {
                "gaming": " Gaming domain includes player movements, interactions, character controls, and game mechanics.",
                "driving": " Driving domain includes vehicle controls, navigation, road rules, and traffic interactions.",
                "cooking": " Cooking domain includes food preparation, ingredient handling, kitchen tools, and cooking techniques.",
                "ui": " UI domain includes interface navigation, button interactions, gestures, and form submissions.",
                "programming": " Programming domain includes coding, debugging, version control, and software design.",
            }
            
            if domain in domain_contexts:
                domain_description += domain_contexts[domain]
            
            # Generate embedding
            inputs = self.embedding_tokenizer(domain_description, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                # Use mean pooling
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = (sum_embeddings / sum_mask).numpy()[0]
            
            self.domain_embeddings[domain] = embedding.tolist()
            return embedding.tolist()
        
        except ImportError:
            # Fallback to simple embedding if transformers not available
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
            try:
                import numpy as np
                # Convert to numpy arrays
                vec1 = np.array(self.domain_embeddings[domain1])
                vec2 = np.array(self.domain_embeddings[domain2])
                
                # Cosine similarity
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 * norm2 == 0:
                    similarity = 0.0
                else:
                    similarity = dot_product / (norm1 * norm2)
                    
                # Convert from numpy float to Python float
                similarity = float(similarity)
            except ImportError:
                # Fallback if numpy not available
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
        
        # Get statistical model
        model = self._build_statistical_mapping_model(procedure.domain, target_domain)
        
        if strategy == "direct_transfer":
            # Simple 1:1 mappings
            for step in procedure.steps:
                # Use the model to map the function if possible
                target_function = await self._map_function_with_model(
                    step["function"], 
                    procedure.domain,
                    target_domain,
                    model
                )
                
                # Use the model to map parameters
                target_params = {}
                for key, value in step.get("parameters", {}).items():
                    mapped_value = await self._map_parameter_with_model(
                        key,
                        value,
                        procedure.domain,
                        target_domain,
                        model
                    )
                    target_params[key] = mapped_value
                
                mappings.append({
                    "source_function": step["function"],
                    "target_function": target_function,
                    "source_parameters": step.get("parameters", {}),
                    "target_parameters": target_params,
                    "confidence": model.get("confidence", 0.9),
                    "mapping_type": "direct"
                })
        elif strategy == "parameter_adaptation":
            # Map functions directly but adapt parameters
            for step in procedure.steps:
                # Use model to map function
                target_function = await self._map_function_with_model(
                    step["function"], 
                    procedure.domain,
                    target_domain,
                    model
                )
                
                # Get adapted parameters
                adapted_params = {}
                for key, value in step.get("parameters", {}).items():
                    mapped_value = await self._map_parameter_with_model(
                        key,
                        value,
                        procedure.domain,
                        target_domain,
                        model
                    )
                    adapted_params[key] = mapped_value
                
                mappings.append({
                    "source_function": step["function"],
                    "target_function": target_function,
                    "source_parameters": step.get("parameters", {}),
                    "target_parameters": adapted_params,
                    "confidence": model.get("confidence", 0.7) * 0.9,  # Slightly lower confidence
                    "mapping_type": "parameter_adapted"
                })
        else:  # structural_adaptation
            # Look for equivalent functions in target domain
            for step in procedure.steps:
                # Use model to map function
                target_function = await self._map_function_with_model(
                    step["function"], 
                    procedure.domain,
                    target_domain,
                    model
                )
                
                # Get adapted parameters with fallbacks and structural changes
                adapted_params = {}
                for key, value in step.get("parameters", {}).items():
                    mapped_value = await self._map_parameter_with_model(
                        key,
                        value,
                        procedure.domain,
                        target_domain,
                        model
                    )
                    
                    # Check if key itself needs translation
                    mapped_key = key
                    if f"param_key_mappings" in model and key in model["param_key_mappings"]:
                        candidate_keys = model["param_key_mappings"][key]
                        if candidate_keys:
                            mapped_key = max(candidate_keys.items(), key=lambda x: x[1])[0]
                    
                    adapted_params[mapped_key] = mapped_value
                
                # Check for additional required parameters in target domain
                if target_function in model.get("additional_params", {}):
                    additional_params = model["additional_params"][target_function]
                    for add_key, add_value in additional_params.items():
                        if add_key not in adapted_params:
                            adapted_params[add_key] = add_value
                
                mappings.append({
                    "source_function": step["function"],
                    "target_function": target_function,
                    "source_parameters": step.get("parameters", {}),
                    "target_parameters": adapted_params,
                    "confidence": model.get("confidence", 0.5) * 0.8,  # Even lower confidence
                    "mapping_type": "structural_adapted",
                    "possible_alternatives": model.get("function_mappings", {}).get(
                        step["function"], {}
                    )
                })
        
        return mappings
    
    def _build_statistical_mapping_model(self, 
                                   source_domain: str, 
                                   target_domain: str) -> Dict[str, Any]:
        """Build a statistical model for mapping between domains based on history"""
        # Get transfer records between these domains
        relevant_records = [r for r in self.transfer_success_history 
                          if r["source_domain"] == source_domain and r["target_domain"] == target_domain]
        
        if not relevant_records:
            return {"model_type": "default", "confidence": 0.3}
        
        # Build function mapping probabilities
        function_mappings = {}
        for record in relevant_records:
            for source_func, target_func in record.get("function_mappings", {}).items():
                if source_func not in function_mappings:
                    function_mappings[source_func] = {}
                
                if target_func not in function_mappings[source_func]:
                    function_mappings[source_func][target_func] = 0
                    
                function_mappings[source_func][target_func] += 1
        
        # Convert counts to probabilities
        function_probs = {}
        for source_func, targets in function_mappings.items():
            total = sum(targets.values())
            function_probs[source_func] = {target: count/total for target, count in targets.items()}
        
        # Build parameter mapping probabilities
        param_mappings = {}
        for record in relevant_records:
            for param_key, value_mappings in record.get("parameter_mappings", {}).items():
                if param_key not in param_mappings:
                    param_mappings[param_key] = {}
                    
                for source_val, target_val in value_mappings.items():
                    if source_val not in param_mappings[param_key]:
                        param_mappings[param_key][source_val] = {}
                        
                    if target_val not in param_mappings[param_key][source_val]:
                        param_mappings[param_key][source_val][target_val] = 0
                        
                    param_mappings[param_key][source_val][target_val] += 1
        
        # Convert counts to probabilities
        param_probs = {}
        for param_key, source_values in param_mappings.items():
            param_probs[param_key] = {}
            for source_val, targets in source_values.items():
                total = sum(targets.values())
                param_probs[param_key][source_val] = {target: count/total for target, count in targets.items()}
        
        # Build parameter key mappings (some keys may need to be renamed)
        param_key_mappings = {}
        for record in relevant_records:
            for old_key, new_key in record.get("param_key_mappings", {}).items():
                if old_key not in param_key_mappings:
                    param_key_mappings[old_key] = {}
                
                if new_key not in param_key_mappings[old_key]:
                    param_key_mappings[old_key][new_key] = 0
                
                param_key_mappings[old_key][new_key] += 1
        
        # Convert to probabilities
        key_probs = {}
        for old_key, new_keys in param_key_mappings.items():
            total = sum(new_keys.values())
            key_probs[old_key] = {new_key: count/total for new_key, count in new_keys.items()}
        
        # Calculate overall model confidence
        avg_success = sum(r.get("success_rate", 0) for r in relevant_records) / len(relevant_records)
        
        # Build the model
        model = {
            "model_type": "statistical",
            "function_mappings": function_probs,
            "parameter_mappings": param_probs,
            "param_key_mappings": key_probs,
            "confidence": avg_success,
            "record_count": len(relevant_records),
            "created_at": datetime.datetime.now().isoformat()
        }
        
        # Store model for future use
        self.transfer_models[(source_domain, target_domain)] = model
        
        return model

    async def _map_function_with_model(self, 
                              function: str, 
                              source_domain: str, 
                              target_domain: str,
                              model: Dict[str, Any] = None) -> Optional[str]:
        """Map a function from source to target domain using statistical model"""
        # Get model if not provided
        if model is None:
            model = self._build_statistical_mapping_model(source_domain, target_domain)
        
        # Check if we have mappings for this function
        if model["model_type"] == "statistical" and function in model.get("function_mappings", {}):
            # Get probability distribution
            probs = model["function_mappings"][function]
            
            # Select highest probability mapping
            if probs:
                return max(probs.items(), key=lambda x: x[1])[0]
        
        # Fall back to default mapping (same function name)
        return function

    async def _map_parameter_with_model(self,
                               param_key: str,
                               param_value: Any,
                               source_domain: str,
                               target_domain: str,
                               model: Dict[str, Any] = None) -> Any:
        """Map a parameter value from source to target domain using statistical model"""
        # Get model if not provided
        if model is None:
            model = self._build_statistical_mapping_model(source_domain, target_domain)
        
        # Convert value to string for lookup
        str_value = str(param_value)
        
        # Check if we have mappings for this parameter
        if (model["model_type"] == "statistical" and 
            param_key in model.get("parameter_mappings", {}) and
            str_value in model["parameter_mappings"][param_key]):
            # Get probability distribution
            probs = model["parameter_mappings"][param_key][str_value]
            
            # Select highest probability mapping
            if probs:
                mapped_value = max(probs.items(), key=lambda x: x[1])[0]
                
                # Try to convert back to original type
                if isinstance(param_value, int):
                    try:
                        return int(mapped_value)
                    except ValueError:
                        pass
                elif isinstance(param_value, float):
                    try:
                        return float(mapped_value)
                    except ValueError:
                        pass
                elif isinstance(param_value, bool):
                    return mapped_value.lower() == "true"
                
                # Return as is if can't convert
                return mapped_value
        
        # Fall back to default mapping (same value)
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
    
    def _recommend_validation_steps(
        self,
        source_procedure: Procedure,
        target_domain: str,
        similarity: float
    ) -> List[Dict[str, Any]]:
        """Recommend validation steps based on transfer difficulty"""
        validation_steps = []
        
        # Base recommendations on similarity
        if similarity < 0.3:
            # Low similarity - extensive validation needed
            validation_steps.append({
                "type": "manual_review",
                "description": "Manually review all transferred steps before execution",
                "importance": "critical"
            })
            validation_steps.append({
                "type": "test_execution",
                "description": "Perform test execution in a safe environment",
                "importance": "critical"
            })
            validation_steps.append({
                "type": "step_verification",
                "description": "Verify each step individually before combining",
                "importance": "high"
            })
        elif similarity < 0.7:
            # Medium similarity - moderate validation
            validation_steps.append({
                "type": "manual_review",
                "description": "Manually review key transferred steps",
                "importance": "high"
            })
            validation_steps.append({
                "type": "test_execution",
                "description": "Perform test execution with monitoring",
                "importance": "high"
            })
        else:
            # High similarity - minimal validation
            validation_steps.append({
                "type": "test_execution",
                "description": "Perform a single test execution",
                "importance": "medium"
            })
        
        # Add procedure-specific recommendations
        if len(source_procedure.steps) > 5:
            validation_steps.append({
                "type": "incremental_testing",
                "description": "Test procedure in segments before full execution",
                "importance": "medium"
            })
        
        # If procedure has been executed many times, it's important
        if source_procedure.execution_count > 10:
            validation_steps.append({
                "type": "success_criteria",
                "description": "Define clear success criteria before execution",
                "importance": "high"
            })
        
        return validation_steps
        
    def update_from_transfer_result(
        self,
        source_domain: str,
        target_domain: str,
        success_rate: float,
        mappings: List[Dict[str, Any]]
    ) -> None:
        """Update optimizer based on transfer results"""
        # Record transfer result
        transfer_record = {
            "source_domain": source_domain,
            "target_domain": target_domain,
            "success_rate": success_rate,
            "function_mappings": {},
            "parameter_mappings": {},
            "param_key_mappings": {},
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
            
            # Check for parameter key mappings (renamed keys)
            source_keys = set(source_params.keys())
            target_keys = set(target_params.keys())
            
            # If keys don't match, record mappings
            if source_keys != target_keys:
                # Try to determine renamed keys
                for source_key in source_keys:
                    if source_key not in target_keys:
                        # This key was renamed or removed
                        for target_key in target_keys:
                            if target_key not in source_keys:
                                # Potential match - check value similarity
                                if source_key in source_params and target_key in target_params:
                                    source_val = source_params[source_key]
                                    target_val = target_params[target_key]
                                    
                                    # Simple check - if values are the same type, likely a rename
                                    if type(source_val) == type(target_val):
                                        transfer_record["param_key_mappings"][source_key] = target_key
            
            # Record value mappings
            for source_key, source_val in source_params.items():
                # Find matching target key
                target_key = source_key  # Default: same key
                
                # Check if key was renamed
                if source_key in transfer_record["param_key_mappings"]:
                    target_key = transfer_record["param_key_mappings"][source_key]
                
                # If target key exists, record mapping
                if target_key in target_params:
                    if source_key not in transfer_record["parameter_mappings"]:
                        transfer_record["parameter_mappings"][source_key] = {}
                        
                    # String conversion for keys
                    str_source_val = str(source_val)
                    transfer_record["parameter_mappings"][source_key][str_source_val] = target_params[target_key]
        
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
            
        # Rebuild transfer model with new data
        _ = self._build_statistical_mapping_model(source_domain, target_domain)
