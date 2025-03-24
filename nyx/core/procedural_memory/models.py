# nyx/core/procedural_memory/models.py

import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from pydantic import BaseModel, Field
import random
from collections import Counter

logger = logging.getLogger(__name__)

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

class CausalModel(BaseModel):
    """Causal model for reasoning about procedure failures"""
    causes: Dict[str, List[Dict[str, float]]] = Field(default_factory=dict)
    interventions: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    error_history: List[Dict[str, Any]] = Field(default_factory=list)
    max_history: int = 50

class ProcedureGraph(BaseModel):
    """Graph representation of a procedure for flexible execution"""
    nodes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    entry_points: List[str] = Field(default_factory=list)
    exit_points: List[str] = Field(default_factory=list)

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
