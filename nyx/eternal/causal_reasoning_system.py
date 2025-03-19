# nyx/eternal/causal_reasoning_system.py

import asyncio
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import math
import time
import networkx as nx
import random
from collections import defaultdict

logger = logging.getLogger(__name__)

class CausalNode:
    """Represents a node in a causal graph"""
    
    def __init__(self, node_id: str, name: str, domain: str = "", 
                node_type: str = "variable", metadata: Dict[str, Any] = None):
        self.id = node_id
        self.name = name
        self.domain = domain
        self.type = node_type  # variable, event, action, state, etc.
        self.metadata = metadata or {}
        self.states = []  # Possible states for discrete variables
        self.default_state = None
        self.timestamp = datetime.now()
        self.observations = []  # Observed values
        self.distribution = {}  # Probability distribution
        
    def add_state(self, state: Any) -> None:
        """Add a possible state for this node"""
        if state not in self.states:
            self.states.append(state)
            
        # Set as default if it's the first state
        if len(self.states) == 1:
            self.default_state = state
    
    def set_default_state(self, state: Any) -> None:
        """Set the default state for this node"""
        if state not in self.states and self.states:
            raise ValueError(f"State {state} not in possible states: {self.states}")
        
        self.default_state = state
    
    def add_observation(self, value: Any, confidence: float = 1.0, 
                      timestamp: Optional[datetime] = None) -> None:
        """Add an observed value for this node"""
        self.observations.append({
            "value": value,
            "confidence": confidence,
            "timestamp": timestamp or datetime.now()
        })
        
        # Update distribution
        self._update_distribution()
    
    def _update_distribution(self) -> None:
        """Update probability distribution based on observations"""
        if not self.states or not self.observations:
            return
            
        # Initialize distribution
        self.distribution = {state: 0.0 for state in self.states}
        total_confidence = 0.0
        
        # Aggregate observations
        for observation in self.observations:
            value = observation["value"]
            confidence = observation["confidence"]
            
            if value in self.distribution:
                self.distribution[value] += confidence
                total_confidence += confidence
        
        # Normalize distribution
        if total_confidence > 0:
            for state in self.distribution:
                self.distribution[state] /= total_confidence
    
    def get_current_state(self) -> Any:
        """Get the most likely current state based on observations"""
        if not self.distribution:
            return self.default_state
            
        # Get state with highest probability
        return max(self.distribution.items(), key=lambda x: x[1])[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain,
            "type": self.type,
            "metadata": self.metadata,
            "states": self.states,
            "default_state": self.default_state,
            "timestamp": self.timestamp.isoformat(),
            "observations": [
                {
                    "value": obs["value"],
                    "confidence": obs["confidence"],
                    "timestamp": obs["timestamp"].isoformat()
                }
                for obs in self.observations
            ],
            "distribution": self.distribution
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CausalNode':
        """Create from dictionary representation"""
        node = cls(
            node_id=data["id"],
            name=data["name"],
            domain=data["domain"],
            node_type=data["type"],
            metadata=data["metadata"]
        )
        
        for state in data["states"]:
            node.add_state(state)
            
        node.default_state = data["default_state"]
        node.timestamp = datetime.fromisoformat(data["timestamp"])
        node.distribution = data["distribution"]
        
        for obs in data["observations"]:
            node.observations.append({
                "value": obs["value"],
                "confidence": obs["confidence"],
                "timestamp": datetime.fromisoformat(obs["timestamp"])
            })
            
        return node

class CausalRelation:
    """Represents a causal relation between nodes"""
    
    def __init__(self, relation_id: str, source_id: str, target_id: str, 
                relation_type: str = "causal", strength: float = 0.5,
                mechanism: str = "", evidence: List[Dict[str, Any]] = None):
        self.id = relation_id
        self.source_id = source_id
        self.target_id = target_id
        self.type = relation_type  # causal, correlation, mediation, etc.
        self.strength = strength  # 0.0 to 1.0
        self.mechanism = mechanism  # Description of causal mechanism
        self.evidence = evidence or []
        self.timestamp = datetime.now()
        self.conditional_probabilities = {}  # State combinations -> probabilities
        self.parameters = {}  # For parametric models
        
    def add_evidence(self, description: str, strength: float, 
                   source: str = "", timestamp: Optional[datetime] = None) -> None:
        """Add evidence supporting this causal relation"""
        self.evidence.append({
            "description": description,
            "strength": strength,
            "source": source,
            "timestamp": timestamp or datetime.now()
        })
        
        # Update strength based on evidence
        if self.evidence:
            avg_strength = sum(e["strength"] for e in self.evidence) / len(self.evidence)
            self.strength = avg_strength
    
    def add_conditional_probability(self, source_state: Any, target_state: Any, 
                                 probability: float) -> None:
        """Add a conditional probability P(target_state|source_state)"""
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability must be between 0 and 1, got {probability}")
            
        key = f"{source_state}|{target_state}"
        self.conditional_probabilities[key] = probability
    
    def get_conditional_probability(self, source_state: Any, target_state: Any) -> float:
        """Get P(target_state|source_state)"""
        key = f"{source_state}|{target_state}"
        return self.conditional_probabilities.get(key, 0.5)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "strength": self.strength,
            "mechanism": self.mechanism,
            "evidence": [
                {
                    "description": e["description"],
                    "strength": e["strength"],
                    "source": e["source"],
                    "timestamp": e["timestamp"].isoformat()
                }
                for e in self.evidence
            ],
            "timestamp": self.timestamp.isoformat(),
            "conditional_probabilities": self.conditional_probabilities,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CausalRelation':
        """Create from dictionary representation"""
        relation = cls(
            relation_id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["type"],
            strength=data["strength"],
            mechanism=data["mechanism"]
        )
        
        for e in data["evidence"]:
            relation.evidence.append({
                "description": e["description"],
                "strength": e["strength"],
                "source": e["source"],
                "timestamp": datetime.fromisoformat(e["timestamp"])
            })
            
        relation.timestamp = datetime.fromisoformat(data["timestamp"])
        relation.conditional_probabilities = data["conditional_probabilities"]
        relation.parameters = data["parameters"]
        
        return relation

class CausalModel:
    """Represents a causal model for a specific domain"""
    
    def __init__(self, model_id: str, name: str, domain: str = "", 
                metadata: Dict[str, Any] = None):
        self.id = model_id
        self.name = name
        self.domain = domain
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.nodes = {}  # id -> CausalNode
        self.relations = {}  # id -> CausalRelation
        self.graph = nx.DiGraph()  # Directed graph for causal structure
        self.next_node_id = 1
        self.next_relation_id = 1
        self.assumptions = []  # Model assumptions
        self.validation_results = []  # Model validation results
        
    def add_node(self, name: str, domain: str = "", node_type: str = "variable",
               metadata: Dict[str, Any] = None) -> str:
        """Add a node to the causal model"""
        # Generate node ID
        node_id = f"node_{self.next_node_id}"
        self.next_node_id += 1
        
        # Create node
        node = CausalNode(
            node_id=node_id,
            name=name,
            domain=domain or self.domain,
            node_type=node_type,
            metadata=metadata
        )
        
        # Add to model
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.to_dict())
        
        return node_id
    
    def add_relation(self, source_id: str, target_id: str, relation_type: str = "causal",
                   strength: float = 0.5, mechanism: str = "") -> str:
        """Add a relation between nodes"""
        # Check nodes exist
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError(f"Source or target node not found: {source_id} -> {target_id}")
            
        # Generate relation ID
        relation_id = f"relation_{self.next_relation_id}"
        self.next_relation_id += 1
        
        # Create relation
        relation = CausalRelation(
            relation_id=relation_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            strength=strength,
            mechanism=mechanism
        )
        
        # Add to model
        self.relations[relation_id] = relation
        self.graph.add_edge(source_id, target_id, **relation.to_dict())
        
        return relation_id
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the model"""
        if node_id not in self.nodes:
            return False
            
        # Remove from nodes
        del self.nodes[node_id]
        
        # Remove from graph
        self.graph.remove_node(node_id)
        
        # Remove related relations
        to_remove = []
        for relation_id, relation in self.relations.items():
            if relation.source_id == node_id or relation.target_id == node_id:
                to_remove.append(relation_id)
                
        for relation_id in to_remove:
            del self.relations[relation_id]
            
        return True
    
    def remove_relation(self, relation_id: str) -> bool:
        """Remove a relation from the model"""
        if relation_id not in self.relations:
            return False
            
        relation = self.relations[relation_id]
        
        # Remove from relations
        del self.relations[relation_id]
        
        # Remove from graph
        self.graph.remove_edge(relation.source_id, relation.target_id)
        
        return True
    
    def get_node(self, node_id: str) -> Optional[CausalNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_relation(self, relation_id: str) -> Optional[CausalRelation]:
        """Get a relation by ID"""
        return self.relations.get(relation_id)
    
    def get_relations_between(self, source_id: str, target_id: str) -> List[CausalRelation]:
        """Get all relations between two nodes"""
        relations = []
        for relation in self.relations.values():
            if relation.source_id == source_id and relation.target_id == target_id:
                relations.append(relation)
        return relations
    
    def get_ancestors(self, node_id: str) -> List[str]:
        """Get all ancestor nodes of a node"""
        return list(nx.ancestors(self.graph, node_id))
    
    def get_descendants(self, node_id: str) -> List[str]:
        """Get all descendant nodes of a node"""
        return list(nx.descendants(self.graph, node_id))
    
    def get_markov_blanket(self, node_id: str) -> List[str]:
        """Get the Markov blanket of a node (parents, children, children's parents)"""
        if node_id not in self.nodes:
            return []
            
        # Get parents
        parents = list(self.graph.predecessors(node_id))
        
        # Get children
        children = list(self.graph.successors(node_id))
        
        # Get parents of children
        parents_of_children = []
        for child in children:
            for parent in self.graph.predecessors(child):
                if parent != node_id and parent not in parents_of_children:
                    parents_of_children.append(parent)
        
        # Combine all
        return list(set(parents + children + parents_of_children))
    
    def add_assumption(self, description: str, confidence: float = 0.5) -> None:
        """Add a model assumption"""
        self.assumptions.append({
            "description": description,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_validation_result(self, method: str, result: Dict[str, Any]) -> None:
        """Add a model validation result"""
        self.validation_results.append({
            "method": method,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "relations": {rel_id: rel.to_dict() for rel_id, rel in self.relations.items()},
            "next_node_id": self.next_node_id,
            "next_relation_id": self.next_relation_id,
            "assumptions": self.assumptions,
            "validation_results": self.validation_results
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CausalModel':
        """Create from dictionary representation"""
        model = cls(
            model_id=data["id"],
            name=data["name"],
            domain=data["domain"],
            metadata=data["metadata"]
        )
        
        model.timestamp = datetime.fromisoformat(data["timestamp"])
        model.next_node_id = data["next_node_id"]
        model.next_relation_id = data["next_relation_id"]
        model.assumptions = data["assumptions"]
        model.validation_results = data["validation_results"]
        
        # Load nodes
        for node_id, node_data in data["nodes"].items():
            model.nodes[node_id] = CausalNode.from_dict(node_data)
            model.graph.add_node(node_id, **node_data)
            
        # Load relations
        for rel_id, rel_data in data["relations"].items():
            model.relations[rel_id] = CausalRelation.from_dict(rel_data)
            model.graph.add_edge(
                rel_data["source_id"], 
                rel_data["target_id"], 
                **rel_data
            )
            
        return model

class Intervention:
    """Represents an intervention in a causal model"""
    
    def __init__(self, intervention_id: str, name: str, 
                target_node: str, target_value: Any,
                description: str = ""):
        self.id = intervention_id
        self.name = name
        self.target_node = target_node
        self.target_value = target_value
        self.description = description
        self.timestamp = datetime.now()
        self.effects = {}  # node_id -> effect
        self.side_effects = {}  # node_id -> side effect
        
    def add_effect(self, node_id: str, expected_value: Any, 
                 probability: float, mechanism: str = "") -> None:
        """Add an expected effect of this intervention"""
        self.effects[node_id] = {
            "expected_value": expected_value,
            "probability": probability,
            "mechanism": mechanism,
            "observed": False,
            "actual_value": None
        }
    
    def add_side_effect(self, node_id: str, expected_value: Any,
                      probability: float, severity: float = 0.5,
                      mechanism: str = "") -> None:
        """Add an expected side effect of this intervention"""
        self.side_effects[node_id] = {
            "expected_value": expected_value,
            "probability": probability,
            "severity": severity,
            "mechanism": mechanism,
            "observed": False,
            "actual_value": None
        }
    
    def record_observation(self, node_id: str, observed_value: Any, 
                         is_side_effect: bool = False) -> None:
        """Record an observed value for an effect or side effect"""
        if is_side_effect:
            if node_id in self.side_effects:
                self.side_effects[node_id]["observed"] = True
                self.side_effects[node_id]["actual_value"] = observed_value
        else:
            if node_id in self.effects:
                self.effects[node_id]["observed"] = True
                self.effects[node_id]["actual_value"] = observed_value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "target_node": self.target_node,
            "target_value": self.target_value,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "effects": self.effects,
            "side_effects": self.side_effects
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Intervention':
        """Create from dictionary representation"""
        intervention = cls(
            intervention_id=data["id"],
            name=data["name"],
            target_node=data["target_node"],
            target_value=data["target_value"],
            description=data["description"]
        )
        
        intervention.timestamp = datetime.fromisoformat(data["timestamp"])
        intervention.effects = data["effects"]
        intervention.side_effects = data["side_effects"]
        
        return intervention

class CausalReasoningSystem:
    """System for identifying and reasoning about causal relationships"""
    
    def __init__(self):
        self.causal_models = {}  # domain -> CausalModel
        self.interventions = {}  # id -> Intervention
        self.counterfactuals = {}  # id -> Counterfactual analysis
        self.observations = []  # List of observed data points
        
        # Configuration
        self.config = {
            "min_relation_strength": 0.3,  # Minimum strength to consider a causal relation
            "max_model_depth": 5,  # Maximum depth for model inference
            "default_confidence": 0.7,  # Default confidence in absence of evidence
            "enable_auto_discovery": True,  # Automatically discover causal relations
            "discovery_threshold": 0.6,  # Threshold for automatic discovery
            "min_data_points": 10  # Minimum data points for causal discovery
        }
        
        # Statistics
        self.stats = {
            "models_created": 0,
            "nodes_created": 0,
            "relations_created": 0,
            "interventions_performed": 0,
            "counterfactuals_analyzed": 0,
            "discovery_runs": 0
        }
        
        # Initialize next ID counters
        self.next_model_id = 1
        self.next_intervention_id = 1
        self.next_counterfactual_id = 1
    
    async def create_causal_model(self, name: str, domain: str, 
                            metadata: Dict[str, Any] = None) -> str:
        """Create a new causal model"""
        # Generate model ID
        model_id = f"model_{self.next_model_id}"
        self.next_model_id += 1
        
        # Create model
        model = CausalModel(
            model_id=model_id,
            name=name,
            domain=domain,
            metadata=metadata
        )
        
        # Store model
        self.causal_models[model_id] = model
        
        # Update statistics
        self.stats["models_created"] += 1
        
        return model_id
    
    async def add_node_to_model(self, model_id: str, name: str, 
                          domain: str = "", node_type: str = "variable",
                          metadata: Dict[str, Any] = None) -> str:
        """Add a node to a causal model"""
        if model_id not in self.causal_models:
            raise ValueError(f"Model {model_id} not found")
            
        # Add node to model
        node_id = self.causal_models[model_id].add_node(
            name=name,
            domain=domain,
            node_type=node_type,
            metadata=metadata
        )
        
        # Update statistics
        self.stats["nodes_created"] += 1
        
        return node_id
    
    async def add_relation_to_model(self, model_id: str, source_id: str, target_id: str,
                             relation_type: str = "causal", strength: float = 0.5,
                             mechanism: str = "") -> str:
        """Add a causal relation to a model"""
        if model_id not in self.causal_models:
            raise ValueError(f"Model {model_id} not found")
            
        # Add relation to model
        relation_id = self.causal_models[model_id].add_relation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            strength=strength,
            mechanism=mechanism
        )
        
        # Update statistics
        self.stats["relations_created"] += 1
        
        return relation_id
    
    async def add_evidence_to_relation(self, model_id: str, relation_id: str,
                                 description: str, strength: float,
                                 source: str = "") -> bool:
        """Add evidence to a causal relation"""
        if model_id not in self.causal_models:
            raise ValueError(f"Model {model_id} not found")
            
        model = self.causal_models[model_id]
        
        if relation_id not in model.relations:
            raise ValueError(f"Relation {relation_id} not found in model {model_id}")
            
        # Add evidence
        model.relations[relation_id].add_evidence(
            description=description,
            strength=strength,
            source=source
        )
        
        return True
    
    async def add_observation_to_node(self, model_id: str, node_id: str,
                               value: Any, confidence: float = 1.0) -> bool:
        """Add an observed value to a node"""
        if model_id not in self.causal_models:
            raise ValueError(f"Model {model_id} not found")
            
        model = self.causal_models[model_id]
        
        if node_id not in model.nodes:
            raise ValueError(f"Node {node_id} not found in model {model_id}")
            
        # Add observation
        model.nodes[node_id].add_observation(
            value=value,
            confidence=confidence
        )
        
        # Add to global observations
        self.observations.append({
            "model_id": model_id,
            "node_id": node_id,
            "value": value,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
        
        # If auto-discovery is enabled, check if we have enough data
        if (self.config["enable_auto_discovery"] and 
            len(self.observations) % self.config["min_data_points"] == 0):
            asyncio.create_task(self.discover_causal_relations(model_id))
        
        return True
    
    async def infer_causal_structure(self, model_id: str, 
                              observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Infer causal structure from observations"""
        if model_id not in self.causal_models:
            raise ValueError(f"Model {model_id} not found")
            
        model = self.causal_models[model_id]
        
        # Process observations
        for observation in observations:
            node_id = observation.get("node_id")
            value = observation.get("value")
            confidence = observation.get("confidence", 1.0)
            
            if node_id and node_id in model.nodes and value is not None:
                model.nodes[node_id].add_observation(value, confidence)
        
        # Identify potential causal relationships
        potential_relations = await self._identify_potential_relationships(model, observations)
        
        # Test causal hypotheses
        tested_relations = await self._test_causal_hypotheses(model, potential_relations)
        
        # Build causal model
        updated_model = await self._build_causal_model(model, tested_relations)
        
        # Validate model against observations
        validation = await self._validate_causal_model(updated_model, observations)
        
        # Update statistics
        self.stats["discovery_runs"] += 1
        
        return {
            "model_id": model_id,
            "potential_relations": len(potential_relations),
            "accepted_relations": len(tested_relations),
            "validation_score": validation.get("score", 0.0),
            "validation_details": validation
        }
    
    async def _identify_potential_relationships(self, model: CausalModel, 
                                       observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify potential causal relationships from observations"""
        # Extract unique node IDs from observations
        node_ids = set()
        for obs in observations:
            if "node_id" in obs and obs["node_id"] in model.nodes:
                node_ids.add(obs["node_id"])
        
        # Group observations by node
        node_observations = defaultdict(list)
        for obs in observations:
            if "node_id" in obs and "value" in obs:
                node_observations[obs["node_id"]].append(obs["value"])
        
        # Check for pairwise correlations
        potential_relations = []
        for node1 in node_ids:
            for node2 in node_ids:
                if node1 != node2:
                    # Skip if relation already exists
                    if model.get_relations_between(node1, node2):
                        continue
                    
                    # Calculate correlation if we have enough data
                    values1 = node_observations.get(node1, [])
                    values2 = node_observations.get(node2, [])
                    
                    if len(values1) >= 5 and len(values2) >= 5:
                        # Use the minimum length
                        min_length = min(len(values1), len(values2))
                        
                        # Convert to numeric if possible
                        try:
                            numeric1 = [float(v) if isinstance(v, (int, float)) else 0.0 for v in values1[:min_length]]
                            numeric2 = [float(v) if isinstance(v, (int, float)) else 0.0 for v in values2[:min_length]]
                            
                            # Calculate correlation
                            correlation = np.corrcoef(numeric1, numeric2)[0, 1]
                            
                            # If significant correlation, add as potential relation
                            if abs(correlation) >= 0.3:
                                potential_relations.append({
                                    "source_id": node1,
                                    "target_id": node2,
                                    "type": "correlation",
                                    "strength": abs(correlation),
                                    "direction": "positive" if correlation > 0 else "negative"
                                })
                        except:
                            # Skip if numeric conversion fails
                            pass
        
        return potential_relations
    
    async def _test_causal_hypotheses(self, model: CausalModel, 
                              potential_relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Test causal hypotheses based on potential relationships"""
        tested_relations = []
        
        for relation in potential_relations:
            source_id = relation["source_id"]
            target_id = relation["target_id"]
            strength = relation["strength"]
            
            # Skip if strength is below threshold
            if strength < self.config["min_relation_strength"]:
                continue
            
            # Check temporal ordering if available
            temporal_evidence = await self._check_temporal_ordering(model, source_id, target_id)
            
            # Apply structural constraints
            structurally_valid = await self._check_structural_constraints(model, source_id, target_id)
            
            # Calculate causal score
            causal_score = strength
            if temporal_evidence:
                causal_score *= 1.2  # Boost for temporal evidence
            
            if not structurally_valid:
                causal_score *= 0.5  # Penalty for structural issues
            
            # Add if score is above threshold
            if causal_score >= self.config["discovery_threshold"]:
                tested_relations.append({
                    "source_id": source_id,
                    "target_id": target_id,
                    "type": "causal" if causal_score >= 0.7 else "potential_causal",
                    "strength": causal_score,
                    "temporal_evidence": temporal_evidence,
                    "structurally_valid": structurally_valid
                })
        
        return tested_relations
    
    async def _check_temporal_ordering(self, model: CausalModel, 
                              source_id: str, target_id: str) -> bool:
        """Check if temporal ordering supports causality (cause precedes effect)"""
        source_node = model.nodes.get(source_id)
        target_node = model.nodes.get(target_id)
        
        if not source_node or not target_node:
            return False
        
        # Check observations with timestamps
        source_times = [obs["timestamp"] for obs in source_node.observations if "timestamp" in obs]
        target_times = [obs["timestamp"] for obs in target_node.observations if "timestamp" in obs]
        
        if not source_times or not target_times:
            return False
        
        # Get median timestamps
        source_median = sorted(source_times)[len(source_times) // 2]
        target_median = sorted(target_times)[len(target_times) // 2]
        
        # Check if source typically precedes target
        return source_median < target_median
    
    async def _check_structural_constraints(self, model: CausalModel, 
                                  source_id: str, target_id: str) -> bool:
        """Check structural constraints for causal relationships"""
        # Check for cycles
        if source_id in model.get_descendants(target_id):
            return False  # Would create a cycle
        
        # Check for domain compatibility
        source_node = model.nodes.get(source_id)
        target_node = model.nodes.get(target_id)
        
        if source_node and target_node:
            # If domains are specified and different, lower probability of causation
            if source_node.domain and target_node.domain and source_node.domain != target_node.domain:
                return False
        
        return True
    
    async def _build_causal_model(self, model: CausalModel, 
                          tested_relations: List[Dict[str, Any]]) -> CausalModel:
        """Build or update a causal model based on tested relations"""
        # Add relations to model
        for relation in tested_relations:
            source_id = relation["source_id"]
            target_id = relation["target_id"]
            relation_type = relation["type"]
            strength = relation["strength"]
            
            # Check if relation already exists
            existing_relations = model.get_relations_between(source_id, target_id)
            
            if existing_relations:
                # Update existing relation
                for rel in existing_relations:
                    rel_obj = model.relations.get(rel.id)
                    if rel_obj:
                        # Update strength
                        rel_obj.strength = (rel_obj.strength + strength) / 2
                        
                        # Add evidence
                        rel_obj.add_evidence(
                            description="Automatically discovered by causal reasoning system",
                            strength=strength,
                            source="causal_inference"
                        )
            else:
                # Create new relation
                model.add_relation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    strength=strength,
                    mechanism="Automatically discovered by causal reasoning system"
                )
        
        return model
    
async def _validate_causal_model(self, model: CausalModel, 
                            observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate causal model against observations"""
        # Calculate model quality metrics
        validation = {
            "node_count": len(model.nodes),
            "relation_count": len(model.relations),
            "coverage": 0.0,  # Percentage of observed variables covered by the model
            "accuracy": 0.0,  # Accuracy of predictions
            "complexity": 0.0,  # Model complexity
            "score": 0.0      # Overall validation score
        }
        
        # Calculate coverage
        observed_nodes = set(obs["node_id"] for obs in observations if "node_id" in obs)
        model_nodes = set(model.nodes.keys())
        covered_nodes = observed_nodes.intersection(model_nodes)
        
        if observed_nodes:
            validation["coverage"] = len(covered_nodes) / len(observed_nodes)
        
        # Calculate prediction accuracy (hold out some observations and test)
        accuracy_scores = []
        for node_id in covered_nodes:
            node = model.nodes[node_id]
            if len(node.observations) >= 4:
                # Use last observation as test
                test_obs = node.observations[-1]
                predicted_state = self._predict_node_state(model, node_id)
                
                # Calculate accuracy
                if predicted_state is not None and test_obs["value"] == predicted_state:
                    accuracy_scores.append(1.0)
                else:
                    accuracy_scores.append(0.0)
        
        if accuracy_scores:
            validation["accuracy"] = sum(accuracy_scores) / len(accuracy_scores)
        
        # Calculate complexity
        validation["complexity"] = len(model.relations) / (len(model.nodes) if model.nodes else 1)
        
        # Calculate overall score
        validation["score"] = (
            validation["coverage"] * 0.4 +
            validation["accuracy"] * 0.4 +
            (1.0 / (1.0 + validation["complexity"])) * 0.2  # Penalize complexity
        )
        
        # Add validation result to model
        model.add_validation_result("system_validation", validation)
        
        return validation
    
    def _predict_node_state(self, model: CausalModel, node_id: str) -> Any:
        """Predict the state of a node based on the causal model"""
        if node_id not in model.nodes:
            return None
            
        node = model.nodes[node_id]
        
        # If node has observations, use most likely state
        if node.observations:
            return node.get_current_state()
            
        # Otherwise, infer from causal parents
        parents = list(model.graph.predecessors(node_id))
        if not parents:
            return node.default_state
        
        # Get parent states
        parent_states = {}
        for parent_id in parents:
            parent = model.nodes.get(parent_id)
            if parent:
                parent_states[parent_id] = parent.get_current_state()
        
        # Simple inference: majority vote over parent relations
        target_probabilities = {}
        
        for parent_id, parent_state in parent_states.items():
            # Get relations from this parent
            for relation_id, relation in model.relations.items():
                if relation.source_id == parent_id and relation.target_id == node_id:
                    # Get conditional probability if available
                    for target_state in node.states:
                        prob = relation.get_conditional_probability(parent_state, target_state)
                        target_probabilities[target_state] = target_probabilities.get(target_state, 0) + prob
        
        # Return most likely state
        if target_probabilities:
            return max(target_probabilities.items(), key=lambda x: x[1])[0]
        
        return node.default_state
    
    async def discover_causal_relations(self, model_id: str) -> Dict[str, Any]:
        """Discover causal relations in a model automatically"""
        if model_id not in self.causal_models:
            raise ValueError(f"Model {model_id} not found")
            
        model = self.causal_models[model_id]
        
        # Collect observations from model
        observations = []
        for node_id, node in model.nodes.items():
            for obs in node.observations:
                observations.append({
                    "node_id": node_id,
                    "value": obs["value"],
                    "confidence": obs["confidence"],
                    "timestamp": obs["timestamp"]
                })
        
        # If not enough observations, return empty result
        if len(observations) < self.config["min_data_points"]:
            return {
                "status": "insufficient_data",
                "required": self.config["min_data_points"],
                "available": len(observations)
            }
        
        # Run causal structure inference
        result = await self.infer_causal_structure(model_id, observations)
        
        return result
    
    async def define_intervention(self, model_id: str, target_node: str, 
                           target_value: Any, name: str,
                           description: str = "") -> str:
        """Define an intervention on a causal model"""
        if model_id not in self.causal_models:
            raise ValueError(f"Model {model_id} not found")
            
        model = self.causal_models[model_id]
        
        if target_node not in model.nodes:
            raise ValueError(f"Node {target_node} not found in model {model_id}")
            
        # Generate intervention ID
        intervention_id = f"intervention_{self.next_intervention_id}"
        self.next_intervention_id += 1
        
        # Create intervention
        intervention = Intervention(
            intervention_id=intervention_id,
            name=name,
            target_node=target_node,
            target_value=target_value,
            description=description
        )
        
        # Predict effects
        effects = await self._predict_intervention_effects(model, target_node, target_value)
        
        # Add effects to intervention
        for node_id, effect in effects.items():
            if effect["is_side_effect"]:
                intervention.add_side_effect(
                    node_id=node_id,
                    expected_value=effect["expected_value"],
                    probability=effect["probability"],
                    severity=effect.get("severity", 0.5),
                    mechanism=effect.get("mechanism", "")
                )
            else:
                intervention.add_effect(
                    node_id=node_id,
                    expected_value=effect["expected_value"],
                    probability=effect["probability"],
                    mechanism=effect.get("mechanism", "")
                )
        
        # Store intervention
        self.interventions[intervention_id] = intervention
        
        # Update statistics
        self.stats["interventions_performed"] += 1
        
        return intervention_id
    
    async def _predict_intervention_effects(self, model: CausalModel, 
                                   target_node: str, target_value: Any) -> Dict[str, Dict[str, Any]]:
        """Predict the effects of an intervention"""
        effects = {}
        
        # Get descendants of target node
        descendants = model.get_descendants(target_node)
        
        # Predict effects on each descendant
        for desc_id in descendants:
            descendant = model.nodes.get(desc_id)
            if not descendant:
                continue
                
            # Get causal path from target to descendant
            paths = list(nx.all_simple_paths(model.graph, target_node, desc_id, 
                                         cutoff=self.config["max_model_depth"]))
            
            if not paths:
                continue
                
            # Calculate effect probability based on path strengths
            path_probabilities = []
            expected_values = {}
            
            for path in paths:
                # Calculate path strength
                path_strength = 1.0
                
                for i in range(len(path) - 1):
                    source = path[i]
                    target = path[i + 1]
                    
                    # Get relation between these nodes
                    relations = model.get_relations_between(source, target)
                    if relations:
                        # Use max strength relation
                        max_strength = max(rel.strength for rel in relations)
                        path_strength *= max_strength
                
                path_probabilities.append(path_strength)
                
                # Determine expected value through this path
                if path_strength > 0.2:  # Only consider significant paths
                    # Simplified logic - in a real system would use conditional probabilities
                    if descendant.states:
                        # For discrete states, choose most likely next state
                        current_state_idx = descendant.states.index(descendant.get_current_state()) if descendant.get_current_state() in descendant.states else 0
                        next_state_idx = (current_state_idx + 1) % len(descendant.states)
                        expected_value = descendant.states[next_state_idx]
                    else:
                        # For continuous values, increment/decrement based on path structure
                        current_value = descendant.get_current_state() or 0
                        if isinstance(current_value, (int, float)):
                            expected_value = current_value + (0.1 * len(path))  # Simple increment
                        else:
                            expected_value = current_value
                    
                    # Count occurrences of expected values
                    expected_values[expected_value] = expected_values.get(expected_value, 0) + path_strength
            
            # Calculate overall probability
            overall_probability = max(path_probabilities) if path_probabilities else 0.0
            
            # Determine most likely expected value
            most_likely_value = None
            if expected_values:
                most_likely_value = max(expected_values.items(), key=lambda x: x[1])[0]
            else:
                most_likely_value = descendant.get_current_state()
            
            # Determine if this is a direct effect or side effect
            is_side_effect = len(paths[0]) > 2  # If path length > 2, it's not a direct effect
            
            effects[desc_id] = {
                "expected_value": most_likely_value,
                "probability": overall_probability,
                "is_side_effect": is_side_effect,
                "severity": 0.3 if is_side_effect else 0.0,  # Simplified severity calculation
                "mechanism": f"Causal path via {len(paths)} paths"
            }
        
        return effects
    
    async def record_intervention_outcome(self, intervention_id: str, 
                                  outcomes: Dict[str, Any]) -> Dict[str, Any]:
        """Record the outcomes of an intervention"""
        if intervention_id not in self.interventions:
            raise ValueError(f"Intervention {intervention_id} not found")
            
        intervention = self.interventions[intervention_id]
        
        # Record observations for each outcome
        for node_id, value in outcomes.items():
            # Check if this is an effect or side effect
            is_side_effect = node_id in intervention.side_effects and node_id not in intervention.effects
            
            # Record observation
            intervention.record_observation(node_id, value, is_side_effect)
        
        # Evaluate intervention success
        success_score = await self._evaluate_intervention_success(intervention)
        
        return {
            "intervention_id": intervention_id,
            "success_score": success_score,
            "recorded_outcomes": len(outcomes)
        }
    
    async def _evaluate_intervention_success(self, intervention: Intervention) -> float:
        """Evaluate the success of an intervention"""
        # Count correct predictions
        total_effects = len(intervention.effects)
        correct_effects = 0
        
        for node_id, effect in intervention.effects.items():
            if effect["observed"]:
                if effect["actual_value"] == effect["expected_value"]:
                    correct_effects += 1
        
        # Calculate success score
        success_score = correct_effects / total_effects if total_effects > 0 else 0.0
        
        # Adjust for side effects
        side_effect_penalty = 0.0
        for node_id, effect in intervention.side_effects.items():
            if effect["observed"]:
                side_effect_penalty += effect["severity"] * 0.1
        
        # Final adjusted score
        adjusted_score = max(0.0, success_score - side_effect_penalty)
        
        return adjusted_score
    
    async def reason_counterfactually(self, model_id: str, 
                              query: Dict[str, Any]) -> Dict[str, Any]:
        """Perform counterfactual reasoning using causal model"""
        if model_id not in self.causal_models:
            raise ValueError(f"Model {model_id} not found")
            
        model = self.causal_models[model_id]
        
        # Extract query components
        counterfactual_id = f"counterfactual_{self.next_counterfactual_id}"
        self.next_counterfactual_id += 1
        
        factual_values = query.get("factual_values", {})
        counterfactual_values = query.get("counterfactual_values", {})
        target_nodes = query.get("target_nodes", [])
        
        # Validate query
        if not counterfactual_values:
            return {
                "error": "No counterfactual values provided",
                "counterfactual_id": counterfactual_id
            }
        
        # Identify relevant variables
        relevant_vars = await self._identify_relevant_variables(model, counterfactual_values, target_nodes)
        
        # Build counterfactual world
        counterfactual_world = await self._set_counterfactual_values(model, factual_values, counterfactual_values)
        
        # Propagate effects through causal graph
        results = await self._propagate_effects(model, counterfactual_world, relevant_vars)
        
        # Analyze results
        analysis = await self._analyze_counterfactual_results(model, factual_values, counterfactual_values, results)
        
        # Store counterfactual analysis
        self.counterfactuals[counterfactual_id] = {
            "timestamp": datetime.now().isoformat(),
            "model_id": model_id,
            "query": query,
            "relevant_variables": relevant_vars,
            "results": results,
            "analysis": analysis
        }
        
        # Update statistics
        self.stats["counterfactuals_analyzed"] += 1
        
        return {
            "counterfactual_id": counterfactual_id,
            "relevant_variables": relevant_vars,
            "results": results,
            "analysis": analysis
        }
    
    async def _identify_relevant_variables(self, model: CausalModel,
                                  counterfactual_values: Dict[str, Any],
                                  target_nodes: List[str]) -> List[str]:
        """Identify variables relevant to the counterfactual query"""
        relevant_vars = set()
        
        # Add counterfactual nodes
        for node_id in counterfactual_values.keys():
            if node_id in model.nodes:
                relevant_vars.add(node_id)
                
                # Add descendants of counterfactual nodes
                descendants = model.get_descendants(node_id)
                relevant_vars.update(descendants)
        
        # Add target nodes
        for node_id in target_nodes:
            if node_id in model.nodes:
                relevant_vars.add(node_id)
                
                # Add ancestors of target nodes
                ancestors = model.get_ancestors(node_id)
                relevant_vars.update(ancestors)
        
        # If no target nodes specified, use all nodes affected by counterfactual
        if not target_nodes:
            for cf_node_id in counterfactual_values.keys():
                if cf_node_id in model.nodes:
                    descendants = model.get_descendants(cf_node_id)
                    relevant_vars.update(descendants)
        
        return list(relevant_vars)
    
    async def _set_counterfactual_values(self, model: CausalModel,
                                factual_values: Dict[str, Any],
                                counterfactual_values: Dict[str, Any]) -> Dict[str, Any]:
        """Set up the counterfactual world with modified values"""
        # Start with factual values
        world = factual_values.copy()
        
        # Get current states for nodes not in factual values
        for node_id, node in model.nodes.items():
            if node_id not in world:
                world[node_id] = node.get_current_state()
        
        # Apply counterfactual changes
        for node_id, value in counterfactual_values.items():
            if node_id in model.nodes:
                world[node_id] = value
        
        return world
    
    async def _propagate_effects(self, model: CausalModel,
                         counterfactual_world: Dict[str, Any],
                         relevant_vars: List[str]) -> Dict[str, Any]:
        """Propagate effects through the causal graph"""
        # Create a copy of the world to modify
        results = counterfactual_world.copy()
        
        # Get topological ordering of nodes
        try:
            node_order = list(nx.topological_sort(model.graph))
        except nx.NetworkXUnfeasible:
            # Graph has cycles, use a default order
            node_order = list(model.nodes.keys())
        
        # Filter to relevant variables
        node_order = [n for n in node_order if n in relevant_vars]
        
        # Process nodes in topological order
        for node_id in node_order:
            # Skip nodes with counterfactual values (they're fixed)
            if node_id in counterfactual_world:
                continue
                
            # Get parents
            parents = list(model.graph.predecessors(node_id))
            
            if not parents:
                continue
                
            # Get parent values
            parent_values = {p: results.get(p) for p in parents}
            
            # Predict node value based on parents
            node = model.nodes.get(node_id)
            if node:
                prediction = await self._predict_from_parents(model, node_id, parent_values)
                results[node_id] = prediction
        
        return results
    
    async def _predict_from_parents(self, model: CausalModel,
                            node_id: str,
                            parent_values: Dict[str, Any]) -> Any:
        """Predict a node's value based on parent values"""
        node = model.nodes.get(node_id)
        if not node:
            return None
            
        # Get relevant relations
        incoming_relations = []
        for relation_id, relation in model.relations.items():
            if relation.target_id == node_id and relation.source_id in parent_values:
                incoming_relations.append(relation)
        
        if not incoming_relations:
            return node.get_current_state()
            
        # For discrete states
        if node.states:
            state_scores = {state: 0.0 for state in node.states}
            
            for relation in incoming_relations:
                source_id = relation.source_id
                source_value = parent_values.get(source_id)
                
                if source_value is not None:
                    for state in node.states:
                        # Get conditional probability
                        prob = relation.get_conditional_probability(source_value, state)
                        
                        # Weight by relation strength
                        weighted_prob = prob * relation.strength
                        
                        # Add to score
                        state_scores[state] += weighted_prob
            
            # Return state with highest score
            if state_scores:
                return max(state_scores.items(), key=lambda x: x[1])[0]
            
        # For continuous values
        current_value = node.get_current_state()
        if isinstance(current_value, (int, float)):
            # Calculate weighted influence from parents
            weighted_sum = 0.0
            total_weight = 0.0
            
            for relation in incoming_relations:
                source_id = relation.source_id
                source_value = parent_values.get(source_id)
                
                if isinstance(source_value, (int, float)):
                    source_node = model.nodes.get(source_id)
                    if source_node:
                        # Calculate effect direction (simple approximation)
                        source_current = source_node.get_current_state()
                        if source_current is not None and isinstance(source_current, (int, float)):
                            source_diff = source_value - source_current
                            
                            # Apply causal effect
                            effect = source_diff * relation.strength
                            weighted_sum += effect
                            total_weight += relation.strength
            
            # Apply weighted effect to current value
            if total_weight > 0:
                new_value = current_value + (weighted_sum / total_weight)
                return new_value
        
        return current_value
    
    async def _analyze_counterfactual_results(self, model: CausalModel,
                                     factual_values: Dict[str, Any],
                                     counterfactual_values: Dict[str, Any],
                                     results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the results of counterfactual reasoning"""
        analysis = {
            "summary": "",
            "changes": {},
            "significant_effects": [],
            "pathways": [],
            "confidence": 0.0
        }
        
        # Identify changes from factual to counterfactual
        changes = {}
        for node_id, value in results.items():
            if node_id in factual_values and factual_values[node_id] != value:
                node = model.nodes.get(node_id)
                if node:
                    changes[node_id] = {
                        "name": node.name,
                        "factual": factual_values[node_id],
                        "counterfactual": value,
                        "significant": self._is_significant_change(node, factual_values[node_id], value)
                    }
        
        analysis["changes"] = changes
        
        # Identify significant effects
        significant_effects = []
        for node_id, change in changes.items():
            if change["significant"]:
                node = model.nodes.get(node_id)
                if node:
                    significant_effects.append({
                        "node_id": node_id,
                        "name": node.name,
                        "factual": change["factual"],
                        "counterfactual": change["counterfactual"],
                        "type": node.type
                    })
        
        analysis["significant_effects"] = significant_effects
        
        # Identify causal pathways
        for cf_node_id in counterfactual_values.keys():
            for effect_node_id in significant_effects:
                effect_id = effect_node_id["node_id"]
                
                # Find paths from counterfactual node to effect
                paths = list(nx.all_simple_paths(model.graph, cf_node_id, effect_id, 
                                             cutoff=self.config["max_model_depth"]))
                
                for path in paths:
                    # Calculate path strength
                    path_strength = 1.0
                    path_nodes = []
                    
                    for i in range(len(path) - 1):
                        source = path[i]
                        target = path[i + 1]
                        
                        # Get relation between these nodes
                        relations = model.get_relations_between(source, target)
                        if relations:
                            # Use max strength relation
                            max_rel = max(relations, key=lambda r: r.strength)
                            path_strength *= max_rel.strength
                            
                            # Add to path nodes
                            source_node = model.nodes.get(source)
                            target_node = model.nodes.get(target)
                            
                            if source_node and target_node:
                                path_nodes.append({
                                    "from": {
                                        "id": source,
                                        "name": source_node.name,
                                        "value": results.get(source)
                                    },
                                    "to": {
                                        "id": target,
                                        "name": target_node.name,
                                        "value": results.get(target)
                                    },
                                    "relation": {
                                        "type": max_rel.type,
                                        "strength": max_rel.strength,
                                        "mechanism": max_rel.mechanism
                                    }
                                })
                    
                    # Add pathway if significant
                    if path_strength >= 0.2:  # Only include significant paths
                        analysis["pathways"].append({
                            "from": cf_node_id,
                            "to": effect_id,
                            "strength": path_strength,
                            "nodes": path_nodes
                        })
        
        # Calculate overall confidence
        if significant_effects:
            # Average of path strengths for significant effects
            path_strengths = [p["strength"] for p in analysis["pathways"]]
            if path_strengths:
                avg_strength = sum(path_strengths) / len(path_strengths)
                analysis["confidence"] = avg_strength
            else:
                analysis["confidence"] = 0.3  # Low confidence if no clear paths
        else:
            analysis["confidence"] = 0.5  # Moderate confidence in no significant effects
        
        # Generate summary
        changed_count = len(changes)
        significant_count = len(significant_effects)
        
        if changed_count == 0:
            analysis["summary"] = "The counterfactual change would have no effect."
        elif significant_count == 0:
            analysis["summary"] = f"The counterfactual change would affect {changed_count} variables, but none significantly."
        else:
            analysis["summary"] = f"The counterfactual change would significantly affect {significant_count} out of {changed_count} changed variables."
        
        return analysis
    
    def _is_significant_change(self, node: CausalNode, factual_value: Any, 
                            counterfactual_value: Any) -> bool:
        """Determine if a change is significant"""
        # For categorical values, any change is significant
        if node.states and factual_value != counterfactual_value:
            return True
            
        # For numeric values, check if change is substantial
        if isinstance(factual_value, (int, float)) and isinstance(counterfactual_value, (int, float)):
            current_value = factual_value
            if current_value == 0:
                # Avoid division by zero
                return abs(counterfactual_value) > 0.1
            else:
                relative_change = abs((counterfactual_value - current_value) / current_value)
                return relative_change > 0.1  # More than 10% change
        
        # Default case
        return factual_value != counterfactual_value
    
    async def get_causal_model(self, model_id: str) -> Dict[str, Any]:
        """Get a causal model by ID"""
        if model_id not in self.causal_models:
            raise ValueError(f"Model {model_id} not found")
            
        return self.causal_models[model_id].to_dict()
    
    async def get_all_causal_models(self) -> List[Dict[str, Any]]:
        """Get all causal models"""
        return [model.to_dict() for model in self.causal_models.values()]
    
    async def get_intervention(self, intervention_id: str) -> Dict[str, Any]:
        """Get an intervention by ID"""
        if intervention_id not in self.interventions:
            raise ValueError(f"Intervention {intervention_id} not found")
            
        return self.interventions[intervention_id].to_dict()
    
    async def get_counterfactual(self, counterfactual_id: str) -> Dict[str, Any]:
        """Get a counterfactual analysis by ID"""
        if counterfactual_id not in self.counterfactuals:
            raise ValueError(f"Counterfactual {counterfactual_id} not found")
            
        return self.counterfactuals[counterfactual_id]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the causal reasoning system"""
        return {
            "timestamp": datetime.now().isoformat(),
            "models": self.stats["models_created"],
            "nodes": self.stats["nodes_created"],
            "relations": self.stats["relations_created"],
            "interventions": self.stats["interventions_performed"],
            "counterfactuals": self.stats["counterfactuals_analyzed"],
            "discovery_runs": self.stats["discovery_runs"],
            "configuration": self.config
        }
    
    async def save_state(self, file_path: str) -> bool:
        """Save current state to file"""
        try:
            state = {
                "causal_models": {model_id: model.to_dict() for model_id, model in self.causal_models.items()},
                "interventions": {int_id: intervention.to_dict() for int_id, intervention in self.interventions.items()},
                "counterfactuals": self.counterfactuals,
                "observations": self.observations,
                "config": self.config,
                "stats": self.stats,
                "next_model_id": self.next_model_id,
                "next_intervention_id": self.next_intervention_id,
                "next_counterfactual_id": self.next_counterfactual_id,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving causal reasoning state: {e}")
            return False
        
    async def load_state(self, file_path: str) -> bool:
        """Load state from file"""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Load models
            self.causal_models = {}
            for model_id, model_data in state["causal_models"].items():
                self.causal_models[model_id] = CausalModel.from_dict(model_data)
            
            # Load interventions
            self.interventions = {}
            for int_id, int_data in state["interventions"].items():
                self.interventions[int_id] = Intervention.from_dict(int_data)
            
            # Load other attributes
            self.counterfactuals = state["counterfactuals"]
            self.observations = state["observations"]
            self.config = state["config"]
            self.stats = state["stats"]
            self.next_model_id = state["next_model_id"]
            self.next_intervention_id = state["next_intervention_id"]
            self.next_counterfactual_id = state["next_counterfactual_id"]
            
            return True
        except Exception as e:
            logger.error(f"Error loading causal reasoning state: {e}")
            return False
