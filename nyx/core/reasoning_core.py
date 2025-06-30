# nyx/core/reasoning_core.py

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
import re
from collections import defaultdict
import threading
from nyx.core.multimodal_integrator import (
    MultimodalIntegrator, Modality, SensoryInput, ExpectationSignal, IntegratedPercept
)

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

class ConceptSpace:
    """Represents a conceptual space with concepts and their properties"""
    
    def __init__(self, space_id: str, name: str, domain: str = ""):
        self.id = space_id
        self.name = name
        self.domain = domain
        self.concepts = {}  # id -> concept
        self.relations = []  # relations between concepts
        self.properties = {}  # property -> [value ranges]
        self.organizing_principles = []  # principles organizing the space
        self.created_at = datetime.now()
        self.last_modified = datetime.now()
        self.metadata = {}
    
    def add_concept(self, concept_id: str, name: str, 
                  properties: Dict[str, Any] = None) -> None:
        """Add a concept to the space"""
        self.concepts[concept_id] = {
            "id": concept_id,
            "name": name,
            "properties": properties or {},
            "added_at": datetime.now().isoformat()
        }
        
        # Update property ranges
        for prop, value in (properties or {}).items():
            if prop not in self.properties:
                self.properties[prop] = []
                
            if isinstance(value, (int, float)):
                # For numeric properties, track [min, max]
                ranges = self.properties[prop]
                if not ranges:
                    ranges.append([value, value])
                else:
                    ranges[0][0] = min(ranges[0][0], value)
                    ranges[0][1] = max(ranges[0][1], value)
            elif value not in self.properties[prop]:
                # For categorical properties, track unique values
                self.properties[prop].append(value)
                
        self.last_modified = datetime.now()
    
    def add_relation(self, source_id: str, target_id: str, 
                   relation_type: str, strength: float = 1.0) -> None:
        """Add a relation between concepts"""
        # Ensure concepts exist
        if source_id not in self.concepts or target_id not in self.concepts:
            return
            
        relation = {
            "source": source_id,
            "target": target_id,
            "type": relation_type,
            "strength": strength,
            "added_at": datetime.now().isoformat()
        }
        
        self.relations.append(relation)
        self.last_modified = datetime.now()
    
    def add_organizing_principle(self, name: str, description: str, 
                              properties: List[str] = None) -> None:
        """Add an organizing principle to the conceptual space"""
        principle = {
            "name": name,
            "description": description,
            "properties": properties or [],
            "added_at": datetime.now().isoformat()
        }
        
        self.organizing_principles.append(principle)
        self.last_modified = datetime.now()
    
    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get a concept by ID"""
        return self.concepts.get(concept_id)
    
    def get_related_concepts(self, concept_id: str, 
                          relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get concepts related to a concept"""
        related = []
        
        # Check each relation
        for relation in self.relations:
            matches_type = relation_type is None or relation["type"] == relation_type
            
            if relation["source"] == concept_id and matches_type:
                target_id = relation["target"]
                if target_id in self.concepts:
                    related.append({
                        "concept": self.concepts[target_id],
                        "relation": relation
                    })
            elif relation["target"] == concept_id and matches_type:
                source_id = relation["source"]
                if source_id in self.concepts:
                    related.append({
                        "concept": self.concepts[source_id],
                        "relation": relation
                    })
        
        return related
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conceptual space to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain,
            "concepts": self.concepts,
            "relations": self.relations,
            "properties": self.properties,
            "organizing_principles": self.organizing_principles,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConceptSpace':
        """Create from dictionary representation"""
        space = cls(
            space_id=data["id"],
            name=data["name"],
            domain=data["domain"]
        )
        
        space.concepts = data["concepts"]
        space.relations = data["relations"]
        space.properties = data["properties"]
        space.organizing_principles = data["organizing_principles"]
        space.created_at = datetime.fromisoformat(data["created_at"])
        space.last_modified = datetime.fromisoformat(data["last_modified"])
        space.metadata = data["metadata"]
        
        return space

class ConceptualBlend:
    """Represents a blend of multiple conceptual spaces"""
    
    def __init__(self, blend_id: str, name: str, 
               input_spaces: List[str], blend_type: str = "composition"):
        self.id = blend_id
        self.name = name
        self.input_spaces = input_spaces  # IDs of input spaces
        self.type = blend_type  # composition, fusion, extraction, etc.
        self.created_at = datetime.now()
        self.last_modified = datetime.now()
        self.concepts = {}  # Concepts in the blend
        self.relations = []  # Relations between concepts in the blend
        self.mappings = []  # Mappings between input and blend concepts
        self.emergent_structure = []  # Structure that emerges from blending
        self.organizing_principles = []  # Principles organizing the blend
        self.evaluation = {}  # Evaluation metrics for the blend
        self.elaborations = []  # Elaborations of the blend
        self.metadata = {}
    
    def add_concept(self, concept_id: str, name: str,
                  properties: Dict[str, Any] = None,
                  source_concepts: List[Dict[str, str]] = None) -> None:
        """Add a concept to the blend"""
        self.concepts[concept_id] = {
            "id": concept_id,
            "name": name,
            "properties": properties or {},
            "source_concepts": source_concepts or [],
            "added_at": datetime.now().isoformat()
        }
        
        self.last_modified = datetime.now()
    
    def add_relation(self, source_id: str, target_id: str, 
                   relation_type: str, strength: float = 1.0,
                   source_relation: Optional[Dict[str, Any]] = None) -> None:
        """Add a relation between concepts in the blend"""
        # Ensure concepts exist
        if source_id not in self.concepts or target_id not in self.concepts:
            return
            
        relation = {
            "source": source_id,
            "target": target_id,
            "type": relation_type,
            "strength": strength,
            "source_relation": source_relation,
            "added_at": datetime.now().isoformat()
        }
        
        self.relations.append(relation)
        self.last_modified = datetime.now()
    
    def add_mapping(self, input_space_id: str, input_concept_id: str,
                 blend_concept_id: str, mapping_type: str,
                 mapping_strength: float = 1.0) -> None:
        """Add a mapping between input and blend concepts"""
        # Ensure blend concept exists
        if blend_concept_id not in self.concepts:
            return
            
        mapping = {
            "input_space": input_space_id,
            "input_concept": input_concept_id,
            "blend_concept": blend_concept_id,
            "type": mapping_type,
            "strength": mapping_strength,
            "added_at": datetime.now().isoformat()
        }
        
        self.mappings.append(mapping)
        self.last_modified = datetime.now()
    
    def add_emergent_structure(self, name: str, description: str,
                           concept_ids: List[str] = None,
                           relation_ids: List[Dict[str, str]] = None) -> None:
        """Add emergent structure identified in the blend"""
        structure = {
            "name": name,
            "description": description,
            "concepts": concept_ids or [],
            "relations": relation_ids or [],
            "added_at": datetime.now().isoformat()
        }
        
        self.emergent_structure.append(structure)
        self.last_modified = datetime.now()
    
    def add_elaboration(self, elaboration_type: str, description: str,
                    affected_concepts: List[str] = None,
                    affected_relations: List[Dict[str, str]] = None) -> None:
        """Add an elaboration of the blend"""
        elaboration = {
            "type": elaboration_type,
            "description": description,
            "affected_concepts": affected_concepts or [],
            "affected_relations": affected_relations or [],
            "added_at": datetime.now().isoformat()
        }
        
        self.elaborations.append(elaboration)
        self.last_modified = datetime.now()
    
    def update_evaluation(self, metrics: Dict[str, float]) -> None:
        """Update evaluation metrics for the blend"""
        self.evaluation = {
            **self.evaluation,
            **metrics,
            "updated_at": datetime.now().isoformat()
        }
        
        self.last_modified = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert blend to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "input_spaces": self.input_spaces,
            "type": self.type,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "concepts": self.concepts,
            "relations": self.relations,
            "mappings": self.mappings,
            "emergent_structure": self.emergent_structure,
            "organizing_principles": self.organizing_principles,
            "evaluation": self.evaluation,
            "elaborations": self.elaborations,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConceptualBlend':
        """Create from dictionary representation"""
        blend = cls(
            blend_id=data["id"],
            name=data["name"],
            input_spaces=data["input_spaces"],
            blend_type=data["type"]
        )
        
        blend.concepts = data["concepts"]
        blend.relations = data["relations"]
        blend.mappings = data["mappings"]
        blend.emergent_structure = data["emergent_structure"]
        blend.organizing_principles = data["organizing_principles"]
        blend.evaluation = data["evaluation"]
        blend.elaborations = data["elaborations"]
        blend.created_at = datetime.fromisoformat(data["created_at"])
        blend.last_modified = datetime.fromisoformat(data["last_modified"])
        blend.metadata = data["metadata"]
        
        return blend

class ReasoningCore:
    """
    Core system for causal reasoning, inference, and explanation.
    Extended with conceptual blending capabilities for creative reasoning.
    """
    from nyx.core.multimodal_integrator import MultimodalIntegratorContext, MultimodalIntegrator
    
    def __init__(self, knowledge_core=None):
        """
        Initialize the reasoning core, optionally with a reference to the knowledge core.
        
        Args:
            knowledge_core: Reference to the KnowledgeCore instance for integrating causal knowledge
        """
        self.knowledge_core = knowledge_core
        
        # Causal reasoning components
        self.causal_models = {}  # domain -> CausalModel
        self.interventions = {}  # id -> Intervention
        self.counterfactuals = {}  # id -> Counterfactual analysis
        self.observations = []  # List of observed data points
        
        # Conceptual blending components
        self.concept_spaces = {}  # id -> ConceptSpace
        self.blends = {}  # id -> ConceptualBlend
        
        # Cross-system mappings
        self.concept_to_node_mappings = {}  # concept_id → node_id
        self.node_to_concept_mappings = {}  # node_id → concept_id
        self.blend_to_model_mappings = {}  # blend_id → model_id
        self.model_to_blend_mappings = {}  # model_id → blend_id
        
        # Causal reasoning configuration
        self.causal_config = {
            "min_relation_strength": 0.3,  # Minimum strength to consider a causal relation
            "max_model_depth": 5,  # Maximum depth for model inference
            "default_confidence": 0.7,  # Default confidence in absence of evidence
            "enable_auto_discovery": True,  # Automatically discover causal relations
            "discovery_threshold": 0.6,  # Threshold for automatic discovery
            "min_data_points": 10  # Minimum data points for causal discovery
        }
        
        # Conceptual blending configuration
        self.blending_config = {
            "default_mapping_threshold": 0.5,  # Minimum similarity for default mapping
            "emergent_structure_threshold": 0.7,  # Threshold for identifying emergent structure
            "elaboration_iterations": 3,  # Number of elaboration iterations
            "evaluation_weights": {
                "novelty": 0.25,
                "coherence": 0.25,
                "practicality": 0.2,
                "expressiveness": 0.15,
                "surprise": 0.15
            }
        }
        
        # Integrated reasoning configuration
        self.integrated_config = {
            "enable_auto_blending": True,  # Automatically create blends from causal patterns
            "enable_causal_elaboration": True,  # Use causal reasoning to elaborate blends
            "cross_system_mapping_threshold": 0.6,  # Threshold for mapping between systems
            "max_blend_to_causal_depth": 3,  # Maximum depth for recursive blend→causal conversion
            "max_causal_to_blend_depth": 3,  # Maximum depth for recursive causal→blend conversion
        }
        
        # Statistics
        self.stats = {
            # Causal reasoning stats
            "models_created": 0,
            "nodes_created": 0,
            "relations_created": 0,
            "interventions_performed": 0,
            "counterfactuals_analyzed": 0,
            "discovery_runs": 0,
            
            # Conceptual blending stats
            "spaces_created": 0,
            "concepts_created": 0, 
            "concept_relations_created": 0,
            "blends_created": 0,
            "blend_evaluations": 0,
            
            # Integrated reasoning stats
            "causal_to_conceptual_conversions": 0,
            "conceptual_to_causal_conversions": 0,
            "integrated_analyses": 0,
            "creative_interventions": 0
        }
        
        # Initialize next ID counters
        self.next_model_id = 1
        self.next_intervention_id = 1
        self.next_counterfactual_id = 1
        self.next_space_id = 1
        self.next_blend_id = 1
    
        self._stats_lock = threading.Lock()
        self._monitoring_task = None
        
        # Auto-load general model if available
        self._load_general_model()
    
    # ==================================================================================
    # Causal Reasoning Methods
    # ==================================================================================

    def _load_general_model(self):
        """Load the general causal model on initialization"""
        try:
            from nyx.models.general_causal_model import GENERAL_CAUSAL_MODEL
            self.add_causal_model("general", GENERAL_CAUSAL_MODEL)
            logger.info("General causal model loaded successfully")
        except ImportError:
            logger.info("General causal model not found; skipping auto-load")

    def sync_general_model(self):
        """Sync the general model nodes with actual system values"""
        # Find the general model
        general_model = None
        for model in self.causal_models.values():
            if model.domain == "general":
                general_model = model
                break
        
        if not general_model:
            return
        
        # Update control nodes from config
        config_mappings = {
            "min_relation_strength": self.causal_config["min_relation_strength"],
            "max_model_depth": self.causal_config["max_model_depth"] / 10.0,
            "default_confidence": self.causal_config["default_confidence"],
            "enable_auto_discovery": 1.0 if self.causal_config["enable_auto_discovery"] else 0.0,
            "discovery_threshold": self.causal_config["discovery_threshold"],
            "min_data_points": self.causal_config["min_data_points"] / 25.0,  # Normalize
            "default_mapping_threshold": self.blending_config["default_mapping_threshold"],
            "emergent_structure_thresh": self.blending_config["emergent_structure_threshold"],
            "elaboration_iterations": self.blending_config["elaboration_iterations"] / 5.0,
            "enable_auto_blending": 1.0 if self.integrated_config["enable_auto_blending"] else 0.0,
            "cross_system_map_thresh": self.integrated_config["cross_system_mapping_threshold"]
        }
        
        # Update variable nodes from stats (with thread safety)
        with self._stats_lock:
            stats_mappings = {
                "models_created": min(1.0, self.stats["models_created"] / 100.0) if self.stats["models_created"] > 0 else 0.0,
                "nodes_created": min(1.0, self.stats["nodes_created"] / 1000.0) if self.stats["nodes_created"] > 0 else 0.0,
                "relations_created": min(1.0, self.stats["relations_created"] / 1000.0) if self.stats["relations_created"] > 0 else 0.0,
                "interventions_performed": min(1.0, self.stats["interventions_performed"] / 100.0) if self.stats["interventions_performed"] > 0 else 0.0,
                "counterfactuals_analyzed": min(1.0, self.stats["counterfactuals_analyzed"] / 100.0) if self.stats["counterfactuals_analyzed"] > 0 else 0.0,
                "blends_created": min(1.0, self.stats["blends_created"] / 100.0) if self.stats["blends_created"] > 0 else 0.0,
                "integrated_analyses": min(1.0, self.stats["integrated_analyses"] / 100.0) if self.stats["integrated_analyses"] > 0 else 0.0,
                "creative_interventions": min(1.0, self.stats["creative_interventions"] / 100.0) if self.stats["creative_interventions"] > 0 else 0.0
            }
        
        # Batch update observations
        for node_id, node in general_model.nodes.items():
            original_id = node.metadata.get("original_id")
            
            if original_id in config_mappings:
                value = config_mappings[original_id]
                node.add_observation(value, confidence=1.0)
            elif original_id in stats_mappings:
                value = stats_mappings[original_id]
                node.add_observation(value, confidence=0.9)

    def add_causal_model(self, domain: str, model_dict: Dict[str, Any]) -> str:
        """Register a pre-defined causal model from a dictionary"""
        # Create the model
        model_id = model_dict.get("id", f"model_{self.next_model_id}")
        self.next_model_id += 1
        
        model = CausalModel(
            model_id=model_id,
            name=model_dict["name"],
            domain=model_dict.get("domain", domain),
            metadata=model_dict.get("metadata", {})
        )
        
        # Add nodes
        node_mappings = {}  # original_id -> actual_id
        for node_id, node_data in model_dict.get("nodes", {}).items():
            actual_node_id = model.add_node(
                name=node_id,
                domain=domain,
                node_type=node_data["type"],
                metadata={"original_id": node_id}
            )
            
            node = model.nodes[actual_node_id]
            
            # Add states based on node type
            if node_data["type"] == "control":
                # Control nodes get discrete states
                node.add_state(0.0)
                node.add_state(0.5)
                node.add_state(1.0)
            else:
                # Other nodes just use current value
                node.add_state(node_data["current_value"])
            
            # Set current state
            node.set_default_state(node_data["current_value"])
            
            # Add initial observation
            node.add_observation(
                value=node_data["current_value"],
                confidence=1.0
            )
            
            node_mappings[node_id] = actual_node_id
        
        # Add relations
        for relation_data in model_dict.get("relations", []):
            source_id = node_mappings.get(relation_data["source"])
            target_id = node_mappings.get(relation_data["target"])
            
            if source_id and target_id:
                model.add_relation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type="causal",
                    strength=relation_data["strength"],
                    mechanism=relation_data.get("comment", "")
                )
        
        # Store the model
        self.causal_models[model_id] = model
        
        # Update statistics HERE, not later
        self.stats["models_created"] += 1
        
        return model_id
    
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

        try:
            loop = asyncio.get_running_loop()
            loop.call_soon(self.sync_general_model)
        except RuntimeError:
            # No event loop, call directly
            self.sync_general_model()                
                
        return model_id

    async def analyze_system_performance(self) -> Dict[str, Any]:
        """Use the general model to analyze system performance"""
        # Find general model
        general_model = None
        for model in self.causal_models.values():
            if model.domain == "general":
                general_model = model
                break
        
        if not general_model:
            return {"error": "General model not found"}
        
        # Resolve actual node IDs at runtime
        node_id_map = {}
        for node_id, node in general_model.nodes.items():
            original_id = node.metadata.get("original_id")
            if original_id:
                node_id_map[original_id] = node_id
        
        suggestions = []
        
        # Test increasing discovery threshold
        discovery_node_id = node_id_map.get("discovery_threshold")
        
        if discovery_node_id:
            # Define intervention returns ID, not results
            intervention_id = await self.define_intervention(
                model_id=general_model.id,
                target_node=discovery_node_id,
                target_value=0.8,
                name="Increase discovery threshold",
                description="Test impact of stricter discovery"
            )
            
            # Get intervention object
            intervention = self.interventions[intervention_id]
            
            for effect_node_id, effect in intervention.effects.items():
                node = general_model.nodes.get(effect_node_id)
                if node and effect["probability"] > 0.6:
                    original_name = node.metadata.get("original_id", node.name)
                    suggestions.append({
                        "control": "discovery_threshold",
                        "change": "increase to 0.8",
                        "expected_effect": f"{original_name}: {effect['expected_value']}",
                        "confidence": effect["probability"]
                    })
        
        return {
            "current_performance": await self.get_stats(),
            "optimization_suggestions": suggestions
        }

    async def start_system_monitoring(self, interval: float = 60.0):
        """Start monitoring system performance with the general model"""
        # Cancel existing task if any
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        async def _monitor():
            while True:
                try:
                    # Sync is synchronous, so just call it
                    self.sync_general_model()
                    
                    # Analysis is async
                    analysis = await self.analyze_system_performance()
                    
                    if analysis.get("optimization_suggestions"):
                        logger.info(f"System optimization suggestions: {analysis['optimization_suggestions']}")
                    
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in system monitoring: {e}")
                    await asyncio.sleep(interval)
        
        self._monitoring_task = asyncio.create_task(_monitor())
    
    async def shutdown(self):
        """Clean shutdown of background tasks"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass


    
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
        
        # If knowledge core is available, add as a knowledge node as well
        if self.knowledge_core:
            # Create knowledge node
            knowledge_id = await self.knowledge_core.add_knowledge(
                type=node_type,
                content={
                    "name": name,
                    "domain": domain,
                    "causal_node_id": node_id,
                    "causal_model_id": model_id
                },
                source="causal_reasoning",
                confidence=self.causal_config["default_confidence"]
            )
            
            # Add link from causal node to knowledge node in metadata
            model = self.causal_models[model_id]
            if node_id in model.nodes:
                model.nodes[node_id].metadata["knowledge_node_id"] = knowledge_id
        
        return node_id

    # Add to nyx/core/reasoning_core.py
    
    async def generate_perceptual_expectations(self, modality: str = None) -> List[ExpectationSignal]:
        """
        Generate top-down expectations for perceptual processing based on
        current causal models and conceptual understanding.
        
        Args:
            modality: Optional specific modality to generate expectations for
            
        Returns:
            List of expectation signals to influence perception
        """
        expectations = []
        
        # 1. Generate expectations from active causal models
        for model_id, model in self.causal_models.items():
            # Find highly predictive nodes in the model
            try:
                # Get nodes with high causal influence
                central_nodes = []
                
                try:
                    # Use network centrality if available
                    centrality = nx.betweenness_centrality(model.graph)
                    # Get top nodes by centrality
                    central_nodes = sorted(centrality.keys(), key=lambda n: centrality[n], reverse=True)[:5]
                except:
                    # Fallback to random selection of nodes
                    central_nodes = list(model.nodes.keys())[:min(5, len(model.nodes))]
                
                # For each central node, generate expectations
                for node_id in central_nodes:
                    node = model.nodes.get(node_id)
                    if not node:
                        continue
                        
                    # Get current state and high-probability descendants
                    current_state = node.get_current_state()
                    descendants = model.get_descendants(node_id)
                    
                    for desc_id in descendants[:3]:  # Limit to 3 descendants
                        desc_node = model.nodes.get(desc_id)
                        if not desc_node:
                            continue
                            
                        # Generate expectation
                        target_modality = desc_node.metadata.get("modality", "unknown")
                        
                        # Skip if modality filter applied and doesn't match
                        if modality and target_modality != modality:
                            continue
                        
                        # Calculate expectation strength based on causal relation strength
                        relations = model.get_relations_between(node_id, desc_id)
                        if not relations:
                            continue
                            
                        # Use strongest relation
                        strongest_relation = max(relations, key=lambda r: r.strength)
                        
                        # Create expectation signal
                        expectation = ExpectationSignal(
                            target_modality=target_modality,
                            pattern=desc_node.get_current_state(),
                            strength=strongest_relation.strength,
                            source=f"causal_model:{model_id}:{node_id}",
                            priority=0.7  # High priority for causal expectations
                        )
                        
                        expectations.append(expectation)
            except Exception as e:
                logger.error(f"Error generating causal expectations: {e}")
        
        # 2. Generate expectations from conceptual blends
        for blend_id, blend in self.blends.items():
            try:
                # Focus on emergent structures as they often represent novel patterns
                for structure in blend.emergent_structure:
                    # Get concepts involved in the structure
                    concept_ids = structure.get("concepts", [])
                    
                    for concept_id in concept_ids:
                        if concept_id not in blend.concepts:
                            continue
                            
                        concept = blend.concepts[concept_id]
                        
                        # Determine target modality from concept properties
                        target_modality = "unknown"
                        for prop_name, prop_value in concept.get("properties", {}).items():
                            if "modality" in prop_name.lower():
                                target_modality = str(prop_value)
                                break
                        
                        # Skip if modality filter applied and doesn't match
                        if modality and target_modality != modality:
                            continue
                        
                        # Get most salient property for expectation
                        salient_prop = None
                        for prop_name, prop_value in concept.get("properties", {}).items():
                            if isinstance(prop_value, (str, int, float, bool)):
                                salient_prop = prop_value
                                break
                        
                        if salient_prop is None:
                            continue
                        
                        # Create expectation signal
                        expectation = ExpectationSignal(
                            target_modality=target_modality,
                            pattern=salient_prop,
                            strength=0.6,  # Moderate strength for conceptual expectations
                            source=f"conceptual_blend:{blend_id}:{concept_id}",
                            priority=0.5  # Medium priority
                        )
                        
                        expectations.append(expectation)
            except Exception as e:
                logger.error(f"Error generating conceptual expectations: {e}")
        
        # 3. Limit number of expectations to avoid overwhelming the system
        if len(expectations) > 10:
            # Prioritize by strength and priority
            expectations.sort(key=lambda x: x.strength * x.priority, reverse=True)
            expectations = expectations[:10]
        
        return expectations
    
    async def update_with_perception(self, percept: IntegratedPercept):
        """
        Update causal models and conceptual understanding
        based on new perceptual information.
        
        Args:
            percept: Integrated percept from multimodal integrator
        """
        try:
            # 1. Update relevant causal models
            updated_models = []
            
            for model_id, model in self.causal_models.items():
                # Look for nodes that match this modality
                matching_nodes = []
                
                for node_id, node in model.nodes.items():
                    # Check if node modality matches percept modality
                    if node.metadata.get("modality") == percept.modality:
                        matching_nodes.append(node_id)
                
                if matching_nodes:
                    # Add observation to matching nodes
                    for node_id in matching_nodes:
                        # Add observation
                        model.nodes[node_id].add_observation(
                            value=percept.content,
                            confidence=percept.bottom_up_confidence * percept.attention_weight
                        )
                    
                    updated_models.append(model_id)
                    
                    # If automatic discovery is enabled, check for new relations
                    if self.causal_config["enable_auto_discovery"] and \
                       len(matching_nodes) > 1 and percept.attention_weight > 0.7:
                        # Schedule discovery task
                        asyncio.create_task(self.discover_causal_relations(model_id))
            
            # 2. Update conceptual spaces
            for space_id, space in self.concept_spaces.items():
                # Find concepts that might relate to this percept
                updated = False
                
                for concept_id, concept in space.concepts.items():
                    # Check if concept relates to this modality
                    modality_match = False
                    
                    for prop_name, prop_value in concept.get("properties", {}).items():
                        if "modality" in prop_name.lower() and str(prop_value) == percept.modality:
                            modality_match = True
                            break
                    
                    if modality_match:
                        # Update concept with new property based on percept
                        property_name = f"observed_{percept.modality}_{len(concept.get('properties', {}))}"
                        concept["properties"][property_name] = percept.content
                        updated = True
                        
                        # If percept has high attention weight, update more properties
                        if percept.attention_weight > 0.8:
                            concept["properties"]["attention_weight"] = percept.attention_weight
                            concept["properties"]["bottom_up_confidence"] = percept.bottom_up_confidence
                            concept["properties"]["top_down_influence"] = percept.top_down_influence
                
                if updated:
                    # If blending is enabled, consider creating new blend
                    if self.integrated_config["enable_auto_blending"] and random.random() < 0.2:
                        # Schedule background task to avoid blocking
                        asyncio.create_task(self._create_background_blend(space_id))
        
        except Exception as e:
            logger.error(f"Error updating with perception: {e}")
    
        async def _create_background_blend(self, space_id: str):
            """Create a background conceptual blend based on recent updates"""
            try:
                # Find another space to blend with
                other_spaces = [s_id for s_id in self.concept_spaces.keys() if s_id != space_id]
                
                if not other_spaces:
                    return
                    
                # Pick random other space
                other_space_id = random.choice(other_spaces)
                
                # Create blend
                space1 = self.concept_spaces[space_id]
                space2 = self.concept_spaces[other_space_id]
                
                # Find mappings between spaces
                mappings = []
                
                for concept1_id, concept1 in space1.concepts.items():
                    for concept2_id, concept2 in space2.concepts.items():
                        # Calculate similarity
                        similarity = self._calculate_concept_similarity(
                            concept1, concept2, space1, space2
                        )
                        
                        # If above threshold, add mapping
                        if similarity >= self.blending_config["default_mapping_threshold"]:
                            mappings.append({
                                "concept1": concept1_id,
                                "concept2": concept2_id,
                                "similarity": similarity
                            })
                
                if mappings:
                    # Generate a blend
                    blend_types = ["composition", "fusion", "elaboration", "contrast"]
                    blend_type = random.choice(blend_types)
                    
                    if blend_type == "composition":
                        self._generate_composition_blend(space1, space2, mappings)
                    elif blend_type == "fusion":
                        self._generate_fusion_blend(space1, space2, mappings)
                    elif blend_type == "elaboration":
                        self._generate_elaboration_blend(space1, space2, mappings)
                    elif blend_type == "contrast":
                        self._generate_contrast_blend(space1, space2, mappings)
                    
                    # Update statistics
                    self.stats["blends_created"] += 1
            
            except Exception as e:
                logger.error(f"Error creating background blend: {e}")
    
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
        
        # If knowledge core is available, add as a knowledge relation as well
        if self.knowledge_core:
            model = self.causal_models[model_id]
            
            # Get knowledge node IDs if available
            source_knowledge_id = None
            target_knowledge_id = None
            
            if source_id in model.nodes and "knowledge_node_id" in model.nodes[source_id].metadata:
                source_knowledge_id = model.nodes[source_id].metadata["knowledge_node_id"]
                
            if target_id in model.nodes and "knowledge_node_id" in model.nodes[target_id].metadata:
                target_knowledge_id = model.nodes[target_id].metadata["knowledge_node_id"]
                
            # Add relation to knowledge core if both knowledge nodes exist
            if source_knowledge_id and target_knowledge_id:
                await self.knowledge_core.add_relation(
                    source_id=source_knowledge_id,
                    target_id=target_knowledge_id,
                    type=relation_type,
                    weight=strength,
                    metadata={
                        "causal_relation_id": relation_id,
                        "causal_model_id": model_id,
                        "mechanism": mechanism
                    }
                )
        
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
        if (self.causal_config["enable_auto_discovery"] and 
            len(self.observations) % self.causal_config["min_data_points"] == 0):
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
            if strength < self.causal_config["min_relation_strength"]:
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
            if causal_score >= self.causal_config["discovery_threshold"]:
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
                            description="Automatically discovered by reasoning core",
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
                    mechanism="Automatically discovered by reasoning core"
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
        if len(observations) < self.causal_config["min_data_points"]:
            return {
                "status": "insufficient_data",
                "required": self.causal_config["min_data_points"],
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
                                         cutoff=self.causal_config["max_model_depth"]))
            
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
                                             cutoff=self.causal_config["max_model_depth"]))
                
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
    
    # ==================================================================================
    # Conceptual Blending Methods
    # ==================================================================================
    
    async def create_concept_space(self, name: str, domain: str = "",
                            metadata: Dict[str, Any] = None) -> str:
        """Create a new conceptual space"""
        # Generate space ID
        space_id = f"space_{self.next_space_id}"
        self.next_space_id += 1
        
        # Create space
        space = ConceptSpace(
            space_id=space_id,
            name=name,
            domain=domain
        )
        
        # Add metadata if provided
        if metadata:
            space.metadata = metadata
            
        # Store space
        self.concept_spaces[space_id] = space
        
        # Update statistics
        self.stats["spaces_created"] += 1
        
        return space_id
    
    async def add_concept_to_space(self, space_id: str, name: str,
                            properties: Dict[str, Any] = None) -> str:
        """Add a concept to a conceptual space"""
        if space_id not in self.concept_spaces:
            raise ValueError(f"Concept space {space_id} not found")
            
        space = self.concept_spaces[space_id]
        
        # Generate concept ID
        concept_id = f"{space_id}_concept_{len(space.concepts) + 1}"
        
        # Add concept
        space.add_concept(
            concept_id=concept_id,
            name=name,
            properties=properties
        )
        
        # Update statistics
        self.stats["concepts_created"] += 1
        
        return concept_id
    
    async def add_relation_to_space(self, space_id: str, source_id: str,
                             target_id: str, relation_type: str,
                             strength: float = 1.0) -> None:
        """Add a relation between concepts in a space"""
        if space_id not in self.concept_spaces:
            raise ValueError(f"Concept space {space_id} not found")
            
        space = self.concept_spaces[space_id]
        
        # Add relation
        space.add_relation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            strength=strength
        )
        
        # Update statistics
        self.stats["concept_relations_created"] += 1
    
    async def add_organizing_principle(self, space_id: str, name: str,
                               description: str,
                               properties: List[str] = None) -> None:
        """Add an organizing principle to a conceptual space"""
        if space_id not in self.concept_spaces:
            raise ValueError(f"Concept space {space_id} not found")
            
        space = self.concept_spaces[space_id]
        
        # Add principle
        space.add_organizing_principle(
            name=name,
            description=description,
            properties=properties
        )
    
    def _generate_candidate_blends(self, spaces: List[ConceptSpace], 
                                  mappings: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
            """Generate potential blends from conceptual spaces and mappings"""
            candidate_blends = []
            
            # For each pair of spaces with mappings
            for space_pair, space_mappings in mappings.items():
                if not space_mappings:
                    continue
                    
                # Extract space IDs
                space_ids = space_pair.split('_')
                if len(space_ids) != 2:
                    continue
                    
                space1_id, space2_id = space_ids
                
                # Get spaces
                space1 = self.concept_spaces.get(space1_id)
                space2 = self.concept_spaces.get(space2_id)
                
                if not space1 or not space2:
                    continue
                    
                # Generate different blend types
                
                # 1. Composition blend (preserves most structure)
                composition_blend = self._generate_composition_blend(
                    space1, space2, space_mappings
                )
                if composition_blend:
                    candidate_blends.append(composition_blend)
                    
                # 2. Fusion blend (more integrated)
                fusion_blend = self._generate_fusion_blend(
                    space1, space2, space_mappings
                )
                if fusion_blend:
                    candidate_blends.append(fusion_blend)
                    
                # 3. Completion blend (completes partial structures)
                completion_blend = self._generate_completion_blend(
                    space1, space2, space_mappings
                )
                if completion_blend:
                    candidate_blends.append(completion_blend)
                    
                # 4. Elaboration blend (develops emergent structure)
                elaboration_blend = self._generate_elaboration_blend(
                    space1, space2, space_mappings
                )
                if elaboration_blend:
                    candidate_blends.append(elaboration_blend)
                    
                # 5. Contrast blend (exploits differences)
                contrast_blend = self._generate_contrast_blend(
                    space1, space2, space_mappings
                )
                if contrast_blend:
                    candidate_blends.append(contrast_blend)
            
            return candidate_blends
    
    def _generate_composition_blend(self, space1: ConceptSpace, space2: ConceptSpace,
                                 mappings: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate a composition blend that preserves input space structure"""
        # Generate blend ID
        blend_id = f"blend_{self.next_blend_id}"
        self.next_blend_id += 1
        
        # Create blend name
        blend_name = f"{space1.name} + {space2.name} Composition"
        
        # Create blend
        blend = ConceptualBlend(
            blend_id=blend_id,
            name=blend_name,
            input_spaces=[space1.id, space2.id],
            blend_type="composition"
        )
        
        # Add mapped concepts to the blend
        mapped_concepts = {}
        
        for mapping in mappings:
            concept1_id = mapping["concept1"]
            concept2_id = mapping["concept2"]
            
            concept1 = space1.concepts.get(concept1_id)
            concept2 = space2.concepts.get(concept2_id)
            
            if not concept1 or not concept2:
                continue
                
            # Create blend concept with properties from both input concepts
            blend_concept_id = f"{blend_id}_concept_{len(blend.concepts) + 1}"
            blend_concept_name = self._generate_blend_concept_name(concept1["name"], concept2["name"])
            
            # Merge properties
            merged_properties = self._merge_properties(
                concept1.get("properties", {}),
                concept2.get("properties", {})
            )
            
            # Add concept to blend
            blend.add_concept(
                concept_id=blend_concept_id,
                name=blend_concept_name,
                properties=merged_properties,
                source_concepts=[
                    {"space_id": space1.id, "concept_id": concept1_id},
                    {"space_id": space2.id, "concept_id": concept2_id}
                ]
            )
            
            # Record mapping
            mapped_concepts[concept1_id] = blend_concept_id
            mapped_concepts[concept2_id] = blend_concept_id
            
            # Add mappings
            blend.add_mapping(
                input_space_id=space1.id,
                input_concept_id=concept1_id,
                blend_concept_id=blend_concept_id,
                mapping_type="direct",
                mapping_strength=mapping["similarity"]
            )
            
            blend.add_mapping(
                input_space_id=space2.id,
                input_concept_id=concept2_id,
                blend_concept_id=blend_concept_id,
                mapping_type="direct",
                mapping_strength=mapping["similarity"]
            )
        
        # Add relations based on input space relations
        self._transfer_relations(space1, blend, mapped_concepts)
        self._transfer_relations(space2, blend, mapped_concepts)
        
        # If no concepts were mapped, return None
        if not blend.concepts:
            return None
            
        # Store the blend
        self.blends[blend_id] = blend
        
        return blend.to_dict()
    
    def _generate_fusion_blend(self, space1: ConceptSpace, space2: ConceptSpace,
                           mappings: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate a fusion blend with deeper integration of input spaces"""
        # Generate blend ID
        blend_id = f"blend_{self.next_blend_id}"
        self.next_blend_id += 1
        
        # Create blend name
        blend_name = f"{space1.name} × {space2.name} Fusion"
        
        # Create blend
        blend = ConceptualBlend(
            blend_id=blend_id,
            name=blend_name,
            input_spaces=[space1.id, space2.id],
            blend_type="fusion"
        )
        
        # Add mapped concepts to the blend with deeper integration
        mapped_concepts = {}
        
        for mapping in mappings:
            concept1_id = mapping["concept1"]
            concept2_id = mapping["concept2"]
            
            concept1 = space1.concepts.get(concept1_id)
            concept2 = space2.concepts.get(concept2_id)
            
            if not concept1 or not concept2:
                continue
                
            # Create blend concept with more integrated properties
            blend_concept_id = f"{blend_id}_concept_{len(blend.concepts) + 1}"
            
            # Create more integrated name
            blend_concept_name = self._generate_integrated_name(concept1["name"], concept2["name"])
            
            # Integrate properties more deeply
            integrated_properties = self._integrate_properties(
                concept1.get("properties", {}),
                concept2.get("properties", {})
            )
            
            # Add concept to blend
            blend.add_concept(
                concept_id=blend_concept_id,
                name=blend_concept_name,
                properties=integrated_properties,
                source_concepts=[
                    {"space_id": space1.id, "concept_id": concept1_id},
                    {"space_id": space2.id, "concept_id": concept2_id}
                ]
            )
            
            # Record mapping
            mapped_concepts[concept1_id] = blend_concept_id
            mapped_concepts[concept2_id] = blend_concept_id
            
            # Add mappings
            blend.add_mapping(
                input_space_id=space1.id,
                input_concept_id=concept1_id,
                blend_concept_id=blend_concept_id,
                mapping_type="integration",
                mapping_strength=mapping["similarity"]
            )
            
            blend.add_mapping(
                input_space_id=space2.id,
                input_concept_id=concept2_id,
                blend_concept_id=blend_concept_id,
                mapping_type="integration",
                mapping_strength=mapping["similarity"]
            )
        
        # Transfer and integrate relations
        self._integrate_relations(space1, space2, blend, mapped_concepts)
        
        # If no concepts were mapped, return None
        if not blend.concepts:
            return None
            
        # Look for potential emergent structure
        self._identify_emergent_structure(blend, space1, space2)
        
        # Store the blend
        self.blends[blend_id] = blend
        
        return blend.to_dict()
    
    def _generate_completion_blend(self, space1: ConceptSpace, space2: ConceptSpace,
                             mappings: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate a completion blend that completes partial structures"""
        # Generate blend ID
        blend_id = f"blend_{self.next_blend_id}"
        self.next_blend_id += 1
        
        # Create blend name
        blend_name = f"{space1.name} ⊕ {space2.name} Completion"
        
        # Create blend
        blend = ConceptualBlend(
            blend_id=blend_id,
            name=blend_name,
            input_spaces=[space1.id, space2.id],
            blend_type="completion"
        )
        
        # Add all concepts from space1 to the blend
        mapped_concepts = {}
        
        # First, add all mapped concepts
        for mapping in mappings:
            concept1_id = mapping["concept1"]
            concept2_id = mapping["concept2"]
            
            concept1 = space1.concepts.get(concept1_id)
            concept2 = space2.concepts.get(concept2_id)
            
            if not concept1 or not concept2:
                continue
                
            # Create blend concept
            blend_concept_id = f"{blend_id}_concept_{len(blend.concepts) + 1}"
            blend_concept_name = concept1["name"]  # Use space1 name as base
            
            # Complete properties from both spaces
            completed_properties = self._complete_properties(
                concept1.get("properties", {}),
                concept2.get("properties", {})
            )
            
            # Add concept to blend
            blend.add_concept(
                concept_id=blend_concept_id,
                name=blend_concept_name,
                properties=completed_properties,
                source_concepts=[
                    {"space_id": space1.id, "concept_id": concept1_id},
                    {"space_id": space2.id, "concept_id": concept2_id}
                ]
            )
            
            # Record mapping
            mapped_concepts[concept1_id] = blend_concept_id
            mapped_concepts[concept2_id] = blend_concept_id
            
            # Add mappings
            blend.add_mapping(
                input_space_id=space1.id,
                input_concept_id=concept1_id,
                blend_concept_id=blend_concept_id,
                mapping_type="completion",
                mapping_strength=mapping["similarity"]
            )
            
            blend.add_mapping(
                input_space_id=space2.id,
                input_concept_id=concept2_id,
                blend_concept_id=blend_concept_id,
                mapping_type="completion",
                mapping_strength=mapping["similarity"]
            )
        
        # Add remaining concepts from space1
        for concept1_id, concept1 in space1.concepts.items():
            if concept1_id in mapped_concepts:
                continue
                
            # Create blend concept
            blend_concept_id = f"{blend_id}_concept_{len(blend.concepts) + 1}"
            
            # Add concept to blend
            blend.add_concept(
                concept_id=blend_concept_id,
                name=concept1["name"],
                properties=concept1.get("properties", {}),
                source_concepts=[
                    {"space_id": space1.id, "concept_id": concept1_id}
                ]
            )
            
            # Record mapping
            mapped_concepts[concept1_id] = blend_concept_id
            
            # Add mapping
            blend.add_mapping(
                input_space_id=space1.id,
                input_concept_id=concept1_id,
                blend_concept_id=blend_concept_id,
                mapping_type="direct",
                mapping_strength=1.0
            )
        
        # Transfer relations from space1
        self._transfer_relations(space1, blend, mapped_concepts)
        
        # Complete the structure with relevant relations from space2
        self._complete_relations(space1, space2, blend, mapped_concepts)
        
        # If no concepts were mapped, return None
        if not blend.concepts:
            return None
            
        # Store the blend
        self.blends[blend_id] = blend
        
        return blend.to_dict()
    
    def _generate_elaboration_blend(self, space1: ConceptSpace, space2: ConceptSpace,
                               mappings: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate an elaboration blend that develops emergent structure"""
        # Generate blend ID
        blend_id = f"blend_{self.next_blend_id}"
        self.next_blend_id += 1
        
        # Create blend name
        blend_name = f"{space1.name} ⋆ {space2.name} Elaboration"
        
        # Create blend
        blend = ConceptualBlend(
            blend_id=blend_id,
            name=blend_name,
            input_spaces=[space1.id, space2.id],
            blend_type="elaboration"
        )
        
        # Add selected concepts from both spaces
        mapped_concepts = {}
        
        # Use the top mappings
        top_mappings = mappings[:5] if len(mappings) > 5 else mappings
        
        for mapping in top_mappings:
            concept1_id = mapping["concept1"]
            concept2_id = mapping["concept2"]
            
            concept1 = space1.concepts.get(concept1_id)
            concept2 = space2.concepts.get(concept2_id)
            
            if not concept1 or not concept2:
                continue
                
            # Create blend concept
            blend_concept_id = f"{blend_id}_concept_{len(blend.concepts) + 1}"
            
            # Create elaborate name
            blend_concept_name = self._generate_elaborate_name(concept1["name"], concept2["name"])
            
            # Elaborate properties
            elaborated_properties = self._elaborate_properties(
                concept1.get("properties", {}),
                concept2.get("properties", {})
            )
            
            # Add concept to blend
            blend.add_concept(
                concept_id=blend_concept_id,
                name=blend_concept_name,
                properties=elaborated_properties,
                source_concepts=[
                    {"space_id": space1.id, "concept_id": concept1_id},
                    {"space_id": space2.id, "concept_id": concept2_id}
                ]
            )
            
            # Record mapping
            mapped_concepts[concept1_id] = blend_concept_id
            mapped_concepts[concept2_id] = blend_concept_id
            
            # Add mappings
            blend.add_mapping(
                input_space_id=space1.id,
                input_concept_id=concept1_id,
                blend_concept_id=blend_concept_id,
                mapping_type="elaboration",
                mapping_strength=mapping["similarity"]
            )
            
            blend.add_mapping(
                input_space_id=space2.id,
                input_concept_id=concept2_id,
                blend_concept_id=blend_concept_id,
                mapping_type="elaboration",
                mapping_strength=mapping["similarity"]
            )
        
        # Add some additional concepts from each space to enable more elaboration
        self._add_supporting_concepts(space1, blend, mapped_concepts)
        self._add_supporting_concepts(space2, blend, mapped_concepts)
        
        # Transfer relations from both spaces
        self._transfer_relations(space1, blend, mapped_concepts)
        self._transfer_relations(space2, blend, mapped_concepts)
        
        # Create new emergent relations
        self._create_emergent_relations(blend)
        
        # If no concepts were mapped, return None
        if not blend.concepts:
            return None
            
        # Store the blend
        self.blends[blend_id] = blend
        
        return blend.to_dict()

    def _generate_contrast_blend(self, space1: ConceptSpace, space2: ConceptSpace,
                              mappings: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate a contrast blend that exploits differences between spaces"""
        # Generate blend ID
        blend_id = f"blend_{self.next_blend_id}"
        self.next_blend_id += 1
        
        # Create blend name
        blend_name = f"{space1.name} ⟷ {space2.name} Contrast"
        
        # Create blend
        blend = ConceptualBlend(
            blend_id=blend_id,
            name=blend_name,
            input_spaces=[space1.id, space2.id],
            blend_type="contrast"
        )
        
        # Find concepts with significant differences but some similarity
        contrast_mappings = []
        
        for mapping in mappings:
            similarity = mapping["similarity"]
            
            # Look for moderate similarity (not too similar, not too different)
            if 0.3 <= similarity <= 0.7:
                contrast_mappings.append(mapping)
        
        # If not enough contrast mappings, return None
        if len(contrast_mappings) < 2:
            return None
            
        # Add selected contrasting concepts
        mapped_concepts = {}
        
        for mapping in contrast_mappings:
            concept1_id = mapping["concept1"]
            concept2_id = mapping["concept2"]
            
            concept1 = space1.concepts.get(concept1_id)
            concept2 = space2.concepts.get(concept2_id)
            
            if not concept1 or not concept2:
                continue
                
            # Create blend concept
            blend_concept_id = f"{blend_id}_concept_{len(blend.concepts) + 1}"
            
            # Create contrasting name
            blend_concept_name = self._generate_contrast_name(concept1["name"], concept2["name"])
            
            # Contrast properties
            contrasted_properties = self._contrast_properties(
                concept1.get("properties", {}),
                concept2.get("properties", {})
            )
            
            # Add concept to blend
            blend.add_concept(
                concept_id=blend_concept_id,
                name=blend_concept_name,
                properties=contrasted_properties,
                source_concepts=[
                    {"space_id": space1.id, "concept_id": concept1_id},
                    {"space_id": space2.id, "concept_id": concept2_id}
                ]
            )
            
            # Record mapping
            mapped_concepts[concept1_id] = blend_concept_id
            mapped_concepts[concept2_id] = blend_concept_id
            
            # Add mappings
            blend.add_mapping(
                input_space_id=space1.id,
                input_concept_id=concept1_id,
                blend_concept_id=blend_concept_id,
                mapping_type="contrast",
                mapping_strength=mapping["similarity"]
            )
            
            blend.add_mapping(
                input_space_id=space2.id,
                input_concept_id=concept2_id,
                blend_concept_id=blend_concept_id,
                mapping_type="contrast",
                mapping_strength=mapping["similarity"]
            )
        
        # Add contrasting relations
        self._add_contrasting_relations(space1, space2, blend, mapped_concepts)
        
        # If no concepts were mapped, return None
        if not blend.concepts:
            return None
            
        # Look for emergent structure from contrasts
        self._identify_emergent_structure(blend, space1, space2)
        
        # Store the blend
        self.blends[blend_id] = blend
        
        return blend.to_dict()
    
    def _generate_blend_concept_name(self, name1: str, name2: str) -> str:
        """Generate a name for a blend concept from two input names"""
        # Simple combination with plus
        return f"{name1}-{name2}"
    
    def _generate_integrated_name(self, name1: str, name2: str) -> str:
        """Generate a more integrated name for fusion blends"""
        # Extract words
        words1 = re.findall(r'\w+', name1)
        words2 = re.findall(r'\w+', name2)
        
        if not words1 or not words2:
            return f"{name1}-{name2}"
            
        # Try different integration patterns
        
        # 1. First half of word1 + second half of word2
        if len(words1[0]) >= 2 and len(words2[-1]) >= 2:
            half1 = words1[0][:len(words1[0])//2 + 1]
            half2 = words2[-1][len(words2[-1])//2:]
            return half1 + half2
            
        # 2. Alternate words
        result = []
        for i in range(max(len(words1), len(words2))):
            if i < len(words1):
                result.append(words1[i])
            if i < len(words2):
                result.append(words2[i])
        
        return " ".join(result)
    
    def _generate_elaborate_name(self, name1: str, name2: str) -> str:
        """Generate an elaborate name for elaboration blends"""
        # Extract words
        words1 = re.findall(r'\w+', name1)
        words2 = re.findall(r'\w+', name2)
        
        if not words1 or not words2:
            return f"{name1}-{name2}"
        
        # Use a more descriptive pattern
        if len(words1) == 1 and len(words2) == 1:
            return f"{words1[0]}-inspired {words2[0]}"
            
        # Use adjectives from one with nouns from another
        # (simplified assumption that last word is noun, earlier words are adjectives)
        if len(words1) > 1 and len(words2) >= 1:
            adjectives = words1[:-1]
            noun = words2[-1]
            return " ".join(adjectives + [noun])
            
        # Default to combination
        return f"{name1} {name2}"
    
    def _generate_contrast_name(self, name1: str, name2: str) -> str:
        """Generate a contrasting name for contrast blends"""
        # Extract words
        words1 = re.findall(r'\w+', name1)
        words2 = re.findall(r'\w+', name2)
        
        if not words1 or not words2:
            return f"{name1} vs {name2}"
            
        # Create contrasting patterns
        
        # 1. "Anti-X Y" pattern
        if random.random() < 0.3:
            return f"Anti-{words1[0]} {words2[0]}"
            
        # 2. "X-but-Y" pattern
        if random.random() < 0.5:
            return f"{words1[0]}-but-{words2[0]}"
            
        # 3. "Neither-X-nor-Y" pattern
        return f"Neither-{words1[0]}-nor-{words2[0]}"
    
    def _merge_properties(self, props1: Dict[str, Any], props2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge properties from two concepts for composition blend"""
        merged = {}
        
        # Add all properties from props1
        merged.update(props1)
        
        # Add/override with props2
        for key, value in props2.items():
            if key not in merged:
                # New property
                merged[key] = value
            else:
                # Existing property - use average if numeric
                if isinstance(merged[key], (int, float)) and isinstance(value, (int, float)):
                    merged[key] = (merged[key] + value) / 2
                else:
                    # For non-numeric, prefer props1
                    pass
        
        return merged
    
    def _integrate_properties(self, props1: Dict[str, Any], props2: Dict[str, Any]) -> Dict[str, Any]:
        """More deeply integrate properties for fusion blend"""
        integrated = {}
        
        # Find all property keys
        all_keys = set(props1.keys()).union(set(props2.keys()))
        
        for key in all_keys:
            # Get values (or None if not present)
            val1 = props1.get(key)
            val2 = props2.get(key)
            
            if key in props1 and key in props2:
                # Both have this property
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # For numeric, use weighted average
                    integrated[key] = (val1 * 0.6) + (val2 * 0.4)
                elif isinstance(val1, str) and isinstance(val2, str):
                    # For strings, try to blend
                    if len(val1) < 10 and len(val2) < 10:
                        # Short strings can be combined
                        integrated[key] = val1[:len(val1)//2] + val2[len(val2)//2:]
                    else:
                        # Longer strings keep one
                        integrated[key] = val1
                else:
                    # Default to first value
                    integrated[key] = val1
            elif key in props1:
                # Only in props1
                integrated[key] = val1
            else:
                # Only in props2
                integrated[key] = val2
        
        return integrated
    
    def _complete_properties(self, props1: Dict[str, Any], props2: Dict[str, Any]) -> Dict[str, Any]:
        """Complete properties from one space with another"""
        completed = props1.copy()  # Start with all props1
        
        # Add properties from props2 that don't exist in props1
        for key, value in props2.items():
            if key not in completed:
                completed[key] = value
        
        return completed
    
    def _elaborate_properties(self, props1: Dict[str, Any], props2: Dict[str, Any]) -> Dict[str, Any]:
        """Elaborate properties for elaboration blend"""
        elaborated = {}
        
        # Start with properties from both
        elaborated.update(props1)
        
        for key, value in props2.items():
            if key not in elaborated:
                elaborated[key] = value
            else:
                # For existing properties, use combination
                val1 = elaborated[key]
                
                if isinstance(val1, (int, float)) and isinstance(value, (int, float)):
                    # For numeric, use average and add a small random factor
                    avg = (val1 + value) / 2
                    elaborated[key] = avg * random.uniform(0.9, 1.1)
                elif isinstance(val1, str) and isinstance(value, str):
                    # For strings, combine where appropriate
                    if len(val1) + len(value) < 50:
                        elaborated[key] = f"{val1} {value}"
                    else:
                        elaborated[key] = val1
        
        # Create one or two new properties based on combinations
        property_keys = list(elaborated.keys())
        if len(property_keys) >= 2:
            key1 = random.choice(property_keys)
            key2 = random.choice([k for k in property_keys if k != key1])
            
            new_key = f"{key1}_{key2}"
            val1 = elaborated[key1]
            val2 = elaborated[key2]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                elaborated[new_key] = val1 * val2
            else:
                elaborated[new_key] = f"{val1}_{val2}"
        
        return elaborated
    
    def _contrast_properties(self, props1: Dict[str, Any], props2: Dict[str, Any]) -> Dict[str, Any]:
        """Contrast properties for contrast blend"""
        contrasted = {}
        
        # Find common properties to contrast
        common_keys = set(props1.keys()).intersection(set(props2.keys()))
        
        for key in common_keys:
            val1 = props1[key]
            val2 = props2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # For numeric, use difference or ratio
                if random.random() < 0.5:
                    contrasted[key] = val1 - val2
                else:
                    if val2 != 0:
                        contrasted[key] = val1 / val2
                    else:
                        contrasted[key] = val1
            elif val1 != val2:
                # For different values, use contrasting format
                contrasted[key] = f"{val1}-not-{val2}"
        
        # Add some unique properties from each
        only_in_1 = set(props1.keys()) - set(props2.keys())
        only_in_2 = set(props2.keys()) - set(props1.keys())
        
        for key in list(only_in_1)[:2]:  # Limit to 2 properties
            contrasted[f"{key}_unique_1"] = props1[key]
            
        for key in list(only_in_2)[:2]:  # Limit to 2 properties
            contrasted[f"{key}_unique_2"] = props2[key]
        
        return contrasted
    
    def _transfer_relations(self, space: ConceptSpace, blend: ConceptualBlend, 
                        mapped_concepts: Dict[str, str]) -> None:
        """Transfer relations from an input space to the blend"""
        for relation in space.relations:
            source_id = relation["source"]
            target_id = relation["target"]
            
            # Skip if either concept is not mapped
            if source_id not in mapped_concepts or target_id not in mapped_concepts:
                continue
                
            # Get blend concept IDs
            blend_source_id = mapped_concepts[source_id]
            blend_target_id = mapped_concepts[target_id]
            
            # Add relation to blend
            blend.add_relation(
                source_id=blend_source_id,
                target_id=blend_target_id,
                relation_type=relation["type"],
                strength=relation["strength"],
                source_relation=relation
            )
    
    def _integrate_relations(self, space1: ConceptSpace, space2: ConceptSpace,
                         blend: ConceptualBlend, mapped_concepts: Dict[str, str]) -> None:
        """Integrate relations from both input spaces for fusion blend"""
        # First transfer relations from both spaces
        self._transfer_relations(space1, blend, mapped_concepts)
        self._transfer_relations(space2, blend, mapped_concepts)
        
        # Then look for relations to merge
        relations_by_concepts = defaultdict(list)
        
        for relation in blend.relations:
            key = (relation["source"], relation["target"])
            relations_by_concepts[key].append(relation)
        
        # Merge duplicate relations
        for (source_id, target_id), rels in relations_by_concepts.items():
            if len(rels) > 1:
                # Get average strength
                avg_strength = sum(rel["strength"] for rel in rels) / len(rels)
                
                # Combine relation types
                types = [rel["type"] for rel in rels]
                if len(set(types)) == 1:
                    # Same type - keep it
                    relation_type = types[0]
                else:
                    # Different types - combine
                    relation_type = "_".join(sorted(set(types)))
                
                # Create new merged relation
                merged_relation = {
                    "source": source_id,
                    "target": target_id,
                    "type": relation_type,
                    "strength": avg_strength,
                    "source_relation": {
                        "merged_from": [rel.get("source_relation") for rel in rels if rel.get("source_relation")]
                    },
                    "added_at": datetime.now().isoformat()
                }
                
                # Remove original relations
                blend.relations = [rel for rel in blend.relations 
                               if not (rel["source"] == source_id and rel["target"] == target_id)]
                
                # Add merged relation
                blend.relations.append(merged_relation)
    
    def _complete_relations(self, space1: ConceptSpace, space2: ConceptSpace,
                        blend: ConceptualBlend, mapped_concepts: Dict[str, str]) -> None:
        """Complete the structure with relevant relations from space2"""
        # Look for relations in space2 that involve mapped concepts
        space2_ids = {id for id, blend_id in mapped_concepts.items() 
                   if any(space2_concept_id == id for space2_concept_id in space2.concepts)}
        
        for relation in space2.relations:
            source_id = relation["source"]
            target_id = relation["target"]
            
            # Check if at least one concept is mapped
            if source_id in space2_ids or target_id in space2_ids:
                # Get or create blend concepts
                if source_id in mapped_concepts:
                    blend_source_id = mapped_concepts[source_id]
                else:
                    # Create new concept in blend
                    source_concept = space2.concepts.get(source_id)
                    if not source_concept:
                        continue
                        
                    blend_source_id = f"{blend.id}_concept_{len(blend.concepts) + 1}"
                    
                    # Add concept to blend
                    blend.add_concept(
                        concept_id=blend_source_id,
                        name=source_concept["name"],
                        properties=source_concept.get("properties", {}),
                        source_concepts=[
                            {"space_id": space2.id, "concept_id": source_id}
                        ]
                    )
                    
                    # Record mapping
                    mapped_concepts[source_id] = blend_source_id
                    
                    # Add mapping
                    blend.add_mapping(
                        input_space_id=space2.id,
                        input_concept_id=source_id,
                        blend_concept_id=blend_source_id,
                        mapping_type="completion",
                        mapping_strength=1.0
                    )
                
                if target_id in mapped_concepts:
                    blend_target_id = mapped_concepts[target_id]
                else:
                    # Create new concept in blend
                    target_concept = space2.concepts.get(target_id)
                    if not target_concept:
                        continue
                        
                    blend_target_id = f"{blend.id}_concept_{len(blend.concepts) + 1}"
                    
                    # Add concept to blend
                    blend.add_concept(
                        concept_id=blend_target_id,
                        name=target_concept["name"],
                        properties=target_concept.get("properties", {}),
                        source_concepts=[
                            {"space_id": space2.id, "concept_id": target_id}
                        ]
                    )
                    
                    # Record mapping
                    mapped_concepts[target_id] = blend_target_id
                    
                    # Add mapping
                    blend.add_mapping(
                        input_space_id=space2.id,
                        input_concept_id=target_id,
                        blend_concept_id=blend_target_id,
                        mapping_type="completion",
                        mapping_strength=1.0
                    )
                
                # Add relation to blend
                blend.add_relation(
                    source_id=blend_source_id,
                    target_id=blend_target_id,
                    relation_type=relation["type"],
                    strength=relation["strength"],
                    source_relation=relation
                )
    
    def _add_supporting_concepts(self, space: ConceptSpace, blend: ConceptualBlend,
                            mapped_concepts: Dict[str, str]) -> None:
        """Add supporting concepts from a space to enable more elaboration"""
        # Find concepts related to already mapped concepts
        space_ids = {id for id in space.concepts if id in mapped_concepts}
        
        for space_id in space_ids:
            # Get related concepts
            related = space.get_related_concepts(space_id)
            
            for related_data in related:
                related_concept = related_data["concept"]
                related_id = related_concept["id"]
                
                # Skip if already mapped
                if related_id in mapped_concepts:
                    continue
                    
                # Create blend concept
                blend_concept_id = f"{blend.id}_concept_{len(blend.concepts) + 1}"
                
                # Add concept to blend
                blend.add_concept(
                    concept_id=blend_concept_id,
                    name=related_concept["name"],
                    properties=related_concept.get("properties", {}),
                    source_concepts=[
                        {"space_id": space.id, "concept_id": related_id}
                    ]
                )
                
                # Record mapping
                mapped_concepts[related_id] = blend_concept_id
                
                # Add mapping
                blend.add_mapping(
                    input_space_id=space.id,
                    input_concept_id=related_id,
                    blend_concept_id=blend_concept_id,
                    mapping_type="supporting",
                    mapping_strength=0.8
                )
                
                # Limit to a few supporting concepts
                if len(mapped_concepts) >= len(space.concepts) * 0.5:
                    break
    
    def _create_emergent_relations(self, blend: ConceptualBlend) -> None:
        """Create new emergent relations in the blend"""
        # Get all concepts in the blend
        concept_ids = list(blend.concepts.keys())
        
        # Skip if too few concepts
        if len(concept_ids) < 3:
            return
            
        # Create a few emergent relations
        num_emergent = min(len(concept_ids) // 2, 5)
        
        for _ in range(num_emergent):
            # Select random concepts
            source_id = random.choice(concept_ids)
            target_id = random.choice([c for c in concept_ids if c != source_id])
            
            # Skip if relation already exists
            if any(r["source"] == source_id and r["target"] == target_id for r in blend.relations):
                continue
                
            # Create a new relation type
            emergent_types = [
                "emergent_association",
                "unexpected_similarity",
                "novel_causality",
                "creative_transformation",
                "imagined_interaction"
            ]
            relation_type = random.choice(emergent_types)
            
            # Add relation to blend
            blend.add_relation(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                strength=random.uniform(0.6, 0.9)
            )
    
    def _add_contrasting_relations(self, space1: ConceptSpace, space2: ConceptSpace,
                               blend: ConceptualBlend, mapped_concepts: Dict[str, str]) -> None:
        """Add contrasting relations for contrast blend"""
        # Find relations that exist in one space but not the other
        relations_space1 = set()
        relations_space2 = set()
        
        for relation in space1.relations:
            source_id = relation["source"]
            target_id = relation["target"]
            
            if source_id in mapped_concepts and target_id in mapped_concepts:
                relations_space1.add((source_id, target_id, relation["type"]))
        
        for relation in space2.relations:
            source_id = relation["source"]
            target_id = relation["target"]
            
            if source_id in mapped_concepts and target_id in mapped_concepts:
                relations_space2.add((source_id, target_id, relation["type"]))
        
        # Find relations in space1 but not in space2
        for source_id, target_id, rel_type in relations_space1:
            # Check if relation exists with same concepts in space2
            corresponding_relations = [r for r in relations_space2 
                                     if r[0] == source_id and r[1] == target_id]
            
            if not corresponding_relations:
                # Relation exists only in space1
                blend_source_id = mapped_concepts[source_id]
                blend_target_id = mapped_concepts[target_id]
                
                # Add relation to blend with "only_in_space1" annotation
                blend.add_relation(
                    source_id=blend_source_id,
                    target_id=blend_target_id,
                    relation_type=f"{rel_type}_only_in_space1",
                    strength=0.8
                )
        
        # Find relations in space2 but not in space1
        for source_id, target_id, rel_type in relations_space2:
            # Check if relation exists with same concepts in space1
            corresponding_relations = [r for r in relations_space1 
                                     if r[0] == source_id and r[1] == target_id]
            
            if not corresponding_relations:
                # Relation exists only in space2
                blend_source_id = mapped_concepts[source_id]
                blend_target_id = mapped_concepts[target_id]
                
                # Add relation to blend with "only_in_space2" annotation
                blend.add_relation(
                    source_id=blend_source_id,
                    target_id=blend_target_id,
                    relation_type=f"{rel_type}_only_in_space2",
                    strength=0.8
                )
        
        # Find relations that have different types
        common_pairs = set((s, t) for s, t, _ in relations_space1) & set((s, t) for s, t, _ in relations_space2)
        
        for source_id, target_id in common_pairs:
            # Get relation types from each space
            types_space1 = [r[2] for r in relations_space1 if r[0] == source_id and r[1] == target_id]
            types_space2 = [r[2] for r in relations_space2 if r[0] == source_id and r[1] == target_id]
            
            # Check if types are different
            if set(types_space1) != set(types_space2):
                blend_source_id = mapped_concepts[source_id]
                blend_target_id = mapped_concepts[target_id]
                
                # Add contrasting relation
                contrast_type = f"{types_space1[0]}_vs_{types_space2[0]}"
                
                blend.add_relation(
                    source_id=blend_source_id,
                    target_id=blend_target_id,
                    relation_type=contrast_type,
                    strength=0.9
                )
    
    def _identify_emergent_structure(self, blend: ConceptualBlend,
                               space1: ConceptSpace, space2: ConceptSpace) -> None:
        """Identify emergent structure in the blend"""
        # Skip if too few concepts
        if len(blend.concepts) < 3:
            return
            
        # Look for patterns in relations
        
        # 1. Cycles (conceptual loops)
        cycles = self._find_cycles(blend)
        
        for cycle in cycles:
            if len(cycle) >= 3:  # Only consider cycles of at least 3 concepts
                cycle_concepts = cycle
                cycle_relations = []
                
                # Find relations in the cycle
                for i in range(len(cycle)):
                    source_id = cycle[i]
                    target_id = cycle[(i + 1) % len(cycle)]
                    
                    for rel in blend.relations:
                        if rel["source"] == source_id and rel["target"] == target_id:
                            cycle_relations.append({
                                "source": source_id,
                                "target": target_id
                            })
                
                # Add emergent structure
                blend.add_emergent_structure(
                    name=f"Conceptual Cycle of {len(cycle)} Elements",
                    description=f"A cycle of concepts forming a closed loop of relationships",
                    concept_ids=cycle_concepts,
                    relation_ids=cycle_relations
                )
        
        # 2. Hubs (concepts with many connections)
        concept_connections = defaultdict(int)
        
        for relation in blend.relations:
            concept_connections[relation["source"]] += 1
            concept_connections[relation["target"]] += 1
        
        # Find concepts with high connectivity
        hub_threshold = max(3, len(blend.concepts) // 3)
        
        for concept_id, connection_count in concept_connections.items():
            if connection_count >= hub_threshold:
                # Find all relations involving this hub
                hub_relations = []
                connected_concepts = [concept_id]  # Include the hub itself
                
                for relation in blend.relations:
                    if relation["source"] == concept_id:
                        hub_relations.append({
                            "source": relation["source"],
                            "target": relation["target"]
                        })
                        if relation["target"] not in connected_concepts:
                            connected_concepts.append(relation["target"])
                    elif relation["target"] == concept_id:
                        hub_relations.append({
                            "source": relation["source"],
                            "target": relation["target"]
                        })
                        if relation["source"] not in connected_concepts:
                            connected_concepts.append(relation["source"])
                
                # Add emergent structure
                blend.add_emergent_structure(
                    name=f"Conceptual Hub: {blend.concepts[concept_id]['name']}",
                    description=f"A central concept with many connections to other concepts",
                    concept_ids=connected_concepts,
                    relation_ids=hub_relations
                )
        
        # 3. Novel property combinations
        novel_combinations = self._find_novel_property_combinations(blend, space1, space2)
        
        for combo in novel_combinations:
            # Add emergent structure
            blend.add_emergent_structure(
                name=f"Novel Property Combination: {combo['name']}",
                description=combo["description"],
                concept_ids=combo["concept_ids"],
                relation_ids=[]
            )
    
    def _find_cycles(self, blend: ConceptualBlend) -> List[List[str]]:
        """Find cycles in the blend relations"""
        # Build a directed graph
        graph = defaultdict(list)
        
        for relation in blend.relations:
            source_id = relation["source"]
            target_id = relation["target"]
            graph[source_id].append(target_id)
        
        # Find cycles using DFS
        cycles = []
        visited = set()
        path_stack = []
        
        def dfs(node, parent):
            if node in path_stack:
                # Found a cycle
                cycle_start = path_stack.index(node)
                cycles.append(path_stack[cycle_start:])
                return
                
            if node in visited:
                return
                
            visited.add(node)
            path_stack.append(node)
            
            for neighbor in graph[node]:
                if neighbor != parent:  # Avoid immediate backtracking
                    dfs(neighbor, node)
            
            path_stack.pop()
        
        # Start DFS from each node
        for node in graph:
            if node not in visited:
                dfs(node, None)
        
        return cycles
    
    def _find_novel_property_combinations(self, blend: ConceptualBlend,
                                     space1: ConceptSpace, 
                                     space2: ConceptSpace) -> List[Dict[str, Any]]:
        """Find novel combinations of properties in the blend"""
        novel_combinations = []
        
        # Build property sets for original spaces
        props_space1 = set()
        props_space2 = set()
        
        for concept in space1.concepts.values():
            for prop_name, prop_value in concept.get("properties", {}).items():
                if isinstance(prop_value, (str, int, float, bool)):
                    props_space1.add(f"{prop_name}:{prop_value}")
        
        for concept in space2.concepts.values():
            for prop_name, prop_value in concept.get("properties", {}).items():
                if isinstance(prop_value, (str, int, float, bool)):
                    props_space2.add(f"{prop_name}:{prop_value}")
        
        # Look for novel combinations in blend
        for concept_id, concept in blend.concepts.items():
            novel_props = []
            
            for prop_name, prop_value in concept.get("properties", {}).items():
                if isinstance(prop_value, (str, int, float, bool)):
                    prop_str = f"{prop_name}:{prop_value}"
                    
                    # Check if this property combination is novel
                    if prop_str not in props_space1 and prop_str not in props_space2:
                        novel_props.append({
                            "name": prop_name,
                            "value": prop_value
                        })
            
            if len(novel_props) >= 2:  # At least 2 novel properties
                novel_combinations.append({
                    "name": concept["name"],
                    "description": f"Novel combination of properties in {concept['name']}: " + 
                                   ", ".join([f"{p['name']}={p['value']}" for p in novel_props]),
                    "concept_ids": [concept_id],
                    "properties": novel_props
                })
        
        return novel_combinations
    
    def _apply_constraints(self, candidate_blends: List[Dict[str, Any]], 
                       constraints: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Apply constraints and filters to candidate blends"""
        if not constraints:
            return candidate_blends
            
        filtered_blends = []
        
        for blend_data in candidate_blends:
            blend_id = blend_data["id"]
            blend = self.blends.get(blend_id)
            
            if not blend:
                continue
                
            # Apply constraints
            passes_constraints = True
            
            # Constraint: minimum number of concepts
            if "min_concepts" in constraints:
                min_concepts = constraints["min_concepts"]
                if len(blend.concepts) < min_concepts:
                    passes_constraints = False
            
            # Constraint: required properties
            if "required_properties" in constraints:
                required_props = constraints["required_properties"]
                
                # Check if any concept has the required properties
                has_required_props = False
                
                for concept in blend.concepts.values():
                    props = concept.get("properties", {})
                    
                    # Check if all required properties are present
                    if all(prop in props for prop in required_props):
                        has_required_props = True
                        break
                
                if not has_required_props:
                    passes_constraints = False
            
            # Constraint: excluded relation types
            if "excluded_relations" in constraints:
                excluded_relations = constraints["excluded_relations"]
                
                # Check if any relation has excluded type
                for relation in blend.relations:
                    if relation["type"] in excluded_relations:
                        passes_constraints = False
                        break
            
            # Constraint: concept name pattern
            if "concept_name_pattern" in constraints:
                pattern = constraints["concept_name_pattern"]
                
                # Check if any concept matches the pattern
                has_matching_concept = False
                
                for concept in blend.concepts.values():
                    if re.search(pattern, concept["name"]):
                        has_matching_concept = True
                        break
                
                if not has_matching_concept:
                    passes_constraints = False
            
            # Add blend to filtered list if it passes all constraints
            if passes_constraints:
                filtered_blends.append(blend_data)
        
        return filtered_blends
    
    async def _elaborate_blends(self, blends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Elaborate and develop promising blends"""
        elaborated_blends = []
        
        for blend_data in blends:
            blend_id = blend_data["id"]
            blend = self.blends.get(blend_id)
            
            if not blend:
                continue
                
            # Perform multiple iterations of elaboration
            for i in range(self.blending_config["elaboration_iterations"]):
                # 1. Add detail to concepts
                for concept_id, concept in blend.concepts.items():
                    self._elaborate_concept(blend, concept_id, i)
                
                # 2. Add new relations
                self._elaborate_relations(blend, i)
                
                # 3. Identify new emergent structure
                if i == self.blending_config["elaboration_iterations"] - 1:  # Only on last iteration
                    space_ids = blend.input_spaces
                    input_spaces = [self.concept_spaces.get(space_id) for space_id in space_ids]
                    input_spaces = [space for space in input_spaces if space]
                    
                    if len(input_spaces) >= 2:
                        self._identify_emergent_structure(blend, input_spaces[0], input_spaces[1])
                
                # 4. Add an elaboration record
                blend.add_elaboration(
                    elaboration_type=f"iteration_{i+1}",
                    description=f"Elaboration iteration {i+1}",
                    affected_concepts=list(blend.concepts.keys()),
                    affected_relations=[{"source": r["source"], "target": r["target"]} 
                                      for r in blend.relations]
                )
            
            # Update the blend in storage
            self.blends[blend_id] = blend
            
            # Add to elaborated list
            elaborated_blends.append(blend.to_dict())
        
        return elaborated_blends
    
    def _elaborate_concept(self, blend: ConceptualBlend, concept_id: str, iteration: int) -> None:
        """Elaborate a concept in the blend"""
        concept = blend.concepts.get(concept_id)
        if not concept:
            return
            
        properties = concept.get("properties", {})
        
        # 1. Enhance existing properties
        for prop_name, prop_value in list(properties.items()):
            if isinstance(prop_value, (int, float)):
                # For numeric properties, add small variations in later iterations
                if iteration > 0:
                    variation = prop_value * random.uniform(-0.2, 0.2)
                    properties[f"{prop_name}_variation"] = prop_value + variation
            elif isinstance(prop_value, str) and len(prop_value) < 100:
                # For short string properties, add elaborations
                if iteration > 0 and random.random() < 0.3:
                    elaborations = [
                        f"enhanced {prop_value}",
                        f"{prop_value} (refined)",
                        f"{prop_value} with nuance",
                        f"{prop_value} in detail"
                    ]
                    properties[f"{prop_name}_elaborated"] = random.choice(elaborations)
        
        # 2. Add new properties in later iterations
        if iteration > 0:
            # Add 1-2 new properties per iteration
            new_props_count = random.randint(1, 2)
            
            for i in range(new_props_count):
                prop_name = f"elaborated_property_{iteration}_{i}"
                
                # Generate different property types
                prop_type = random.choice(["numeric", "categorical", "descriptive"])
                
                if prop_type == "numeric":
                    properties[prop_name] = random.uniform(0, 10)
                elif prop_type == "categorical":
                    categories = ["primary", "secondary", "tertiary", "quaternary"]
                    properties[prop_name] = random.choice(categories)
                else:  # descriptive
                    descriptions = [
                        "emerging characteristic",
                        "novel attribute",
                        "unexpected quality",
                        "derived property"
                    ]
                    properties[prop_name] = random.choice(descriptions)
        
        # Update concept
        concept["properties"] = properties
    
    def _elaborate_relations(self, blend: ConceptualBlend, iteration: int) -> None:
        """Elaborate relations in the blend"""
        concept_ids = list(blend.concepts.keys())
        if len(concept_ids) < 2:
            return
            
        # 1. Strengthen existing relations
        for relation in blend.relations:
            # Gradually increase strength with iterations
            relation["strength"] = min(1.0, relation["strength"] + (0.1 * iteration))
            
            # Add detail to relation type in later iterations
            if iteration > 0 and random.random() < 0.3:
                elaborations = [
                    f"{relation['type']}_refined",
                    f"{relation['type']}_strengthened",
                    f"{relation['type']}_advanced"
                ]
                relation["type"] = random.choice(elaborations)
        
        # 2. Add new relations in later iterations
        if iteration > 0:
            # Add 1-3 new relations per iteration
            new_relations_count = random.randint(1, 3)
            
            for _ in range(new_relations_count):
                source_id = random.choice(concept_ids)
                target_id = random.choice([c for c in concept_ids if c != source_id])
                
                # Skip if relation already exists
                if any(r["source"] == source_id and r["target"] == target_id for r in blend.relations):
                    continue
                    
                # Create relation types for elaboration
                elaboration_types = [
                    "emergent_connection",
                    "novel_association",
                    "elaborated_link",
                    "developed_relation",
                    "refined_connection"
                ]
                
                relation_type = f"{random.choice(elaboration_types)}_{iteration}"
                
                # Add relation
                blend.add_relation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    strength=0.5 + (0.1 * iteration)  # Stronger with iterations
                )
    
    def _evaluate_blends(self, blends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate novelty and utility of blends"""
        evaluated_blends = []
        
        for blend_data in blends:
            blend_id = blend_data["id"]
            blend = self.blends.get(blend_id)
            
            if not blend:
                continue
                
            # Evaluate metrics
            evaluation = {}
            
            # Calculate novelty score
            evaluation["novelty"] = self._evaluate_novelty(blend)
            
            # Calculate coherence score
            evaluation["coherence"] = self._evaluate_coherence(blend)
            
            # Calculate practicality score
            evaluation["practicality"] = self._evaluate_practicality(blend)
            
            # Calculate expressiveness score
            evaluation["expressiveness"] = self._evaluate_expressiveness(blend)
            
            # Calculate surprise score
            evaluation["surprise"] = self._evaluate_surprise(blend)
            
            # Calculate overall score
            overall_score = 0.0
            weights = self.blending_config["evaluation_weights"]
            
            for metric, score in evaluation.items():
                weight = weights.get(metric, 0.2)  # Default weight
                overall_score += score * weight
            
            # Update blend evaluation
            blend.update_evaluation(evaluation)
            blend.evaluation["overall_score"] = overall_score
            
            # Update the blend in storage
            self.blends[blend_id] = blend
            
            # Update statistics
            self.stats["blend_evaluations"] += 1
            
            # Add to evaluated list
            evaluated_data = blend.to_dict()
            evaluated_data["overall_score"] = overall_score
            evaluated_blends.append(evaluated_data)
        
        return evaluated_blends
    
    def _evaluate_novelty(self, blend: ConceptualBlend) -> float:
        """Evaluate novelty of a blend"""
        # 1. Check for novel concepts
        novel_concept_count = 0
        
        for concept in blend.concepts.values():
            source_concepts = concept.get("source_concepts", [])
            
            # Concepts from multiple sources are potentially more novel
            if len(source_concepts) > 1:
                novel_concept_count += 1
                
            # Check if concept has novel properties
            source_concepts_data = []
            
            for source in source_concepts:
                space_id = source.get("space_id")
                concept_id = source.get("concept_id")
                
                if space_id in self.concept_spaces and concept_id in self.concept_spaces[space_id].concepts:
                    source_concepts_data.append(self.concept_spaces[space_id].concepts[concept_id])
            
            # Compare properties with source concepts
            if source_concepts_data:
                novel_properties = self._count_novel_properties(
                    concept.get("properties", {}),
                    [s.get("properties", {}) for s in source_concepts_data]
                )
                
                if novel_properties > 0:
                    novel_concept_count += novel_properties / 5  # Boost for novel properties
        
        # 2. Check for novel relations
        novel_relation_count = 0
        
        for relation in blend.relations:
            # Relations with novel types are more novel
            if "emergent" in relation["type"] or "novel" in relation["type"]:
                novel_relation_count += 1
            
            # Check if relation source differs from original
            if relation.get("source_relation") is None:
                novel_relation_count += 0.5
        
        # 3. Check for emergent structure
        emergent_structure_score = len(blend.emergent_structure) * 0.2
        
        # Calculate novelty score
        concept_novelty = novel_concept_count / max(1, len(blend.concepts))
        relation_novelty = novel_relation_count / max(1, len(blend.relations))
        
        # Combine scores
        novelty_score = (
            concept_novelty * 0.4 +
            relation_novelty * 0.4 +
            emergent_structure_score * 0.2
        )
        
        # Normalize to 0-1 range
        return min(1.0, novelty_score)
    
    def _count_novel_properties(self, properties: Dict[str, Any], 
                             source_properties_list: List[Dict[str, Any]]) -> int:
        """Count properties that don't appear in source concepts"""
        novel_count = 0
        
        for prop_name, prop_value in properties.items():
            property_is_novel = True
            
            for source_props in source_properties_list:
                if prop_name in source_props:
                    # Property exists in source
                    source_value = source_props[prop_name]
                    
                    # Check if values are similar
                    if isinstance(prop_value, (int, float)) and isinstance(source_value, (int, float)):
                        # For numeric, check if within 20%
                        if source_value != 0 and abs((prop_value - source_value) / source_value) < 0.2:
                            property_is_novel = False
                            break
                    elif prop_value == source_value:
                        property_is_novel = False
                        break
            
            if property_is_novel:
                novel_count += 1
        
        return novel_count
    
    def _evaluate_coherence(self, blend: ConceptualBlend) -> float:
        """Evaluate coherence of a blend"""
        # 1. Check structural coherence
        
        # Build a graph to check connectivity
        graph = defaultdict(list)
        
        for relation in blend.relations:
            source_id = relation["source"]
            target_id = relation["target"]
            
            graph[source_id].append(target_id)
            graph[target_id].append(source_id)  # Treat as undirected for connectivity
        
        # Check if graph is connected
        concept_ids = set(blend.concepts.keys())
        if not concept_ids:
            return 0.0
            
        # BFS to check connectivity
        visited = set()
        start = next(iter(concept_ids))
        
        queue = [start]
        visited.add(start)
        
        while queue:
            node = queue.pop(0)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        connectivity_score = len(visited) / len(concept_ids)
        
        # 2. Check conceptual coherence (consistent property values)
        property_consistency = 1.0
        
        if len(blend.concepts) >= 2:
            # Get all property names
            all_props = set()
            
            for concept in blend.concepts.values():
                all_props.update(concept.get("properties", {}).keys())
            
            # Check consistency for each property
            for prop in all_props:
                prop_values = []
                
                for concept in blend.concepts.values():
                    if prop in concept.get("properties", {}):
                        prop_values.append(concept["properties"][prop])
                
                if len(prop_values) >= 2:
                    # Calculate consistency
                    if all(isinstance(v, (int, float)) for v in prop_values):
                        # For numeric, check variance
                        mean = sum(prop_values) / len(prop_values)
                        variance = sum((v - mean) ** 2 for v in prop_values) / len(prop_values)
                        
                        # Normalize variance to 0-1 range (0 = consistent, 1 = inconsistent)
                        if mean != 0:
                            normalized_variance = min(1.0, variance / (mean ** 2))
                        else:
                            normalized_variance = min(1.0, variance)
                            
                        prop_consistency = 1.0 - normalized_variance
                    elif all(isinstance(v, str) for v in prop_values):
                        # For strings, count unique values
                        unique_values = len(set(prop_values))
                        prop_consistency = 1.0 - (unique_values - 1) / len(prop_values)
                    else:
                        # Mixed types are less consistent
                        prop_consistency = 0.5
                        
                    property_consistency *= prop_consistency
        
        # 3. Check relation coherence
        relation_coherence = 1.0
        
        if blend.relations:
            # Check if relations form consistent patterns
            relation_types = [r["type"] for r in blend.relations]
            unique_types = len(set(relation_types))
            
            # Fewer unique relation types relative to total is more coherent
            relation_type_consistency = 1.0 - (unique_types - 1) / len(relation_types) if len(relation_types) > 1 else 1.0
            
            # Check relation strengths
            strengths = [r["strength"] for r in blend.relations]
            mean_strength = sum(strengths) / len(strengths)
            
            # Calculate strength variance
            strength_variance = sum((s - mean_strength) ** 2 for s in strengths) / len(strengths)
            strength_consistency = 1.0 - min(1.0, strength_variance * 4)  # Scale for sensitivity
            
            relation_coherence = 0.5 * relation_type_consistency + 0.5 * strength_consistency
        
        # Combine scores
        coherence_score = (
            connectivity_score * 0.4 +
            property_consistency * 0.3 +
            relation_coherence * 0.3
        )
        
        return coherence_score
    
    def _evaluate_practicality(self, blend: ConceptualBlend) -> float:
        """Evaluate practicality of a blend"""
        # 1. Check if blend has clear, defined concepts
        concept_clarity = 0.0
        
        for concept in blend.concepts.values():
            # Clear concepts have substantial properties
            prop_count = len(concept.get("properties", {}))
            concept_clarity += min(1.0, prop_count / 5)  # Cap at 5 properties
        
        if blend.concepts:
            concept_clarity /= len(blend.concepts)
        
        # 2. Check for concrete relations
        relation_concreteness = 0.0
        
        for relation in blend.relations:
            # Relations with high strength are more concrete
            relation_concreteness += relation["strength"]
            
            # Relations with specific types (not emergent or abstract) are more practical
            if "emergent" not in relation["type"] and "novel" not in relation["type"] and "abstract" not in relation["type"]:
                relation_concreteness += 0.5
        
        if blend.relations:
            relation_concreteness /= len(blend.relations) * 1.5  # Normalize by max possible score
        
        # 3. Check if blend has organizing principles
        organizing_score = min(1.0, len(blend.organizing_principles) / 3)  # Cap at 3 principles
        
        # 4. Check if blend has been elaborated
        elaboration_score = min(1.0, len(blend.elaborations) / 5)  # Cap at 5 elaborations
        
        # 5. Source concept proportion (blends with more source material may be more practical)
        source_coverage = 0.0
        total_mappings = 0
        
        for concept in blend.concepts.values():
            source_concepts = concept.get("source_concepts", [])
            total_mappings += len(source_concepts)
        
        if blend.concepts:
            avg_mappings_per_concept = total_mappings / len(blend.concepts)
            source_coverage = min(1.0, avg_mappings_per_concept)
        
        # Combine scores
        practicality_score = (
            concept_clarity * 0.3 +
            relation_concreteness * 0.3 +
            organizing_score * 0.2 +
            elaboration_score * 0.1 +
            source_coverage * 0.1
        )
        
        return practicality_score

    async def reason_about(self, text: str) -> Dict[str, Any]:
        """
        Perform reasoning about the given text input.
        This is a simplified reasoning interface for the global workspace.
        """
        # Try to find relevant causal models
        relevant_models = []
        for model_id, model in self.causal_models.items():
            # Simple relevance check based on text content
            if any(node.name.lower() in text.lower() for node in model.nodes.values()):
                relevant_models.append(model)
        
        # If we have relevant models, analyze them
        if relevant_models:
            model = relevant_models[0]  # Use most relevant
            
            # Find mentioned nodes
            mentioned_nodes = []
            for node_id, node in model.nodes.items():
                if node.name.lower() in text.lower():
                    mentioned_nodes.append({
                        "node_id": node_id,
                        "name": node.name,
                        "current_state": node.get_current_state(),
                        "type": node.type
                    })
            
            # Get causal relationships
            causal_insights = []
            for node_info in mentioned_nodes:
                node_id = node_info["node_id"]
                
                # Get causes
                parents = list(model.graph.predecessors(node_id))
                causes = [model.nodes[p].name for p in parents if p in model.nodes]
                
                # Get effects  
                children = list(model.graph.successors(node_id))
                effects = [model.nodes[c].name for c in children if c in model.nodes]
                
                if causes or effects:
                    causal_insights.append({
                        "concept": node_info["name"],
                        "causes": causes,
                        "effects": effects
                    })
            
            return {
                "model_used": model.name,
                "mentioned_concepts": mentioned_nodes,
                "causal_insights": causal_insights,
                "reasoning_type": "causal"
            }
        
        # Try conceptual reasoning if no causal models match
        relevant_spaces = []
        for space_id, space in self.concept_spaces.items():
            if any(concept["name"].lower() in text.lower() for concept in space.concepts.values()):
                relevant_spaces.append(space)
        
        if relevant_spaces:
            space = relevant_spaces[0]
            
            # Find mentioned concepts
            mentioned_concepts = []
            for concept_id, concept in space.concepts.items():
                if concept["name"].lower() in text.lower():
                    mentioned_concepts.append({
                        "id": concept_id,
                        "name": concept["name"],
                        "properties": concept.get("properties", {})
                    })
            
            return {
                "space_used": space.name,
                "mentioned_concepts": mentioned_concepts,
                "reasoning_type": "conceptual"
            }
        
        # Default response if no models or spaces match
        return {
            "reasoning_type": "default",
            "insight": f"Analyzing: {text[:100]}...",
            "confidence": 0.5
        }
    
    def _evaluate_expressiveness(self, blend: ConceptualBlend) -> float:
        """Evaluate expressiveness of a blend"""
        # 1. Concept richness (diversity of concept types)
        concept_names = [c["name"] for c in blend.concepts.values()]
        name_diversity = 0.0
        
        if concept_names:
            # Count unique words in concept names
            all_words = []
            for name in concept_names:
                all_words.extend(re.findall(r'\w+', name.lower()))
            
            unique_words = len(set(all_words))
            total_words = len(all_words)
            
            if total_words > 0:
                name_diversity = min(1.0, unique_words / total_words * (1 + math.log(len(concept_names)) / 10))
        
        # 2. Property richness (variety of properties)
        property_diversity = 0.0
        all_props = set()
        prop_values = []
        
        for concept in blend.concepts.values():
            props = concept.get("properties", {})
            all_props.update(props.keys())
            
            for value in props.values():
                if isinstance(value, (str, int, float, bool)):
                    prop_values.append(str(value))
        
        if blend.concepts:
            # Normalize by number of concepts
            property_diversity = min(1.0, len(all_props) / (len(blend.concepts) * 3))
            
            # Boost for diverse values
            if prop_values:
                unique_values = len(set(prop_values))
                value_diversity = unique_values / len(prop_values)
                property_diversity = (property_diversity + value_diversity) / 2
        
        # 3. Relation expressiveness (diversity of relation types)
        relation_diversity = 0.0
        
        if blend.relations:
            relation_types = [r["type"] for r in blend.relations]
            unique_types = len(set(relation_types))
            
            # More unique types relative to total = more expressive
            relation_diversity = min(1.0, unique_types / len(relation_types) * 2)
        
        # 4. Emergent structure expressiveness
        emergent_score = min(1.0, len(blend.emergent_structure) / 2)  # Cap at 2 structures
        
        # 5. Naming expressiveness (creative names are more expressive)
        naming_expressiveness = 0.0
        
        for concept in blend.concepts.values():
            name = concept["name"]
            
            # Count special characters in name
            special_chars = sum(1 for c in name if not c.isalnum() and not c.isspace())
            
            # Count word count
            word_count = len(re.findall(r'\w+', name))
            
            # Higher word count and moderate special chars = more expressive
            word_score = min(1.0, word_count / 4)  # Cap at 4 words
            special_score = max(0, min(1.0, special_chars / 3))  # Cap at 3 special chars
            
            naming_expressiveness += (word_score * 0.7 + special_score * 0.3)
        
        if blend.concepts:
            naming_expressiveness /= len(blend.concepts)
        
        # Combine scores
        expressiveness_score = (
            name_diversity * 0.2 +
            property_diversity * 0.3 +
            relation_diversity * 0.2 +
            emergent_score * 0.2 +
            naming_expressiveness * 0.1
        )
        
        return expressiveness_score
    
    def _evaluate_surprise(self, blend: ConceptualBlend) -> float:
        """Evaluate surprise or unexpectedness of a blend"""
        # 1. Concept unexpectedness
        concept_surprise = 0.0
        
        for concept in blend.concepts.values():
            source_concepts = concept.get("source_concepts", [])
            
            # Concepts with multiple, distant sources are more surprising
            if len(source_concepts) >= 2:
                # Get source spaces
                space_ids = [s.get("space_id") for s in source_concepts if s.get("space_id")]
                unique_spaces = len(set(space_ids))
                
                # More unique spaces = more surprising
                concept_surprise += min(1.0, unique_spaces / 2)
            
            # Check for property surprises
            properties = concept.get("properties", {})
            
            # Unusual property combinations are surprising
            if "contrast" in concept["name"] or "unexpected" in concept["name"]:
                concept_surprise += 0.2
                
            # Properties with extreme values are surprising
            for prop_name, prop_value in properties.items():
                if isinstance(prop_value, (int, float)) and abs(prop_value) > 10:
                    concept_surprise += 0.1
        
        if blend.concepts:
            concept_surprise /= len(blend.concepts)
        
        # 2. Relation unexpectedness
        relation_surprise = 0.0
        
        for relation in blend.relations:
            # Relations marked as emergent, unexpected, or novel are surprising
            if any(term in relation["type"].lower() for term in ["emergent", "unexpected", "novel", "surprise"]):
                relation_surprise += 0.3
            
            # Relations that only exist in the blend (not in sources) are surprising
            if relation.get("source_relation") is None:
                relation_surprise += 0.2
            
            # Relations with extreme strengths are surprising
            strength = relation["strength"]
            if strength < 0.2 or strength > 0.8:
                relation_surprise += 0.1
        
        if blend.relations:
            relation_surprise /= len(blend.relations)
        
        # 3. Emergent structure surprise
        emergent_surprise = 0.0
        
        for structure in blend.emergent_structure:
            # Large emergent structures are more surprising
            concept_count = len(structure.get("concepts", []))
            relation_count = len(structure.get("relations", []))
            
            size_score = min(1.0, (concept_count + relation_count) / 10)
            
            # Names with surprise-related terms are more surprising
            surprise_terms = ["unexpected", "novel", "surprise", "emergent", "creative"]
            term_match = any(term in structure["name"].lower() for term in surprise_terms)
            
            term_score = 0.3 if term_match else 0.0
            
            emergent_surprise += size_score * 0.7 + term_score * 0.3
        
        if blend.emergent_structure:
            emergent_surprise /= len(blend.emergent_structure)
        
        # 4. Overall blend unexpectedness
        
        # Blend type contributes to surprise
        type_score = 0.0
        if blend.type == "contrast":
            type_score = 0.8  # Contrast blends are very surprising
        elif blend.type == "elaboration":
            type_score = 0.6  # Elaboration blends are moderately surprising
        elif blend.type == "fusion":
            type_score = 0.4  # Fusion blends are somewhat surprising
        else:
            type_score = 0.2  # Other blend types are less surprising
        
        # Combine scores
        surprise_score = (
            concept_surprise * 0.3 +
            relation_surprise * 0.3 +
            emergent_surprise * 0.2 +
            type_score * 0.2
        )
        
        return surprise_score
    
    async def get_concept_space(self, space_id: str) -> Dict[str, Any]:
        """Get a concept space by ID"""
        if space_id not in self.concept_spaces:
            raise ValueError(f"Concept space {space_id} not found")
            
        return self.concept_spaces[space_id].to_dict()
    
    async def get_all_concept_spaces(self) -> List[Dict[str, Any]]:
        """Get all concept spaces"""
        return [space.to_dict() for space in self.concept_spaces.values()]
    
    async def get_blend(self, blend_id: str) -> Dict[str, Any]:
        """Get a blend by ID"""
        if blend_id not in self.blends:
            raise ValueError(f"Blend {blend_id} not found")
            
        return self.blends[blend_id].to_dict()
    
    # ==================================================================================
    # Integrated Reasoning Methods
    # ==================================================================================
    
    async def convert_blend_to_causal_model(self, blend_id: str, 
                                      name: str = None, domain: str = None) -> str:
        """Convert a conceptual blend to a causal model"""
        if blend_id not in self.blends:
            raise ValueError(f"Blend {blend_id} not found")
            
        blend = self.blends[blend_id]
        
        # Create causal model
        model_name = name or f"Causal model from {blend.name}"
        model_domain = domain or blend.input_spaces[0] if blend.input_spaces else ""
        
        model_id = await self.create_causal_model(
            name=model_name,
            domain=model_domain,
            metadata={"source_blend": blend_id}
        )
        
        # Map concepts to causal nodes
        for concept_id, concept in blend.concepts.items():
            # Create causal node
            node_id = await self.add_node_to_model(
                model_id=model_id,
                name=concept["name"],
                domain=model_domain,
                node_type="concept",
                metadata={"source_concept": concept_id}
            )
            
            # Add properties as node states
            causal_node = self.causal_models[model_id].nodes[node_id]
            
            for prop_name, prop_value in concept.get("properties", {}).items():
                if isinstance(prop_value, (str, int, float, bool)):
                    causal_node.add_state(prop_value)
                    
            # Record mapping
            self.concept_to_node_mappings[concept_id] = node_id
            self.node_to_concept_mappings[node_id] = concept_id
        
        # Map relations to causal relations
        for relation in blend.relations:
            source_id = relation["source"]
            target_id = relation["target"]
            
            # Get corresponding node IDs
            if source_id in self.concept_to_node_mappings and target_id in self.concept_to_node_mappings:
                node_source_id = self.concept_to_node_mappings[source_id]
                node_target_id = self.concept_to_node_mappings[target_id]
                
                # Determine relation type and strength
                relation_type = self._map_relation_type_to_causal(relation["type"])
                strength = relation["strength"]
                
                # Add causal relation
                await self.add_relation_to_model(
                    model_id=model_id,
                    source_id=node_source_id,
                    target_id=node_target_id,
                    relation_type=relation_type,
                    strength=strength,
                    mechanism=f"Derived from {relation['type']} relation in blend"
                )
        
        # Map emergent structure to causal patterns
        for structure in blend.emergent_structure:
            # Add as model assumption
            self.causal_models[model_id].add_assumption(
                description=f"Emergent structure: {structure['name']} - {structure['description']}",
                confidence=0.7
            )
        
        # Record model-blend mapping
        self.blend_to_model_mappings[blend_id] = model_id
        self.model_to_blend_mappings[model_id] = blend_id
        
        # Update statistics
        self.stats["conceptual_to_causal_conversions"] += 1
        
        return model_id
    
    async def convert_causal_model_to_concept_space(self, model_id: str,
                                            name: str = None, domain: str = None) -> str:
        """Convert a causal model to a conceptual space"""
        if model_id not in self.causal_models:
            raise ValueError(f"Model {model_id} not found")
            
        model = self.causal_models[model_id]
        
        # Create concept space
        space_name = name or f"Concept space from {model.name}"
        space_domain = domain or model.domain
        
        space_id = await self.create_concept_space(
            name=space_name,
            domain=space_domain,
            metadata={"source_model": model_id}
        )
        
        # Map nodes to concepts
        for node_id, node in model.nodes.items():
            # Create properties from node states and observations
            properties = {}
            
            # Add states as properties
            for i, state in enumerate(node.states):
                properties[f"state_{i}"] = state
                
            # Add current state
            current_state = node.get_current_state()
            if current_state is not None:
                properties["current_state"] = current_state
                
            # Add observation data if available
            if node.observations:
                recent_obs = node.observations[-1]
                properties["observed_value"] = recent_obs["value"]
                properties["observation_confidence"] = recent_obs["confidence"]
            
            # Create concept
            concept_id = await self.add_concept_to_space(
                space_id=space_id,
                name=node.name,
                properties=properties
            )
            
            # Record mapping
            self.node_to_concept_mappings[node_id] = concept_id
            self.concept_to_node_mappings[concept_id] = node_id
        
        # Map causal relations to concept relations
        for relation_id, relation in model.relations.items():
            source_id = relation.source_id
            target_id = relation.target_id
            
            # Get corresponding concept IDs
            if source_id in self.node_to_concept_mappings and target_id in self.node_to_concept_mappings:
                concept_source_id = self.node_to_concept_mappings[source_id]
                concept_target_id = self.node_to_concept_mappings[target_id]
                
                # Determine relation type
                relation_type = self._map_causal_type_to_relation(relation.type)
                
                # Add relation
                await self.add_relation_to_space(
                    space_id=space_id,
                    source_id=concept_source_id,
                    target_id=concept_target_id,
                    relation_type=relation_type,
                    strength=relation.strength
                )
        
        # Add organizing principles from model assumptions
        for assumption in model.assumptions:
            await self.add_organizing_principle(
                space_id=space_id,
                name=f"Principle from assumption",
                description=assumption["description"]
            )
        
        # Update statistics
        self.stats["causal_to_conceptual_conversions"] += 1
        
        return space_id
    
    def _map_relation_type_to_causal(self, blend_relation_type: str) -> str:
        """Map blend relation type to causal relation type"""
        mapping = {
            "association": "correlation",
            "similarity": "similarity",
            "contrast": "negative_correlation",
            "transformation": "transformation",
            "emergent": "emergent",
            "integration": "integration",
            "elaboration": "elaboration",
            "completion": "completion",
            "fusion": "fusion",
            "unexpected_similarity": "potential_causal",
            "novel_causality": "causal",
            "creative_transformation": "transformation"
        }
        
        # Check for exact match
        if blend_relation_type in mapping:
            return mapping[blend_relation_type]
            
        # Check for partial match
        for key, value in mapping.items():
            if key in blend_relation_type:
                return value
        
        # Default to association
        return "association"
    
    def _map_causal_type_to_relation(self, causal_relation_type: str) -> str:
        """Map causal relation type to blend relation type"""
        mapping = {
            "causal": "causal_influence",
            "correlation": "association",
            "similarity": "similarity",
            "negative_correlation": "contrast",
            "transformation": "transformation",
            "potential_causal": "potential_causality",
            "mediation": "mediation",
            "emergent": "emergent_relation"
        }
        
        # Check for exact match
        if causal_relation_type in mapping:
            return mapping[causal_relation_type]
            
        # Check for partial match
        for key, value in mapping.items():
            if key in causal_relation_type:
                return value
        
        # Default to association
        return "association"
    
    async def create_integrated_model(self, domain: str, 
                              base_on_causal: bool = True) -> Dict[str, Any]:
        """Create an integrated model with both causal and conceptual reasoning"""
        # Find relevant inputs from both systems
        causal_models = []
        concept_spaces = []
        
        # Find causal models in domain
        for model_id, model in self.causal_models.items():
            if model.domain == domain or self._domains_are_related(model.domain, domain):
                causal_models.append(model)
        
        # Find concept spaces in domain
        for space_id, space in self.concept_spaces.items():
            if space.domain == domain or self._domains_are_related(space.domain, domain):
                concept_spaces.append(space)
        
        # If no inputs, return error
        if not causal_models and not concept_spaces:
            return {"error": f"No causal models or concept spaces found for domain: {domain}"}
        
        # Determine base system (if both available)
        if base_on_causal and causal_models:
            # Use causal model as base
            base_model = causal_models[0]
            result = await self._integrate_from_causal(base_model, concept_spaces)
        elif concept_spaces:
            # Use concept space as base
            base_space = concept_spaces[0]
            result = await self._integrate_from_conceptual(base_space, causal_models)
        else:
            # Default to creating from scratch
            # First create a causal model
            model_id = await self.create_causal_model(
                name=f"Integrated {domain} Model", 
                domain=domain
            )
            
            # Then create a concept space
            space_id = await self.create_concept_space(
                name=f"Integrated {domain} Space",
                domain=domain
            )
            
            # Link them
            self.model_to_blend_mappings[model_id] = space_id
            self.blend_to_model_mappings[space_id] = model_id
            
            result = {
                "causal_model_id": model_id,
                "concept_space_id": space_id,
                "nodes_created": 0,
                "concepts_created": 0,
                "relations_created": 0
            }
        
        # Update statistics
        self.stats["integrated_analyses"] += 1
        
        return result
    
    async def _integrate_from_causal(self, base_model: CausalModel, 
                            concept_spaces: List[ConceptSpace]) -> Dict[str, Any]:
        """Create integrated model starting from a causal model"""
        # Step 1: Create concept space from causal model
        space_id = await self.convert_causal_model_to_concept_space(
            model_id=base_model.id,
            name=f"Integrated {base_model.domain} Space"
        )
        
        # Step 2: Blend with existing concept spaces
        if concept_spaces:
            # Find potential mappings
            space = self.concept_spaces[space_id]
            
            # For each concept space, find mappings and integrate
            for other_space in concept_spaces:
                # Skip if same
                if other_space.id == space_id:
                    continue
                    
                # Find mappings between spaces
                mappings = []
                
                for concept1_id, concept1 in space.concepts.items():
                    for concept2_id, concept2 in other_space.concepts.items():
                        # Calculate similarity
                        similarity = self._calculate_concept_similarity(
                            concept1, concept2, space, other_space
                        )
                        
                        # If above threshold, add mapping
                        if similarity >= self.integrated_config["cross_system_mapping_threshold"]:
                            mappings.append({
                                "space1": space.id,
                                "concept1": concept1_id,
                                "space2": other_space.id,
                                "concept2": concept2_id,
                                "similarity": similarity
                            })
                
                # Generate fusion blend from mappings
                if mappings:
                    fusion_blend = self._generate_fusion_blend(space, other_space, mappings)
                    
                    if fusion_blend:
                        blend_id = fusion_blend["id"]
                        
                        # Convert back to causal model
                        await self.convert_blend_to_causal_model(
                            blend_id=blend_id,
                            name=f"Enhanced {base_model.name}"
                        )
        
        # Step 3: Return result
        return {
            "causal_model_id": base_model.id,
            "concept_space_id": space_id,
            "nodes_created": len(base_model.nodes),
            "concepts_created": len(self.concept_spaces[space_id].concepts),
            "relations_created": len(base_model.relations)
        }
    

    async def _integrate_from_conceptual(self, base_space: ConceptSpace,
                                     causal_models: List[CausalModel]) -> Dict[str, Any]:
            """Create integrated model starting from a concept space"""
            # Step 1: Create a blend from concept space
            spaces = [base_space]
            
            # Find additional spaces to blend with
            for space_id, space in self.concept_spaces.items():
                if space_id != base_space.id and (space.domain == base_space.domain 
                                            or self._domains_are_related(space.domain, base_space.domain)):
                    spaces.append(space)
                    if len(spaces) >= 3:  # Limit to 3 spaces
                        break
            
            # Perform blending if we have at least 2 spaces
            blend_id = None
            if len(spaces) >= 2:
                # Find mappings between first two spaces
                mappings = []
                space1 = spaces[0]
                space2 = spaces[1]
                
                for concept1_id, concept1 in space1.concepts.items():
                    for concept2_id, concept2 in space2.concepts.items():
                        # Calculate similarity
                        similarity = self._calculate_concept_similarity(
                            concept1, concept2, space1, space2
                        )
                        
                        # If above threshold, add mapping
                        if similarity >= self.integrated_config["cross_system_mapping_threshold"]:
                            mappings.append({
                                "space1": space1.id,
                                "concept1": concept1_id,
                                "space2": space2.id,
                                "concept2": concept2_id,
                                "similarity": similarity
                            })
                
                # Generate elaboration blend if we have mappings
                if mappings:
                    elaboration_blend = self._generate_elaboration_blend(space1, space2, mappings)
                    
                    if elaboration_blend:
                        blend_id = elaboration_blend["id"]
                        
                        # Add input from third space if available
                        if len(spaces) >= 3:
                            space3 = spaces[2]
                            blend = self.blends[blend_id]
                            
                            # Add supporting concepts from third space
                            for concept3_id, concept3 in space3.concepts.items():
                                # Create blend concept
                                blend_concept_id = f"{blend.id}_concept_{len(blend.concepts) + 1}"
                                
                                # Add concept to blend
                                blend.add_concept(
                                    concept_id=blend_concept_id,
                                    name=concept3["name"],
                                    properties=concept3.get("properties", {}),
                                    source_concepts=[
                                        {"space_id": space3.id, "concept_id": concept3_id}
                                    ]
                                )
                                
                                # Add mapping
                                blend.add_mapping(
                                    input_space_id=space3.id,
                                    input_concept_id=concept3_id,
                                    blend_concept_id=blend_concept_id,
                                    mapping_type="supporting",
                                    mapping_strength=0.8
                                )
            
            # Step 2: Convert blend to causal model
            model_id = None
            if blend_id:
                model_id = await self.convert_blend_to_causal_model(
                    blend_id=blend_id,
                    name=f"Causal model from {base_space.name}"
                )
            else:
                # If no blend was created, just create a causal model directly
                model_id = await self.create_causal_model(
                    name=f"Causal model from {base_space.name}",
                    domain=base_space.domain
                )
                
                # Convert space directly to model
                for concept_id, concept in base_space.concepts.items():
                    # Create causal node
                    node_id = await self.add_node_to_model(
                        model_id=model_id,
                        name=concept["name"],
                        domain=base_space.domain,
                        node_type="concept",
                        metadata={"source_concept": concept_id}
                    )
                    
                    # Add properties as node states
                    causal_node = self.causal_models[model_id].nodes[node_id]
                    
                    for prop_name, prop_value in concept.get("properties", {}).items():
                        if isinstance(prop_value, (str, int, float, bool)):
                            causal_node.add_state(prop_value)
                            
                    # Record mapping
                    self.concept_to_node_mappings[concept_id] = node_id
                    self.node_to_concept_mappings[node_id] = concept_id
                
                # Add relations
                for relation in base_space.relations:
                    source_id = relation["source"]
                    target_id = relation["target"]
                    
                    # Get corresponding node IDs
                    if source_id in self.concept_to_node_mappings and target_id in self.concept_to_node_mappings:
                        node_source_id = self.concept_to_node_mappings[source_id]
                        node_target_id = self.concept_to_node_mappings[target_id]
                        
                        # Determine relation type and strength
                        relation_type = self._map_relation_type_to_causal(relation["type"])
                        strength = relation["strength"]
                        
                        # Add causal relation
                        await self.add_relation_to_model(
                            model_id=model_id,
                            source_id=node_source_id,
                            target_id=node_target_id,
                            relation_type=relation_type,
                            strength=strength,
                            mechanism=f"Derived from {relation['type']} relation in concept space"
                        )
            
            # Step 3: Integrate with existing causal models
            if model_id and causal_models:
                model = self.causal_models[model_id]
                
                # Find strongest causal model
                target_model = max(causal_models, key=lambda m: len(m.nodes))
                
                # Map nodes between models
                node_mappings = {}
                for node_id, node in model.nodes.items():
                    # Find similar node in target model
                    best_match = None
                    best_similarity = 0.0
                    
                    for target_node_id, target_node in target_model.nodes.items():
                        # Calculate name similarity
                        similarity = self._calculate_string_similarity(
                            node.name.lower(),
                            target_node.name.lower()
                        )
                        
                        if similarity > best_similarity and similarity >= self.integrated_config["cross_system_mapping_threshold"]:
                            best_match = target_node_id
                            best_similarity = similarity
                    
                    if best_match:
                        node_mappings[node_id] = best_match
                
                # Copy unique nodes and relations from one model to another
                for node_id, node in model.nodes.items():
                    if node_id not in node_mappings:
                        # Create new node in target model
                        new_node_id = target_model.add_node(
                            name=node.name,
                            domain=node.domain,
                            node_type=node.type,
                            metadata=node.metadata
                        )
                        
                        # Add states
                        target_node = target_model.nodes[new_node_id]
                        for state in node.states:
                            target_node.add_state(state)
                        
                        # Record mapping
                        node_mappings[node_id] = new_node_id
                
                # Add unique relations
                for relation_id, relation in model.relations.items():
                    source_id = relation.source_id
                    target_id = relation.target_id
                    
                    if source_id in node_mappings and target_id in node_mappings:
                        mapped_source = node_mappings[source_id]
                        mapped_target = node_mappings[target_id]
                        
                        # Skip if relation already exists
                        if not target_model.get_relations_between(mapped_source, mapped_target):
                            target_model.add_relation(
                                source_id=mapped_source,
                                target_id=mapped_target,
                                relation_type=relation.type,
                                strength=relation.strength,
                                mechanism=relation.mechanism
                            )
            
            # Step 4: Return result
            nodes_created = len(self.causal_models[model_id].nodes) if model_id else 0
            relations_created = len(self.causal_models[model_id].relations) if model_id else 0
            concepts_created = len(base_space.concepts)
            
            return {
                "causal_model_id": model_id,
                "concept_space_id": base_space.id,
                "nodes_created": nodes_created,
                "concepts_created": concepts_created,
                "relations_created": relations_created
            }
        
    async def create_creative_intervention(self, model_id: str, target_node: str,
                                     description: str = "", 
                                     use_blending: bool = True) -> Dict[str, Any]:
        """Create a creative intervention using conceptual blending and causal reasoning"""
        if model_id not in self.causal_models:
            raise ValueError(f"Model {model_id} not found")
            
        model = self.causal_models[model_id]
        
        if target_node not in model.nodes:
            raise ValueError(f"Target node {target_node} not found in model {model_id}")
        
        target_node_obj = model.nodes[target_node]
        
        # Step 1: Get current state and possible states for target node
        current_state = target_node_obj.get_current_state()
        possible_states = target_node_obj.states.copy()
        
        if current_state in possible_states:
            possible_states.remove(current_state)
        
        # If no states to intervene with, return error
        if not possible_states:
            return {"error": "No alternative states available for target node"}
        
        # Step 2: If blending enabled, use conceptual blending to generate creative intervention
        blend_id = None
        novel_intervention_value = None
        
        if use_blending:
            # Check if we have a concept space for this model
            space_id = None
            
            # If model is already mapped to a concept space, use that
            if model_id in self.model_to_blend_mappings:
                space_id = self.model_to_blend_mappings[model_id]
            else:
                # Create concept space from model
                space_id = await self.convert_causal_model_to_concept_space(
                    model_id=model_id,
                    name=f"Concept space for {model.name}"
                )
            
            # Get space
            space = self.concept_spaces.get(space_id)
            if space:
                # Find concept corresponding to target node
                target_concept_id = None
                
                # Check mappings
                if target_node in self.node_to_concept_mappings:
                    target_concept_id = self.node_to_concept_mappings[target_node]
                else:
                    # Find by name similarity
                    for concept_id, concept in space.concepts.items():
                        if concept["name"] == target_node_obj.name:
                            target_concept_id = concept_id
                            break
                
                if target_concept_id:
                    # Find other spaces to blend with
                    blend_spaces = []
                    for other_space_id, other_space in self.concept_spaces.items():
                        if other_space_id != space_id and (other_space.domain == space.domain 
                                                    or self._domains_are_related(other_space.domain, space.domain)):
                            blend_spaces.append(other_space)
                            if len(blend_spaces) >= 2:  # Limit to 2 additional spaces
                                break
                    
                    if blend_spaces:
                        # Generate contrast blend for creative intervention
                        # First, find concept mappings
                        mappings = []
                        for other_space in blend_spaces:
                            for other_concept_id, other_concept in other_space.concepts.items():
                                # Get target concept
                                target_concept = space.concepts[target_concept_id]
                                
                                # Calculate similarity
                                similarity = self._calculate_concept_similarity(
                                    target_concept, other_concept, space, other_space
                                )
                                
                                # Add mapping if above threshold
                                if similarity >= self.integrated_config["cross_system_mapping_threshold"]:
                                    mappings.append({
                                        "space1": space.id,
                                        "concept1": target_concept_id,
                                        "space2": other_space.id,
                                        "concept2": other_concept_id,
                                        "similarity": similarity,
                                        "mapping_type": "contrast"
                                    })
                        
                        if mappings:
                            # Generate contrast blend
                            other_space = blend_spaces[0]
                            blend_data = self._generate_contrast_blend(space, other_space, mappings)
                            
                            if blend_data:
                                blend_id = blend_data["id"]
                                blend = self.blends[blend_id]
                                
                                # Extract novel state from blend
                                for concept in blend.concepts.values():
                                    # Look for concept related to target
                                    source_concepts = concept.get("source_concepts", [])
                                    
                                    for source in source_concepts:
                                        if source.get("space_id") == space.id and source.get("concept_id") == target_concept_id:
                                            # Found relevant concept
                                            properties = concept.get("properties", {})
                                            
                                            # Look for novel property value that doesn't match any existing state
                                            for prop_value in properties.values():
                                                if (isinstance(prop_value, (str, int, float, bool)) 
                                                        and prop_value not in target_node_obj.states):
                                                    novel_intervention_value = prop_value
                                                    break
                                            
                                            break
                                    
                                    if novel_intervention_value:
                                        break
        
        # Step 3: Create intervention
        intervention_value = novel_intervention_value or random.choice(possible_states)
        intervention_name = f"Creative intervention on {target_node_obj.name}"
        
        if description:
            intervention_name = description
        elif blend_id:
            intervention_name = f"Blend-based intervention on {target_node_obj.name}"
        
        # Create the intervention
        intervention_id = await self.define_intervention(
            model_id=model_id,
            target_node=target_node,
            target_value=intervention_value,
            name=intervention_name,
            description=f"Generated via integrated reasoning system" + 
                       (f" using blend {blend_id}" if blend_id else "")
        )
        
        # Update statistics
        self.stats["creative_interventions"] += 1
        
        # Return intervention details
        return {
            "intervention_id": intervention_id,
            "model_id": model_id,
            "target_node": target_node,
            "target_value": intervention_value,
            "blend_id": blend_id,
            "is_novel": novel_intervention_value is not None
        }
        
    async def perform_integrated_analysis(self, domain: str, 
                                    query: Dict[str, Any]) -> Dict[str, Any]:
        """Perform integrated analysis using both causal and conceptual reasoning"""
        # Find relevant models and spaces
        causal_models = []
        concept_spaces = []
        
        # Find causal models in domain
        for model_id, model in self.causal_models.items():
            if model.domain == domain or self._domains_are_related(model.domain, domain):
                causal_models.append(model)
        
        # Find concept spaces in domain
        for space_id, space in self.concept_spaces.items():
            if space.domain == domain or self._domains_are_related(space.domain, domain):
                concept_spaces.append(space)
        
        # If no inputs, return error
        if not causal_models and not concept_spaces:
            return {"error": f"No causal models or concept spaces found for domain: {domain}"}
        
        # Process query
        query_type = query.get("type", "causal")
        
        if query_type == "causal":
            # Perform causal analysis
            return await self._integrated_causal_analysis(causal_models, concept_spaces, query)
        elif query_type == "conceptual":
            # Perform conceptual analysis
            return await self._integrated_conceptual_analysis(causal_models, concept_spaces, query)
        elif query_type == "counterfactual":
            # Perform counterfactual analysis
            return await self._integrated_counterfactual_analysis(causal_models, concept_spaces, query)
        else:
            return {"error": f"Unknown query type: {query_type}"}
    
    async def _integrated_causal_analysis(self, causal_models: List[CausalModel], 
                                  concept_spaces: List[ConceptSpace],
                                  query: Dict[str, Any]) -> Dict[str, Any]:
        """Perform causal analysis with integrated conceptual reasoning"""
        # If no causal models but concept spaces, convert a space to model
        if not causal_models and concept_spaces:
            # Use first space
            space = concept_spaces[0]
            
            # Convert to causal model
            model_id = await self.convert_causal_model_to_concept_space(
                space_id=space.id,
                name=f"Causal model from {space.name}"
            )
            
            # Get newly created model
            causal_models = [self.causal_models[model_id]]
        
        # If still no models, return error
        if not causal_models:
            return {"error": "No causal models available for analysis"}
        
        # Use model with most nodes
        model = max(causal_models, key=lambda m: len(m.nodes))
        
        # Get query parameters
        target_nodes = query.get("target_nodes", [])
        
        # If no target nodes specified but we have nodes, use central nodes
        if not target_nodes and model.nodes:
            # Find central nodes by degree centrality
            graph = model.graph
            
            try:
                centrality = nx.degree_centrality(graph)
                # Get top 3 nodes by centrality
                target_nodes = sorted(centrality.keys(), key=lambda n: centrality[n], reverse=True)[:3]
            except:
                # If network analysis fails, just use some nodes
                target_nodes = list(model.nodes.keys())[:3]
        
        # Perform causal analysis
        result = {
            "model_id": model.id,
            "model_name": model.name,
            "target_nodes": []
        }
        
        for node_id in target_nodes:
            if node_id in model.nodes:
                node = model.nodes[node_id]
                
                # Get node information
                node_info = {
                    "id": node_id,
                    "name": node.name,
                    "type": node.type,
                    "current_state": node.get_current_state(),
                    "possible_states": node.states,
                    "causes": [],
                    "effects": [],
                    "related_concepts": []
                }
                
                # Get causes (parents)
                parents = list(model.graph.predecessors(node_id))
                for parent_id in parents:
                    parent = model.nodes.get(parent_id)
                    if parent:
                        relations = model.get_relations_between(parent_id, node_id)
                        
                        for relation in relations:
                            node_info["causes"].append({
                                "id": parent_id,
                                "name": parent.name,
                                "relation_type": relation.type,
                                "strength": relation.strength,
                                "mechanism": relation.mechanism
                            })
                
                # Get effects (children)
                children = list(model.graph.successors(node_id))
                for child_id in children:
                    child = model.nodes.get(child_id)
                    if child:
                        relations = model.get_relations_between(node_id, child_id)
                        
                        for relation in relations:
                            node_info["effects"].append({
                                "id": child_id,
                                "name": child.name,
                                "relation_type": relation.type,
                                "strength": relation.strength,
                                "mechanism": relation.mechanism
                            })
                
                # Find related concepts in concept spaces
                if node_id in self.node_to_concept_mappings:
                    concept_id = self.node_to_concept_mappings[node_id]
                    
                    for space in concept_spaces:
                        # Find related concepts
                        if concept_id in space.concepts:
                            related = space.get_related_concepts(concept_id)
                            
                            for related_data in related:
                                related_concept = related_data["concept"]
                                relation = related_data["relation"]
                                
                                node_info["related_concepts"].append({
                                    "id": related_concept["id"],
                                    "name": related_concept["name"],
                                    "space_id": space.id,
                                    "space_name": space.name,
                                    "relation_type": relation["type"],
                                    "strength": relation["strength"]
                                })
                
                result["target_nodes"].append(node_info)
        
        # Update statistics
        self.stats["integrated_analyses"] += 1
        
        return result
    
    async def _integrated_conceptual_analysis(self, causal_models: List[CausalModel], 
                                     concept_spaces: List[ConceptSpace],
                                     query: Dict[str, Any]) -> Dict[str, Any]:
        """Perform conceptual analysis with integrated causal reasoning"""
        # If no concept spaces but causal models, convert a model to space
        if not concept_spaces and causal_models:
            # Use first model
            model = causal_models[0]
            
            # Convert to concept space
            space_id = await self.convert_causal_model_to_concept_space(
                model_id=model.id,
                name=f"Concept space from {model.name}"
            )
            
            # Get newly created space
            concept_spaces = [self.concept_spaces[space_id]]
        
        # If still no spaces, return error
        if not concept_spaces:
            return {"error": "No concept spaces available for analysis"}
        
        # Use space with most concepts
        space = max(concept_spaces, key=lambda s: len(s.concepts))
        
        # Get query parameters
        concept_query = query.get("concept_query", "")
        limit = query.get("limit", 5)
        
        # Find relevant concepts based on query string
        relevant_concepts = []
        
        if concept_query:
            # Find concepts by name or property match
            for concept_id, concept in space.concepts.items():
                name_match = concept_query.lower() in concept["name"].lower()
                
                # Check properties
                property_match = False
                for prop_name, prop_value in concept.get("properties", {}).items():
                    if isinstance(prop_value, str) and concept_query.lower() in prop_value.lower():
                        property_match = True
                        break
                
                if name_match or property_match:
                    relevant_concepts.append(concept_id)
        else:
            # If no query, use random concepts
            concept_ids = list(space.concepts.keys())
            sample_size = min(limit, len(concept_ids))
            relevant_concepts = random.sample(concept_ids, sample_size)
        
        # Perform conceptual analysis
        result = {
            "space_id": space.id,
            "space_name": space.name,
            "concepts": []
        }
        
        for concept_id in relevant_concepts[:limit]:
            concept = space.concepts[concept_id]
            
            # Get concept information
            concept_info = {
                "id": concept_id,
                "name": concept["name"],
                "properties": concept.get("properties", {}),
                "related_concepts": [],
                "causal_factors": []
            }
            
            # Get related concepts
            related = space.get_related_concepts(concept_id)
            
            for related_data in related:
                related_concept = related_data["concept"]
                relation = related_data["relation"]
                
                concept_info["related_concepts"].append({
                    "id": related_concept["id"],
                    "name": related_concept["name"],
                    "relation_type": relation["type"],
                    "strength": relation["strength"]
                })
            
            # Find causal factors in causal models
            if concept_id in self.concept_to_node_mappings:
                node_id = self.concept_to_node_mappings[concept_id]
                
                for model in causal_models:
                    if node_id in model.nodes:
                        # Get node
                        node = model.nodes[node_id]
                        
                        # Get causes (parents)
                        parents = list(model.graph.predecessors(node_id))
                        for parent_id in parents:
                            parent = model.nodes.get(parent_id)
                            if parent:
                                relations = model.get_relations_between(parent_id, node_id)
                                
                                for relation in relations:
                                    concept_info["causal_factors"].append({
                                        "id": parent_id,
                                        "name": parent.name,
                                        "model_id": model.id,
                                        "model_name": model.name,
                                        "relation_type": relation.type,
                                        "strength": relation.strength,
                                        "direction": "cause"
                                    })
                        
                        # Get effects (children)
                        children = list(model.graph.successors(node_id))
                        for child_id in children:
                            child = model.nodes.get(child_id)
                            if child:
                                relations = model.get_relations_between(node_id, child_id)
                                
                                for relation in relations:
                                    concept_info["causal_factors"].append({
                                        "id": child_id,
                                        "name": child.name,
                                        "model_id": model.id,
                                        "model_name": model.name,
                                        "relation_type": relation.type,
                                        "strength": relation.strength,
                                        "direction": "effect"
                                    })
            
            result["concepts"].append(concept_info)
        
        # Update statistics
        self.stats["integrated_analyses"] += 1
        
        return result
    
    async def _integrated_counterfactual_analysis(self, causal_models: List[CausalModel], 
                                         concept_spaces: List[ConceptSpace],
                                         query: Dict[str, Any]) -> Dict[str, Any]:
        """Perform counterfactual analysis with both causal and conceptual reasoning"""
        # Need at least one causal model
        if not causal_models:
            # Try to create from concept space
            if concept_spaces:
                space = concept_spaces[0]
                
                # Convert to causal model
                model_id = await self.convert_blend_to_causal_model(
                    space_id=space.id,
                    name=f"Causal model from {space.name}"
                )
                
                # Get newly created model
                causal_models = [self.causal_models[model_id]]
            else:
                return {"error": "No causal models or concept spaces available for counterfactual analysis"}
        
        # Get model for counterfactual
        model = causal_models[0]
        
        # Get counterfactual parameters
        cf_values = query.get("counterfactual_values", {})
        
        # If no counterfactual values, generate creative ones using conceptual blending
        if not cf_values and self.integrated_config["enable_auto_blending"]:
            # First, generate a concept space if needed
            space_id = None
            
            if model.id in self.model_to_blend_mappings:
                space_id = self.model_to_blend_mappings[model.id]
            else:
                space_id = await self.convert_causal_model_to_concept_space(
                    model_id=model.id,
                    name=f"Concept space from {model.name}"
                )
            
            # Generate blend with other spaces
            if space_id and concept_spaces:
                space = self.concept_spaces[space_id]
                other_spaces = [s for s in concept_spaces if s.id != space_id]
                
                if other_spaces:
                    # Find mappings
                    mappings = []
                    other_space = other_spaces[0]
                    
                    for concept1_id, concept1 in space.concepts.items():
                        for concept2_id, concept2 in other_space.concepts.items():
                            # Calculate similarity
                            similarity = self._calculate_concept_similarity(
                                concept1, concept2, space, other_space
                            )
                            
                            # If above threshold, add mapping
                            if similarity >= self.integrated_config["cross_system_mapping_threshold"]:
                                mappings.append({
                                    "space1": space.id,
                                    "concept1": concept1_id,
                                    "space2": other_space.id,
                                    "concept2": concept2_id,
                                    "similarity": similarity
                                })
                    
                    # Generate contrast blend
                    if mappings:
                        blend_data = self._generate_contrast_blend(space, other_space, mappings)
                        
                        if blend_data:
                            blend_id = blend_data["id"]
                            blend = self.blends[blend_id]
                            
                            # Generate counterfactual values from blend
                            for concept_id, concept in blend.concepts.items():
                                source_concepts = concept.get("source_concepts", [])
                                
                                # Find source concept from model
                                for source in source_concepts:
                                    if source.get("space_id") == space_id:
                                        # Get corresponding causal node
                                        concept_id = source.get("concept_id")
                                        
                                        if concept_id in self.concept_to_node_mappings:
                                            node_id = self.concept_to_node_mappings[concept_id]
                                            
                                            # Find novel property values
                                            node = model.nodes.get(node_id)
                                            
                                            if node:
                                                for prop_name, prop_value in concept.get("properties", {}).items():
                                                    if prop_name.startswith("contrast_") and isinstance(prop_value, (str, int, float, bool)):
                                                        # Add as counterfactual value
                                                        cf_values[node_id] = prop_value
                                                        break
        
        # If still no counterfactual values, generate some from model
        if not cf_values:
            # Find central nodes
            try:
                centrality = nx.degree_centrality(model.graph)
                # Get top 2 nodes by centrality
                central_nodes = sorted(centrality.keys(), key=lambda n: centrality[n], reverse=True)[:2]
                
                # Set counterfactual values
                for node_id in central_nodes:
                    node = model.nodes.get(node_id)
                    if node and node.states:
                        current_state = node.get_current_state()
                        
                        # Choose a different state
                        alternative_states = [s for s in node.states if s != current_state]
                        if alternative_states:
                            cf_values[node_id] = random.choice(alternative_states)
            except:
                # If network analysis fails, just use some random nodes
                node_ids = list(model.nodes.keys())
                
                if node_ids:
                    # Choose a random node
                    node_id = random.choice(node_ids)
                    node = model.nodes.get(node_id)
                    
                    if node and node.states:
                        current_state = node.get_current_state()
                        
                        # Choose a different state
                        alternative_states = [s for s in node.states if s != current_state]
                        if alternative_states:
                            cf_values[node_id] = random.choice(alternative_states)
        
        # Perform counterfactual reasoning
        factual_values = query.get("factual_values", {})
        target_nodes = query.get("target_nodes", [])
        
        result = await self.reason_counterfactually(
            model_id=model.id,
            query={
                "factual_values": factual_values,
                "counterfactual_values": cf_values,
                "target_nodes": target_nodes
            }
        )
        
        # Add conceptual insights if available
        if concept_spaces:
            conceptual_insights = []
            
            # Find impacted nodes from counterfactual
            affected_nodes = set()
            
            for node_id, change in result.get("changes", {}).items():
                if change.get("significant", False):
                    affected_nodes.add(node_id)
            
            # For each affected node, find conceptual insights
            for node_id in affected_nodes:
                # If node has concept mapping, find related concepts
                if node_id in self.node_to_concept_mappings:
                    concept_id = self.node_to_concept_mappings[node_id]
                    
                    for space in concept_spaces:
                        if concept_id in space.concepts:
                            # Get related concepts
                            related = space.get_related_concepts(concept_id)
                            
                            if related:
                                conceptual_insights.append({
                                    "node_id": node_id,
                                    "node_name": model.nodes[node_id].name if node_id in model.nodes else "",
                                    "space_id": space.id,
                                    "space_name": space.name,
                                    "related_concepts": [
                                        {
                                            "id": r["concept"]["id"],
                                            "name": r["concept"]["name"],
                                            "relation_type": r["relation"]["type"],
                                            "strength": r["relation"]["strength"]
                                        }
                                        for r in related[:5]  # Limit to 5 related concepts
                                    ]
                                })
            
            # Add insights to result
            result["conceptual_insights"] = conceptual_insights
        
        # Update statistics
        self.stats["integrated_analyses"] += 1
        
        return result
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the integrated reasoning system"""
        return {
            "timestamp": datetime.now().isoformat(),
            "causal_stats": {
                "models": self.stats["models_created"],
                "nodes": self.stats["nodes_created"],
                "relations": self.stats["relations_created"],
                "interventions": self.stats["interventions_performed"],
                "counterfactuals": self.stats["counterfactuals_analyzed"],
                "discovery_runs": self.stats["discovery_runs"]
            },
            "conceptual_stats": {
                "spaces": self.stats["spaces_created"],
                "concepts": self.stats["concepts_created"],
                "concept_relations": self.stats["concept_relations_created"],
                "blends": self.stats["blends_created"],
                "blend_evaluations": self.stats["blend_evaluations"]
            },
            "integrated_stats": {
                "causal_to_conceptual": self.stats["causal_to_conceptual_conversions"],
                "conceptual_to_causal": self.stats["conceptual_to_causal_conversions"],
                "integrated_analyses": self.stats["integrated_analyses"],
                "creative_interventions": self.stats["creative_interventions"]
            },
            "configuration": {
                "causal": self.causal_config,
                "blending": self.blending_config,
                "integrated": self.integrated_config
            }
        }
    
    async def save_state(self, file_path: str) -> bool:
        """Save current state to file"""
        try:
            state = {
                "causal_models": {model_id: model.to_dict() for model_id, model in self.causal_models.items()},
                "interventions": {int_id: intervention.to_dict() for int_id, intervention in self.interventions.items()},
                "counterfactuals": self.counterfactuals,
                "observations": self.observations,
                "concept_spaces": {space_id: space.to_dict() for space_id, space in self.concept_spaces.items()},
                "blends": {blend_id: blend.to_dict() for blend_id, blend in self.blends.items()},
                "concept_to_node_mappings": self.concept_to_node_mappings,
                "node_to_concept_mappings": self.node_to_concept_mappings,
                "blend_to_model_mappings": self.blend_to_model_mappings,
                "model_to_blend_mappings": self.model_to_blend_mappings,
                "causal_config": self.causal_config,
                "blending_config": self.blending_config,
                "integrated_config": self.integrated_config,
                "stats": self.stats,
                "next_model_id": self.next_model_id,
                "next_intervention_id": self.next_intervention_id,
                "next_counterfactual_id": self.next_counterfactual_id,
                "next_space_id": self.next_space_id,
                "next_blend_id": self.next_blend_id,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving integrated reasoning state: {e}")
            return False
        
    async def load_state(self, file_path: str) -> bool:
        """Load state from file"""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Load causal models
            self.causal_models = {}
            for model_id, model_data in state["causal_models"].items():
                self.causal_models[model_id] = CausalModel.from_dict(model_data)
            
            # Load interventions
            self.interventions = {}
            for int_id, int_data in state["interventions"].items():
                self.interventions[int_id] = Intervention.from_dict(int_data)
            
            # Load concept spaces
            self.concept_spaces = {}
            for space_id, space_data in state["concept_spaces"].items():
                self.concept_spaces[space_id] = ConceptSpace.from_dict(space_data)
            
            # Load blends
            self.blends = {}
            for blend_id, blend_data in state["blends"].items():
                self.blends[blend_id] = ConceptualBlend.from_dict(blend_data)
            
            # Load mappings
            self.concept_to_node_mappings = state["concept_to_node_mappings"]
            self.node_to_concept_mappings = state["node_to_concept_mappings"]
            self.blend_to_model_mappings = state["blend_to_model_mappings"]
            self.model_to_blend_mappings = state["model_to_blend_mappings"]
            
            # Load other attributes
            self.counterfactuals = state["counterfactuals"]
            self.observations = state["observations"]
            self.causal_config = state["causal_config"]
            self.blending_config = state["blending_config"]
            self.integrated_config = state["integrated_config"]
            self.stats = state["stats"]
            self.next_model_id = state["next_model_id"]
            self.next_intervention_id = state["next_intervention_id"]
            self.next_counterfactual_id = state["next_counterfactual_id"]
            self.next_space_id = state["next_space_id"]
            self.next_blend_id = state["next_blend_id"]
            
            return True
        except Exception as e:
            logger.error(f"Error loading integrated reasoning state: {e}")
            return False

def test_general_model_registration():
    """Test that general model can be registered successfully"""
    from nyx.models.general_causal_model import GENERAL_CAUSAL_MODEL
    
    rc = ReasoningCore()
    model_id = rc.add_causal_model("general", GENERAL_CAUSAL_MODEL)
    
    # Check model was added
    assert model_id in rc.causal_models
    assert any(m.domain == "general" for m in rc.causal_models.values())
    
    # Check nodes were created correctly
    model = rc.causal_models[model_id]
    assert len(model.nodes) == len(GENERAL_CAUSAL_MODEL["nodes"])
    
    # Check relations were created
    assert len(model.relations) == len(GENERAL_CAUSAL_MODEL["relations"])
    
    # Test sync doesn't crash
    rc.sync_general_model()
    
    print("✓ General model registration test passed")
