# nyx/eternal/conceptual_blending_system.py

import asyncio
import json
import logging
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import math
import re
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

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

class ConceptualBlendingSystem:
    """System for creative conceptual blending and novel idea generation"""
    
    def __init__(self, knowledge_system=None):
        self.knowledge_system = knowledge_system
        self.concept_spaces = {}  # id -> ConceptSpace
        self.blends = {}  # id -> ConceptualBlend
        self.evaluation_metrics = {
            "novelty": self._evaluate_novelty,
            "coherence": self._evaluate_coherence,
            "practicality": self._evaluate_practicality,
            "expressiveness": self._evaluate_expressiveness,
            "surprise": self._evaluate_surprise
        }
        
        # Configuration
        self.config = {
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
        
        # System state
        self.next_space_id = 1
        self.next_blend_id = 1
    
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
    
    async def generate_novel_concepts(self, domain: str, 
                              constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate novel concepts through conceptual blending"""
        # Retrieve relevant concept spaces
        spaces = await self._retrieve_concept_spaces(domain)
        
        if len(spaces) < 2:
            return {"error": "Not enough concept spaces found for blending"}
            
        # Identify potential mappings between spaces
        mappings = self._identify_mappings(spaces)
        
        # Generate potential blends
        candidate_blends = self._generate_candidate_blends(spaces, mappings)
        
        # Apply constraints and filters
        filtered_blends = self._apply_constraints(candidate_blends, constraints)
        
        # Elaborate and develop promising blends
        developed_blends = await self._elaborate_blends(filtered_blends)
        
        # Evaluate novelty and utility
        evaluated_blends = self._evaluate_blends(developed_blends)
        
        # Sort by overall score
        sorted_blends = sorted(evaluated_blends, key=lambda x: x["overall_score"], reverse=True)
        
        return {
            "blends": sorted_blends,
            "count": len(sorted_blends),
            "domain": domain,
            "constraints": constraints
        }
    
    async def _retrieve_concept_spaces(self, domain: str) -> List[ConceptSpace]:
        """Retrieve concept spaces relevant to the domain"""
        # Get spaces with matching domain
        domain_spaces = [space for space in self.concept_spaces.values() 
                       if space.domain == domain]
        
        # If not enough spaces with exact domain match, find related domains
        if len(domain_spaces) < 2:
            # Try to find spaces with related domains
            for space in self.concept_spaces.values():
                if space.domain != domain and space not in domain_spaces:
                    # Check if domain is related
                    if self._domains_are_related(space.domain, domain):
                        domain_spaces.append(space)
                        
                        # Stop once we have enough spaces
                        if len(domain_spaces) >= 4:
                            break
        
        # If still not enough, add some random spaces for cross-domain blending
        if len(domain_spaces) < 2:
            other_spaces = [space for space in self.concept_spaces.values() 
                          if space not in domain_spaces]
            
            if other_spaces:
                # Add random spaces up to a maximum of 4 total spaces
                num_to_add = min(4 - len(domain_spaces), len(other_spaces))
                if num_to_add > 0:
                    random_spaces = random.sample(other_spaces, num_to_add)
                    domain_spaces.extend(random_spaces)
        
        return domain_spaces
    
    def _domains_are_related(self, domain1: str, domain2: str) -> bool:
        """Check if two domains are related"""
        # Simple string comparison for now
        if not domain1 or not domain2:
            return False
            
        # Check if one is a subdomain of the other
        if domain1.startswith(domain2) or domain2.startswith(domain1):
            return True
            
        # Check if they share common words
        words1 = set(re.findall(r'\w+', domain1.lower()))
        words2 = set(re.findall(r'\w+', domain2.lower()))
        
        return len(words1.intersection(words2)) > 0
    
    def _identify_mappings(self, spaces: List[ConceptSpace]) -> Dict[str, List[Dict[str, Any]]]:
        """Identify potential mappings between concept spaces"""
        mappings = {}
        
        # For each pair of spaces
        for i, space1 in enumerate(spaces):
            for j, space2 in enumerate(spaces):
                if i >= j:  # Skip same space and duplicates
                    continue
                    
                space_pair = f"{space1.id}_{space2.id}"
                mappings[space_pair] = []
                
                # For each concept in space1
                for concept1_id, concept1 in space1.concepts.items():
                    # For each concept in space2
                    for concept2_id, concept2 in space2.concepts.items():
                        # Calculate similarity
                        similarity = self._calculate_concept_similarity(
                            concept1, concept2, space1, space2
                        )
                        
                        # If similarity is above threshold, add mapping
                        if similarity >= self.config["default_mapping_threshold"]:
                            mappings[space_pair].append({
                                "space1": space1.id,
                                "concept1": concept1_id,
                                "space2": space2.id,
                                "concept2": concept2_id,
                                "similarity": similarity,
                                "mapping_type": "property_similarity"
                            })
                
                # Sort mappings by similarity
                mappings[space_pair].sort(key=lambda x: x["similarity"], reverse=True)
        
        return mappings
    
    def _calculate_concept_similarity(self, concept1: Dict[str, Any], 
                                    concept2: Dict[str, Any],
                                    space1: ConceptSpace, 
                                    space2: ConceptSpace) -> float:
        """Calculate similarity between two concepts"""
        # Name similarity
        name_similarity = self._calculate_string_similarity(
            concept1["name"].lower(), 
            concept2["name"].lower()
        )
        
        # Property similarity
        props1 = concept1.get("properties", {})
        props2 = concept2.get("properties", {})
        
        property_similarity = self._calculate_property_similarity(props1, props2)
        
        # Combine similarities
        return 0.3 * name_similarity + 0.7 * property_similarity
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Levenshtein distance"""
        if not str1 or not str2:
            return 0.0
            
        # Calculate Levenshtein distance
        m, n = len(str1), len(str2)
        
        # Create distance matrix
        d = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Initialize first row and column
        for i in range(m + 1):
            d[i][0] = i
        for j in range(n + 1):
            d[0][j] = j
            
        # Fill distance matrix
        for j in range(1, n + 1):
            for i in range(1, m + 1):
                if str1[i-1] == str2[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(
                        d[i-1][j] + 1,  # deletion
                        d[i][j-1] + 1,  # insertion
                        d[i-1][j-1] + 1  # substitution
                    )
        
        # Calculate similarity from distance
        max_len = max(m, n)
        if max_len == 0:
            return 1.0
            
        distance = d[m][n]
        similarity = 1.0 - (distance / max_len)
        
        return similarity
    
    def _calculate_property_similarity(self, props1: Dict[str, Any], 
                                    props2: Dict[str, Any]) -> float:
        """Calculate similarity between two property sets"""
        if not props1 or not props2:
            return 0.0
            
        # Find common properties
        common_props = set(props1.keys()).intersection(set(props2.keys()))
        
        if not common_props:
            return 0.0
            
        # Calculate similarity for each common property
        prop_similarities = []
        
        for prop in common_props:
            val1 = props1[prop]
            val2 = props2[prop]
            
            # Calculate value similarity based on type
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    prop_similarity = 1.0 - (abs(val1 - val2) / max_val)
                else:
                    prop_similarity = 1.0
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity
                prop_similarity = self._calculate_string_similarity(val1.lower(), val2.lower())
            elif val1 == val2:
                # Exact match for other types
                prop_similarity = 1.0
            else:
                # Different values
                prop_similarity = 0.0
                
            prop_similarities.append(prop_similarity)
        
        # Calculate average similarity
        avg_similarity = sum(prop_similarities) / len(prop_similarities)
        
        # Adjust for proportion of matching properties
        total_props = len(set(props1.keys()).union(set(props2.keys())))
        proportion_matching = len(common_props) / total_props
        
        return avg_similarity * proportion_matching
    
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
            for i in range(self.config["elaboration_iterations"]):
                # 1. Add detail to concepts
                for concept_id, concept in blend.concepts.items():
                    self._elaborate_concept(blend, concept_id, i)
                
                # 2. Add new relations
                self._elaborate_relations(blend, i)
                
                # 3. Identify new emergent structure
                if i == self.config["elaboration_iterations"] - 1:  # Only on last iteration
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
            
            for metric_name, metric_func in self.evaluation_metrics.items():
                evaluation[metric_name] = metric_func(blend)
            
            # Calculate overall score
            overall_score = 0.0
            weights = self.config["evaluation_weights"]
            
            for metric, score in evaluation.items():
                weight = weights.get(metric, 0.2)  # Default weight
                overall_score += score * weight
            
            # Update blend evaluation
            blend.update_evaluation(evaluation)
            blend.evaluation["overall_score"] = overall_score
            
            # Update the blend in storage
            self.blends[blend_id] = blend
            
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
