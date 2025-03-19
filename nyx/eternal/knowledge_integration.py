# nyx/eternal/knowledge_integration.py

import asyncio
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import math
import time
import networkx as nx

logger = logging.getLogger(__name__)

class KnowledgeNode:
    """Represents a node in the knowledge graph"""
    
    def __init__(self, id: str, type: str, content: Dict[str, Any], 
                source: str, confidence: float = 0.5, timestamp: Optional[datetime] = None):
        self.id = id
        self.type = type  # concept, fact, rule, hypothesis, etc.
        self.content = content
        self.source = source  # which system generated this knowledge
        self.confidence = confidence  # 0.0 to 1.0
        self.timestamp = timestamp or datetime.now()
        self.last_accessed = self.timestamp
        self.access_count = 0
        self.conflicting_nodes = []
        self.supporting_nodes = []
        self.metadata = {}
        
    def update(self, new_content: Dict[str, Any], 
              new_confidence: Optional[float] = None, 
              source: Optional[str] = None) -> None:
        """Update node with new content and optionally new confidence"""
        self.content.update(new_content)
        if new_confidence is not None:
            self.confidence = new_confidence
        if source:
            self.source = source
        self.last_accessed = datetime.now()
        
    def access(self) -> None:
        """Record an access of this knowledge node"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to a dictionary representation"""
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "conflicting_nodes": self.conflicting_nodes,
            "supporting_nodes": self.supporting_nodes,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeNode':
        """Create a node from dictionary representation"""
        node = cls(
            id=data["id"],
            type=data["type"],
            content=data["content"],
            source=data["source"],
            confidence=data["confidence"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
        node.last_accessed = datetime.fromisoformat(data["last_accessed"])
        node.access_count = data["access_count"]
        node.conflicting_nodes = data["conflicting_nodes"]
        node.supporting_nodes = data["supporting_nodes"]
        node.metadata = data["metadata"]
        return node

class KnowledgeRelation:
    """Represents a relation between knowledge nodes"""
    
    def __init__(self, source_id: str, target_id: str, type: str, 
                weight: float = 1.0, metadata: Dict[str, Any] = None):
        self.source_id = source_id
        self.target_id = target_id
        self.type = type  # supports, contradicts, specializes, similar_to, etc.
        self.weight = weight  # strength of the relation
        self.timestamp = datetime.now()
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert relation to a dictionary representation"""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "weight": self.weight,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeRelation':
        """Create a relation from dictionary representation"""
        relation = cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            type=data["type"],
            weight=data["weight"],
            metadata=data["metadata"]
        )
        relation.timestamp = datetime.fromisoformat(data["timestamp"])
        return relation

class KnowledgeIntegrationSystem:
    """System that integrates knowledge across different components"""
    
    def __init__(self):
        # Initialize graph using NetworkX
        self.graph = nx.DiGraph()
        self.nodes = {}  # id -> KnowledgeNode
        self.node_embeddings = {}  # id -> vector embedding
        
        # Performance tracking
        self.integration_stats = {
            "nodes_added": 0,
            "relations_added": 0,
            "conflicts_resolved": 0,
            "knowledge_queries": 0,
            "integration_cycles": 0
        }
        
        # Configuration
        self.config = {
            "conflict_threshold": 0.7,  # Threshold to consider nodes in conflict
            "support_threshold": 0.7,   # Threshold to consider nodes supporting
            "similarity_threshold": 0.8, # Threshold for considering nodes similar
            "decay_rate": 0.01,         # Rate at which confidence decays for unused nodes
            "integration_cycle_interval": 10,  # Intervals between integration cycles
            "enable_embeddings": True,  # Whether to use vector embeddings
            "max_node_age_days": 30,    # Maximum age for nodes before pruning
            "pruning_confidence_threshold": 0.3  # Nodes below this confidence may be pruned
        }
        
        # Initialize counters
        self.last_integration_cycle = datetime.now()
        self.next_node_id = 1
        
        # Cache for integration
        self.integration_cache = {}
        self.query_cache = {}
        
        # System references
        self.memory_system = None
        self.reasoning_system = None
        
    async def initialize(self, system_references: Dict[str, Any]) -> None:
        """Initialize the knowledge integration system"""
        # Store references to other systems
        if "memory_system" in system_references:
            self.memory_system = system_references["memory_system"]
        if "reasoning_system" in system_references:
            self.reasoning_system = system_references["reasoning_system"]
            
        # Load existing knowledge if available
        await self._load_knowledge()
        
        logger.info("Knowledge Integration System initialized")
    
    async def _load_knowledge(self) -> None:
        """Load existing knowledge from storage"""
        try:
            # This would load from a database or file in a real implementation
            # For now, we'll initialize with empty knowledge
            pass
        except Exception as e:
            logger.error(f"Error loading knowledge: {str(e)}")
    
    async def add_knowledge(self, 
                          type: str, 
                          content: Dict[str, Any], 
                          source: str, 
                          confidence: float = 0.5,
                          relations: List[Dict[str, Any]] = None) -> str:
        """
        Add new knowledge to the integration system
        
        Args:
            type: Type of knowledge (concept, fact, rule, etc.)
            content: Content of the knowledge
            source: Source system that generated this knowledge
            confidence: Confidence level (0.0 to 1.0)
            relations: List of relations to other nodes
            
        Returns:
            ID of the new knowledge node
        """
        # Generate node ID
        node_id = f"node_{self.next_node_id}"
        self.next_node_id += 1
        
        # Create node
        node = KnowledgeNode(
            id=node_id,
            type=type,
            content=content,
            source=source,
            confidence=confidence
        )
        
        # Check for similar existing nodes
        similar_nodes = await self._find_similar_nodes(node)
        
        # If we have very similar nodes, handle integration
        if similar_nodes:
            most_similar_id, similarity = similar_nodes[0]
            if similarity > self.config["similarity_threshold"]:
                # Instead of adding new node, update the existing similar node
                return await self._integrate_with_existing(node, most_similar_id, similarity)
        
        # Add node to graph
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.to_dict())
        
        # Add embeddings if enabled
        if self.config["enable_embeddings"]:
            await self._add_node_embedding(node)
        
        # Add relations if provided
        if relations:
            for relation in relations:
                await self.add_relation(
                    source_id=node_id,
                    target_id=relation["target_id"],
                    type=relation["type"],
                    weight=relation.get("weight", 1.0),
                    metadata=relation.get("metadata", {})
                )
        
        # Add relations to similar nodes
        for similar_id, similarity in similar_nodes:
            if similarity > self.config["similarity_threshold"] * 0.6:  # Relaxed threshold
                await self.add_relation(
                    source_id=node_id,
                    target_id=similar_id,
                    type="similar_to",
                    weight=similarity,
                    metadata={"auto_generated": True}
                )
        
        # Update stats
        self.integration_stats["nodes_added"] += 1
        
        # Check if we should run an integration cycle
        await self._check_integration_cycle()
        
        return node_id
    
    async def add_relation(self, 
                         source_id: str, 
                         target_id: str, 
                         type: str, 
                         weight: float = 1.0,
                         metadata: Dict[str, Any] = None) -> bool:
        """
        Add a relation between two knowledge nodes
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            type: Type of relation
            weight: Strength of the relation
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        # Check if nodes exist
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.warning(f"Cannot add relation: one or both nodes don't exist")
            return False
        
        # Create relation
        relation = KnowledgeRelation(
            source_id=source_id,
            target_id=target_id,
            type=type,
            weight=weight,
            metadata=metadata or {}
        )
        
        # Add to graph
        self.graph.add_edge(
            source_id, 
            target_id, 
            type=type, 
            weight=weight, 
            **relation.to_dict()
        )
        
        # Update supporting/conflicting lists
        if type == "supports":
            if target_id not in self.nodes[source_id].supporting_nodes:
                self.nodes[source_id].supporting_nodes.append(target_id)
        elif type == "contradicts":
            if target_id not in self.nodes[source_id].conflicting_nodes:
                self.nodes[source_id].conflicting_nodes.append(target_id)
            await self._handle_contradiction(source_id, target_id)
        
        # Update stats
        self.integration_stats["relations_added"] += 1
        
        return True
    
    async def _integrate_with_existing(self, 
                                    new_node: KnowledgeNode, 
                                    existing_id: str, 
                                    similarity: float) -> str:
        """Integrate a new node with an existing similar node"""
        existing_node = self.nodes[existing_id]
        
        # Update access counters
        existing_node.access()
        
        # Merge content based on confidence
        if new_node.confidence > existing_node.confidence:
            # New knowledge has higher confidence, prioritize it
            updated_content = new_node.content.copy()
            # Keep any existing content keys that aren't in the new content
            for key, value in existing_node.content.items():
                if key not in updated_content:
                    updated_content[key] = value
                    
            # Update confidence as weighted average
            updated_confidence = (new_node.confidence * 0.7 + existing_node.confidence * 0.3)
            
            # Update the existing node
            existing_node.update(updated_content, updated_confidence, 
                               f"{existing_node.source}+{new_node.source}")
        else:
            # Existing knowledge has higher confidence, but still integrate new details
            for key, value in new_node.content.items():
                if key not in existing_node.content:
                    existing_node.content[key] = value
            
            # Update confidence as weighted average
            updated_confidence = (existing_node.confidence * 0.7 + new_node.confidence * 0.3)
            existing_node.confidence = updated_confidence
        
        # Update the graph
        self.graph.nodes[existing_id].update(existing_node.to_dict())
        
        # Update embeddings if necessary
        if self.config["enable_embeddings"]:
            await self._update_node_embedding(existing_node)
        
        logger.debug(f"Integrated new node into existing node {existing_id}")
        
        return existing_id
    
    async def _handle_contradiction(self, node1_id: str, node2_id: str) -> None:
        """Handle contradiction between two nodes"""
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        
        # If both nodes have high confidence, mark as conflicting
        if node1.confidence > 0.7 and node2.confidence > 0.7:
            # This is a genuine conflict, mark for resolution
            logger.info(f"Detected high-confidence conflict between {node1_id} and {node2_id}")
            
            # Try to resolve with reasoning if available
            if self.reasoning_system:
                try:
                    resolution = await self.reasoning_system.resolve_contradiction(
                        node1.to_dict(), node2.to_dict())
                    
                    if resolution["resolved"]:
                        # Apply resolution
                        if resolution["preferred_node"] == node1_id:
                            node2.confidence *= 0.7  # Reduce confidence of non-preferred node
                        else:
                            node1.confidence *= 0.7
                            
                        # Update stats
                        self.integration_stats["conflicts_resolved"] += 1
                        logger.info(f"Resolved conflict between {node1_id} and {node2_id}")
                    
                except Exception as e:
                    logger.error(f"Error resolving contradiction: {str(e)}")
        
        # For conflicts with clear confidence difference, adjust the lower confidence
        elif abs(node1.confidence - node2.confidence) > 0.3:
            if node1.confidence < node2.confidence:
                node1.confidence *= 0.9  # Slightly decrease confidence of lower node
            else:
                node2.confidence *= 0.9
                
            logger.debug(f"Adjusted confidence in conflict between {node1_id} and {node2_id}")
    
    async def _find_similar_nodes(self, node: KnowledgeNode) -> List[Tuple[str, float]]:
        """Find nodes similar to the given node"""
        similar_nodes = []
        
        if self.config["enable_embeddings"]:
            # Use embeddings for similarity if available
            embedding = await self._generate_embedding(node)
            
            for node_id, node_embedding in self.node_embeddings.items():
                similarity = self._calculate_embedding_similarity(embedding, node_embedding)
                if similarity > self.config["similarity_threshold"] * 0.5:  # Lower threshold for candidates
                    similar_nodes.append((node_id, similarity))
        else:
            # Fallback to keyword matching
            for node_id, existing_node in self.nodes.items():
                if existing_node.type == node.type:
                    similarity = self._calculate_content_similarity(node.content, existing_node.content)
                    if similarity > self.config["similarity_threshold"] * 0.5:
                        similar_nodes.append((node_id, similarity))
        
        # Sort by similarity (descending)
        similar_nodes.sort(key=lambda x: x[1], reverse=True)
        
        return similar_nodes
    
    async def _add_node_embedding(self, node: KnowledgeNode) -> None:
        """Generate and store embedding for a node"""
        embedding = await self._generate_embedding(node)
        self.node_embeddings[node.id] = embedding
    
    async def _update_node_embedding(self, node: KnowledgeNode) -> None:
        """Update embedding for a node"""
        embedding = await self._generate_embedding(node)
        self.node_embeddings[node.id] = embedding
    
    async def _generate_embedding(self, node: KnowledgeNode) -> np.ndarray:
        """Generate embedding vector for a node"""
        # In a real implementation, this would use a language model
        # Here we'll just create a simple mock embedding
        
        # Convert node content to string
        content_str = json.dumps(node.content)
        
        # Create a simple mock embedding (128-dimensional)
        # In a real implementation, this would be a call to an embedding model
        mock_embedding = np.zeros(128)
        
        # Fill with values derived from content string hash
        for i, char in enumerate(content_str):
            position = i % 128
            mock_embedding[position] += ord(char) / 1000
            
        # Normalize
        norm = np.linalg.norm(mock_embedding)
        if norm > 0:
            mock_embedding = mock_embedding / norm
            
        return mock_embedding
    
    def _calculate_embedding_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        return np.dot(embedding1, embedding2)
    
    def _calculate_content_similarity(self, content1: Dict[str, Any], content2: Dict[str, Any]) -> float:
        """Calculate similarity between content dictionaries"""
        # Get all keys
        all_keys = set(content1.keys()) | set(content2.keys())
        if not all_keys:
            return 0.0
            
        # Count matching keys and values
        matching_keys = set(content1.keys()) & set(content2.keys())
        matching_values = sum(1 for k in matching_keys if content1[k] == content2[k])
        
        # Calculate Jaccard similarity for keys
        key_similarity = len(matching_keys) / len(all_keys)
        
        # Calculate value similarity for matching keys
        value_similarity = matching_values / len(matching_keys) if matching_keys else 0.0
        
        # Combine similarities
        return 0.7 * key_similarity + 0.3 * value_similarity
    
    async def _check_integration_cycle(self) -> None:
        """Check if we should run an integration cycle"""
        now = datetime.now()
        time_since_last = (now - self.last_integration_cycle).total_seconds()
        
        if time_since_last > self.config["integration_cycle_interval"] * 60:  # Convert to seconds
            await self._run_integration_cycle()
    
    async def _run_integration_cycle(self) -> None:
        """Run a full integration cycle to optimize knowledge"""
        logger.info("Running knowledge integration cycle")
        
        # Update timestamp
        self.last_integration_cycle = datetime.now()
        self.integration_stats["integration_cycles"] += 1
        
        # 1. Prune old or low-confidence nodes
        await self._prune_nodes()
        
        # 2. Identify and resolve conflicts
        await self._resolve_conflicts()
        
        # 3. Consolidate similar nodes
        await self._consolidate_similar_nodes()
        
        # 4. Infer new relations
        await self._infer_relations()
        
        # 5. Apply confidence decay
        self._apply_confidence_decay()
        
        # 6. Update knowledge stats
        await self._update_knowledge_stats()
        
        logger.info("Knowledge integration cycle completed")
    
    async def _prune_nodes(self) -> None:
        """Prune old or low-confidence nodes"""
        now = datetime.now()
        nodes_to_prune = []
        
        for node_id, node in self.nodes.items():
            # Check age
            age_days = (now - node.timestamp).total_seconds() / (24 * 3600)
            
            # Check conditions for pruning
            if (age_days > self.config["max_node_age_days"] and 
                node.confidence < self.config["pruning_confidence_threshold"] and
                node.access_count < 3):
                nodes_to_prune.append(node_id)
        
        # Prune nodes
        for node_id in nodes_to_prune:
            await self._remove_node(node_id)
            
        if nodes_to_prune:
            logger.info(f"Pruned {len(nodes_to_prune)} old or low-confidence nodes")
    
    async def _remove_node(self, node_id: str) -> None:
        """Remove a node and its relations"""
        if node_id in self.nodes:
            # Remove from nodes dict
            del self.nodes[node_id]
            
            # Remove from graph
            self.graph.remove_node(node_id)
            
            # Remove embeddings
            if node_id in self.node_embeddings:
                del self.node_embeddings[node_id]
            
            # Clean up references in other nodes
            for other_node in self.nodes.values():
                if node_id in other_node.supporting_nodes:
                    other_node.supporting_nodes.remove(node_id)
                if node_id in other_node.conflicting_nodes:
                    other_node.conflicting_nodes.remove(node_id)
    
    async def _resolve_conflicts(self) -> None:
        """Resolve conflicts between nodes"""
        # Get all contradicting relations
        contradictions = []
        for source, target, data in self.graph.edges(data=True):
            if data.get("type") == "contradicts":
                contradictions.append((source, target))
        
        for source_id, target_id in contradictions:
            # Skip if either node has been removed
            if source_id not in self.nodes or target_id not in self.nodes:
                continue
                
            await self._handle_contradiction(source_id, target_id)
    
    async def _consolidate_similar_nodes(self) -> None:
        """Consolidate very similar nodes"""
        # Find similar nodes using embeddings
        if not self.config["enable_embeddings"] or len(self.nodes) < 2:
            return
            
        # Get all nodes with embeddings
        node_ids = list(self.node_embeddings.keys())
        
        # Check for nodes to merge
        nodes_to_merge = []
        
        # Compare all pairs (limit to 1000 comparisons for performance)
        max_comparisons = min(1000, len(node_ids) * (len(node_ids) - 1) // 2)
        comparison_count = 0
        
        for i in range(len(node_ids)):
            for j in range(i+1, len(node_ids)):
                # Check comparison limit
                comparison_count += 1
                if comparison_count > max_comparisons:
                    break
                    
                id1 = node_ids[i]
                id2 = node_ids[j]
                
                # Skip if either node doesn't exist
                if id1 not in self.nodes or id2 not in self.nodes:
                    continue
                
                # Skip if nodes are already in conflict
                if id2 in self.nodes[id1].conflicting_nodes:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_embedding_similarity(
                    self.node_embeddings[id1], self.node_embeddings[id2])
                
                # If very similar, mark for merging
                if similarity > self.config["similarity_threshold"]:
                    nodes_to_merge.append((id1, id2, similarity))
        
        # Sort by similarity (descending)
        nodes_to_merge.sort(key=lambda x: x[2], reverse=True)
        
        # Merge similar nodes (maintain a set of already merged nodes)
        merged_nodes = set()
        merges_performed = 0
        
        for id1, id2, similarity in nodes_to_merge:
            # Skip if either node has already been merged or doesn't exist anymore
            if id1 in merged_nodes or id2 in merged_nodes:
                continue
            if id1 not in self.nodes or id2 not in self.nodes:
                continue
            
            # Choose higher confidence node as target
            if self.nodes[id1].confidence >= self.nodes[id2].confidence:
                target_id, source_id = id1, id2
            else:
                target_id, source_id = id2, id1
            
            # Merge nodes
            await self._merge_nodes(target_id, source_id)
            
            # Update merged set
            merged_nodes.add(source_id)
            merges_performed += 1
        
        if merges_performed > 0:
            logger.info(f"Consolidated {merges_performed} similar nodes")
    
    async def _merge_nodes(self, target_id: str, source_id: str) -> None:
        """Merge source node into target node"""
        target_node = self.nodes[target_id]
        source_node = self.nodes[source_id]
        
        # Combine content
        for key, value in source_node.content.items():
            if key not in target_node.content:
                target_node.content[key] = value
        
        # Update metadata
        target_node.metadata["merged_from"] = target_node.metadata.get("merged_from", []) + [source_id]
        target_node.metadata["merged_timestamp"] = datetime.now().isoformat()
        
        # Update confidence (weighted average)
        combined_confidence = (target_node.confidence * 0.7 + source_node.confidence * 0.3)
        target_node.confidence = min(1.0, combined_confidence)
        
        # Update source
        if source_node.source != target_node.source:
            target_node.source = f"{target_node.source}+{source_node.source}"
        
        # Transfer relations from source to target
        # Get all edges from source
        for successor in list(self.graph.successors(source_id)):
            if successor != target_id:  # Avoid self-loops
                edge_data = self.graph.get_edge_data(source_id, successor)
                # Add edge from target to successor
                self.graph.add_edge(target_id, successor, **edge_data)
        
        # Get all edges to source
        for predecessor in list(self.graph.predecessors(source_id)):
            if predecessor != target_id:  # Avoid self-loops
                edge_data = self.graph.get_edge_data(predecessor, source_id)
                # Add edge from predecessor to target
                self.graph.add_edge(predecessor, target_id, **edge_data)
        
        # Update supporting/conflicting lists
        for node_id in source_node.supporting_nodes:
            if node_id != target_id and node_id not in target_node.supporting_nodes:
                target_node.supporting_nodes.append(node_id)
        
        for node_id in source_node.conflicting_nodes:
            if node_id != target_id and node_id not in target_node.conflicting_nodes:
                target_node.conflicting_nodes.append(node_id)
        
        # Update graph node data
        self.graph.nodes[target_id].update(target_node.to_dict())
        
        # Remove source node
        await self._remove_node(source_id)
    
    async def _infer_relations(self) -> None:
        """Infer new relations between nodes"""
        # This would use logical inference rules in a real implementation
        # Here we'll implement a simple transitive inference
        
        # Find potential transitive relations (A relates to B, B relates to C => A relates to C)
        inferred_relations = []
        
        for node1_id in self.nodes:
            # Get all successors (nodes that node1 has relations to)
            for node2_id in self.graph.successors(node1_id):
                # Get relation type
                relation1_type = self.graph.get_edge_data(node1_id, node2_id)["type"]
                
                # Get all successors of node2
                for node3_id in self.graph.successors(node2_id):
                    # Skip if node3 is node1
                    if node3_id == node1_id:
                        continue
                        
                    # Get relation type
                    relation2_type = self.graph.get_edge_data(node2_id, node3_id)["type"]
                    
                    # Check if we already have a direct relation
                    if self.graph.has_edge(node1_id, node3_id):
                        continue
                    
                    # Infer relation type based on relation composition
                    inferred_type = self._infer_relation_type(relation1_type, relation2_type)
                    if inferred_type:
                        inferred_relations.append((node1_id, node3_id, inferred_type))
        
        # Add inferred relations
        for source_id, target_id, relation_type in inferred_relations:
            # Verify nodes still exist
            if source_id in self.nodes and target_id in self.nodes:
                await self.add_relation(
                    source_id=source_id,
                    target_id=target_id,
                    type=relation_type,
                    weight=0.7,  # Lower confidence for inferred relations
                    metadata={"inferred": True}
                )
                
        if inferred_relations:
            logger.info(f"Inferred {len(inferred_relations)} new relations")
    
    def _infer_relation_type(self, relation1_type: str, relation2_type: str) -> Optional[str]:
        """Infer the type of relation based on two existing relations"""
        # Define relation composition rules
        composition_rules = {
            # Transitive supports: A supports B, B supports C => A supports C
            ("supports", "supports"): "supports",
            
            # Conflicting support: A supports B, B contradicts C => A contradicts C
            ("supports", "contradicts"): "contradicts",
            
            # Conflicting support (reversed): A contradicts B, B supports C => A contradicts C
            ("contradicts", "supports"): "contradicts",
            
            # Double contradiction: A contradicts B, B contradicts C => A supports C
            ("contradicts", "contradicts"): "supports",
            
            # Transitive specialization: A specializes B, B specializes C => A specializes C
            ("specializes", "specializes"): "specializes",
            
            # Inheritance of properties: A has_property B, B specializes C => A has_property C
            ("has_property", "specializes"): "has_property"
        }
        
        return composition_rules.get((relation1_type, relation2_type))
    
    def _apply_confidence_decay(self) -> None:
        """Apply confidence decay to unused nodes"""
        now = datetime.now()
        
        for node in self.nodes.values():
            # Calculate days since last access
            days_since_access = (now - node.last_accessed).total_seconds() / (24 * 3600)
            
            # Apply decay if not accessed recently
            if days_since_access > 7:  # One week threshold
                decay_factor = 1.0 - (self.config["decay_rate"] * min(days_since_access / 30, 1.0))
                node.confidence *= decay_factor
                
                # Update node in graph
                self.graph.nodes[node.id].update(node.to_dict())
    
    async def _update_knowledge_stats(self) -> None:
        """Update knowledge statistics"""
        # Count nodes by type
        node_types = {}
        for node in self.nodes.values():
            node_types[node.type] = node_types.get(node.type, 0) + 1
        
        # Count relations by type
        relation_types = {}
        for source, target, data in self.graph.edges(data=True):
            rel_type = data.get("type")
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
        
        # Update stats
        self.integration_stats["node_count"] = len(self.nodes)
        self.integration_stats["relation_count"] = self.graph.number_of_edges()
        self.integration_stats["node_types"] = node_types
        self.integration_stats["relation_types"] = relation_types
    
    async def query_knowledge(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph
        
        Args:
            query: Query specification
                - type: optional node type to filter by
                - content_filter: dictionary of key-value pairs to match
                - relation_filter: filter based on relations
                - limit: maximum number of results to return
                
        Returns:
            List of matching knowledge nodes
        """
        # Update stats
        self.integration_stats["knowledge_queries"] += 1
        
        # Extract query parameters
        node_type = query.get("type")
        content_filter = query.get("content_filter", {})
        relation_filter = query.get("relation_filter", {})
        limit = query.get("limit", 10)
        
        # Build cache key
        cache_key = json.dumps({
            "type": node_type,
            "content_filter": content_filter,
            "relation_filter": relation_filter,
            "limit": limit
        })
        
        # Check cache
        if cache_key in self.query_cache:
            cache_entry = self.query_cache[cache_key]
            if (datetime.now() - cache_entry["timestamp"]).total_seconds() < 60:  # 1 minute cache
                return cache_entry["results"]
        
        # Filter nodes
        matching_nodes = []
        
        for node_id, node in self.nodes.items():
            # Filter by type
            if node_type and node.type != node_type:
                continue
                
            # Filter by content
            content_match = True
            for key, value in content_filter.items():
                if key not in node.content or node.content[key] != value:
                    content_match = False
                    break
            
            if not content_match:
                continue
                
            # Filter by relations
            relation_match = True
            if relation_filter:
                relation_type = relation_filter.get("type")
                related_node_id = relation_filter.get("node_id")
                direction = relation_filter.get("direction", "outgoing")
                
                if relation_type and related_node_id:
                    if direction == "outgoing":
                        # Check if this node has outgoing relation to related_node
                        if not self.graph.has_edge(node_id, related_node_id):
                            relation_match = False
                        else:
                            edge_data = self.graph.get_edge_data(node_id, related_node_id)
                            if edge_data["type"] != relation_type:
                                relation_match = False
                    else:  # incoming
                        # Check if this node has incoming relation from related_node
                        if not self.graph.has_edge(related_node_id, node_id):
                            relation_match = False
                        else:
                            edge_data = self.graph.get_edge_data(related_node_id, node_id)
                            if edge_data["type"] != relation_type:
                                relation_match = False
            
            if not relation_match:
                continue
                
            # This node matches all filters
            matching_nodes.append(node.to_dict())
            
            # Update access timestamp
            node.access()
            
            # Update node in graph
            self.graph.nodes[node_id].update(node.to_dict())
        
        # Sort by confidence (descending)
        matching_nodes.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Apply limit
        results = matching_nodes[:limit]
        
        # Update cache
        self.query_cache[cache_key] = {
            "timestamp": datetime.now(),
            "results": results
        }
        
        return results
    
    async def get_related_knowledge(self, node_id: str, relation_type: Optional[str] = None, 
                                 direction: str = "both", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get knowledge nodes related to a specific node
        
        Args:
            node_id: ID of the node to get relations for
            relation_type: Optional type of relation to filter by
            direction: Direction of relations ('incoming', 'outgoing', or 'both')
            limit: Maximum number of results to return
            
        Returns:
            List of related knowledge nodes
        """
        if node_id not in self.nodes:
            return []
            
        # Update access timestamp for the node
        self.nodes[node_id].access()
        
        related_nodes = []
        
        # Get outgoing relations
        if direction in ["outgoing", "both"]:
            for target_id in self.graph.successors(node_id):
                edge_data = self.graph.get_edge_data(node_id, target_id)
                if relation_type is None or edge_data["type"] == relation_type:
                    if target_id in self.nodes:
                        related_nodes.append({
                            "node": self.nodes[target_id].to_dict(),
                            "relation": {
                                "type": edge_data["type"],
                                "direction": "outgoing",
                                "weight": edge_data["weight"]
                            }
                        })
        
        # Get incoming relations
        if direction in ["incoming", "both"]:
            for source_id in self.graph.predecessors(node_id):
                edge_data = self.graph.get_edge_data(source_id, node_id)
                if relation_type is None or edge_data["type"] == relation_type:
                    if source_id in self.nodes:
                        related_nodes.append({
                            "node": self.nodes[source_id].to_dict(),
                            "relation": {
                                "type": edge_data["type"],
                                "direction": "incoming",
                                "weight": edge_data["weight"]
                            }
                        })
        
        # Sort by relation weight (descending)
        related_nodes.sort(key=lambda x: x["relation"]["weight"], reverse=True)
        
        # Apply limit
        return related_nodes[:limit]
    
    async def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        return {
            "timestamp": datetime.now().isoformat(),
            "node_count": len(self.nodes),
            "edge_count": self.graph.number_of_edges(),
            "node_types": self.integration_stats.get("node_types", {}),
            "relation_types": self.integration_stats.get("relation_types", {}),
            "nodes_added": self.integration_stats["nodes_added"],
            "relations_added": self.integration_stats["relations_added"],
            "conflicts_resolved": self.integration_stats["conflicts_resolved"],
            "knowledge_queries": self.integration_stats["knowledge_queries"],
