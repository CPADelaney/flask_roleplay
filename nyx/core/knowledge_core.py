# nyx/core/knowledge_core.py

"""
KnowledgeCore refactored with OpenAI Agent SDK.
This module provides an agentic framework for knowledge management, integrating
the original KnowledgeCore functionality with the OpenAI Agent SDK architecture.
"""

import asyncio
import json
import logging
import os
import re
import math
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Annotated
import random
from collections import Counter

from agents import Agent, Runner, function_tool, handoff, FunctionTool, InputGuardrail, GuardrailFunctionOutput, ModelSettings, trace, RunContextWrapper
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)



class KnowledgeQuery(BaseModel):
    type: Optional[str] = Field(
        None, description="If set, only return nodes of this type"
    )
    content_filter: Optional[Dict[str, Any]] = Field(
        None, description="Key/value pairs that node.content must match"
    )
    relation_filter: Optional[Dict[str, Any]] = Field(
        None,
        description=("Optional relationship constraints: "
                     "{type, node_id, direction ('outgoing'|'incoming')}")
    )
    limit: int = Field(
        10, ge=1, le=100, description="Maximum number of results"
    )

# Keep the original data models
class KnowledgeNode:
    """Represents a node in the knowledge graph, storing a piece of knowledge."""
    
    def __init__(self,
                 id: str,
                 type: str,
                 content: Dict[str, Any],
                 source: str,
                 confidence: float = 0.5,
                 timestamp: Optional[datetime] = None):
        """
        :param id: Unique ID for the node.
        :param type: The type/category (e.g. 'concept', 'fact', 'rule', 'hypothesis').
        :param content: Arbitrary dictionary storing the knowledge data.
        :param source: Which system or user provided this knowledge.
        :param confidence: Float in [0.0..1.0] indicating how reliable or certain this knowledge is.
        :param timestamp: Creation timestamp. Defaults to now if None.
        """
        self.id = id
        self.type = type
        self.content = content
        self.source = source
        self.confidence = confidence
        self.timestamp = timestamp or datetime.now()
        self.last_accessed = self.timestamp
        self.access_count = 0

        # Track nodes that conflict or support this node
        self.conflicting_nodes: List[str] = []
        self.supporting_nodes: List[str] = []

        # Arbitrary metadata (e.g., merged_from info, domain tags, etc.)
        self.metadata: Dict[str, Any] = {}

    def update(self,
               new_content: Dict[str, Any],
               new_confidence: Optional[float] = None,
               source: Optional[str] = None) -> None:
        """Merge new content into the node's existing content, optionally adjusting confidence and source."""
        self.content.update(new_content)
        if new_confidence is not None:
            self.confidence = new_confidence
        if source:
            self.source = source
        self.last_accessed = datetime.now()

    def access(self) -> None:
        """Record an access operation on the node, for usage statistics or confidence decay logic."""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert this node to a dictionary for JSON serialization or logging."""
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
        """Create a KnowledgeNode from a dictionary (e.g., loading from JSON)."""
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
    """
    Represents a directed relation (edge) between two knowledge nodes in the graph.
    """

    def __init__(self,
                 source_id: str,
                 target_id: str,
                 type: str,
                 weight: float = 1.0,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        :param source_id: ID of the node that is the source of the relation (directed edge).
        :param target_id: ID of the node that is the target of the relation (directed edge).
        :param type: The type/category of the relation (e.g. 'supports', 'contradicts', 'similar_to').
        :param weight: Float representing strength or importance of the relation.
        :param metadata: Arbitrary dictionary storing extra info about the relation.
        """
        self.source_id = source_id
        self.target_id = target_id
        self.type = type
        self.weight = weight
        self.timestamp = datetime.now()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert this relation to a dictionary for JSON serialization or logging."""
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
        """Create a KnowledgeRelation from a dictionary (e.g., loading from JSON)."""
        relation = cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            type=data["type"],
            weight=data["weight"],
            metadata=data["metadata"]
        )
        relation.timestamp = datetime.fromisoformat(data["timestamp"])
        return relation


class KnowledgeMap:
    """Maps knowledge domains and gaps"""
    
    def __init__(self):
        self.domains = {}  # domain -> {topic -> level}
        self.connections = {}  # (domain1, topic1) -> [(domain2, topic2, strength), ...]
        self.importance_levels = {}  # (domain, topic) -> importance
        self.last_updated = {}  # (domain, topic) -> datetime
        
    def add_knowledge_to_map(self, domain: str, topic: str, 
                    level: float = 0.0, importance: float = 0.5) -> None:
        """Add or update knowledge in the map"""
        # Ensure domain exists
        if domain not in self.domains:
            self.domains[domain] = {}
            
        # Update knowledge level
        self.domains[domain][topic] = level
        
        # Update importance and timestamp
        key = (domain, topic)
        self.importance_levels[key] = importance
        self.last_updated[key] = datetime.now()
    
    def add_connection(self, domain1: str, topic1: str, 
                     domain2: str, topic2: str, 
                     strength: float = 0.5) -> None:
        """Add a connection between knowledge topics"""
        # Create key for first topic
        key1 = (domain1, topic1)
        
        # Ensure key exists in connections
        if key1 not in self.connections:
            self.connections[key1] = []
            
        # Add connection
        connection = (domain2, topic2, strength)
        if connection not in self.connections[key1]:
            self.connections[key1].append(connection)
            
        # Add reverse connection
        key2 = (domain2, topic2)
        if key2 not in self.connections:
            self.connections[key2] = []
            
        reverse_connection = (domain1, topic1, strength)
        if reverse_connection not in self.connections[key2]:
            self.connections[key2].append(reverse_connection)
    
    def get_knowledge_level(self, domain: str, topic: str) -> float:
        """Get knowledge level for a domain/topic"""
        if domain in self.domains and topic in self.domains[domain]:
            return self.domains[domain][topic]
        return 0.0
    
    def get_importance(self, domain: str, topic: str) -> float:
        """Get importance level for a domain/topic"""
        key = (domain, topic)
        return self.importance_levels.get(key, 0.5)
    
    def get_knowledge_gaps(self) -> List[Tuple[str, str, float, float]]:
        """Get all knowledge gaps (domain, topic, level, importance)"""
        gaps = []
        
        for domain, topics in self.domains.items():
            for topic, level in topics.items():
                if level < 0.7:  # Consider anything below 0.7 as a gap
                    gap_size = 1.0 - level
                    importance = self.get_importance(domain, topic)
                    gaps.append((domain, topic, gap_size, importance))
        
        # Sort by gap_size * importance (descending)
        gaps.sort(key=lambda x: x[2] * x[3], reverse=True)
        
        return gaps
    
    def get_related_topics(self, domain: str, topic: str) -> List[Tuple[str, str, float]]:
        """Get topics related to a given topic"""
        key = (domain, topic)
        return self.connections.get(key, [])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "domains": {domain: dict(topics) for domain, topics in self.domains.items()},
            "connections": {f"{k[0]}|{k[1]}": v for k, v in self.connections.items()},
            "importance_levels": {f"{k[0]}|{k[1]}": v for k, v in self.importance_levels.items()},
            "last_updated": {f"{k[0]}|{k[1]}": v.isoformat() for k, v in self.last_updated.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeMap':
        """Create from dictionary representation"""
        knowledge_map = cls()
        
        # Load domains
        for domain, topics in data["domains"].items():
            knowledge_map.domains[domain] = topics
        
        # Load connections
        for key_str, connections in data["connections"].items():
            domain, topic = key_str.split("|")
            knowledge_map.connections[(domain, topic)] = connections
        
        # Load importance levels
        for key_str, importance in data["importance_levels"].items():
            domain, topic = key_str.split("|")
            knowledge_map.importance_levels[(domain, topic)] = importance
        
        # Load timestamps
        for key_str, timestamp in data["last_updated"].items():
            domain, topic = key_str.split("|")
            knowledge_map.last_updated[(domain, topic)] = datetime.fromisoformat(timestamp)
            
        return knowledge_map


class ExplorationTarget:
    """Represents a target for exploration"""
    
    def __init__(self, target_id: str, domain: str, topic: str, 
                importance: float = 0.5, urgency: float = 0.5,
                knowledge_gap: float = 0.5):
        self.id = target_id
        self.domain = domain
        self.topic = topic
        self.importance = importance  # How important is this knowledge (0.0-1.0)
        self.urgency = urgency  # How urgent is filling this gap (0.0-1.0)
        self.knowledge_gap = knowledge_gap  # How large is the gap (0.0-1.0)
        self.created_at = datetime.now()
        self.last_explored = None
        self.exploration_count = 0
        self.exploration_results = []
        self.related_questions = []
        self.priority_score = self._calculate_priority()
        
    def _calculate_priority(self) -> float:
        """Calculate priority score based on factors"""
        # Weights for different factors
        weights = {
            "importance": 0.4,
            "urgency": 0.3,
            "knowledge_gap": 0.3
        }
        
        # Calculate weighted score
        priority = (
            self.importance * weights["importance"] +
            self.urgency * weights["urgency"] +
            self.knowledge_gap * weights["knowledge_gap"]
        )
        
        return priority
    
    def update_priority(self) -> float:
        """Update priority based on current factors"""
        # Apply decay factor for previously explored targets
        if self.exploration_count > 0:
            # Decay based on exploration count (diminishing returns)
            exploration_factor = 1.0 / (1.0 + self.exploration_count * 0.5)
            
            # Decay based on recency (more recent = less urgent to revisit)
            if self.last_explored:
                days_since = (datetime.now() - self.last_explored).total_seconds() / (24 * 3600)
                recency_factor = min(1.0, days_since / 30)  # Max effect after 30 days
            else:
                recency_factor = 1.0
                
            # Apply factors to base priority
            self.priority_score = self._calculate_priority() * exploration_factor * recency_factor
        else:
            # Never explored, use base priority
            self.priority_score = self._calculate_priority()
            
        return self.priority_score
    
    def record_exploration(self, result: Dict[str, Any]) -> None:
        """Record an exploration of this target"""
        self.exploration_count += 1
        self.last_explored = datetime.now()
        self.exploration_results.append({
            "timestamp": self.last_explored.isoformat(),
            "result": result
        })
        
        # Update knowledge gap based on result
        if "knowledge_gained" in result:
            knowledge_gained = result["knowledge_gained"]
            
            # Reduce knowledge gap proportionally to knowledge gained
            self.knowledge_gap = max(0.0, self.knowledge_gap - knowledge_gained)
        
        # Update priority score
        self.update_priority()
    
    def add_related_question(self, question: str) -> None:
        """Add a related question to this exploration target"""
        if question not in self.related_questions:
            self.related_questions.append(question)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "domain": self.domain,
            "topic": self.topic,
            "importance": self.importance,
            "urgency": self.urgency,
            "knowledge_gap": self.knowledge_gap,
            "created_at": self.created_at.isoformat(),
            "last_explored": self.last_explored.isoformat() if self.last_explored else None,
            "exploration_count": self.exploration_count,
            "exploration_results": self.exploration_results,
            "related_questions": self.related_questions,
            "priority_score": self.priority_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExplorationTarget':
        """Create from dictionary representation"""
        target = cls(
            target_id=data["id"],
            domain=data["domain"],
            topic=data["topic"],
            importance=data["importance"],
            urgency=data["urgency"],
            knowledge_gap=data["knowledge_gap"]
        )
        target.created_at = datetime.fromisoformat(data["created_at"])
        if data["last_explored"]:
            target.last_explored = datetime.fromisoformat(data["last_explored"])
        target.exploration_count = data["exploration_count"]
        target.exploration_results = data["exploration_results"]
        target.related_questions = data["related_questions"]
        target.priority_score = data["priority_score"]
        return target

# Pydantic models for the Agents SDK
class NodeContent(BaseModel):
    """Content of a knowledge node"""
    data: Dict[str, Any] = Field(description="The content data")
    
class NodeInfo(BaseModel):
    """Information about a knowledge node"""
    id: str = Field(description="Unique ID of the node")
    type: str = Field(description="Type/category of the node")
    content: Dict[str, Any] = Field(description="Content of the node")
    source: str = Field(description="Source of the knowledge")
    confidence: float = Field(description="Confidence level (0.0-1.0)")
    
class RelationInfo(BaseModel):
    """Information about a knowledge relation"""
    source_id: str = Field(description="ID of the source node")
    target_id: str = Field(description="ID of the target node")
    type: str = Field(description="Type of relation")
    weight: float = Field(description="Weight of the relation (0.0-1.0)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
class QueryParams(BaseModel):
    """Parameters for querying knowledge"""
    type: Optional[str] = Field(default=None, description="Filter by node type")
    content_filter: Optional[Dict[str, Any]] = Field(default={}, description="Filter by content fields")
    relation_filter: Optional[Dict[str, Any]] = Field(default={}, description="Filter by relation")
    limit: int = Field(default=10, description="Maximum number of results")
    
class ConflictResolutionResult(BaseModel):
    """Result of a conflict resolution operation"""
    resolved: bool = Field(description="Whether the conflict was resolved")
    preferred_node: Optional[str] = Field(default=None, description="ID of the preferred node if resolved")
    
class ExplorationTargetInfo(BaseModel):
    """Information about an exploration target"""
    domain: str = Field(description="Knowledge domain")
    topic: str = Field(description="Knowledge topic")
    importance: float = Field(default=0.5, description="Importance (0.0-1.0)")
    urgency: float = Field(default=0.5, description="Urgency (0.0-1.0)")
    knowledge_gap: Optional[float] = Field(default=None, description="Knowledge gap size (0.0-1.0)")
    
class ExplorationResult(BaseModel):
    """Result of an exploration"""
    success: bool = Field(description="Whether the exploration was successful")
    knowledge_gained: float = Field(description="Amount of knowledge gained (0.0-1.0)")
    content: Optional[Dict[str, Any]] = Field(default=None, description="Content discovered")

# Context class for storing shared state
class KnowledgeCoreContext:
    """Context object for KnowledgeCore operations"""
    
    def __init__(self, 
                 knowledge_store_file: str = "knowledge_store.json",
                 embedding_model_name: Optional[str] = "all-mpnet-base-v2"):
        # Graph and nodes
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.node_embeddings: Dict[str, np.ndarray] = {}
        
        # Stats and tracking
        self.integration_stats = {
            "nodes_added": 0,
            "relations_added": 0,
            "conflicts_resolved": 0,
            "knowledge_queries": 0,
            "integration_cycles": 0,
            "node_count": 0,
            "relation_count": 0,
            "node_types": {},
            "relation_types": {}
        }
        
        # Config
        self.config = {
            "conflict_threshold": 0.7,
            "support_threshold": 0.7,
            "similarity_threshold": 0.8,
            "decay_rate": 0.01,
            "integration_cycle_interval": 10,  # minutes
            "enable_embeddings": True,
            "max_node_age_days": 30,
            "pruning_confidence_threshold": 0.3
        }
        
        # Timestamps and counters
        self.last_integration_cycle = datetime.now()
        self.next_node_id = 1
        
        # Caches
        self.integration_cache: Dict[str, Any] = {}
        self.query_cache: Dict[str, Any] = {}
        
        # File storage
        self.knowledge_store_file = knowledge_store_file
        
        # Curiosity system
        self.knowledge_map = KnowledgeMap()
        self.exploration_targets: Dict[str, ExplorationTarget] = {}
        self.exploration_history = []
        self.next_target_id = 1
        
        # Curiosity config
        self.curiosity_config = {
            "max_active_targets": 20,
            "exploration_budget": 0.5,
            "novelty_bias": 0.7,
            "importance_threshold": 0.3,
            "knowledge_decay_rate": 0.01
        }
        
        # Curiosity stats
        self.curiosity_stats = {
            "total_explorations": 0,
            "successful_explorations": 0,
            "knowledge_gained": 0.0,
            "avg_importance": 0.0
        }
        
        # Embedding model
        self._embedding_model = None
        self._embedding_model_name = embedding_model_name
        if self.config["enable_embeddings"]:
            self._try_load_embedding_model()
    
    def _try_load_embedding_model(self):
        """Try to load an embedding model if available"""
        if self.config["enable_embeddings"]:
            try:
                # First, try to import SentenceTransformer
                try:
                    from sentence_transformers import SentenceTransformer
                    # Use a standard SentenceTransformer model
                    self._embedding_model = SentenceTransformer(self._embedding_model_name)
                    logger.info("Loaded embedding model (SentenceTransformer).")
                except ImportError:
                    # Try to see if we can use transformers directly
                    try:
                        from transformers import AutoTokenizer, AutoModel
                        import torch
                        
                        # A simplified wrapper to make it work like SentenceTransformer
                        class SimpleEmbedder:
                            def __init__(self, model_name):
                                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                                self.model = AutoModel.from_pretrained(model_name)
                                
                            def encode(self, texts, show_progress_bar=False):
                                # Simple mean pooling function
                                def mean_pooling(model_output, attention_mask):
                                    token_embeddings = model_output[0]
                                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                                    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                                
                                # Tokenize
                                encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
                                
                                # Compute token embeddings
                                with torch.no_grad():
                                    model_output = self.model(**encoded_input)
                                
                                # Perform pooling
                                embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                                
                                # Normalize
                                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                                
                                return embeddings.numpy()
                        
                        self._embedding_model = SimpleEmbedder(self._embedding_model_name)
                        logger.info("Loaded embedding model (Transformers).")
                    except ImportError:
                        logger.warning("Neither SentenceTransformer nor Transformers is available.")
                        self._embedding_model = None
            except Exception as e:
                logger.warning(f"Error loading embedding model: {e}")
                self._embedding_model = None


# Function tools for knowledge operations
@function_tool
async def add_knowledge(
    ctx: RunContextWrapper["KnowledgeCoreContext"],
    type: str,
    content_json: str,                 #  <-- JSON–encoded string
    source: str,
    confidence: Optional[float] = None
) -> str:
    """
    Add a new knowledge node to the knowledge graph.
    
    Args:
        type: The type/category of knowledge (e.g. 'concept', 'fact', 'rule')
        content: The content of the knowledge node as a dictionary
        source: The source of the knowledge
        confidence: Confidence level from 0.0 to 1.0
    
    Returns:
        The ID of the new node
    """
    if confidence is None:
        confidence = 0.5

    content = json.loads(content_json) 
    
    core = ctx.context
    node_id = f"node_{core.next_node_id}"
    core.next_node_id += 1
    
    # Create node
    node = KnowledgeNode(
        id=node_id,
        type=type,
        content=content,
        source=source,
        confidence=confidence
    )
    
    # Check for similar nodes
    similar_nodes = await _find_similar_nodes(core_ctx, node)
    
    if similar_nodes:
        most_similar_id, similarity_val = similar_nodes[0]
        if similarity_val > core_ctx.config["similarity_threshold"]:
            return await _integrate_with_existing(core_ctx, node, most_similar_id, similarity_val)
    
    # Add as new node
    core_ctx.nodes[node_id] = node
    core_ctx.graph.add_node(node_id, **node.to_dict())
    
    # Generate embedding
    if core_ctx.config["enable_embeddings"] and core_ctx._embedding_model:
        await _add_node_embedding(core_ctx, node)
    
    # Add "similar_to" edges for moderately similar nodes
    for sim_id, sim_val in similar_nodes:
        if sim_val > core_ctx.config["similarity_threshold"] * 0.6:
            await add_relation(
                ctx,
                source_id=node_id,
                target_id=sim_id,
                type="similar_to",
                weight=sim_val,
                metadata={"auto_generated": True}
            )
    
    # Update stats
    core_ctx.integration_stats["nodes_added"] += 1
    
    # Check if we should run an integration cycle
    await _check_integration_cycle(core_ctx)
    
    return node_id

@function_tool
async def add_relation(
    ctx: RunContextWrapper[KnowledgeCoreContext],
    source_id: str,
    target_id: str,
    type: str,
    weight: Optional[float],
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Add a relation (edge) between two knowledge nodes.
    
    Args:
        source_id: ID of the source node
        target_id: ID of the target node
        type: Type of relation (e.g. 'supports', 'contradicts', 'similar_to')
        weight: Weight/strength of the relation (0.0-1.0)
        metadata: Optional metadata for the relation
    
    Returns:
        True if successful, False if either node doesn't exist
    """
    if weight is None:
        weight = 1.0
    core_ctx = ctx.context
    
    # Check if nodes exist
    if source_id not in core_ctx.nodes or target_id not in core_ctx.nodes:
        logger.warning(f"Cannot add relation: node {source_id} or {target_id} does not exist.")
        return False
    
    # Create relation
    rel = KnowledgeRelation(
        source_id=source_id,
        target_id=target_id,
        type=type,
        weight=weight,
        metadata=metadata or {}
    )
    
    # Add to graph
    core_ctx.graph.add_edge(source_id, target_id, **rel.to_dict())
    
    # Update supporting/conflicting references
    if type == "supports":
        if target_id not in core_ctx.nodes[source_id].supporting_nodes:
            core_ctx.nodes[source_id].supporting_nodes.append(target_id)
    elif type == "contradicts":
        if target_id not in core_ctx.nodes[source_id].conflicting_nodes:
            core_ctx.nodes[source_id].conflicting_nodes.append(target_id)
        await _handle_contradiction(core_ctx, source_id, target_id)
    
    # Update stats
    core_ctx.integration_stats["relations_added"] += 1
    
    # Update curiosity system if appropriate
    if type in ["related", "similar_to", "specializes"]:
        s_node = core_ctx.nodes[source_id]
        t_node = core_ctx.nodes[target_id]
        
        s_domain = s_node.content.get("domain", s_node.type)
        s_topic = s_node.content.get("topic", list(s_node.content.keys())[0] if s_node.content else "unknown")
        
        t_domain = t_node.content.get("domain", t_node.type)
        t_topic = t_node.content.get("topic", list(t_node.content.keys())[0] if t_node.content else "unknown")
        
        core_ctx.knowledge_map.add_connection(s_domain, s_topic, t_domain, t_topic, weight)
    
    return True

@function_tool  # strict schema still ON (default)
async def query_knowledge(
    ctx: RunContextWrapper[KnowledgeCoreContext],
    query: KnowledgeQuery
) -> List[Dict[str, Any]]:
    """
    Search the knowledge graph for nodes matching certain criteria.
    
    Args:
        query: A dict with optional keys:
            - type: Filter by node type
            - content_filter: Filter by content fields
            - relation_filter: Filter by relation
            - limit: Maximum number of results (default: 10)
    
    Returns:
        A list of matching node dictionaries
    """
    core_ctx = ctx.context
    core_ctx.integration_stats["knowledge_queries"] += 1
    
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
    }, sort_keys=True)
    
    # Check cache
    now = datetime.now()
    if cache_key in core_ctx.query_cache:
        entry = core_ctx.query_cache[cache_key]
        cached_time = datetime.fromisoformat(entry["timestamp"])
        if (now - cached_time).total_seconds() < 60:  # Cache valid for 1 minute
            return entry["results"]
    
    # Perform the search
    matching = []
    for nid, node in core_ctx.nodes.items():
        # Type filter
        if node_type and node.type != node_type:
            continue
        
        # Content filter
        content_ok = True
        for k, v in content_filter.items():
            if k not in node.content or node.content[k] != v:
                content_ok = False
                break
        if not content_ok:
            continue
        
        # Relation filter
        relation_ok = True
        if relation_filter:
            rtype = relation_filter.get("type")
            other_id = relation_filter.get("node_id")
            direct = relation_filter.get("direction", "outgoing")
            
            if rtype and other_id:
                if direct == "outgoing":
                    if not core_ctx.graph.has_edge(nid, other_id):
                        relation_ok = False
                    else:
                        edata = core_ctx.graph.get_edge_data(nid, other_id)
                        if edata["type"] != rtype:
                            relation_ok = False
                else:  # incoming
                    if not core_ctx.graph.has_edge(other_id, nid):
                        relation_ok = False
                    else:
                        edata = core_ctx.graph.get_edge_data(other_id, nid)
                        if edata["type"] != rtype:
                            relation_ok = False
        
        if not relation_ok:
            continue
        
        # Node passes all filters
        node.access()
        core_ctx.graph.nodes[nid].update(node.to_dict())
        matching.append(node.to_dict())
    
    # Sort by confidence
    matching.sort(key=lambda x: x["confidence"], reverse=True)
    results = matching[:limit]
    
    # Update cache
    core_ctx.query_cache[cache_key] = {
        "timestamp": now.isoformat(),
        "results": results
    }
    
    return results

@function_tool
async def get_related_knowledge(
    ctx: RunContextWrapper[KnowledgeCoreContext],
    node_id: str,
    relation_type: Optional[str] = None,
    direction: str = "both",
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Get nodes related to a specific knowledge node.
    
    Args:
        node_id: The ID of the node to find neighbors for
        relation_type: Optional filter for relation type
        direction: 'incoming', 'outgoing', or 'both'
        limit: Maximum number of results
    
    Returns:
        List of dicts with 'node' and 'relation' fields
    """
    core_ctx = ctx.context
    
    if node_id not in core_ctx.nodes:
        return []
    
    # Update access stats
    core_ctx.nodes[node_id].access()
    
    neighbors = []
    
    # Outgoing edges
    if direction in ["outgoing", "both"]:
        for tgt in core_ctx.graph.successors(node_id):
            edata = core_ctx.graph.get_edge_data(node_id, tgt)
            if relation_type is None or edata["type"] == relation_type:
                if tgt in core_ctx.nodes:
                    neighbors.append({
                        "node": core_ctx.nodes[tgt].to_dict(),
                        "relation": {
                            "type": edata["type"],
                            "direction": "outgoing",
                            "weight": edata.get("weight", 1.0)
                        }
                    })
    
    # Incoming edges
    if direction in ["incoming", "both"]:
        for src in core_ctx.graph.predecessors(node_id):
            edata = core_ctx.graph.get_edge_data(src, node_id)
            if relation_type is None or edata["type"] == relation_type:
                if src in core_ctx.nodes:
                    neighbors.append({
                        "node": core_ctx.nodes[src].to_dict(),
                        "relation": {
                            "type": edata["type"],
                            "direction": "incoming",
                            "weight": edata.get("weight", 1.0)
                        }
                    })
    
    # Sort by relation weight
    neighbors.sort(key=lambda x: x["relation"]["weight"], reverse=True)
    
    return neighbors[:limit]

@function_tool
async def identify_knowledge_gaps(
    ctx: RunContextWrapper[KnowledgeCoreContext]
) -> List[Dict[str, Any]]:
    """
    Identify knowledge gaps from the knowledge map.
    
    Returns:
        List of knowledge gaps with domain, topic, gap size and importance
    """
    core_ctx = ctx.context
    
    # Get gaps from knowledge map
    gaps = core_ctx.knowledge_map.get_knowledge_gaps()
    
    # Convert to dictionaries
    gap_dicts = []
    for domain, topic, gap_size, importance in gaps:
        if importance >= core_ctx.curiosity_config["importance_threshold"]:
            gap_dicts.append({
                "domain": domain,
                "topic": topic,
                "gap_size": gap_size,
                "importance": importance,
                "priority": gap_size * importance
            })
    
    # Sort by priority
    gap_dicts.sort(key=lambda x: x["priority"], reverse=True)
    
    return gap_dicts

@function_tool
async def create_exploration_target(
    ctx: RunContextWrapper[KnowledgeCoreContext],
    domain: str,
    topic: str,
    importance: float = 0.5,
    urgency: float = 0.5,
    knowledge_gap: Optional[float] = None
) -> str:
    """
    Create a new exploration target for the curiosity system.
    
    Args:
        domain: Knowledge domain
        topic: Knowledge topic
        importance: Importance of the knowledge (0.0-1.0)
        urgency: Urgency of exploration (0.0-1.0)
        knowledge_gap: Size of knowledge gap (0.0-1.0), computed if None
    
    Returns:
        ID of the created exploration target
    """
    core_ctx = ctx.context
    
    # Generate target ID
    target_id = f"target_{core_ctx.next_target_id}"
    core_ctx.next_target_id += 1
    
    # Calculate knowledge gap if not provided
    if knowledge_gap is None:
        level = core_ctx.knowledge_map.get_knowledge_level(domain, topic)
        knowledge_gap = 1.0 - level
    
    # Create target
    target = ExplorationTarget(
        target_id=target_id,
        domain=domain,
        topic=topic,
        importance=importance,
        urgency=urgency,
        knowledge_gap=knowledge_gap
    )
    
    # Store target
    core_ctx.exploration_targets[target_id] = target
    
    return target_id

@function_tool
async def get_exploration_targets(
    ctx: RunContextWrapper[KnowledgeCoreContext],
    limit: int = 10,
    min_priority: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Get current exploration targets sorted by priority.
    
    Args:
        limit: Maximum number of targets to return
        min_priority: Minimum priority threshold
    
    Returns:
        List of exploration target dictionaries
    """
    core_ctx = ctx.context
    
    # Update priorities
    for target in core_ctx.exploration_targets.values():
        target.update_priority()
    
    # Sort by priority
    sorted_targets = sorted(
        [t for t in core_ctx.exploration_targets.values() if t.priority_score >= min_priority],
        key=lambda x: x.priority_score,
        reverse=True
    )
    
    # Apply limit
    targets = sorted_targets[:limit]
    
    # Convert to dictionaries
    return [target.to_dict() for target in targets]

@function_tool
async def record_exploration(
    ctx: RunContextWrapper[KnowledgeCoreContext],
    target_id: str,
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Record the result of an exploration.
    
    Args:
        target_id: ID of the exploration target
        result: Dictionary with exploration results
    
    Returns:
        Dictionary with updated target info
    """
    core_ctx = ctx.context
    
    if target_id not in core_ctx.exploration_targets:
        return {"error": f"Target {target_id} not found"}
    
    # Get target
    target = core_ctx.exploration_targets[target_id]
    
    # Record exploration
    target.record_exploration(result)
    
    # Update knowledge map if knowledge gained
    if "knowledge_gained" in result and result["knowledge_gained"] > 0:
        level = 1.0 - target.knowledge_gap
        core_ctx.knowledge_map.add_knowledge_to_map(
            target.domain,
            target.topic,
            level,
            target.importance
        )
    
    # Update statistics
    core_ctx.curiosity_stats["total_explorations"] += 1
    if result.get("success", False):
        core_ctx.curiosity_stats["successful_explorations"] += 1
    core_ctx.curiosity_stats["knowledge_gained"] += result.get("knowledge_gained", 0.0)
    
    # Calculate average importance
    total_importance = sum(target.importance for target in core_ctx.exploration_targets.values())
    if core_ctx.exploration_targets:
        core_ctx.curiosity_stats["avg_importance"] = total_importance / len(core_ctx.exploration_targets)
    
    # Add to exploration history
    core_ctx.exploration_history.append({
        "timestamp": datetime.now().isoformat(),
        "target_id": target_id,
        "domain": target.domain,
        "topic": target.topic,
        "result": result
    })
    
    # Prune history if needed
    if len(core_ctx.exploration_history) > 1000:
        core_ctx.exploration_history = core_ctx.exploration_history[-1000:]
    
    return {
        "target": target.to_dict(),
        "updated_knowledge_level": 1.0 - target.knowledge_gap
    }

@function_tool
async def generate_questions(
    ctx: RunContextWrapper[KnowledgeCoreContext],
    target_id: str,
    limit: int = 5
) -> List[str]:
    """
    Generate questions to explore a target.
    
    Args:
        target_id: ID of the exploration target
        limit: Maximum number of questions
    
    Returns:
        List of questions
    """
    core_ctx = ctx.context
    
    if target_id not in core_ctx.exploration_targets:
        return []
    
    # Get target
    target = core_ctx.exploration_targets[target_id]
    
    # If there are already questions, return them
    if len(target.related_questions) >= limit:
        return target.related_questions[:limit]
    
    # Generate basic questions
    questions = [
        f"What is {target.topic} in the context of {target.domain}?",
        f"Why is {target.topic} important in {target.domain}?",
        f"How does {target.topic} relate to other topics in {target.domain}?",
        f"What are the key components or aspects of {target.topic}?",
        f"What are common misconceptions about {target.topic}?"
    ]
    
    # Get related topics for more targeted questions
    related_topics = core_ctx.knowledge_map.get_related_topics(target.domain, target.topic)
    for domain, topic, strength in related_topics[:3]:  # Use up to 3 related topics
        questions.append(f"How does {target.topic} relate to {topic}?")
    
    # Add questions to target
    for question in questions:
        target.add_related_question(question)
    
    return questions[:limit]

@function_tool
async def save_knowledge(
    ctx: RunContextWrapper[KnowledgeCoreContext]
) -> bool:
    """
    Save the current knowledge graph to storage.
    
    Returns:
        True if successful, False otherwise
    """
    core_ctx = ctx.context
    
    try:
        # Build list of node dicts
        nodes_data = [node.to_dict() for node in core_ctx.nodes.values()]
        
        # Build list of relation dicts
        relations_data = []
        for source, target, data in core_ctx.graph.edges(data=True):
            relation_dict = {
                "source_id": source,
                "target_id": target,
                "type": data.get("type"),
                "weight": data.get("weight", 1.0),
                "timestamp": data.get("timestamp"),
                "metadata": data.get("metadata", {})
            }
            
            # Handle timestamp format
            if isinstance(relation_dict["timestamp"], datetime):
                relation_dict["timestamp"] = relation_dict["timestamp"].isoformat()
                
            relations_data.append(relation_dict)
        
        store_data = {
            "nodes": nodes_data,
            "relations": relations_data
        }
        
        with open(core_ctx.knowledge_store_file, "w", encoding="utf-8") as f:
            json.dump(store_data, f, indent=2)
            
        logger.info("Knowledge store saved successfully.")
        return True
    except Exception as e:
        logger.error(f"Error saving knowledge: {str(e)}")
        return False

@function_tool
async def load_knowledge(
    ctx: RunContextWrapper[KnowledgeCoreContext]
) -> bool:
    """
    Load knowledge from storage.
    
    Returns:
        True if successful, False otherwise
    """
    core_ctx = ctx.context
    
    if not os.path.exists(core_ctx.knowledge_store_file):
        logger.info("No existing knowledge store found; starting fresh.")
        return False
    
    try:
        with open(core_ctx.knowledge_store_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Load nodes
        for node_data in data.get("nodes", []):
            node = KnowledgeNode.from_dict(node_data)
            core_ctx.nodes[node.id] = node
            core_ctx.graph.add_node(node.id, **node.to_dict())
            
            # Update next_node_id
            try:
                numeric_id = int(node.id.split("_")[-1])
                if numeric_id >= core_ctx.next_node_id:
                    core_ctx.next_node_id = numeric_id + 1
            except:
                pass
        
        # Load relations
        for rel_data in data.get("relations", []):
            rel = KnowledgeRelation.from_dict(rel_data)
            core_ctx.graph.add_edge(
                rel.source_id,
                rel.target_id,
                **rel.to_dict()
            )
            
            # Update supporting/conflicting references
            if rel.type == "supports":
                if rel.target_id not in core_ctx.nodes[rel.source_id].supporting_nodes:
                    core_ctx.nodes[rel.source_id].supporting_nodes.append(rel.target_id)
            elif rel.type == "contradicts":
                if rel.target_id not in core_ctx.nodes[rel.source_id].conflicting_nodes:
                    core_ctx.nodes[rel.source_id].conflicting_nodes.append(rel.target_id)
        
        # Rebuild embeddings
        if core_ctx.config["enable_embeddings"] and core_ctx._embedding_model:
            await _rebuild_embeddings(core_ctx)
        
        logger.info(f"Knowledge store loaded with {len(core_ctx.nodes)} nodes and {core_ctx.graph.number_of_edges()} edges.")
        return True
    except Exception as e:
        logger.error(f"Error loading knowledge: {str(e)}")
        return False

@function_tool
async def get_knowledge_statistics(
    ctx: RunContextWrapper[KnowledgeCoreContext]
) -> Dict[str, Any]:
    """
    Get statistics about the knowledge graph.
    
    Returns:
        Dictionary with statistics
    """
    core_ctx = ctx.context
    
    # Update node and edge counts
    core_ctx.integration_stats["node_count"] = len(core_ctx.nodes)
    core_ctx.integration_stats["relation_count"] = core_ctx.graph.number_of_edges()
    
    # Count node types
    node_types = {}
    for node in core_ctx.nodes.values():
        node_types[node.type] = node_types.get(node.type, 0) + 1
    core_ctx.integration_stats["node_types"] = node_types
    
    # Count relation types
    relation_types = {}
    for s, t, data in core_ctx.graph.edges(data=True):
        rtype = data.get("type")
        if rtype:
            relation_types[rtype] = relation_types.get(rtype, 0) + 1
    core_ctx.integration_stats["relation_types"] = relation_types
    
    # Basic statistics
    stats = {
        "timestamp": datetime.now().isoformat(),
        "node_count": core_ctx.integration_stats["node_count"],
        "edge_count": core_ctx.integration_stats["relation_count"],
        "node_types": core_ctx.integration_stats["node_types"],
        "relation_types": core_ctx.integration_stats["relation_types"],
        "nodes_added": core_ctx.integration_stats["nodes_added"],
        "relations_added": core_ctx.integration_stats["relations_added"],
        "conflicts_resolved": core_ctx.integration_stats["conflicts_resolved"],
        "knowledge_queries": core_ctx.integration_stats["knowledge_queries"],
        "integration_cycles": core_ctx.integration_stats["integration_cycles"]
    }
    
    # Curiosity statistics
    curiosity_stats = {
        "knowledge_map": {
            "domain_count": len(core_ctx.knowledge_map.domains),
            "topic_count": sum(len(topics) for topics in core_ctx.knowledge_map.domains.values()),
            "connection_count": sum(len(connections) for connections in core_ctx.knowledge_map.connections.values()),
            "average_knowledge_level": _calculate_average_knowledge_level(core_ctx),
            "average_gap_size": _calculate_average_gap_size(core_ctx),
            "knowledge_gap_count": len(core_ctx.knowledge_map.get_knowledge_gaps())
        },
        "exploration": {
            "active_targets": len(core_ctx.exploration_targets),
            "explored_targets": sum(1 for t in core_ctx.exploration_targets.values() if t.exploration_count > 0),
            "success_rate": core_ctx.curiosity_stats["successful_explorations"] / core_ctx.curiosity_stats["total_explorations"] 
                           if core_ctx.curiosity_stats["total_explorations"] > 0 else 0.0,
            "total_knowledge_gained": core_ctx.curiosity_stats["knowledge_gained"],
            "average_target_importance": core_ctx.curiosity_stats["avg_importance"]
        },
        "configuration": core_ctx.curiosity_config
    }
    
    stats["curiosity_system"] = curiosity_stats
    
    return stats

@function_tool
async def run_integration_cycle(
    ctx: RunContextWrapper[KnowledgeCoreContext]
) -> Dict[str, Any]:
    """
    Run a knowledge integration cycle (maintenance operations).
    
    Returns:
        Dictionary with results of the integration cycle
    """
    core_ctx = ctx.context
    
    logger.info("Running knowledge integration cycle.")
    core_ctx.last_integration_cycle = datetime.now()
    core_ctx.integration_stats["integration_cycles"] += 1
    
    cycle_results = {}
    
    # 1. Prune old or low-confidence nodes
    pruned = await _prune_nodes(core_ctx)
    cycle_results["pruned_nodes"] = len(pruned)
    
    # 2. Resolve conflicts
    conflicts_resolved = await _resolve_conflicts(core_ctx)
    cycle_results["conflicts_resolved"] = conflicts_resolved
    
    # 3. Consolidate similar nodes
    nodes_merged = await _consolidate_similar_nodes(core_ctx)
    cycle_results["nodes_merged"] = nodes_merged
    
    # 4. Infer new relations
    new_relations = await _infer_relations(core_ctx)
    cycle_results["relations_inferred"] = len(new_relations)
    
    # 5. Apply confidence decay
    decay_stats = _apply_confidence_decay(core_ctx)
    cycle_results["confidence_decay"] = decay_stats
    
    # 6. Apply knowledge decay in curiosity system
    curiosity_decay = await _apply_knowledge_decay(core_ctx)
    cycle_results["knowledge_decay"] = curiosity_decay
    
    # 7. Save knowledge
    await save_knowledge(ctx)
    
    logger.info("Integration cycle completed.")
    
    return cycle_results

# Helper functions for internal operations
async def _find_similar_nodes(ctx: KnowledgeCoreContext, node: KnowledgeNode) -> List[Tuple[str, float]]:
    """Find nodes similar to the given node"""
    if ctx.config["enable_embeddings"] and ctx._embedding_model:
        # Use embeddings for similarity
        embedding = await _generate_embedding(ctx, node)
        similar_list = []
        
        for node_id, emb in ctx.node_embeddings.items():
            sim = _calculate_embedding_similarity(embedding, emb)
            if sim > (ctx.config["similarity_threshold"] * 0.5):
                similar_list.append((node_id, sim))
                
        similar_list.sort(key=lambda x: x[1], reverse=True)
        return similar_list
    else:
        # Fallback: use content-based similarity
        results = []
        for nid, existing_node in ctx.nodes.items():
            if existing_node.type == node.type:
                content_sim = _calculate_content_similarity(node.content, existing_node.content)
                if content_sim > (ctx.config["similarity_threshold"] * 0.5):
                    results.append((nid, content_sim))
                    
        results.sort(key=lambda x: x[1], reverse=True)
        return results

async def _generate_embedding(ctx: KnowledgeCoreContext, node: KnowledgeNode) -> np.ndarray:
    """Generate an embedding for a node"""
    if ctx._embedding_model:
        # Use the real embedding model
        content_str = json.dumps(node.content, ensure_ascii=False)
        embedding = ctx._embedding_model.encode([content_str], show_progress_bar=False)
        return embedding[0]
    else:
        # Fallback to a simple approach
        fallback_vector = np.zeros(128, dtype=np.float32)
        content_str = json.dumps(node.content)
        
        for i, c in enumerate(content_str):
            idx = i % 128
            fallback_vector[idx] += (ord(c) % 96) / 100.0
            
        norm = np.linalg.norm(fallback_vector)
        if norm > 0:
            fallback_vector = fallback_vector / norm
            
        return fallback_vector

def _calculate_embedding_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between embeddings"""
    dot = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
        
    return float(dot / (norm1 * norm2))

def _calculate_content_similarity(c1: Dict[str, Any], c2: Dict[str, Any]) -> float:
    """Calculate similarity between content dictionaries"""
    all_keys = set(c1.keys()).union(set(c2.keys()))
    
    if not all_keys:
        return 0.0
        
    matching_keys = set(c1.keys()).intersection(set(c2.keys()))
    matching_values = 0
    
    for k in matching_keys:
        if c1[k] == c2[k]:
            matching_values += 1
            
    key_similarity = len(matching_keys) / len(all_keys)
    value_similarity = matching_values / len(matching_keys) if matching_keys else 0.0
    
    # Weighted combination
    return 0.7 * key_similarity + 0.3 * value_similarity

async def _rebuild_embeddings(ctx: KnowledgeCoreContext) -> None:
    """Rebuild embeddings for all nodes"""
    for node in ctx.nodes.values():
        await _update_node_embedding(ctx, node)

async def _add_node_embedding(ctx: KnowledgeCoreContext, node: KnowledgeNode) -> None:
    """Create and store an embedding for a node"""
    emb = await _generate_embedding(ctx, node)
    ctx.node_embeddings[node.id] = emb

async def _update_node_embedding(ctx: KnowledgeCoreContext, node: KnowledgeNode) -> None:
    """Update the embedding for a node"""
    emb = await _generate_embedding(ctx, node)
    ctx.node_embeddings[node.id] = emb

async def _integrate_with_existing(
    ctx: KnowledgeCoreContext, 
    new_node: KnowledgeNode, 
    existing_id: str, 
    similarity: float
) -> str:
    """Merge a new node into an existing one"""
    existing_node = ctx.nodes[existing_id]
    existing_node.access()
    
    # Update based on which node has higher confidence
    if new_node.confidence > existing_node.confidence:
        # New node is more reliable
        updated_content = new_node.content.copy()
        
        # Keep old content that's missing in new
        for key, val in existing_node.content.items():
            if key not in updated_content:
                updated_content[key] = val
                
        updated_conf = (new_node.confidence * 0.7 + existing_node.confidence * 0.3)
        existing_node.update(updated_content, updated_conf, f"{existing_node.source}+{new_node.source}")
    else:
        # Keep existing as basis
        for key, val in new_node.content.items():
            if key not in existing_node.content:
                existing_node.content[key] = val
                
        updated_conf = (existing_node.confidence * 0.7 + new_node.confidence * 0.3)
        existing_node.confidence = updated_conf
        existing_node.last_accessed = datetime.now()
    
    # Update node data in graph
    ctx.graph.nodes[existing_id].update(existing_node.to_dict())
    
    # Update embedding
    if ctx.config["enable_embeddings"] and ctx._embedding_model:
        await _update_node_embedding(ctx, existing_node)
        
    logger.debug(f"Integrated node {new_node.id} into existing node {existing_id}.")
    return existing_id

async def _handle_contradiction(ctx: KnowledgeCoreContext, node1_id: str, node2_id: str) -> None:
    """Handle a contradiction between two nodes"""
    node1 = ctx.nodes[node1_id]
    node2 = ctx.nodes[node2_id]
    
    # High confidence conflict needs reasoning
    if node1.confidence > 0.7 and node2.confidence > 0.7:
        logger.info(f"Detected high-confidence conflict between {node1_id} and {node2_id}")
        
        # Simple reasoning approach
        if abs(node1.confidence - node2.confidence) < 0.05:
            # Similar confidence, no clear winner
            logger.info(f"Similar confidence levels, no clear resolution for {node1_id} vs {node2_id}")
        else:
            # Prefer the higher confidence node
            if node1.confidence > node2.confidence:
                node2.confidence *= 0.7
                logger.info(f"Resolved conflict in favor of {node1_id}")
            else:
                node1.confidence *= 0.7
                logger.info(f"Resolved conflict in favor of {node2_id}")
                
            ctx.integration_stats["conflicts_resolved"] += 1
    
    # If there's a big confidence gap, reduce the lower
    elif abs(node1.confidence - node2.confidence) > 0.3:
        if node1.confidence < node2.confidence:
            node1.confidence *= 0.9
        else:
            node2.confidence *= 0.9
        logger.debug(f"Adjusted confidence in conflict: {node1_id} vs {node2_id}.")

async def _check_integration_cycle(ctx: KnowledgeCoreContext) -> None:
    """Check if we should run an integration cycle"""
    now = datetime.now()
    elapsed = (now - ctx.last_integration_cycle).total_seconds()
    
    # If enough time has passed, run cycle
    if elapsed > ctx.config["integration_cycle_interval"] * 60:
        await _run_integration_cycle(ctx)

async def _run_integration_cycle(ctx: KnowledgeCoreContext) -> None:
    """Run the integration cycle"""
    logger.info("Running knowledge integration cycle.")
    ctx.last_integration_cycle = datetime.now()
    ctx.integration_stats["integration_cycles"] += 1
    
    # 1. Prune old or low-confidence nodes
    await _prune_nodes(ctx)
    
    # 2. Resolve conflicts
    await _resolve_conflicts(ctx)
    
    # 3. Consolidate similar nodes
    await _consolidate_similar_nodes(ctx)
    
    # 4. Infer new relations
    await _infer_relations(ctx)
    
    # 5. Apply confidence decay
    _apply_confidence_decay(ctx)
    
    # 6. Apply knowledge decay in curiosity system
    await _apply_knowledge_decay(ctx)
    
    logger.info("Integration cycle completed.")

async def _prune_nodes(ctx: KnowledgeCoreContext) -> List[str]:
    """Remove old or low-confidence nodes"""
    now = datetime.now()
    to_prune = []
    
    for node_id, node in ctx.nodes.items():
        age_days = (now - node.timestamp).total_seconds() / (24 * 3600)
        if (age_days > ctx.config["max_node_age_days"] and
                node.confidence < ctx.config["pruning_confidence_threshold"] and
                node.access_count < 3):
            to_prune.append(node_id)
    
    for nid in to_prune:
        await _remove_node(ctx, nid)
    
    if to_prune:
        logger.info(f"Pruned {len(to_prune)} node(s).")
        
    return to_prune

async def _remove_node(ctx: KnowledgeCoreContext, node_id: str) -> None:
    """Delete a node and all its edges"""
    if node_id in ctx.nodes:
        del ctx.nodes[node_id]
        
    if node_id in ctx.graph:
        ctx.graph.remove_node(node_id)
        
    if node_id in ctx.node_embeddings:
        del ctx.node_embeddings[node_id]
    
    # Remove references from other nodes
    for node in ctx.nodes.values():
        if node_id in node.supporting_nodes:
            node.supporting_nodes.remove(node_id)
        if node_id in node.conflicting_nodes:
            node.conflicting_nodes.remove(node_id)

async def _resolve_conflicts(ctx: KnowledgeCoreContext) -> int:
    """Find and resolve conflicts in the graph"""
    contradictions = []
    count = 0
    
    for s, t, data in ctx.graph.edges(data=True):
        if data.get("type") == "contradicts":
            contradictions.append((s, t))
    
    for (nid1, nid2) in contradictions:
        if nid1 in ctx.nodes and nid2 in ctx.nodes:
            # Track original confidences
            old_conf1 = ctx.nodes[nid1].confidence
            old_conf2 = ctx.nodes[nid2].confidence
            
            await _handle_contradiction(ctx, nid1, nid2)
            
            # Check if confidence was adjusted
            if (old_conf1 != ctx.nodes[nid1].confidence or 
                old_conf2 != ctx.nodes[nid2].confidence):
                count += 1
    
    return count

async def _consolidate_similar_nodes(ctx: KnowledgeCoreContext) -> int:
    """Merge nodes that are very similar"""
    if not ctx.config["enable_embeddings"] or not ctx._embedding_model:
        return 0
        
    node_ids = list(ctx.nodes.keys())
    if len(node_ids) < 2:
        return 0
    
    to_merge = []
    max_comparisons = min(1000, len(node_ids) * (len(node_ids) - 1) // 2)
    comparisons = 0
    
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            comparisons += 1
            if comparisons > max_comparisons:
                break
                
            id1 = node_ids[i]
            id2 = node_ids[j]
            
            if id1 not in ctx.node_embeddings or id2 not in ctx.node_embeddings:
                continue
                
            # Skip if they have a direct contradiction
            if id2 in ctx.nodes[id1].conflicting_nodes:
                continue
                
            # Compare embeddings
            emb1 = ctx.node_embeddings[id1]
            emb2 = ctx.node_embeddings[id2]
            sim = _calculate_embedding_similarity(emb1, emb2)
            
            if sim > ctx.config["similarity_threshold"]:
                to_merge.append((id1, id2, sim))
    
    # Sort by similarity
    to_merge.sort(key=lambda x: x[2], reverse=True)
    merged_nodes = set()
    merges_done = 0
    
    for (id1, id2, sim) in to_merge:
        if id1 in merged_nodes or id2 in merged_nodes:
            continue
            
        if id1 not in ctx.nodes or id2 not in ctx.nodes:
            continue
            
        # Choose target based on confidence
        if ctx.nodes[id1].confidence >= ctx.nodes[id2].confidence:
            target_id, source_id = id1, id2
        else:
            target_id, source_id = id2, id1
            
        await _merge_nodes(ctx, target_id, source_id)
        merged_nodes.add(source_id)
        merges_done += 1
    
    if merges_done > 0:
        logger.info(f"Consolidated {merges_done} similar node pairs.")
        
    return merges_done

async def _merge_nodes(ctx: KnowledgeCoreContext, target_id: str, source_id: str) -> None:
    """Merge source node into target node"""
    if target_id not in ctx.nodes or source_id not in ctx.nodes:
        return
        
    target_node = ctx.nodes[target_id]
    source_node = ctx.nodes[source_id]
    
    # Merge content
    for key, val in source_node.content.items():
        if key not in target_node.content:
            target_node.content[key] = val
    
    # Update metadata
    merged_from = target_node.metadata.get("merged_from", [])
    merged_from.append(source_id)
    target_node.metadata["merged_from"] = merged_from
    target_node.metadata["merged_timestamp"] = datetime.now().isoformat()
    
    # Adjust confidence
    new_conf = (target_node.confidence * 0.7 + source_node.confidence * 0.3)
    target_node.confidence = min(1.0, new_conf)
    
    # Combine sources
    if source_node.source not in target_node.source:
        target_node.source = f"{target_node.source}+{source_node.source}"
    
    # Transfer edges
    # Outgoing edges from source
    for succ in list(ctx.graph.successors(source_id)):
        if succ != target_id:
            edge_data = ctx.graph.get_edge_data(source_id, succ)
            ctx.graph.add_edge(target_id, succ, **edge_data)
    
    # Incoming edges to source
    for pred in list(ctx.graph.predecessors(source_id)):
        if pred != target_id:
            edge_data = ctx.graph.get_edge_data(pred, source_id)
            ctx.graph.add_edge(pred, target_id, **edge_data)
    
    # Merge supporting/conflicting lists
    for n_id in source_node.supporting_nodes:
        if n_id != target_id and n_id not in target_node.supporting_nodes:
            target_node.supporting_nodes.append(n_id)
            
    for n_id in source_node.conflicting_nodes:
        if n_id != target_id and n_id not in target_node.conflicting_nodes:
            target_node.conflicting_nodes.append(n_id)
    
    # Update graph data
    ctx.graph.nodes[target_id].update(target_node.to_dict())
    
    # Remove source node
    await _remove_node(ctx, source_id)
    
    # Update embedding
    if ctx.config["enable_embeddings"] and ctx._embedding_model:
        await _update_node_embedding(ctx, target_node)

async def _infer_relations(ctx: KnowledgeCoreContext) -> List[Tuple[str, str, str]]:
    """Infer new relations from existing ones"""
    new_relations = []
    
    for n1 in ctx.nodes:
        for n2 in ctx.graph.successors(n1):
            data12 = ctx.graph.get_edge_data(n1, n2)
            rel1_type = data12.get("type")
            
            for n3 in ctx.graph.successors(n2):
                if n3 == n1:
                    continue
                    
                data23 = ctx.graph.get_edge_data(n2, n3)
                rel2_type = data23.get("type")
                
                if ctx.graph.has_edge(n1, n3):
                    continue
                
                # Infer relation type
                inferred = _infer_relation_type(rel1_type, rel2_type)
                if inferred:
                    new_relations.append((n1, n3, inferred))
    
    # Add the new relations
    for (src, trg, rtype) in new_relations:
        if src in ctx.nodes and trg in ctx.nodes:
            # Create an edge with inferred=True in metadata
            ctx.graph.add_edge(
                src, trg,
                **KnowledgeRelation(
                    source_id=src,
                    target_id=trg,
                    type=rtype,
                    weight=0.7,
                    metadata={"inferred": True}
                ).to_dict()
            )
            
            # Update supporting/conflicting references
            if rtype == "supports":
                if trg not in ctx.nodes[src].supporting_nodes:
                    ctx.nodes[src].supporting_nodes.append(trg)
            elif rtype == "contradicts":
                if trg not in ctx.nodes[src].conflicting_nodes:
                    ctx.nodes[src].conflicting_nodes.append(trg)
                await _handle_contradiction(ctx, src, trg)
    
    if new_relations:
        logger.info(f"Inferred {len(new_relations)} new relations.")
        
    return new_relations

def _infer_relation_type(t1: str, t2: str) -> Optional[str]:
    """Determine the relation type that follows from two relations"""
    composition = {
        ("supports", "supports"): "supports",
        ("supports", "contradicts"): "contradicts",
        ("contradicts", "supports"): "contradicts",
        ("contradicts", "contradicts"): "supports",
        ("specializes", "specializes"): "specializes",
        ("has_property", "specializes"): "has_property"
    }
    
    return composition.get((t1, t2), None)

def _apply_confidence_decay(ctx: KnowledgeCoreContext) -> Dict[str, Any]:
    """Decrease confidence of old nodes"""
    now = datetime.now()
    decay_stats = {
        "nodes_affected": 0,
        "total_decay": 0.0,
        "average_decay": 0.0
    }
    
    for node in ctx.nodes.values():
        days_inactive = (now - node.last_accessed).total_seconds() / (24 * 3600)
        if days_inactive > 7:  # Decay after 1 week of inactivity
            original = node.confidence
            factor = 1.0 - (ctx.config["decay_rate"] * min(days_inactive / 30.0, 1.0))
            node.confidence *= factor
            
            # Update node in graph
            ctx.graph.nodes[node.id].update(node.to_dict())
            
            # Track stats
            decay = original - node.confidence
            decay_stats["nodes_affected"] += 1
            decay_stats["total_decay"] += decay
    
    if decay_stats["nodes_affected"] > 0:
        decay_stats["average_decay"] = decay_stats["total_decay"] / decay_stats["nodes_affected"]
        
    return decay_stats

async def _apply_knowledge_decay(ctx: KnowledgeCoreContext) -> Dict[str, Any]:
    """Apply decay to knowledge levels in the knowledge map"""
    decay_stats = {
        "domains_affected": 0,
        "topics_affected": 0,
        "total_decay": 0.0,
        "average_decay": 0.0
    }
    
    now = datetime.now()
    decay_factor = ctx.curiosity_config["knowledge_decay_rate"]
    
    # Apply decay to all domains and topics
    for domain, topics in ctx.knowledge_map.domains.items():
        domain_affected = False
        
        for topic, level in list(topics.items()):
            # Get last update time
            key = (domain, topic)
            last_updated = ctx.knowledge_map.last_updated.get(key)
            
            if last_updated:
                # Calculate days since last update
                days_since = (now - last_updated).total_seconds() / (24 * 3600)
                
                # Apply decay based on time
                if days_since > 30:  # Only decay knowledge older than 30 days
                    decay_amount = level * decay_factor * (days_since / 30)
                    new_level = max(0.0, level - decay_amount)
                    
                    # Update knowledge level
                    ctx.knowledge_map.domains[domain][topic] = new_level
                    
                    # Update statistics
                    decay_stats["topics_affected"] += 1
                    decay_stats["total_decay"] += decay_amount
                    domain_affected = True
        
        if domain_affected:
            decay_stats["domains_affected"] += 1
    
    # Calculate average decay
    if decay_stats["topics_affected"] > 0:
        decay_stats["average_decay"] = decay_stats["total_decay"] / decay_stats["topics_affected"]
    
    return decay_stats

def _calculate_average_knowledge_level(ctx: KnowledgeCoreContext) -> float:
    """Calculate the average knowledge level across all domains/topics"""
    levels = []
    for domain, topics in ctx.knowledge_map.domains.items():
        for topic, level in topics.items():
            levels.append(level)
    
    return sum(levels) / len(levels) if levels else 0.0

def _calculate_average_gap_size(ctx: KnowledgeCoreContext) -> float:
    """Calculate the average knowledge gap size"""
    gaps = ctx.knowledge_map.get_knowledge_gaps()
    return sum(gap[2] for gap in gaps) / len(gaps) if gaps else 0.0

# Define the agents
knowledge_agent = Agent(
    name="Knowledge Manager",
    instructions="""You manage a knowledge graph for the system. Your role is to:
1. Add new knowledge to the graph
2. Find similar knowledge and identify relationships
3. Answer questions about the knowledge graph
4. Resolve conflicts between contradictory knowledge
5. Run regular maintenance on the graph

Use your tools to perform these operations efficiently.
""",
    tools=[
        add_knowledge,
        add_relation,
        query_knowledge,
        get_related_knowledge,
        run_integration_cycle,
        save_knowledge,
        load_knowledge,
        get_knowledge_statistics
    ]
)

curiosity_agent = Agent(
    name="Curiosity Manager",
    instructions="""You manage the system's curiosity and knowledge exploration. Your role is to:
1. Identify gaps in the system's knowledge
2. Create exploration targets to fill those gaps
3. Generate questions to explore knowledge domains
4. Track exploration results and prioritize targets
5. Connect related knowledge areas

Use your tools to guide knowledge exploration efficiently.
""",
    tools=[
        identify_knowledge_gaps,
        create_exploration_target,
        get_exploration_targets,
        record_exploration,
        generate_questions
    ]
)

class KnowledgeCoreAgents:
    """
    Orchestrates the knowledge graph and curiosity systems using the Agents SDK.
    This is a drop-in replacement for the original KnowledgeCore class.
    """
    
    def __init__(self, knowledge_store_file: str = "knowledge_store.json"):
        """Initialize the knowledge core with agents"""
        self.context = KnowledgeCoreContext(knowledge_store_file)
        self.knowledge_agent = knowledge_agent
        self.curiosity_agent = curiosity_agent
    
    async def initialize(self, system_references: Dict[str, Any] = None) -> None:
        """Initialize and load knowledge"""
        # Load knowledge
        result = await Runner.run(
            self.knowledge_agent, 
            "Load the knowledge store",
            context=self.context
        )
        logger.info("Knowledge Core initialized with agents.")
    
    async def _send_add_knowledge_prompt(self, type: str, content: Dict[str, Any], 
                           source: str, confidence: float = 0.5, 
                           relations: Optional[List[Dict[str, Any]]] = None) -> str:
        """Add a new knowledge node"""
        prompt = f"Add new knowledge of type '{type}' with content {json.dumps(content)}, " \
                 f"from source '{source}' with confidence {confidence}"
        
        if relations:
            prompt += f" and relations {json.dumps(relations)}"
            
        result = await Runner.run(
            self.knowledge_agent,
            prompt,
            context=self.context
        )
        
        # Extract the node ID from the result
        lines = result.final_output.strip().split('\n')
        for line in lines:
            if line.startswith("node_"):
                return line.strip()
            
        # If we couldn't find a clear node ID, assume it's in the last line
        return lines[-1].strip()
    
    async def add_relation(self, source_id: str, target_id: str, type: str,
                          weight: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a relation between nodes"""
        metadata_str = json.dumps(metadata) if metadata else "null"
        prompt = f"Add relation from node {source_id} to node {target_id} " \
                 f"of type '{type}' with weight {weight} and metadata {metadata_str}"
                 
        result = await Runner.run(
            self.knowledge_agent,
            prompt,
            context=self.context
        )
        
        return "success" in result.final_output.lower() or "true" in result.final_output.lower()
    
    async def query_knowledge(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for knowledge matching criteria"""
        prompt = f"Query knowledge with these parameters: {json.dumps(query)}"
        
        result = await Runner.run(
            self.knowledge_agent,
            prompt,
            context=self.context
        )
        
        # Extract the results from the output
        try:
            # Try to find a JSON array in the output
            import re
            json_match = re.search(r'\[\s*\{.*\}\s*\]', result.final_output, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group(0))
            else:
                # If no clear JSON, assume the whole output is the response
                return json.loads(result.final_output)
        except:
            # Return empty list if we couldn't parse
            logger.error("Failed to parse query results")
            return []
    
    async def get_related_knowledge(self, node_id: str, relation_type: Optional[str] = None,
                                  direction: str = "both", limit: int = 10) -> List[Dict[str, Any]]:
        """Get knowledge related to a node"""
        prompt = f"Get knowledge related to node {node_id}"
        if relation_type:
            prompt += f", with relation type '{relation_type}'"
        prompt += f", direction '{direction}', limit {limit}"
        
        result = await Runner.run(
            self.knowledge_agent,
            prompt,
            context=self.context
        )
        
        # Extract the results from the output
        try:
            # Try to find a JSON array in the output
            import re
            json_match = re.search(r'\[\s*\{.*\}\s*\]', result.final_output, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group(0))
            else:
                # If no clear JSON, assume the whole output is the response
                return json.loads(result.final_output)
        except:
            # Return empty list if we couldn't parse
            logger.error("Failed to parse related knowledge results")
            return []
    
    async def identify_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """Identify gaps in the knowledge"""
        result = await Runner.run(
            self.curiosity_agent,
            "Identify the current knowledge gaps",
            context=self.context
        )
        
        # Extract the results from the output
        try:
            # Try to find a JSON array in the output
            import re
            json_match = re.search(r'\[\s*\{.*\}\s*\]', result.final_output, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group(0))
            else:
                # If no clear JSON, assume the whole output is the response
                return json.loads(result.final_output)
        except:
            # Return empty list if we couldn't parse
            logger.error("Failed to parse knowledge gaps results")
            return []
    
    async def create_exploration_target(self, domain: str, topic: str,
                                      importance: float = 0.5, urgency: float = 0.5,
                                      knowledge_gap: Optional[float] = None) -> str:
        """Create a new exploration target"""
        prompt = f"Create an exploration target for domain '{domain}', topic '{topic}', " \
                 f"importance {importance}, urgency {urgency}"
                 
        if knowledge_gap is not None:
            prompt += f", knowledge gap {knowledge_gap}"
            
        result = await Runner.run(
            self.curiosity_agent,
            prompt,
            context=self.context
        )
        
        # Extract the target ID from the result
        lines = result.final_output.strip().split('\n')
        for line in lines:
            if line.startswith("target_"):
                return line.strip()
                
        # If we couldn't find a clear target ID, assume it's in the last line
        return lines[-1].strip()
    
    async def get_exploration_targets(self, limit: int = 10, min_priority: float = 0.0) -> List[Dict[str, Any]]:
        """Get current exploration targets"""
        prompt = f"Get exploration targets with limit {limit} and minimum priority {min_priority}"
        
        result = await Runner.run(
            self.curiosity_agent,
            prompt,
            context=self.context
        )
        
        # Extract the results from the output
        try:
            # Try to find a JSON array in the output
            import re
            json_match = re.search(r'\[\s*\{.*\}\s*\]', result.final_output, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group(0))
            else:
                # If no clear JSON, assume the whole output is the response
                return json.loads(result.final_output)
        except:
            # Return empty list if we couldn't parse
            logger.error("Failed to parse exploration targets results")
            return []
    
    async def record_exploration(self, target_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Record the result of an exploration"""
        prompt = f"Record exploration for target {target_id} with result {json.dumps(result)}"
        
        result = await Runner.run(
            self.curiosity_agent,
            prompt,
            context=self.context
        )
        
        # Extract the result from the output
        try:
            # Try to find a JSON object in the output
            import re
            json_match = re.search(r'\{.*\}', result.final_output, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group(0))
            else:
                # If no clear JSON, assume the whole output is the response
                return json.loads(result.final_output)
        except:
            # Return empty dict if we couldn't parse
            logger.error("Failed to parse exploration record results")
            return {}
    
    async def generate_questions(self, target_id: str, limit: int = 5) -> List[str]:
        """Generate questions for a target"""
        prompt = f"Generate {limit} questions for exploration target {target_id}"
        
        result = await Runner.run(
            self.curiosity_agent,
            prompt,
            context=self.context
        )
        
        # Extract the questions from the output
        try:
            # Try to find a JSON array in the output
            import re
            json_match = re.search(r'\[\s*".*"\s*\]', result.final_output, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group(0))
            else:
                # Extract questions by looking for numbered or bulleted lists
                lines = result.final_output.strip().split('\n')
                questions = []
                
                for line in lines:
                    # Remove list markers and whitespace
                    clean_line = re.sub(r'^[\s\d\.\-\*]+', '', line).strip()
                    
                    # If it looks like a question (has ? or is substantial)
                    if clean_line.endswith('?') or len(clean_line) > 20:
                        questions.append(clean_line)
                
                return questions[:limit]
        except:
            # Return empty list if we couldn't parse
            logger.error("Failed to parse questions results")
            return []
    
    async def prioritize_domains(self) -> List[Dict[str, Any]]:
        """Prioritize knowledge domains"""
        result = await Runner.run(
            self.curiosity_agent,
            "Prioritize knowledge domains based on gaps and importance",
            context=self.context
        )
        
        # Extract the results from the output
        try:
            # Try to find a JSON array in the output
            import re
            json_match = re.search(r'\[\s*\{.*\}\s*\]', result.final_output, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group(0))
            else:
                # If no clear JSON, assume the whole output is the response
                return json.loads(result.final_output)
        except:
            # Return empty list if we couldn't parse
            logger.error("Failed to parse domain priorities results")
            return []
    
    async def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        result = await Runner.run(
            self.knowledge_agent,
            "Get comprehensive knowledge statistics",
            context=self.context
        )
        
        # Extract the statistics from the output
        try:
            # Try to find a JSON object in the output
            import re
            json_match = re.search(r'\{.*\}', result.final_output, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group(0))
            else:
                # If no clear JSON, assume the whole output is the response
                return json.loads(result.final_output)
        except:
            # Return empty dict if we couldn't parse
            logger.error("Failed to parse knowledge statistics results")
            return {}
    
    async def save_knowledge(self) -> bool:
        """Save the knowledge graph to storage"""
        result = await Runner.run(
            self.knowledge_agent,
            "Save the knowledge graph to storage",
            context=self.context
        )
        
        return "success" in result.final_output.lower() or "true" in result.final_output.lower()
    
    async def run_integration_cycle(self) -> Dict[str, Any]:
        """Manually run an integration cycle"""
        result = await Runner.run(
            self.knowledge_agent,
            "Run a knowledge integration cycle",
            context=self.context
        )
        
        # Extract the result from the output
        try:
            # Try to find a JSON object in the output
            import re
            json_match = re.search(r'\{.*\}', result.final_output, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return {"success": True, "message": result.final_output}
        except:
            # Return basic success dict if we couldn't parse
            return {"success": True}
