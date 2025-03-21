# nyx/core/knowledge_core.py

import asyncio
import json
import logging
import os
import math
import networkx as nx
import numpy as np

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import random
from collections import Counter

logger = logging.getLogger(__name__)

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
        """
        Merge new content into the node's existing content, optionally adjusting confidence and source.

        :param new_content: Additional or overriding fields for the node's content.
        :param new_confidence: If provided, set the node's confidence to this new value.
        :param source: If provided, update the node's source.
        """
        self.content.update(new_content)
        if new_confidence is not None:
            self.confidence = new_confidence
        if source:
            self.source = source
        self.last_accessed = datetime.now()

    def access(self) -> None:
        """
        Record an "access" operation on the node, for usage statistics or confidence decay logic.
        """
        self.last_accessed = datetime.now()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this node to a dictionary for JSON serialization or logging.

        :return: Dictionary representation.
        """
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
        """
        Recreate a KnowledgeNode from a dictionary (e.g., loading from JSON).

        :param data: Dictionary with node fields.
        :return: KnowledgeNode instance.
        """
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
        """
        Convert this relation to a dictionary for JSON serialization or logging.

        :return: Dictionary representation.
        """
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
        """
        Recreate a KnowledgeRelation from a dictionary (e.g., loading from JSON).

        :param data: Dictionary with relation fields.
        :return: KnowledgeRelation instance.
        """
        relation = cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            type=data["type"],
            weight=data["weight"],
            metadata=data["metadata"]
        )
        relation.timestamp = datetime.fromisoformat(data["timestamp"])
        return relation


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


class KnowledgeMap:
    """Maps knowledge domains and gaps"""
    
    def __init__(self):
        self.domains = {}  # domain -> {topic -> level}
        self.connections = {}  # (domain1, topic1) -> [(domain2, topic2, strength), ...]
        self.importance_levels = {}  # (domain, topic) -> importance
        self.last_updated = {}  # (domain, topic) -> datetime
        
    def add_knowledge(self, domain: str, topic: str, 
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


class SimpleReasoningSystem:
    """
    A minimal example of a "reasoning" component that attempts to resolve contradictions.
    Replace/extend with your custom logic or external reasoner calls.
    """

    async def resolve_contradiction(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> Dict[str, Any]:
        """
        A naive contradiction-resolution approach:
        - Always pick the node with higher confidence as "preferred".
        - Return the other node with reduced confidence.

        :param node1: Node dict (from node.to_dict()).
        :param node2: Node dict.
        :return: A dictionary indicating resolution status and which node is preferred.
        """
        # Example: just pick the higher confidence node as "preferred"
        confidence1 = node1["confidence"]
        confidence2 = node2["confidence"]

        if abs(confidence1 - confidence2) < 0.05:
            # If they are almost the same, say we "didn't" resolve it fully
            return {
                "resolved": False,
                "preferred_node": None
            }
        else:
            if confidence1 >= confidence2:
                return {
                    "resolved": True,
                    "preferred_node": node1["id"]
                }
            else:
                return {
                    "resolved": True,
                    "preferred_node": node2["id"]
                }


class CuriositySystem:
    """System for curiosity-driven exploration and knowledge gap identification"""
    
    def __init__(self):
        self.knowledge_map = KnowledgeMap()
        self.exploration_targets = {}  # id -> ExplorationTarget
        self.exploration_history = []
        self.next_target_id = 1
        
        # Configuration
        self.config = {
            "max_active_targets": 20,
            "exploration_budget": 0.5,  # 0.0 to 1.0
            "novelty_bias": 0.7,        # 0.0 to 1.0
            "importance_threshold": 0.3, # Minimum importance to consider
            "knowledge_decay_rate": 0.01 # Rate of knowledge decay over time
        }
        
        # Statistics
        self.stats = {
            "total_explorations": 0,
            "successful_explorations": 0,
            "knowledge_gained": 0.0,
            "avg_importance": 0.0
        }
        
    async def add_knowledge(self, domain: str, topic: str, level: float, 
                         importance: float = 0.5) -> None:
        """Add knowledge to the knowledge map"""
        self.knowledge_map.add_knowledge(domain, topic, level, importance)
        
        # Update related exploration targets
        for target_id, target in self.exploration_targets.items():
            if target.domain == domain and target.topic == topic:
                # Update knowledge gap
                original_gap = target.knowledge_gap
                target.knowledge_gap = max(0.0, 1.0 - level)
                
                # Update priority
                target.update_priority()
                
                # If gap is closed, mark as explored
                if target.knowledge_gap < 0.2 and not target.last_explored:
                    target.record_exploration({
                        "source": "external_knowledge",
                        "knowledge_gained": original_gap - target.knowledge_gap
                    })
        
    async def add_knowledge_connection(self, domain1: str, topic1: str,
                                   domain2: str, topic2: str,
                                   strength: float = 0.5) -> None:
        """Add a connection between knowledge topics"""
        self.knowledge_map.add_connection(domain1, topic1, domain2, topic2, strength)
        
    async def identify_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """Identify knowledge gaps from the knowledge map"""
        # Get gaps from knowledge map
        gaps = self.knowledge_map.get_knowledge_gaps()
        
        # Convert to dictionaries
        gap_dicts = []
        for domain, topic, gap_size, importance in gaps:
            if importance >= self.config["importance_threshold"]:
                gap_dicts.append({
                    "domain": domain,
                    "topic": topic,
                    "gap_size": gap_size,
                    "importance": importance,
                    "priority": gap_size * importance
                })
        
        # Sort by priority (descending)
        gap_dicts.sort(key=lambda x: x["priority"], reverse=True)
        
        return gap_dicts
        
    async def create_exploration_target(self, domain: str, topic: str,
                                     importance: float = 0.5, urgency: float = 0.5,
                                     knowledge_gap: Optional[float] = None) -> str:
        """Create a new exploration target"""
        # Generate target id
        target_id = f"target_{self.next_target_id}"
        self.next_target_id += 1
        
        # Get knowledge gap from knowledge map if not provided
        if knowledge_gap is None:
            level = self.knowledge_map.get_knowledge_level(domain, topic)
            knowledge_gap = 1.0 - level
        
        # Create exploration target
        target = ExplorationTarget(
            target_id=target_id,
            domain=domain,
            topic=topic,
            importance=importance,
            urgency=urgency,
            knowledge_gap=knowledge_gap
        )
        
        # Store target
        self.exploration_targets[target_id] = target
        
        return target_id
        
    async def get_exploration_targets(self, limit: int = 10, 
                                  min_priority: float = 0.0) -> List[Dict[str, Any]]:
        """Get current exploration targets sorted by priority"""
        # Update priorities
        for target in self.exploration_targets.values():
            target.update_priority()
        
        # Sort by priority
        sorted_targets = sorted(
            [t for t in self.exploration_targets.values() if t.priority_score >= min_priority],
            key=lambda x: x.priority_score,
            reverse=True
        )
        
        # Apply limit
        targets = sorted_targets[:limit]
        
        # Convert to dictionaries
        return [target.to_dict() for target in targets]
        
    async def record_exploration(self, target_id: str, 
                             result: Dict[str, Any]) -> Dict[str, Any]:
        """Record the result of an exploration"""
        if target_id not in self.exploration_targets:
            return {"error": f"Target {target_id} not found"}
        
        # Get target
        target = self.exploration_targets[target_id]
        
        # Record exploration
        target.record_exploration(result)
        
        # Update knowledge map if knowledge gained
        if "knowledge_gained" in result and result["knowledge_gained"] > 0:
            level = 1.0 - target.knowledge_gap
            self.knowledge_map.add_knowledge(
                target.domain,
                target.topic,
                level,
                target.importance
            )
        
        # Update statistics
        self.stats["total_explorations"] += 1
        if result.get("success", False):
            self.stats["successful_explorations"] += 1
        self.stats["knowledge_gained"] += result.get("knowledge_gained", 0.0)
        
        # Calculate average importance
        total_importance = sum(target.importance for target in self.exploration_targets.values())
        if self.exploration_targets:
            self.stats["avg_importance"] = total_importance / len(self.exploration_targets)
        
        # Add to exploration history
        self.exploration_history.append({
            "timestamp": datetime.now().isoformat(),
            "target_id": target_id,
            "domain": target.domain,
            "topic": target.topic,
            "result": result
        })
        
        # Prune history if needed
        if len(self.exploration_history) > 1000:
            self.exploration_history = self.exploration_history[-1000:]
        
        return {
            "target": target.to_dict(),
            "updated_knowledge_level": 1.0 - target.knowledge_gap
        }
        
    async def generate_questions(self, target_id: str, limit: int = 5) -> List[str]:
        """Generate questions to explore a target"""
        if target_id not in self.exploration_targets:
            return []
        
        # Get target
        target = self.exploration_targets[target_id]
        
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
        related_topics = self.knowledge_map.get_related_topics(target.domain, target.topic)
        for domain, topic, strength in related_topics[:3]:  # Use up to 3 related topics
            questions.append(f"How does {target.topic} relate to {topic}?")
        
        # Add questions to target
        for question in questions:
            target.add_related_question(question)
        
        return questions[:limit]
        
    async def apply_knowledge_decay(self) -> Dict[str, Any]:
        """Apply decay to knowledge levels over time"""
        decay_stats = {
            "domains_affected": 0,
            "topics_affected": 0,
            "total_decay": 0.0,
            "average_decay": 0.0
        }
        
        now = datetime.now()
        decay_factor = self.config["knowledge_decay_rate"]
        
        # Apply decay to all domains and topics
        for domain, topics in self.knowledge_map.domains.items():
            domain_affected = False
            
            for topic, level in list(topics.items()):
                # Get last update time
                key = (domain, topic)
                last_updated = self.knowledge_map.last_updated.get(key)
                
                if last_updated:
                    # Calculate days since last update
                    days_since = (now - last_updated).total_seconds() / (24 * 3600)
                    
                    # Apply decay based on time
                    if days_since > 30:  # Only decay knowledge older than 30 days
                        decay_amount = level * decay_factor * (days_since / 30)
                        new_level = max(0.0, level - decay_amount)
                        
                        # Update knowledge level
                        self.knowledge_map.domains[domain][topic] = new_level
                        
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
        
    async def get_curiosity_statistics(self) -> Dict[str, Any]:
        """Get statistics about the curiosity system"""
        # Calculate domains and topics
        domain_count = len(self.knowledge_map.domains)
        topic_count = sum(len(topics) for topics in self.knowledge_map.domains.values())
        
        # Calculate knowledge levels
        knowledge_levels = []
        for domain, topics in self.knowledge_map.domains.items():
            for topic, level in topics.items():
                knowledge_levels.append(level)
        
        avg_knowledge = sum(knowledge_levels) / len(knowledge_levels) if knowledge_levels else 0.0
        
        # Calculate knowledge gaps
        gaps = self.knowledge_map.get_knowledge_gaps()
        avg_gap_size = sum(gap[2] for gap in gaps) / len(gaps) if gaps else 0.0
        
        # Calculate exploration statistics
        active_targets = len(self.exploration_targets)
        explored_targets = sum(1 for t in self.exploration_targets.values() if t.exploration_count > 0)
        
        # Create statistics dictionary
        statistics = {
            "knowledge_map": {
                "domain_count": domain_count,
                "topic_count": topic_count,
                "connection_count": sum(len(connections) for connections in self.knowledge_map.connections.values()),
                "average_knowledge_level": avg_knowledge,
                "average_gap_size": avg_gap_size,
                "knowledge_gap_count": len(gaps)
            },
            "exploration": {
                "active_targets": active_targets,
                "explored_targets": explored_targets,
                "exploration_ratio": explored_targets / active_targets if active_targets > 0 else 0.0,
                "success_rate": self.stats["successful_explorations"] / self.stats["total_explorations"] if self.stats["total_explorations"] > 0 else 0.0,
                "total_knowledge_gained": self.stats["knowledge_gained"],
                "average_target_importance": self.stats["avg_importance"]
            },
            "configuration": self.config
        }
        
        return statistics

    async def prioritize_domains(self) -> List[Dict[str, Any]]:
        """Prioritize knowledge domains based on gaps and importance"""
        domain_stats = {}
        
        # Collect statistics for each domain
        for domain, topics in self.knowledge_map.domains.items():
            domain_topics = len(topics)
            domain_knowledge = sum(level for level in topics.values()) / domain_topics if domain_topics > 0 else 0.0
            
            # Calculate domain importance
            domain_importance = 0.0
            for topic in topics:
                key = (domain, topic)
                importance = self.knowledge_map.importance_levels.get(key, 0.5)
                domain_importance += importance
            domain_importance /= domain_topics if domain_topics > 0 else 1.0
            
            # Calculate domain knowledge gap
            domain_gap = 1.0 - domain_knowledge
            
            # Calculate domain priority
            domain_priority = domain_gap * domain_importance
            
            domain_stats[domain] = {
                "topic_count": domain_topics,
                "average_knowledge": domain_knowledge,
                "average_importance": domain_importance,
                "knowledge_gap": domain_gap,
                "priority": domain_priority
            }
        
        # Sort domains by priority
        prioritized_domains = [
            {"domain": domain, **stats}
            for domain, stats in domain_stats.items()
        ]
        prioritized_domains.sort(key=lambda x: x["priority"], reverse=True)
        
        return prioritized_domains
        
    async def save_state(self, file_path: str) -> bool:
        """Save current state to file"""
        try:
            state = {
                "knowledge_map": self.knowledge_map.to_dict(),
                "exploration_targets": {tid: target.to_dict() for tid, target in self.exploration_targets.items()},
                "exploration_history": self.exploration_history,
                "next_target_id": self.next_target_id,
                "config": self.config,
                "stats": self.stats,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving curiosity state: {e}")
            return False
        
    async def load_state(self, file_path: str) -> bool:
        """Load state from file"""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Load knowledge map
            self.knowledge_map = KnowledgeMap.from_dict(state["knowledge_map"])
            
            # Load exploration targets
            self.exploration_targets = {}
            for tid, target_data in state["exploration_targets"].items():
                self.exploration_targets[tid] = ExplorationTarget.from_dict(target_data)
            
            # Load other attributes
            self.exploration_history = state["exploration_history"]
            self.next_target_id = state["next_target_id"]
            self.config = state["config"]
            self.stats = state["stats"]
            
            return True
        except Exception as e:
            logger.error(f"Error loading curiosity state: {e}")
            return False


class KnowledgeCore:
    """
    Main system that integrates knowledge across different components, storing data
    in a directed graph and using embeddings for similarity detection, plus a
    reasoning system for contradiction handling.
    """

    def __init__(self,
                 knowledge_store_file: str = "knowledge_store.json"):
        """
        :param knowledge_store_file: Path to a JSON file used for persisting and loading knowledge at startup.
        """
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.node_embeddings: Dict[str, np.ndarray] = {}

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

        # Default config: you can customize or load from external config
        self.config = {
            "conflict_threshold": 0.7,
            "support_threshold": 0.7,
            "similarity_threshold": 0.8,  # for merges
            "decay_rate": 0.01,
            "integration_cycle_interval": 10,  # in minutes
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

        # System references (can be set via initialize())
        self.memory_system = None
        self.reasoning_system = None  # We'll use the SimpleReasoningSystem by default
        self.knowledge_store_file = knowledge_store_file

        # Setup Curiosity system for exploration
        self.curiosity_system = CuriositySystem()

        # Try to load the model
        self._embedding_model = None
        self._try_load_embedding_model()

    def _try_load_embedding_model(self):
        """Try to load an embedding model if available"""
        if self.config["enable_embeddings"]:
            try:
                # First, try to import SentenceTransformer
                try:
                    from sentence_transformers import SentenceTransformer
                    # Use a standard SentenceTransformer model
                    self._embedding_model = SentenceTransformer('all-mpnet-base-v2')
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
                        
                        self._embedding_model = SimpleEmbedder('sentence-transformers/all-mpnet-base-v2')
                        logger.info("Loaded embedding model (Transformers).")
                    except ImportError:
                        logger.warning("Neither SentenceTransformer nor Transformers is available.")
                        self._embedding_model = None
            except Exception as e:
                logger.warning(f"Error loading embedding model: {e}")
                self._embedding_model = None

    async def initialize(self,
                         system_references: Dict[str, Any]) -> None:
        """
        Initialize the knowledge integration system, load references to other subsystems,
        and attempt to load existing knowledge from disk.

        :param system_references: A dictionary that may contain 'memory_system' or 'reasoning_system' references.
        """
        if "memory_system" in system_references:
            self.memory_system = system_references["memory_system"]
        if "reasoning_system" in system_references:
            self.reasoning_system = system_references["reasoning_system"]
        else:
            # Provide a default simple reasoning system if none is provided
            self.reasoning_system = SimpleReasoningSystem()

        await self._load_knowledge()
        
        # Initialize curiosity system
        if self.memory_system:
            # Pass memory system reference if available
            system_refs = {"memory_system": self.memory_system}
            if self.reasoning_system:
                system_refs["reasoning_system"] = self.reasoning_system
        
        logger.info("Knowledge Core initialized.")

    async def _load_knowledge(self) -> None:
        """
        Load existing nodes/relations from a JSON file (if it exists) and populate
        the graph with them.
        """
        if not os.path.exists(self.knowledge_store_file):
            logger.info("No existing knowledge store found; starting fresh.")
            return

        try:
            with open(self.knowledge_store_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load nodes
            for node_data in data.get("nodes", []):
                node = KnowledgeNode.from_dict(node_data)
                self.nodes[node.id] = node
                self.graph.add_node(node.id, **node.to_dict())
                # Keep track of the highest numeric suffix for next_node_id
                try:
                    numeric_id = int(node.id.split("_")[-1])
                    if numeric_id >= self.next_node_id:
                        self.next_node_id = numeric_id + 1
                except:
                    pass

            # Load relations
            for rel_data in data.get("relations", []):
                rel = KnowledgeRelation.from_dict(rel_data)
                self.graph.add_edge(
                    rel.source_id,
                    rel.target_id,
                    **rel.to_dict()
                )
                # Also update supporting/conflicting references
                if rel.type == "supports":
                    self.nodes[rel.source_id].supporting_nodes.append(rel.target_id)
                elif rel.type == "contradicts":
                    self.nodes[rel.source_id].conflicting_nodes.append(rel.target_id)

            # Optionally rebuild embeddings
            if self.config["enable_embeddings"] and self._embedding_model:
                await self._rebuild_embeddings()

            logger.info(f"Knowledge store loaded with {len(self.nodes)} nodes and {self.graph.number_of_edges()} edges.")
        except Exception as e:
            logger.error(f"Error loading knowledge: {str(e)}")

    async def _rebuild_embeddings(self) -> None:
        """
        Rebuild embeddings for all nodes from scratch, e.g. after loading from disk.
        """
        for node in self.nodes.values():
            await self._update_node_embedding(node)

    async def save_knowledge(self) -> None:
        """
        Save all nodes/relations to a JSON file, to persist knowledge across restarts.
        """
        try:
            # Build list of node dicts
            nodes_data = [node.to_dict() for node in self.nodes.values()]

            # Build list of relation dicts from graph edges
            relations_data = []
            for source, target, data in self.graph.edges(data=True):
                # We only store the direct fields from KnowledgeRelation
                # that match the schema in KnowledgeRelation.to_dict().
                # Some fields like 'source_id' / 'target_id' might be duplicated,
                # so we just re-serialize them carefully.
                relation_dict = {
                    "source_id": source,
                    "target_id": target,
                    "type": data.get("type"),
                    "weight": data.get("weight", 1.0),
                    "timestamp": data.get("timestamp"),
                    "metadata": data.get("metadata", {})
                }
                # timestamp might be isoformat string in 'data'
                # be sure to handle it similarly to from_dict
                if isinstance(relation_dict["timestamp"], datetime):
                    relation_dict["timestamp"] = relation_dict["timestamp"].isoformat()
                relations_data.append(relation_dict)

            store_data = {
                "nodes": nodes_data,
                "relations": relations_data
            }

            with open(self.knowledge_store_file, "w", encoding="utf-8") as f:
                json.dump(store_data, f, indent=2)
            logger.info("Knowledge store saved successfully.")
        except Exception as e:
            logger.error(f"Error saving knowledge: {str(e)}")

    async def add_knowledge(self,
                            type: str,
                            content: Dict[str, Any],
                            source: str,
                            confidence: float = 0.5,
                            relations: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Add a new piece of knowledge (node) to the system.

        :param type: The node's knowledge type (concept, fact, rule, etc.)
        :param content: Main data for the knowledge node.
        :param source: Where this knowledge came from (system name, user ID, etc.)
        :param confidence: Float in [0..1] for how reliable the knowledge is.
        :param relations: Optional list of dicts describing relations to other nodes, e.g.:
                          [{"target_id": "node_3", "type": "supports", "weight": 0.8}, ...]
        :return: The ID of the new or integrated node.
        """
        node_id = f"node_{self.next_node_id}"
        self.next_node_id += 1

        node = KnowledgeNode(
            id=node_id,
            type=type,
            content=content,
            source=source,
            confidence=confidence
        )

        # Check if there's an existing node that's very similar
        similar_nodes = await self._find_similar_nodes(node)

        if similar_nodes:
            most_similar_id, similarity_val = similar_nodes[0]
            if similarity_val > self.config["similarity_threshold"]:
                # Merge into the existing node
                return await self._integrate_with_existing(node, most_similar_id, similarity_val)

        # Otherwise, create a brand-new node
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.to_dict())

        # Generate or update embedding if enabled
        if self.config["enable_embeddings"] and self._embedding_model:
            await self._add_node_embedding(node)

        # Add new relations if provided
        if relations:
            for r in relations:
                await self.add_relation(
                    source_id=node_id,
                    target_id=r["target_id"],
                    type=r["type"],
                    weight=r.get("weight", 1.0),
                    metadata=r.get("metadata", {})
                )

        # Optionally add "similar_to" edges for moderately similar nodes
        for sim_id, sim_val in similar_nodes:
            if sim_val > self.config["similarity_threshold"] * 0.6:  # Use up to 3 related topics
                await self.add_relation(
                    source_id=node_id,
                    target_id=sim_id,
                    type="similar_to",
                    weight=sim_val,
                    metadata={"auto_generated": True}
                )

        self.integration_stats["nodes_added"] += 1
        await self._check_integration_cycle()

        return node_id

    async def add_relation(self,
                           source_id: str,
                           target_id: str,
                           type: str,
                           weight: float = 1.0,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Connect two existing knowledge nodes with a directed relation (edge).

        :param source_id: ID of the source node
        :param target_id: ID of the target node
        :param type: e.g., "supports", "contradicts", "specializes", "similar_to"
        :param weight: Strength of the relation
        :param metadata: Additional fields to store about the relation
        :return: True if successful, False if either node doesn't exist
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.warning(f"Cannot add relation: node {source_id} or {target_id} does not exist.")
            return False

        rel = KnowledgeRelation(
            source_id=source_id,
            target_id=target_id,
            type=type,
            weight=weight,
            metadata=metadata or {}
        )

        self.graph.add_edge(source_id, target_id, **rel.to_dict())

        # Also update supporting/conflicting references in the source node
        if type == "supports":
            if target_id not in self.nodes[source_id].supporting_nodes:
                self.nodes[source_id].supporting_nodes.append(target_id)
        elif type == "contradicts":
            if target_id not in self.nodes[source_id].conflicting_nodes:
                self.nodes[source_id].conflicting_nodes.append(target_id)
            await self._handle_contradiction(source_id, target_id)

        self.integration_stats["relations_added"] += 1
        
        # Update curiosity system with connection if appropriate
        if type in ["related", "similar_to", "specializes"]:
            # Extract domain and topic from node content if available
            source_node = self.nodes[source_id]
            target_node = self.nodes[target_id]
            
            source_domain = source_node.content.get("domain", source_node.type)
            source_topic = source_node.content.get("topic", list(source_node.content.keys())[0] if source_node.content else "unknown")
            
            target_domain = target_node.content.get("domain", target_node.type)
            target_topic = target_node.content.get("topic", list(target_node.content.keys())[0] if target_node.content else "unknown")
            
            # Add connection to curiosity system
            await self.curiosity_system.add_knowledge_connection(
                source_domain, source_topic,
                target_domain, target_topic,
                weight
            )
        
        return True

    async def _integrate_with_existing(self,
                                       new_node: KnowledgeNode,
                                       existing_id: str,
                                       similarity: float) -> str:
        """
        Merge the content of a new node into an existing node if they're highly similar.

        :param new_node: The newly proposed KnowledgeNode
        :param existing_id: ID of the existing node
        :param similarity: A measure of how similar they are
        :return: The existing node's ID (which now contains updated content)
        """
        existing_node = self.nodes[existing_id]
        existing_node.access()

        # Weighted combination of content and confidence
        if new_node.confidence > existing_node.confidence:
            # new_node is more reliable, use it as the basis
            updated_content = new_node.content.copy()
            # keep old content that's missing in new
            for key, val in existing_node.content.items():
                if key not in updated_content:
                    updated_content[key] = val

            updated_conf = (new_node.confidence * 0.7 +
                            existing_node.confidence * 0.3)
            existing_node.update(updated_content, updated_conf,
                                 f"{existing_node.source}+{new_node.source}")
        else:
            # keep existing node as basis
            for key, val in new_node.content.items():
                if key not in existing_node.content:
                    existing_node.content[key] = val

            updated_conf = (existing_node.confidence * 0.7 +
                            new_node.confidence * 0.3)
            existing_node.confidence = updated_conf
            existing_node.last_accessed = datetime.now()

        # Update node data in the graph
        self.graph.nodes[existing_id].update(existing_node.to_dict())

        # Update embeddings if needed
        if self.config["enable_embeddings"] and self._embedding_model:
            await self._update_node_embedding(existing_node)

        logger.debug(f"Integrated node {new_node.id} into existing node {existing_id}.")
        return existing_id

    async def _handle_contradiction(self, node1_id: str, node2_id: str) -> None:
        """
        Called when two nodes have a 'contradicts' relation. Possibly attempt resolution via reasoning system.
        """
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]

        # If both are high confidence, we have a real conflict
        if node1.confidence > 0.7 and node2.confidence > 0.7:
            logger.info(f"Detected high-confidence conflict between {node1_id} and {node2_id}")
            if self.reasoning_system:
                try:
                    resolution = await self.reasoning_system.resolve_contradiction(node1.to_dict(),
                                                                                  node2.to_dict())
                    if resolution["resolved"]:
                        preferred = resolution["preferred_node"]
                        if preferred == node1_id:
                            node2.confidence *= 0.7
                        else:
                            node1.confidence *= 0.7

                        self.integration_stats["conflicts_resolved"] += 1
                        logger.info(f"Resolved conflict. Preferred node: {preferred}")
                except Exception as e:
                    logger.error(f"Error resolving contradiction: {str(e)}")

        # If there's a big confidence gap, just reduce the lower
        elif abs(node1.confidence - node2.confidence) > 0.3:
            if node1.confidence < node2.confidence:
                node1.confidence *= 0.9
            else:
                node2.confidence *= 0.9
            logger.debug(f"Adjusted confidence in conflict: {node1_id} vs {node2_id}.")

    async def _find_similar_nodes(self, node: KnowledgeNode) -> List[Tuple[str, float]]:
        """
        Check existing nodes for similarity to the incoming node.

        :return: List of (node_id, similarity_score) sorted descending by score.
        """
        if self.config["enable_embeddings"] and self._embedding_model:
            # Use real embeddings
            embedding = await self._generate_embedding(node)
            similar_list: List[Tuple[str, float]] = []
            for node_id, emb in self.node_embeddings.items():
                sim = self._calculate_embedding_similarity(embedding, emb)
                if sim > (self.config["similarity_threshold"] * 0.5):
                    similar_list.append((node_id, sim))
            similar_list.sort(key=lambda x: x[1], reverse=True)
            return similar_list
        else:
            # Fallback: do naive content-based matching
            results = []
            for node_id, existing_node in self.nodes.items():
                if existing_node.type == node.type:
                    content_sim = self._calculate_content_similarity(node.content, existing_node.content)
                    if content_sim > (self.config["similarity_threshold"] * 0.5):
                        results.append((node_id, content_sim))
            results.sort(key=lambda x: x[1], reverse=True)
            return results

    async def _add_node_embedding(self, node: KnowledgeNode) -> None:
        """
        Create and store an embedding for a newly added node.
        """
        emb = await self._generate_embedding(node)
        self.node_embeddings[node.id] = emb

    async def _update_node_embedding(self, node: KnowledgeNode) -> None:
        """
        Recompute embeddings for a node whose content changed.
        """
        emb = await self._generate_embedding(node)
        self.node_embeddings[node.id] = emb

    async def _generate_embedding(self, node: KnowledgeNode) -> np.ndarray:
        """
        Convert the node's content to an embedding using a real (or fallback) model.
        """
        # If a real model is loaded, do that:
        if self._embedding_model:
            # Convert content to a "document" string
            content_str = json.dumps(node.content, ensure_ascii=False)
            embedding = self._embedding_model.encode([content_str], show_progress_bar=False)
            # sentence-transformers returns a 2D array [ [vector], ]
            # We'll just return the 1D vector
            return embedding[0]

        # Otherwise fallback: naive numeric approach
        fallback_vector = np.zeros(128, dtype=np.float32)
        content_str = json.dumps(node.content)
        for i, c in enumerate(content_str):
            idx = i % 128
            fallback_vector[idx] += (ord(c) % 96) / 100.0
        norm = np.linalg.norm(fallback_vector)
        if norm > 0:
            fallback_vector = fallback_vector / norm
        return fallback_vector

    def _calculate_embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Standard cosine similarity between two vectors.
        """
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def _calculate_content_similarity(self, c1: Dict[str, Any], c2: Dict[str, Any]) -> float:
        """
        A naive Jaccard/overlap approach for dictionary content if no embeddings are available.
        """
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

    async def _check_integration_cycle(self) -> None:
        """
        See if we should run an integration cycle now (based on time).
        """
        now = datetime.now()
        elapsed = (now - self.last_integration_cycle).total_seconds()
        # If enough time has passed, do it
        if elapsed > self.config["integration_cycle_interval"] * 60:
            await self._run_integration_cycle()

    async def _run_integration_cycle(self) -> None:
        """
        Perform housekeeping tasks: pruning, conflict resolution, merging, inference, etc.
        """
        logger.info("Running knowledge integration cycle.")
        self.last_integration_cycle = datetime.now()
        self.integration_stats["integration_cycles"] += 1

        # 1. prune old or low-confidence nodes
        await self._prune_nodes()

        # 2. resolve conflicts
        await self._resolve_conflicts()

        # 3. consolidate similar nodes
        await self._consolidate_similar_nodes()

        # 4. infer new relations
        await self._infer_relations()

        # 5. apply confidence decay
        self._apply_confidence_decay()

        # 6. update stats
        await self._update_knowledge_stats()
        
        # 7. apply knowledge decay in curiosity system
        await self.curiosity_system.apply_knowledge_decay()

        logger.info("Integration cycle completed.")
        # Optionally save knowledge
        await self.save_knowledge()

    async def _prune_nodes(self) -> None:
        """
        Remove old or low-confidence nodes that are rarely accessed.
        """
        now = datetime.now()
        to_prune = []
        for node_id, node in self.nodes.items():
            age_days = (now - node.timestamp).total_seconds() / (24 * 3600)
            if (age_days > self.config["max_node_age_days"] and
                    node.confidence < self.config["pruning_confidence_threshold"] and
                    node.access_count < 3):
                to_prune.append(node_id)

        for nid in to_prune:
            await self._remove_node(nid)

        if to_prune:
            logger.info(f"Pruned {len(to_prune)} node(s).")

    async def _remove_node(self, node_id: str) -> None:
        """
        Delete a node (and all edges from/to it) from memory.
        """
        if node_id in self.nodes:
            del self.nodes[node_id]
        if node_id in self.graph:
            self.graph.remove_node(node_id)
        if node_id in self.node_embeddings:
            del self.node_embeddings[node_id]

        # Remove from any supporting/conflicting references
        for other_node in self.nodes.values():
            if node_id in other_node.supporting_nodes:
                other_node.supporting_nodes.remove(node_id)
            if node_id in other_node.conflicting_nodes:
                other_node.conflicting_nodes.remove(node_id)

    async def _resolve_conflicts(self) -> None:
        """
        Look for edges that have type=contradicts, handle them.
        """
        contradictions = []
        for s, t, data in self.graph.edges(data=True):
            if data.get("type") == "contradicts":
                contradictions.append((s, t))

        for (nid1, nid2) in contradictions:
            if nid1 in self.nodes and nid2 in self.nodes:
                await self._handle_contradiction(nid1, nid2)

    async def _consolidate_similar_nodes(self) -> None:
        """
        Merge nodes that are extremely similar to reduce duplication.
        """
        if not self.config["enable_embeddings"] or not self._embedding_model:
            # If embeddings are disabled or no model is loaded, skip
            return

        node_ids = list(self.nodes.keys())
        if len(node_ids) < 2:
            return

        to_merge = []
        # For performance, we limit to 1000 comparisons
        max_comparisons = min(1000, len(node_ids) * (len(node_ids) - 1) // 2)
        comparisons = 0

        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                comparisons += 1
                if comparisons > max_comparisons:
                    break

                id1 = node_ids[i]
                id2 = node_ids[j]
                if id1 not in self.node_embeddings or id2 not in self.node_embeddings:
                    continue

                # Skip if they have a direct "contradicts" edge
                if id2 in self.nodes[id1].conflicting_nodes:
                    continue

                emb1 = self.node_embeddings[id1]
                emb2 = self.node_embeddings[id2]
                sim = self._calculate_embedding_similarity(emb1, emb2)
                if sim > self.config["similarity_threshold"]:
                    to_merge.append((id1, id2, sim))

        # Sort merges by descending similarity
        to_merge.sort(key=lambda x: x[2], reverse=True)
        merged_nodes = set()
        merges_done = 0

        for (id1, id2, sim) in to_merge:
            if id1 in merged_nodes or id2 in merged_nodes:
                continue
            if id1 not in self.nodes or id2 not in self.nodes:
                continue

            # pick the node with higher confidence as the target
            if self.nodes[id1].confidence >= self.nodes[id2].confidence:
                target_id, source_id = id1, id2
            else:
                target_id, source_id = id2, id1

            await self._merge_nodes(target_id, source_id)
            merged_nodes.add(source_id)
            merges_done += 1

        if merges_done > 0:
            logger.info(f"Consolidated {merges_done} similar node pairs.")

    async def _merge_nodes(self, target_id: str, source_id: str) -> None:
        """
        Integrate the content of 'source_id' into 'target_id', then remove 'source_id'.
        """
        if target_id not in self.nodes or source_id not in self.nodes:
            return

        target_node = self.nodes[target_id]
        source_node = self.nodes[source_id]

        # Merge content
        for key, val in source_node.content.items():
            if key not in target_node.content:
                target_node.content[key] = val

        # Merge metadata
        merged_from = target_node.metadata.get("merged_from", [])
        merged_from.append(source_id)
        target_node.metadata["merged_from"] = merged_from
        target_node.metadata["merged_timestamp"] = datetime.now().isoformat()

        # Weighted confidence
        new_conf = (target_node.confidence * 0.7 + source_node.confidence * 0.3)
        target_node.confidence = min(1.0, new_conf)

        # Combine sources
        if source_node.source not in target_node.source:
            target_node.source = f"{target_node.source}+{source_node.source}"

        # Transfer edges
        # Outgoing edges from source
        for succ in list(self.graph.successors(source_id)):
            if succ != target_id:
                edge_data = self.graph.get_edge_data(source_id, succ)
                self.graph.add_edge(target_id, succ, **edge_data)

        # Incoming edges to source
        for pred in list(self.graph.predecessors(source_id)):
            if pred != target_id:
                edge_data = self.graph.get_edge_data(pred, source_id)
                self.graph.add_edge(pred, target_id, **edge_data)

        # Merge supporting/conflicting
        for n_id in source_node.supporting_nodes:
            if n_id != target_id and n_id not in target_node.supporting_nodes:
                target_node.supporting_nodes.append(n_id)
        for n_id in source_node.conflicting_nodes:
            if n_id != target_id and n_id not in target_node.conflicting_nodes:
                target_node.conflicting_nodes.append(n_id)

        # Update graph data
        self.graph.nodes[target_id].update(target_node.to_dict())

        # Remove the source node
        await self._remove_node(source_id)

        # Possibly update embedding of the target node
        if self.config["enable_embeddings"] and self._embedding_model:
            await self._update_node_embedding(target_node)

    async def _infer_relations(self) -> None:
        """
        Demonstrate a simple "transitive" inference:
          A -> B, B -> C => maybe A -> C if certain relation types align
        """
        new_relations = []

        for n1 in self.nodes:
            for n2 in self.graph.successors(n1):
                data12 = self.graph.get_edge_data(n1, n2)
                rel1_type = data12.get("type")

                for n3 in self.graph.successors(n2):
                    if n3 == n1:
                        continue
                    data23 = self.graph.get_edge_data(n2, n3)
                    rel2_type = data23.get("type")

                    if self.graph.has_edge(n1, n3):
                        continue

                    # Infer a new relation type from (n1->n2, n2->n3)
                    inferred = self._infer_relation_type(rel1_type, rel2_type)
                    if inferred:
                        new_relations.append((n1, n3, inferred))

        # add them
        for (src, trg, rtype) in new_relations:
            if src in self.nodes and trg in self.nodes:
                await self.add_relation(src, trg, rtype, weight=0.7, metadata={"inferred": True})
        if new_relations:
            logger.info(f"Inferred {len(new_relations)} new relations.")

    def _infer_relation_type(self, t1: str, t2: str) -> Optional[str]:
        """
        Basic example composition rules:
          supports + supports -> supports
          supports + contradicts -> contradicts
          contradicts + supports -> contradicts
          contradicts + contradicts -> supports
          ... etc.
        """
        composition = {
            ("supports", "supports"): "supports",
            ("supports", "contradicts"): "contradicts",
            ("contradicts", "supports"): "contradicts",
            ("contradicts", "contradicts"): "supports",
            ("specializes", "specializes"): "specializes",
            ("has_property", "specializes"): "has_property"
        }
        return composition.get((t1, t2), None)

    def _apply_confidence_decay(self) -> None:
        """
        Decrease the confidence of nodes that have not been accessed for a long period.
        """
        now = datetime.now()
        for node in self.nodes.values():
            days_inactive = (now - node.last_accessed).total_seconds() / (24 * 3600)
            if days_inactive > 7:  # e.g. if not accessed in over a week
                factor = 1.0 - (self.config["decay_rate"] * min(days_inactive / 30.0, 1.0))
                node.confidence *= factor
                self.graph.nodes[node.id].update(node.to_dict())

    async def _update_knowledge_stats(self) -> None:
        """
        Refresh stats about the node/edge counts and types for reporting.
        """
        self.integration_stats["node_count"] = len(self.nodes)
        self.integration_stats["relation_count"] = self.graph.number_of_edges()

        # Tally node types
        node_types = {}
        for n in self.nodes.values():
            node_types[n.type] = node_types.get(n.type, 0) + 1
        self.integration_stats["node_types"] = node_types

        # Tally relation types
        relation_types = {}
        for s, t, data in self.graph.edges(data=True):
            rtype = data.get("type")
            if rtype:
                relation_types[rtype] = relation_types.get(rtype, 0) + 1
        self.integration_stats["relation_types"] = relation_types

    async def query_knowledge(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search the knowledge graph for nodes matching certain criteria (type, content filter, relation filter, limit).
        Results are sorted by confidence descending.

        :param query: e.g.,
            {
                "type": "fact",
                "content_filter": {"domain": "medical"},
                "relation_filter": {"type": "supports", "node_id": "node_5", "direction": "outgoing"},
                "limit": 5
            }
        :return: A list of node dictionaries.
        """
        self.integration_stats["knowledge_queries"] += 1

        node_type = query.get("type")
        content_filter = query.get("content_filter", {})
        relation_filter = query.get("relation_filter", {})
        limit = query.get("limit", 10)

        # Build a cache key
        cache_key = json.dumps({
            "type": node_type,
            "content_filter": content_filter,
            "relation_filter": relation_filter,
            "limit": limit
        }, sort_keys=True)

        # If we already have an answer in our query_cache that's <1 minute old, reuse
        now = datetime.now()
        if cache_key in self.query_cache:
            entry = self.query_cache[cache_key]
            cached_time = datetime.fromisoformat(entry["timestamp"])
            if (now - cached_time).total_seconds() < 60:
                return entry["results"]

        matching = []

        # Evaluate each node
        for nid, node in self.nodes.items():
            # type filter
            if node_type and node.type != node_type:
                continue

            # content filter
            content_ok = True
            for k, v in content_filter.items():
                if k not in node.content or node.content[k] != v:
                    content_ok = False
                    break
            if not content_ok:
                continue

            # relation filter
            relation_ok = True
            if relation_filter:
                rtype = relation_filter.get("type")
                other_id = relation_filter.get("node_id")
                direct = relation_filter.get("direction", "outgoing")
                if rtype and other_id:
                    if direct == "outgoing":
                        if not self.graph.has_edge(nid, other_id):
                            relation_ok = False
                        else:
                            edata = self.graph.get_edge_data(nid, other_id)
                            if edata["type"] != rtype:
                                relation_ok = False
                    else:  # incoming
                        if not self.graph.has_edge(other_id, nid):
                            relation_ok = False
                        else:
                            edata = self.graph.get_edge_data(other_id, nid)
                            if edata["type"] != rtype:
                                relation_ok = False

            if not relation_ok:
                continue

            # If this node passes all filters, return it
            node.access()
            self.graph.nodes[nid].update(node.to_dict())
            matching.append(node.to_dict())

        matching.sort(key=lambda x: x["confidence"], reverse=True)
        results = matching[:limit]

        self.query_cache[cache_key] = {
            "timestamp": now.isoformat(),
            "results": results
        }
        return results

    async def get_related_knowledge(self,
                                    node_id: str,
                                    relation_type: Optional[str] = None,
                                    direction: str = "both",
                                    limit: int = 10) -> List[Dict[str, Any]]:
        """
        Return nodes that have the specified relation to a given node, either incoming, outgoing, or both.

        :param node_id: The ID of the node around which we want to find neighbors.
        :param relation_type: Filter by a certain relation type or None for all.
        :param direction: 'incoming', 'outgoing', or 'both'
        :param limit: Max results
        :return: A list of dicts with "node" and "relation" fields describing the neighbor.
        """
        if node_id not in self.nodes:
            return []

        # update node usage
        self.nodes[node_id].access()

        neighbors = []

        # Outgoing edges
        if direction in ["outgoing", "both"]:
            for tgt in self.graph.successors(node_id):
                edata = self.graph.get_edge_data(node_id, tgt)
                if relation_type is None or edata["type"] == relation_type:
                    if tgt in self.nodes:
                        neighbors.append({
                            "node": self.nodes[tgt].to_dict(),
                            "relation": {
                                "type": edata["type"],
                                "direction": "outgoing",
                                "weight": edata.get("weight", 1.0)
                            }
                        })

        # Incoming edges
        if direction in ["incoming", "both"]:
            for src in self.graph.predecessors(node_id):
                edata = self.graph.get_edge_data(src, node_id)
                if relation_type is None or edata["type"] == relation_type:
                    if src in self.nodes:
                        neighbors.append({
                            "node": self.nodes[src].to_dict(),
                            "relation": {
                                "type": edata["type"],
                                "direction": "incoming",
                                "weight": edata.get("weight", 1.0)
                            }
                        })

        neighbors.sort(key=lambda x: x["relation"]["weight"], reverse=True)
        return neighbors[:limit]

    async def get_knowledge_statistics(self) -> Dict[str, Any]:
        """
        Return a summary of knowledge graph stats, plus a timestamp.
        """
        stats = {
            "timestamp": datetime.now().isoformat(),
            "node_count": len(self.nodes),
            "edge_count": self.graph.number_of_edges(),
            "node_types": self.integration_stats.get("node_types", {}),
            "relation_types": self.integration_stats.get("relation_types", {}),
            "nodes_added": self.integration_stats["nodes_added"],
            "relations_added": self.integration_stats["relations_added"],
            "conflicts_resolved": self.integration_stats["conflicts_resolved"],
            "knowledge_queries": self.integration_stats["knowledge_queries"],
            "integration_cycles": self.integration_stats["integration_cycles"]
        }
        
        # Add curiosity system statistics
        curiosity_stats = await self.curiosity_system.get_curiosity_statistics()
        stats["curiosity_system"] = curiosity_stats
        
        return stats
    
    # Curiosity System Methods
    
    async def identify_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """Identify knowledge gaps from the knowledge graph"""
        return await self.curiosity_system.identify_knowledge_gaps()
    
    async def create_exploration_target(self, domain: str, topic: str,
                                     importance: float = 0.5, urgency: float = 0.5,
                                     knowledge_gap: Optional[float] = None) -> str:
        """Create a new exploration target"""
        return await self.curiosity_system.create_exploration_target(
            domain=domain, topic=topic, importance=importance, 
            urgency=urgency, knowledge_gap=knowledge_gap)
    
    async def get_exploration_targets(self, limit: int = 10, min_priority: float = 0.0) -> List[Dict[str, Any]]:
        """Get current exploration targets sorted by priority"""
        return await self.curiosity_system.get_exploration_targets(limit=limit, min_priority=min_priority)
    
    async def record_exploration(self, target_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Record the result of an exploration"""
        return await self.curiosity_system.record_exploration(target_id=target_id, result=result)
    
    async def generate_questions(self, target_id: str, limit: int = 5) -> List[str]:
        """Generate questions to explore a target"""
        return await self.curiosity_system.generate_questions(target_id=target_id, limit=limit)
    
    async def prioritize_domains(self) -> List[Dict[str, Any]]:
        """Prioritize knowledge domains based on gaps and importance"""
        return await self.curiosity_system.prioritize_domains()

    async def retrieve_relevant_experiences(self, 
                                          current_context: Dict[str, Any],
                                          limit: int = 3,
                                          min_relevance: float = 0.6) -> List[Dict[str, Any]]:
        """
        Retrieve experiences relevant to the current conversation context.
        
        Args:
            current_context: Current conversation context including:
                - query: Search query or current topic
                - scenario_type: Type of scenario (e.g., "teasing", "dark")
                - emotional_state: Current emotional state
                - entities: Entities involved in current context
            limit: Maximum number of experiences to return
            min_relevance: Minimum relevance score (0.0-1.0)
            
        Returns:
            List of relevant experiences with metadata
        """
        # Check cache for identical request
        cache_key = f"{current_context.get('query', '')}_{current_context.get('scenario_type', '')}_{limit}_{min_relevance}"
        if cache_key in self.experience_cache:
            cache_time, cache_result = self.experience_cache[cache_key]
            # Use cache if less than 5 minutes old
            cache_age = (datetime.datetime.now() - cache_time).total_seconds()
            if cache_age < 300:
                return cache_result
        
        # Extract key information from context
        query = current_context.get("query", "")
        
        # Get base memories from memory system
        base_memories = await self.memory_core.retrieve_memories(
            query=query,
            memory_types=["observation", "reflection", "episodic", "experience"],
            limit=limit*3,  # Get more to filter later
            context=current_context
        )
        
        if not base_memories:
            logger.info("No base memories found for experience retrieval")
            return []
        
        # Score memories for relevance
        scored_memories = []
        for memory in base_memories:
            # Calculate relevance score
            relevance_score = await self._calculate_relevance_score(
                memory, current_context
            )
            
            # Skip low-relevance memories
            if relevance_score < min_relevance:
                continue
            
            # Get emotional context for this memory
            emotional_context = await self._get_memory_emotional_context(memory)
            
            # Calculate experiential richness (how much detail/emotion it has)
            experiential_richness = self._calculate_experiential_richness(
                memory, emotional_context
            )
            
            # Add to scored memories
            scored_memories.append({
                "memory": memory,
                "relevance_score": relevance_score,
                "emotional_context": emotional_context,
                "experiential_richness": experiential_richness,
                "final_score": relevance_score * 0.7 + experiential_richness * 0.3
            })
        
        # Sort by final score
        scored_memories.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Process top memories into experiences
        experiences = []
        for item in scored_memories[:limit]:
            experience = await self._convert_memory_to_experience(
                item["memory"],
                item["emotional_context"],
                item["relevance_score"],
                item["experiential_richness"]
            )
            experiences.append(experience)
        
        # Cache the result
        self.experience_cache[cache_key] = (datetime.datetime.now(), experiences)
        self.last_retrieval_time = datetime.datetime.now()
        
        return experiences
    
    async def generate_conversational_recall(self, 
                                          experience: Dict[str, Any],
                                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a natural, conversational recall of an experience.
        
        Args:
            experience: The experience to recall
            context: Current conversation context
            
        Returns:
            Conversational recall with reflection
        """
        # Extract experience data
        content = experience.get("content", experience.get("memory_text", ""))
        emotional_context = experience.get("emotional_context", {})
        scenario_type = experience.get("scenario_type", "general")
        timestamp = experience.get("timestamp", experience.get("metadata", {}).get("timestamp"))
        
        # Get timeframe text
        timeframe = self._get_timeframe_text(timestamp)
        
        # Determine tone for recall
        emotional_tone = self._get_emotional_tone(emotional_context)
        scenario_tone = self._get_scenario_tone(scenario_type)
        
        # Select tone (prioritize scenario tone if available)
        tone = scenario_tone or emotional_tone
        
        # Get templates for this tone
        templates = self.recall_templates.get(tone, self.recall_templates["standard"])
        
        # Select a random template
        template = random.choice(templates)
        
        # Generate simple components for template filling
        sentences = content.split(".")
        brief_summary = sentences[0] if sentences else "something happened"
        if len(brief_summary) > 50:
            brief_summary = brief_summary[:47] + "..."
        
        detail = ""
        if len(sentences) > 1:
            detail = sentences[1]
            if len(detail) > 60:
                detail = detail[:57] + "..."
        else:
            detail = "It was quite memorable."
        
        # Generate a reflection based on scenario type and emotion
        reflection = "I found it quite interesting to observe."
        
        # Fill in the template
        recall_text = template.format(
            timeframe=timeframe,
            brief_summary=brief_summary,
            detail=detail,
            reflection=reflection
        )
        
        return {
            "recall_text": recall_text,
            "confidence": experience.get("relevance_score", 0.5)
        }
    
    async def generate_personality_reflection(self,
                                           experiences: List[Dict[str, Any]],
                                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a personality-driven reflection based on experiences.
        
        Args:
            experiences: List of experiences to reflect on
            context: Current conversation context
            
        Returns:
            Personality-driven reflection
        """
        if not experiences:
            return {
                "reflection": "I don't have specific experiences to reflect on yet.",
                "confidence": 0.3
            }
        
        # Extract key information from top experience
        top_experience = experiences[0]
        
        # For more complex reflections with multiple experiences
        if len(experiences) > 1:
            reflection = await self._generate_complex_reflection(experiences, context)
            return reflection
        
        # For simpler reflections based on a single experience
        scenario_type = top_experience.get("scenario_type", "general")
        emotional_context = top_experience.get("emotional_context", {})
        
        # Generate simple reflection
        reflection = "Reflecting on this experience, I've noticed patterns in how I approach similar situations."
        
        # Determine confidence based on relevance and richness
        relevance = top_experience.get("relevance_score", 0.5)
        richness = top_experience.get("experiential_richness", 0.5)
        confidence = 0.5 + (relevance * 0.25) + (richness * 0.25)
        
        return {
            "reflection": reflection,
            "confidence": confidence,
            "experience_id": top_experience.get("id"),
            "scenario_type": scenario_type,
            "emotional_context": emotional_context
        }
    
    async def _generate_complex_reflection(self,
                                        experiences: List[Dict[str, Any]],
                                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a more complex reflection based on multiple experiences.
        
        Args:
            experiences: List of experiences to reflect on
            context: Current conversation context
            
        Returns:
            Complex reflection
        """
        # Extract key data from experiences
        primary_emotions = [exp.get("emotional_context", {}).get("primary_emotion", "neutral") for exp in experiences]
        scenario_types = [exp.get("scenario_type", "general") for exp in experiences]
        
        # Determine dominant emotion and scenario type
        dominant_emotion = max(set(primary_emotions), key=primary_emotions.count) if primary_emotions else "neutral"
        dominant_scenario = max(set(scenario_types), key=scenario_types.count) if scenario_types else "general"
        
        # Generate a reflection about patterns across experiences
        reflection_text = "I've noticed an interesting pattern across these experiences that highlights my approach to similar situations."
        
        # Calculate confidence based on experience count and relevance
        relevance_scores = [exp.get("relevance_score", 0.5) for exp in experiences]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
        
        # More relevant experiences = higher confidence
        confidence = 0.5 + (avg_relevance * 0.3) + (min(len(experiences), 3) * 0.1 / 3)
        
        return {
            "reflection": reflection_text,
            "confidence": min(1.0, confidence),
            "experience_count": len(experiences),
            "experience_ids": [exp.get("id") for exp in experiences],
            "dominant_emotion": dominant_emotion,
            "dominant_scenario": dominant_scenario
        }
    
    async def construct_narrative(self,
                               experiences: List[Dict[str, Any]],
                               topic: str,
                               chronological: bool = True) -> Dict[str, Any]:
        """
        Construct a coherent narrative from multiple experiences.
        
        Args:
            experiences: List of experiences to include in narrative
            topic: Topic of the narrative
            chronological: Whether to maintain chronological order
            
        Returns:
            Narrative data
        """
        return await self.memory_core.construct_narrative_from_memories(
            topic=topic,
            chronological=chronological,
            limit=len(experiences)
        )
        
    async def store_experience(self,
                            memory_text: str,
                            scenario_type: str = "general",
                            entities: List[str] = None,
                            emotional_context: Dict[str, Any] = None,
                            significance: int = 5,
                            tags: List[str] = None) -> Dict[str, Any]:
        """
        Store a new experience in the memory system.
        
        Args:
            memory_text: The memory text
            scenario_type: Type of scenario
            entities: List of entity IDs involved
            emotional_context: Emotional context data
            significance: Memory significance
            tags: Additional tags
            
        Returns:
            Stored experience information
        """
        # Set default tags if not provided
        tags = tags or []
        
        # Add scenario type to tags if not already present
        if scenario_type not in tags:
            tags.append(scenario_type)
        
        # Add experience tag
        if "experience" not in tags:
            tags.append("experience")
            
        # Prepare metadata
        metadata = {
            "scenario_type": scenario_type,
            "entities": entities or [],
            "is_experience": True
        }
        
        # Add emotional context to metadata if provided
        if emotional_context:
            metadata["emotional_context"] = emotional_context
        elif self.emotional_core:
            # If no emotional context is provided, get current emotional state
            emotional_context = self.emotional_core.get_formatted_emotional_state()
            metadata["emotional_context"] = emotional_context
        
        # Store memory using the memory core
        memory_id = await self.memory_core.add_memory(
            memory_text=memory_text,
            memory_type="experience",
            memory_scope="game",
            significance=significance,
            tags=tags,
            metadata=metadata
        )
        
        return {
            "memory_id": memory_id,
            "memory_text": memory_text,
            "scenario_type": scenario_type,
            "tags": tags,
            "significance": significance
        }
        
    async def _calculate_relevance_score(self, 
                                      memory: Dict[str, Any], 
                                      context: Dict[str, Any]) -> float:
        """
        Calculate how relevant a memory is to the current context.
        
        Args:
            memory: Memory to evaluate
            context: Current conversation context
            
        Returns:
            Relevance score (0.0-1.0)
        """
        # Check cache for this calculation
        memory_id = memory.get("id", "")
        context_hash = hash(str(context))
        cache_key = f"{memory_id}_{context_hash}"
        
        if cache_key in self.relevance_cache:
            return self.relevance_cache[cache_key]
        
        # Extract memory attributes
        memory_text = memory.get("memory_text", "")
        memory_tags = memory.get("tags", [])
        memory_metadata = memory.get("metadata", {})
        
        # Initialize score components
        semantic_score = 0.0
        tag_score = 0.0
        temporal_score = 0.0
        emotional_score = 0.0
        entity_score = 0.0
        
        # Semantic relevance (via embedding similarity)
        # This would ideally use the embedding similarity from your memory system
        if "relevance" in memory:
            semantic_score = memory.get("relevance", 0.0)
        elif "embedding" in memory and "query_embedding" in context:
            # Calculate cosine similarity if both embeddings available
            memory_embedding = np.array(memory["embedding"])
            query_embedding = np.array(context["query_embedding"])
            
            # Cosine similarity
            dot_product = np.dot(memory_embedding, query_embedding)
            norm_product = np.linalg.norm(memory_embedding) * np.linalg.norm(query_embedding)
            semantic_score = dot_product / norm_product if norm_product > 0 else 0.0
        else:
            # Fallback: keyword matching
            query = context.get("query", "").lower()
            text = memory_text.lower()
            
            # Simple word overlap
            query_words = set(query.split())
            text_words = set(text.split())
            
            if query_words:
                matches = query_words.intersection(text_words)
                semantic_score = len(matches) / len(query_words)
            else:
                semantic_score = 0.1  # Minimal score for empty query
        
        # Tag matching
        scenario_type = context.get("scenario_type", "").lower()
        context_tags = context.get("tags", [])
        
        # Count matching tags
        matching_tags = 0
        for tag in memory_tags:
            if tag.lower() in scenario_type or tag in context_tags:
                matching_tags += 1
        
        tag_score = min(1.0, matching_tags / 3) if memory_tags else 0.0
        
        # Temporal relevance (newer memories score higher)
        if "timestamp" in memory:
            memory_time = datetime.datetime.fromisoformat(memory["timestamp"]) if isinstance(memory["timestamp"], str) else memory["timestamp"]
            age_in_days = (datetime.datetime.now() - memory_time).total_seconds() / 86400
            # Newer memories get higher scores, but not too dominant
            temporal_score = max(0.0, 1.0 - (age_in_days / 180))  # 6 month scale
        
        # Emotional relevance
        memory_emotions = memory_metadata.get("emotions", {})
        context_emotions = context.get("emotional_state", {})
        
        if memory_emotions and context_emotions:
            # Compare primary emotions
            memory_primary = memory_emotions.get("primary", {}).get("name", "neutral")
            context_primary = context_emotions.get("primary_emotion", "neutral")
            
            # Emotion match bonus
            if memory_primary == context_primary:
                emotional_score += 0.5
                
            # Emotional intensity comparison
            memory_intensity = memory_emotions.get("primary", {}).get("intensity", 0.5)
            context_intensity = context_emotions.get("intensity", 0.5)
            
            # Similar intensity bonus
            emotional_score += 0.5 * (1.0 - abs(memory_intensity - context_intensity))
        
        # Entity relevance
        memory_entities = memory_metadata.get("entities", [])
        context_entities = context.get("entities", [])
        
        if memory_entities and context_entities:
            matching_entities = len(set(memory_entities).intersection(set(context_entities)))
            entity_score = min(1.0, matching_entities / len(context_entities)) if context_entities else 0.0
        
        # Combine scores with weights
        final_score = (
            semantic_score * 0.35 +
            tag_score * 0.20 +
            temporal_score * 0.10 +
            emotional_score * 0.20 +
            entity_score * 0.15
        )
        
        # Cache the result
        self.relevance_cache[cache_key] = final_score
        
        return min(1.0, max(0.0, final_score))
    
    async def _get_memory_emotional_context(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get or infer emotional context for a memory.
        
        Args:
            memory: Memory to analyze
            
        Returns:
            Emotional context information
        """
        # Check if memory already has emotional data
        metadata = memory.get("metadata", {})
        if "emotions" in metadata:
            return metadata["emotions"]
        
        if "emotional_context" in metadata:
            return metadata["emotional_context"]
        
        # If memory has emotional intensity but no detailed emotions, infer them
        emotional_intensity = memory.get("emotional_intensity", 0)
        if emotional_intensity > 0:
            # Get memory tags for context
            tags = memory.get("tags", [])
            
            # For best results, analyze the text directly
            # Use emotional_core if available
            if self.emotional_core:
                try:
                    analysis = self.emotional_core.analyze_text_sentiment(memory.get("memory_text", ""))
                    
                    # Get primary emotion
                    emotions = {k: v for k, v in analysis.items() if v > 0}
                    if emotions:
                        primary_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                        primary_intensity = emotions[primary_emotion]
                        
                        # Other emotions as secondary
                        secondary_emotions = {k: v for k, v in emotions.items() if k != primary_emotion}
                        
                        # Calculate valence
                        positive_emotions = ["Joy", "Trust", "Anticipation", "Love"]
                        negative_emotions = ["Sadness", "Fear", "Anger", "Disgust", "Frustration"]
                        
                        valence = 0.0
                        if primary_emotion in positive_emotions:
                            valence = 0.5 + (primary_intensity * 0.5)
                        elif primary_emotion in negative_emotions:
                            valence = -0.5 - (primary_intensity * 0.5)
                        
                        return {
                            "primary_emotion": primary_emotion,
                            "primary_intensity": primary_intensity,
                            "secondary_emotions": secondary_emotions,
                            "valence": valence,
                            "arousal": primary_intensity
                        }
                except Exception as e:
                    logger.error(f"Error analyzing emotional content: {e}")
            
            # Fallback: infer from intensity and tags
            primary_emotion = "neutral"
            for tag in tags:
                if tag in ["joy", "sadness", "anger", "fear", "disgust", 
                          "surprise", "anticipation", "trust"]:
                    primary_emotion = tag
                    break
            
            return {
                "primary_emotion": primary_emotion,
                "primary_intensity": emotional_intensity / 100,  # Convert to 0-1 scale
                "secondary_emotions": {},
                "valence": 0.5 if primary_emotion in ["joy", "anticipation", "trust"] else -0.5,
                "arousal": 0.7 if emotional_intensity > 50 else 0.3
            }
        
        # Default emotional context if nothing else available
        return {
            "primary_emotion": "neutral",
            "primary_intensity": 0.3,
            "secondary_emotions": {},
            "valence": 0.0,
            "arousal": 0.2
        }
    
    def _calculate_experiential_richness(self, 
                                       memory: Dict[str, Any], 
                                       emotional_context: Dict[str, Any]) -> float:
        """
        Calculate how rich and detailed the experience is.
        Higher scores mean the memory has more emotional and sensory detail.
        
        Args:
            memory: Memory to evaluate
            emotional_context: Emotional context of the memory
            
        Returns:
            Experiential richness score (0.0-1.0)
        """
        # Extract memory attributes
        memory_text = memory.get("memory_text", "")
        memory_tags = memory.get("tags", [])
        significance = memory.get("significance", 3)
        
        # Initialize richness factors
        detail_score = 0.0
        emotional_depth = 0.0
        sensory_richness = 0.0
        significance_score = 0.0
        
        # Text length as a proxy for detail (longer memories might have more detail)
        # Capped at reasonable limits to avoid overly long memories dominating
        word_count = len(memory_text.split())
        detail_score = min(1.0, word_count / 100)  # Cap at 100 words
        
        # Emotional depth from context
        if emotional_context:
            # Primary emotion intensity
            primary_intensity = emotional_context.get("primary_intensity", 0.0)
            
            # Count secondary emotions
            secondary_count = len(emotional_context.get("secondary_emotions", {}))
            
            # Combine for emotional depth
            emotional_depth = 0.7 * primary_intensity + 0.3 * min(1.0, secondary_count / 3)
        
        # Sensory richness via presence of sensory words
        sensory_words = ["see", "saw", "look", "hear", "heard", "sound", 
                        "feel", "felt", "touch", "smell", "scent", "taste"]
        
        sensory_count = sum(1 for word in sensory_words if word in memory_text.lower())
        sensory_richness = min(1.0, sensory_count / 5)  # Cap at 5 sensory words
        
        # Significance as a direct factor
        significance_score = significance / 10.0  # Convert to 0-1 scale
        
        # Combine scores with weights
        richness_score = (
            detail_score * 0.3 +
            emotional_depth * 0.4 +
            sensory_richness * 0.2 +
            significance_score * 0.1
        )
        
        return min(1.0, max(0.0, richness_score))
    
    async def _convert_memory_to_experience(self,
                                         memory: Dict[str, Any],
                                         emotional_context: Dict[str, Any],
                                         relevance_score: float,
                                         experiential_richness: float) -> Dict[str, Any]:
        """
        Convert a raw memory into a rich experience format.
        
        Args:
            memory: The base memory
            emotional_context: Emotional context information
            relevance_score: How relevant this memory is
            experiential_richness: How rich and detailed the experience is
            
        Returns:
            Formatted experience
        """
        # Get memory participants/entities
        metadata = memory.get("metadata", {})
        entities = metadata.get("entities", [])
        
        # Get scenario type from tags
        tags = memory.get("tags", [])
        scenario_types = [tag for tag in tags if tag in [
            "teasing", "dark", "indulgent", "psychological", "nurturing",
            "training", "discipline", "service", "worship", "punishment"
        ]]
        
        scenario_type = scenario_types[0] if scenario_types else "general"
        
        # Format the experience
        experience = {
            "id": memory.get("id"),
            "content": memory.get("memory_text", ""),
            "emotional_context": emotional_context,
            "scenario_type": scenario_type,
            "entities": entities,
            "timestamp": memory.get("timestamp"),
            "relevance_score": relevance_score,
            "experiential_richness": experiential_richness,
            "tags": tags,
            "significance": memory.get("significance", 3)
        }
        
        # Add confidence marker based on relevance
        for (min_val, max_val), marker in self.confidence_markers.items():
            if min_val <= relevance_score < max_val:
                experience["confidence_marker"] = marker
                break
                
        if "confidence_marker" not in experience:
            experience["confidence_marker"] = "remember"  # Default
        
        return experience
        
    def _get_confidence_marker(self, relevance: float) -> str:
        """Get confidence marker text based on relevance score"""
        for (min_val, max_val), marker in self.confidence_markers.items():
            if min_val <= relevance < max_val:
                return marker
        return "remember"  # Default
    
    def _get_timeframe_text(self, timestamp: Optional[str]) -> str:
        """Get conversational timeframe text from timestamp"""
        if not timestamp:
            return "a while back"
            
        try:
            if isinstance(timestamp, str):
                memory_time = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:
                memory_time = timestamp
                
            days_ago = (datetime.datetime.now() - memory_time).days
            
            if days_ago < 1:
                return "earlier today"
            elif days_ago < 2:
                return "yesterday"
            elif days_ago < 7:
                return f"{days_ago} days ago"
            elif days_ago < 14:
                return "last week"
            elif days_ago < 30:
                return "a couple weeks ago"
            elif days_ago < 60:
                return "a month ago"
            elif days_ago < 365:
                return f"{days_ago // 30} months ago"
            else:
                return "a while back"
                
        except Exception as e:
            logger.error(f"Error processing timestamp: {e}")
            return "a while back"
    
    def _get_emotional_tone(self, emotional_context: Dict[str, Any]) -> str:
        """Determine the emotional tone for recall based on the experience's emotions"""
        if not emotional_context:
            return "standard"
            
        primary = emotional_context.get("primary_emotion", "neutral")
        intensity = emotional_context.get("primary_intensity", 0.5)
        valence = emotional_context.get("valence", 0.0)
        
        # High intensity experiences
        if intensity > 0.8:
            return "intense"
            
        # Positive emotions
        if valence > 0.3 or primary in ["Joy", "Anticipation", "Trust", "Love"]:
            return "positive"
            
        # Negative emotions
        if valence < -0.3 or primary in ["Anger", "Fear", "Disgust", "Sadness", "Frustration"]:
            return "negative"
            
        # Default to standard
        return "standard"
    
    def _get_scenario_tone(self, scenario_type: str) -> Optional[str]:
        """Get tone based on scenario type"""
        scenario_type = scenario_type.lower()
        
        if scenario_type in ["teasing", "indulgent"]:
            return "teasing"
        elif scenario_type in ["discipline", "punishment", "training"]:
            return "disciplinary"
        elif scenario_type in ["dark", "fear"]:
            return "intense"
        
        # No specific tone for this scenario type
        return None
3. Dynamic Adaptation System in NyxBrain
Here's how to integrate the dynamic_adaptation_system.py functionality into nyx_brain.py:
pythonCopy# nyx/core/nyx_brain.py

import logging
import asyncio
import json
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from nyx.core.emotional_core import EmotionalCore
from nyx.core.memory_core import MemoryCore
from nyx.core.reflection_engine import ReflectionEngine
from nyx.core.experience_interface import ExperienceInterface
from nyx.core.internal_feedback_system import InternalFeedbackSystem

logger = logging.getLogger(__name__)

class DynamicAdaptationSystem:
    """
    System for dynamically adapting to changing contexts and selecting optimal strategies.
    Detects context changes and adapts behavior accordingly.
    """
    
    def __init__(self):
        self.context_history = []
        self.max_history_size = 20
        self.strategies = {}
        self.strategy_history = []
        self.context_change_threshold = 0.3
        self.performance_history = []
        
        # Initialize with default strategies
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self):
        """Initialize system with default strategies"""
        self.register_strategy({
            "id": "balanced",
            "name": "Balanced Approach",
            "description": "A balanced approach with moderate exploration and adaptation",
            "parameters": {
                "exploration_rate": 0.2,
                "adaptation_rate": 0.15,
                "risk_tolerance": 0.5,
                "innovation_level": 0.5,
                "precision_focus": 0.5
            }
        })
        
        self.register_strategy({
            "id": "exploratory",
            "name": "Exploratory Strategy",
            "description": "High exploration rate with focus on discovering new patterns",
            "parameters": {
                "exploration_rate": 0.4,
                "adaptation_rate": 0.2,
                "risk_tolerance": 0.7,
                "innovation_level": 0.8,
                "precision_focus": 0.3
            }
        })
        
        self.register_strategy({
            "id": "conservative",
            "name": "Conservative Strategy",
            "description": "Low risk with high precision focus",
            "parameters": {
                "exploration_rate": 0.1,
                "adaptation_rate": 0.1,
                "risk_tolerance": 0.2,
                "innovation_level": 0.3,
                "precision_focus": 0.8
            }
        })
        
        self.register_strategy({
            "id": "adaptive",
            "name": "Highly Adaptive Strategy",
            "description": "Focuses on quick adaptation to changes",
            "parameters": {
                "exploration_rate": 0.3,
                "adaptation_rate": 0.3,
                "risk_tolerance": 0.6,
                "innovation_level": 0.6,
                "precision_focus": 0.4
            }
        })
    
    def register_strategy(self, strategy: Dict[str, Any]) -> None:
        """
        Register a new strategy in the system.
        
        Args:
            strategy: Strategy definition to register
        """
        if "id" not in strategy:
            strategy["id"] = f"strategy_{len(self.strategies) + 1}"
            
        self.strategies[strategy["id"]] = strategy
    
    async def detect_context_change(self, 
                              context: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Detect if there has been a significant change in context.
        
        Args:
            context: Current context information
            
        Returns:
            Tuple containing:
            - Whether a significant change was detected
            - Magnitude of the change (0.0-1.0)
            - Description of the change
        """
        # Add current context to history
        self.context_history.append(context)
        if len(self.context_history) > self.max_history_size:
            self.context_history.pop(0)
        
        # If we don't have enough history, no change detected
        if len(self.context_history) < 2:
            return (False, 0.0, "Insufficient context history")
        
        # Compare current context with previous
        current = context
        previous = self.context_history[-2]
        
        # Extract relevant features for comparison
        change_magnitude = self._calculate_context_difference(current, previous)
        
        # Determine if change is significant
        significant_change = change_magnitude > self.context_change_threshold
        
        # Generate change description
        if significant_change:
            description = self._generate_change_description(current, previous, change_magnitude)
        else:
            description = "No significant context change detected"
        
        return (significant_change, change_magnitude, description)
    
    async def monitor_performance(self, 
                            metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Monitor performance metrics and detect trends.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Performance analysis
        """
        # Add metrics to history
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        })
        
        # Trim history if needed
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)
        
        # Calculate trends
        trends = self._calculate_performance_trends(metrics)
        
        # Generate insights
        insights = self._generate_performance_insights(metrics, trends)
        
        return {
            "current": metrics,
            "trends": trends,
            "insights": insights,
            "history_points": len(self.performance_history)
        }
    
    async def select_strategy(self, 
                       context: Dict[str, Any], 
                       performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select the optimal strategy for the current context.
        
        Args:
            context: Current context information
            performance: Current performance metrics
            
        Returns:
            Selected strategy
        """
        # Extract context features
        context_features = self._extract_context_features(context)
        
        # Calculate context complexity
        complexity = self._calculate_context_complexity(context)
        
        # Calculate volatility
        volatility = self._calculate_context_volatility()
        
        # Calculate strategy scores
        strategy_scores = {}
        for strategy_id, strategy in self.strategies.items():
            strategy_scores[strategy_id] = self._calculate_strategy_score(
                strategy, context_features, performance, complexity, volatility
            )
        
        # Select best strategy
        if not strategy_scores:
            # If no strategies, return balanced
            selected_id = "balanced"
            if selected_id not in self.strategies:
                return {
                    "id": "default",
                    "name": "Default Strategy",
                    "parameters": {
                        "exploration_rate": 0.2,
                        "adaptation_rate": 0.15,
                        "risk_tolerance": 0.5,
                        "innovation_level": 0.5,
                        "precision_focus": 0.5
                    }
                }
        else:
            selected_id = max(strategy_scores.items(), key=lambda x: x[1])[0]
        
        # Get the selected strategy
        selected_strategy = self.strategies[selected_id]
        
        # Record strategy selection
        self.strategy_history.append({
            "timestamp": datetime.now().isoformat(),
            "strategy_id": selected_id,
            "context_summary": self._summarize_context(context),
            "performance_summary": self._summarize_performance(performance),
            "score": strategy_scores[selected_id]
        })
        
        return selected_strategy
    
    def _calculate_context_difference(self, 
                                    current: Dict[str, Any], 
                                    previous: Dict[str, Any]) -> float:
        """Calculate the difference between two contexts"""
        # Focus on key elements common to both contexts
        common_keys = set(current.keys()) & set(previous.keys())
        if not common_keys:
            return 1.0  # Maximum difference if no common keys
        
        differences = []
        for key in common_keys:
            # Skip complex nested structures, consider only scalar values
            if isinstance(current[key], (str, int, float, bool)) and isinstance(previous[key], (str, int, float, bool)):
                if isinstance(current[key], bool) or isinstance(previous[key], bool):
                    # For boolean values, difference is either 0 or 1
                    diff = 0.0 if current[key] == previous[key] else 1.0
                elif isinstance(current[key], str) or isinstance(previous[key], str):
                    # For string values, difference is either 0 or 1
                    diff = 0.0 if str(current[key]) == str(previous[key]) else 1.0
                else:
                    # For numeric values, calculate normalized difference
                    max_val = max(abs(float(current[key])), abs(float(previous[key])))
                    if max_val > 0:
                        diff = abs(float(current[key]) - float(previous[key])) / max_val
                    else:
                        diff = 0.0
                differences.append(diff)
        
        if not differences:
            return 0.5  # Middle value if no comparable elements
            
        # Return average difference
        return sum(differences) / len(differences)
    
    def _generate_change_description(self, 
                                   current: Dict[str, Any], 
                                   previous: Dict[str, Any], 
                                   magnitude: float) -> str:
        """Generate a description of the context change"""
        changes = []
        
        # Check for new or modified keys
        for key in current:
            if key in previous:
                if current[key] != previous[key] and isinstance(current[key], (str, int, float, bool)):
                    changes.append(f"{key} changed from {previous[key]} to {current[key]}")
            else:
                changes.append(f"New element: {key}")
        
        # Check for removed keys
        for key in previous:
            if key not in current:
                changes.append(f"Removed element: {key}")
        
        if not changes:
            return f"Context changed with magnitude {magnitude:.2f}"
            
        change_desc = ", ".join(changes[:3])  # Limit to first 3 changes
        if len(changes) > 3:
            change_desc += f", and {len(changes) - 3} more changes"
            
        return f"Context changed ({magnitude:.2f}): {change_desc}"
    
    def _calculate_performance_trends(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Calculate trends in performance metrics"""
        trends = {}
        
        if len(self.performance_history) < 2:
            # Not enough history for trends
            for metric, value in metrics.items():
                trends[metric] = {
                    "direction": "stable",
                    "magnitude": 0.0
                }
            return trends
        
        # Calculate trends for each metric
        for metric, current_value in metrics.items():
            # Find previous values for this metric
            previous_values = []
            for history_point in self.performance_history[:-1]:  # Skip current point
                if metric in history_point["metrics"]:
                    previous_values.append(history_point["metrics"][metric])
            
            if not previous_values:
                trends[metric] = {
                    "direction": "stable",
                    "magnitude": 0.0
                }
                continue
                
            # Calculate average of previous values
            avg_previous = sum(previous_values) / len(previous_values)
            
            # Calculate difference
            diff = current_value - avg_previous
            
            # Determine direction and magnitude
            if abs(diff) < 0.05:  # Small threshold for stability
                direction = "stable"
                magnitude = 0.0
            else:
                direction = "improving" if diff > 0 else "declining"
                magnitude = min(1.0, abs(diff))
                
            trends[metric] = {
                "direction": direction,
                "magnitude": magnitude,
                "diff_from_avg": diff
            }
        
        return trends
    
    def _generate_performance_insights(self, 
                                     metrics: Dict[str, float], 
                                     trends: Dict[str, Any]) -> List[str]:
        """Generate insights based on performance metrics and trends"""
        insights = []
        
        # Check for significant improvements
        improvements = [metric for metric, trend in trends.items() 
                       if trend["direction"] == "improving" and trend["magnitude"] > 0.1]
        if improvements:
            metrics_list = ", ".join(improvements)
            insights.append(f"Significant improvement in {metrics_list}")
        
        # Check for significant declines
        declines = [metric for metric, trend in trends.items() 
                   if trend["direction"] == "declining" and trend["magnitude"] > 0.1]
        if declines:
            metrics_list = ", ".join(declines)
            insights.append(f"Significant decline in {metrics_list}")
        
        # Check for overall performance
        avg_performance = sum(metrics.values()) / len(metrics) if metrics else 0.5
        if avg_performance > 0.8:
            insights.append("Overall performance is excellent")
        elif avg_performance < 0.4:
            insights.append("Overall performance is concerning")
        
        # Check for volatility
        volatility = self._calculate_performance_volatility()
        if volatility > 0.2:
            insights.append("Performance metrics show high volatility")
        
        return insights
    
    def _extract_context_features(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from context for strategy selection"""
        features = {}
        
        # Extract basic scalars
        for key, value in context.items():
            if isinstance(value, (int, float, bool)):
                if isinstance(value, bool):
                    features[key] = 1.0 if value else 0.0
                else:
                    features[key] = float(value)
        
        # Extract feature from user input if present
        if "user_input" in context and isinstance(context["user_input"], str):
            features["input_length"] = min(1.0, len(context["user_input"]) / 500.0)
            features["input_complexity"] = min(1.0, len(set(context["user_input"].split())) / 100.0)
        
        # Calculate volatility feature
        features["context_volatility"] = self._calculate_context_volatility()
        
        return features
    
    def _calculate_context_complexity(self, context: Dict[str, Any]) -> float:
        """Calculate the complexity of the current context"""
        # Count the number of nested elements and total elements
        total_elements = 0
        nested_elements = 0
        max_depth = 0
        
        def count_elements(obj, depth=0):
            nonlocal total_elements, nested_elements, max_depth
            max_depth = max(max_depth, depth)
            
            if isinstance(obj, dict):
                total_elements += len(obj)
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        nested_elements += 1
                        count_elements(value, depth + 1)
            elif isinstance(obj, list):
                total_elements += len(obj)
                for item in obj:
                    if isinstance(item, (dict, list)):
                        nested_elements += 1
                        count_elements(item, depth + 1)
        
        count_elements(context)
        
        # Calculate complexity factors
        size_factor = min(1.0, total_elements / 50.0)  # Normalize by expecting max 50 elements
        nesting_factor = min(1.0, nested_elements / 10.0)  # Normalize by expecting max 10 nested elements
        depth_factor = min(1.0, max_depth / 5.0)  # Normalize by expecting max depth of 5
        
        # Combine factors with weights
        complexity = (
            size_factor * 0.4 +
            nesting_factor * 0.3 +
            depth_factor * 0.3
        )
        
        return complexity
    
    def _calculate_context_volatility(self) -> float:
        """Calculate the volatility of the context over time"""
        if len(self.context_history) < 3:
            return 0.0  # Not enough history to calculate volatility
        
        # Calculate pairwise differences between consecutive contexts
        differences = []
        for i in range(1, len(self.context_history)):
            diff = self._calculate_context_difference(
                self.context_history[i], 
                self.context_history[i-1]
            )
            differences.append(diff)
        
        # Calculate variance of differences
        mean_diff = sum(differences) / len(differences)
        variance = sum((diff - mean_diff) ** 2 for diff in differences) / len(differences)
        
        # Normalize to [0,1]
        volatility = min(1.0, math.sqrt(variance) * 3.0)  # Scale to make values more meaningful
        
        return volatility
    
    def _calculate_strategy_score(self,
                                strategy: Dict[str, Any], 
                                context_features: Dict[str, float],
                                performance: Dict[str, Any],
                                complexity: float,
                                volatility: float) -> float:
        """Calculate a score for how well a strategy matches the current context"""
        params = strategy["parameters"]
        
        # Base score starts at 0.5
        score = 0.5
        
        # Adjust based on complexity
        # Higher complexity prefers higher adaptation rate
        complexity_match = 1.0 - abs(complexity - params["adaptation_rate"])
        score += complexity_match * 0.1
        
        # Adjust based on volatility
        # Higher volatility prefers higher exploration rate
        volatility_match = 1.0 - abs(volatility - params["exploration_rate"])
        score += volatility_match * 0.1
        
        # Adjust based on performance trends
        trends = performance.get("trends", {})
        
        # If performance is declining, prefer more exploratory strategies
        declining_metrics = sum(1 for t in trends.values() if t.get("direction") == "declining")
        if declining_metrics > 0:
            exploration_bonus = params["exploration_rate"] * 0.1 * declining_metrics
            score += exploration_bonus
        
        # If performance is good and stable, prefer more conservative strategies
        stable_good_metrics = sum(1 for m, t in zip(performance.get("current", {}).values(), trends.values()) 
                                if m > 0.7 and t.get("direction") in ["stable", "improving"])
        if stable_good_metrics > 0:
            precision_bonus = params["precision_focus"] * 0.1 * stable_good_metrics
            score += precision_bonus
        
        # Adjust based on history - avoid using the same strategy too many times in a row
        recency_penalty = 0.0
        for i, history_item in enumerate(reversed(self.strategy_history)):
            if history_item["strategy_id"] == strategy["id"]:
                recency_penalty += 0.05 * (0.8 ** i)  # Exponential decay with distance
        
        score -= min(0.2, recency_penalty)  # Cap penalty
        
        # Ensure score is in [0,1] range
        return min(1.0, max(0.0, score))
    
    def _calculate_performance_volatility(self) -> float:
        """Calculate the volatility of performance metrics over time"""
        if len(self.performance_history) < 3:
            return 0.0  # Not enough history
        
        # Extract all metric values
        metric_values = {}
        
        for history_point in self.performance_history:
            for metric, value in history_point["metrics"].items():
                if metric not in metric_values:
                    metric_values[metric] = []
                metric_values[metric].append(value)
        
        # Calculate standard deviation for each metric
        std_devs = []
        for values in metric_values.values():
            if len(values) >= 3:  # Need at least 3 points
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                std_devs.append(math.sqrt(variance))
        
        if not std_devs:
            return 0.0
            
        # Average standard deviation across metrics
        avg_std_dev = sum(std_devs) / len(std_devs)
        
        # Normalize to [0,1] with reasonable scaling
        volatility = min(1.0, avg_std_dev * 3.0)
        
        return volatility
    
    def _summarize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the context for history records"""
        summary = {}
        
        # Include basic scalar values
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                summary[key] = value
        
        # Add derived measures
        summary["complexity"] = self._calculate_context_complexity(context)
        summary["volatility"] = self._calculate_context_volatility()
        
        return summary
    
    def _summarize_performance(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of performance for history records"""
        summary = {}
        
        # Include current metrics
        current = performance.get("current", {})
        summary["metrics"] = current
        
        # Include average
        if current:
            summary["average"] = sum(current.values()) / len(current)
        
        # Include volatility
        summary["volatility"] = self._calculate_performance_volatility()
        
        return summary
