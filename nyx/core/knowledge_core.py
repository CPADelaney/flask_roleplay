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


class IntrinsicMotivationSystem:
    """
    System for generating intrinsic motivation and curiosity
    
    This extends the CuriositySystem with specific intrinsic motivation capabilities
    including competence-based, novelty-based, and mastery-based motivation.
    """
    
    def __init__(self, curiosity_system: Optional[CuriositySystem] = None):
        # Use provided curiosity system or create a new one
        self.curiosity_system = curiosity_system or CuriositySystem()
        
        # Motivation models
        self.novelty_model = {
            "recent_experiences": [],  # Most recent experiences for novelty calculation
            "novelty_threshold": 0.7,  # Threshold for considering something novel
            "novelty_decay_rate": 0.1  # How quickly novelty diminishes with exposure
        }
        
        self.competence_model = {
            "skill_areas": {},         # Skill areas and competence levels
            "learning_curves": {},     # Learning curve parameters for each skill
            "competence_thresholds": {
                "low": 0.3,            # Threshold for low competence
                "medium": 0.6,         # Threshold for medium competence
                "high": 0.9            # Threshold for high competence
            }
        }
        
        self.curiosity_targets = {} # Current targets of curiosity weighted by interest
        
        # Configuration
        self.config = {
            "novelty_weight": 0.4,     # Weight for novelty-based motivation
            "competence_weight": 0.3,  # Weight for competence-based motivation
            "mastery_weight": 0.3,     # Weight for mastery-based motivation
            "min_motivation_level": 0.2, # Minimum motivation level to consider
            "max_active_targets": 10,   # Maximum number of active curiosity targets
            "motivation_boost_rate": 0.2, # Rate at which motivation can be boosted
            "motivation_decay_rate": 0.1  # Rate at which motivation decays over time
        }
        
        # Statistics and state
        self.motivation_levels = {}    # Current motivation levels for different areas
        self.exploration_history = []  # History of explorations
        self.motivation_history = []   # History of motivation levels over time
    
    async def initialize(self, skill_areas: List[Dict[str, Any]] = None) -> None:
        """Initialize the intrinsic motivation system with skill areas"""
        if skill_areas:
            for skill in skill_areas:
                skill_id = skill.get("id", f"skill_{len(self.competence_model['skill_areas'])}")
                self.competence_model["skill_areas"][skill_id] = {
                    "name": skill.get("name", skill_id),
                    "level": skill.get("level", 0.1),
                    "experience": skill.get("experience", 0),
                    "last_practiced": datetime.now().isoformat(),
                    "related_domains": skill.get("related_domains", [])
                }
                
                # Create learning curve parameters
                self.competence_model["learning_curves"][skill_id] = {
                    "learning_rate": skill.get("learning_rate", 0.1),
                    "plateau": skill.get("plateau", 0.9),
                    "difficulty": skill.get("difficulty", 0.5)
                }
                
                # Initialize motivation level for this skill
                self.motivation_levels[skill_id] = 0.5
    
    async def update_competence(self, skill_id: str, experience_gained: float, 
                           success_rate: float) -> Dict[str, Any]:
        """Update competence level for a skill based on experience and success"""
        if skill_id not in self.competence_model["skill_areas"]:
            return {"error": f"Skill {skill_id} not found"}
        
        # Get current skill data
        skill = self.competence_model["skill_areas"][skill_id]
        learning_curve = self.competence_model["learning_curves"][skill_id]
        
        # Calculate effective experience gain based on success rate
        effective_experience = experience_gained * (0.5 + 0.5 * success_rate)
        
        # Update experience
        skill["experience"] += effective_experience
        
        # Calculate new competence level using a learning curve model
        old_level = skill["level"]
        learning_rate = learning_curve["learning_rate"]
        difficulty = learning_curve["difficulty"]
        plateau = learning_curve["plateau"]
        
        # Learning curve formula: Level approaches plateau at a decreasing rate
        progress_factor = 1.0 - (skill["level"] / plateau)
        level_gain = effective_experience * learning_rate * progress_factor * (1.0 - difficulty)
        skill["level"] = min(plateau, skill["level"] + level_gain)
        
        # Update last practiced timestamp
        skill["last_practiced"] = datetime.now().isoformat()
        
        # Update motivation based on competence change
        motivation_change = await self._calculate_competence_motivation_change(
            skill_id, old_level, skill["level"]
        )
        self.motivation_levels[skill_id] += motivation_change
        self.motivation_levels[skill_id] = max(0.0, min(1.0, self.motivation_levels[skill_id]))
        
        # Store motivation history
        self.motivation_history.append({
            "timestamp": datetime.now().isoformat(),
            "skill_id": skill_id,
            "motivation_level": self.motivation_levels[skill_id],
            "cause": "competence_update",
            "change": motivation_change
        })
        
        return {
            "skill_id": skill_id,
            "old_level": old_level,
            "new_level": skill["level"],
            "experience_gained": effective_experience,
            "motivation_change": motivation_change,
            "current_motivation": self.motivation_levels[skill_id]
        }
    
    async def _calculate_competence_motivation_change(self, skill_id: str, 
                                               old_level: float, new_level: float) -> float:
        """Calculate motivation change based on competence change"""
        # No change in level means no motivation change
        if old_level == new_level:
            return 0.0
        
        # Calculate level change
        level_change = new_level - old_level
        
        # Calculate motivation change based on competence thresholds
        thresholds = self.competence_model["competence_thresholds"]
        
        # Higher motivation gain in the mid-range of competence (flow state theory)
        if new_level < thresholds["low"]:
            # Low competence - moderate motivation for easy wins
            motivation_factor = 0.5
        elif new_level < thresholds["medium"]:
            # Medium-low competence - high motivation (rapid growth)
            motivation_factor = 1.0
        elif new_level < thresholds["high"]:
            # Medium-high competence - highest motivation (flow state)
            motivation_factor = 1.2
        else:
            # High competence - decreasing motivation (diminishing returns)
            motivation_factor = 0.3
        
        # Calculate motivation change
        motivation_change = level_change * motivation_factor * self.config["competence_weight"]
        
        # Boost small gains for positive reinforcement
        if 0 < motivation_change < 0.01:
            motivation_change = 0.01
        
        return motivation_change
    
    async def track_novelty(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Track a new experience for novelty calculation"""
        # Extract features
        features = self._extract_experience_features(experience)
        
        # Calculate novelty based on recent experiences
        novelty_score = await self._calculate_novelty_score(features)
        
        # Add to recent experiences
        self.novelty_model["recent_experiences"].append({
            "timestamp": datetime.now().isoformat(),
            "features": features,
            "novelty_score": novelty_score,
            "experience_id": experience.get("id", f"exp_{len(self.novelty_model['recent_experiences'])}")
        })
        
        # Limit size of recent experiences
        if len(self.novelty_model["recent_experiences"]) > 100:
            self.novelty_model["recent_experiences"] = self.novelty_model["recent_experiences"][-100:]
        
        # Update motivation based on novelty
        domains = experience.get("domains", [])
        topics = experience.get("topics", [])
        
        # Create combined domains+topics for motivation update
        areas = list(set(domains + topics))
        
        # Update motivation for each area
        motivation_changes = {}
        for area in areas:
            # Initialize motivation level if not exists
            if area not in self.motivation_levels:
                self.motivation_levels[area] = 0.5
            
            # Calculate motivation change based on novelty
            if novelty_score > self.novelty_model["novelty_threshold"]:
                # High novelty increases motivation
                motivation_change = novelty_score * self.config["novelty_weight"]
            else:
                # Low novelty slightly decreases motivation
                motivation_change = -0.05 * self.config["novelty_weight"]
            
            # Apply change
            self.motivation_levels[area] += motivation_change
            self.motivation_levels[area] = max(0.0, min(1.0, self.motivation_levels[area]))
            
            motivation_changes[area] = {
                "change": motivation_change,
                "level": self.motivation_levels[area]
            }
            
            # Store motivation history
            self.motivation_history.append({
                "timestamp": datetime.now().isoformat(),
                "area": area,
                "motivation_level": self.motivation_levels[area],
                "cause": "novelty_update",
                "change": motivation_change,
                "novelty_score": novelty_score
            })
        
        return {
            "novelty_score": novelty_score,
            "is_novel": novelty_score > self.novelty_model["novelty_threshold"],
            "motivation_changes": motivation_changes
        }
    
    def _extract_experience_features(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant features from an experience for novelty calculation"""
        features = {}
        
        # Extract basic properties
        for key in ["type", "domain", "topic", "source", "difficulty"]:
            if key in experience:
                features[key] = experience[key]
        
        # Extract content features based on type
        if experience.get("type") == "interaction":
            features["entities"] = experience.get("entities", [])
            features["actions"] = experience.get("actions", [])
            features["outcome"] = experience.get("outcome")
        elif experience.get("type") == "observation":
            features["entities"] = experience.get("entities", [])
            features["environment"] = experience.get("environment")
            features["attributes"] = experience.get("attributes", [])
        elif experience.get("type") == "information":
            features["concepts"] = experience.get("concepts", [])
            features["facts"] = experience.get("facts", [])
            features["connections"] = experience.get("connections", [])
        
        # Extract structural features
        features["complexity"] = experience.get("complexity", 0.5)
        features["emotional_impact"] = experience.get("emotional_impact", 0.0)
        
        return features
    
    async def _calculate_novelty_score(self, features: Dict[str, Any]) -> float:
        """Calculate novelty score by comparing to recent experiences"""
        if not self.novelty_model["recent_experiences"]:
            return 1.0  # First experience is fully novel
        
        # Calculate similarity to each recent experience
        similarities = []
        for recent_exp in self.novelty_model["recent_experiences"]:
            similarity = self._calculate_feature_similarity(features, recent_exp["features"])
            similarities.append(similarity)
        
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities)
        novelty = 1.0 - max_similarity
        
        return novelty
    
    def _calculate_feature_similarity(self, features1: Dict[str, Any], 
                                  features2: Dict[str, Any]) -> float:
        """Calculate similarity between two feature sets"""
        # Get common keys
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0  # No common features
        
        # Calculate similarity for each common feature
        similarities = []
        for key in common_keys:
            # Calculate feature-specific similarity
            if isinstance(features1[key], list) and isinstance(features2[key], list):
                # For lists (e.g., entities, actions), use Jaccard similarity
                set1 = set(features1[key])
                set2 = set(features2[key])
                union_size = len(set1 | set2)
                intersection_size = len(set1 & set2)
                
                if union_size > 0:
                    similarities.append(intersection_size / union_size)
                else:
                    similarities.append(1.0)  # Both empty lists
            elif isinstance(features1[key], (int, float)) and isinstance(features2[key], (int, float)):
                # For numeric values, calculate inverse distance
                distance = abs(features1[key] - features2[key])
                max_range = 1.0  # Assuming normalized values
                similarities.append(1.0 - min(1.0, distance / max_range))
            elif features1[key] == features2[key]:
                # For exact matches
                similarities.append(1.0)
            else:
                # Different values
                similarities.append(0.0)
        
        # Weight the similarities (could be improved with feature-specific weights)
        return sum(similarities) / len(similarities)
    
    async def generate_exploration_targets(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Generate intrinsically motivated exploration targets"""
        # Calculate novelty opportunities
        novelty_opportunities = await self._identify_novelty_opportunities()
        
        # Calculate competence opportunities
        competence_opportunities = await self._identify_competence_opportunities()
        
        # Calculate mastery opportunities
        mastery_opportunities = await self._identify_mastery_opportunities()
        
        # Combine all opportunities with their respective weights
        all_opportunities = []
        
        # Add novelty opportunities
        for opp in novelty_opportunities:
            all_opportunities.append({
                **opp,
                "score": opp["score"] * self.config["novelty_weight"],
                "motivation_type": "novelty"
            })
        
        # Add competence opportunities
        for opp in competence_opportunities:
            all_opportunities.append({
                **opp,
                "score": opp["score"] * self.config["competence_weight"],
                "motivation_type": "competence"
            })
        
        # Add mastery opportunities
        for opp in mastery_opportunities:
            all_opportunities.append({
                **opp,
                "score": opp["score"] * self.config["mastery_weight"],
                "motivation_type": "mastery"
            })
        
        # Sort by score
        all_opportunities.sort(key=lambda x: x["score"], reverse=True)
        
        # Convert top opportunities to exploration targets
        targets = []
        for opp in all_opportunities[:limit]:
            # Create a target in the curiosity system
            target_id = await self.curiosity_system.create_exploration_target(
                domain=opp["domain"],
                topic=opp["topic"],
                importance=opp["importance"],
                urgency=opp["urgency"],
                knowledge_gap=opp.get("knowledge_gap", 0.5)
            )
            
            # Generate questions for the target
            questions = await self.curiosity_system.generate_questions(target_id)
            
            # Add to targets list
            targets.append({
                "target_id": target_id,
                "domain": opp["domain"],
                "topic": opp["topic"],
                "motivation_type": opp["motivation_type"],
                "score": opp["score"],
                "importance": opp["importance"],
                "urgency": opp["urgency"],
                "questions": questions,
                "created_at": datetime.now().isoformat()
            })
            
            # Track as curiosity target
            self.curiosity_targets[target_id] = {
                "domain": opp["domain"],
                "topic": opp["topic"],
                "interest_level": opp["score"],
                "created_at": datetime.now().isoformat(),
                "motivation_type": opp["motivation_type"]
            }
        
        return targets
    
    async def _identify_novelty_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for novelty-based exploration"""
        opportunities = []
        
        # Get knowledge domains from curiosity system
        gaps = await self.curiosity_system.identify_knowledge_gaps()
        
        # Select gaps with high potential for novelty
        for gap in gaps:
            domain = gap["domain"]
            topic = gap["topic"]
            
            # Calculate novelty potential
            novelty_potential = await self._calculate_novelty_potential(domain, topic)
            
            if novelty_potential > 0.3:  # Threshold for considering it a novelty opportunity
                opportunities.append({
                    "domain": domain,
                    "topic": topic,
                    "score": novelty_potential,
                    "importance": gap["importance"],
                    "urgency": novelty_potential,
                    "knowledge_gap": gap["gap_size"],
                    "reason": "High novelty potential"
                })
        
        # Also look for unexplored domains
        domains = list(self.curiosity_system.knowledge_map.domains.keys())
        for domain in domains:
            # Check if we have recent experiences in this domain
            domain_experiences = [exp for exp in self.novelty_model["recent_experiences"] 
                               if exp["features"].get("domain") == domain]
            
            if not domain_experiences:
                # No recent experiences, high novelty potential
                opportunities.append({
                    "domain": domain,
                    "topic": "exploration",
                    "score": 0.9,
                    "importance": 0.7,
                    "urgency": 0.8,
                    "knowledge_gap": 0.8,
                    "reason": "Unexplored domain"
                })
        
        return opportunities
    
    async def _calculate_novelty_potential(self, domain: str, topic: str) -> float:
        """Calculate the potential for novelty in a domain/topic"""
        # Check if we have recent experiences with this domain/topic
        domain_experiences = [exp for exp in self.novelty_model["recent_experiences"] 
                           if exp["features"].get("domain") == domain]
        
        topic_experiences = [exp for exp in self.novelty_model["recent_experiences"] 
                          if exp["features"].get("topic") == topic]
        
        # If no experiences, high novelty potential
        if not domain_experiences and not topic_experiences:
            return 1.0
        
        # Calculate recency factor - more recent experiences decrease novelty potential
        avg_time = None
        if domain_experiences or topic_experiences:
            all_experiences = domain_experiences + topic_experiences
            timestamps = [datetime.fromisoformat(exp["timestamp"]) for exp in all_experiences]
            now = datetime.now()
            time_diffs = [(now - ts).total_seconds() / (24 * 3600) for ts in timestamps]  # days
            avg_time = sum(time_diffs) / len(time_diffs) if time_diffs else 0
        
        # Higher average time means higher novelty potential
        recency_factor = min(1.0, avg_time / 30) if avg_time is not None else 1.0  # Max effect after 30 days
        
        # Calculate exposure factor - more experiences decrease novelty potential
        exposure_count = len(domain_experiences) + len(topic_experiences)
        exposure_factor = 1.0 / (1.0 + 0.2 * exposure_count)  # Decay with more exposure
        
        # Combine factors
        novelty_potential = 0.7 * recency_factor + 0.3 * exposure_factor
        
        return novelty_potential
    
    async def _identify_competence_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for competence-based exploration"""
        opportunities = []
        
        # Check skill areas in the competence model
        for skill_id, skill in self.competence_model["skill_areas"].items():
            level = skill["level"]
            learning_curve = self.competence_model["learning_curves"][skill_id]
            
            # Calculate competence-based motivation
            thresholds = self.competence_model["competence_thresholds"]
            
            # Highest motivation in the "flow" state (not too easy, not too hard)
            if level < thresholds["low"]:
                # Too low - moderate motivation
                motivation = 0.4
                reason = "Foundational skill building"
            elif level < thresholds["medium"]:
                # Medium-low - high motivation (flow state)
                motivation = 0.9
                reason = "Rapid skill development"
            elif level < thresholds["high"]:
                # Medium-high - moderate motivation
                motivation = 0.7
                reason = "Skill mastery"
            else:
                # Very high - low motivation (too easy)
                motivation = 0.2
                reason = "Skill maintenance"
            
            # Adjust based on learning rate and difficulty
            learning_rate = learning_curve["learning_rate"]
            difficulty = learning_curve["difficulty"]
            
            # Fast learning rate and moderate difficulty is motivating
            rate_factor = 1.0 + (learning_rate - 0.5)
            difficulty_factor = 1.0 - abs(difficulty - 0.5) * 2  # Peak at medium difficulty
            
            adjusted_motivation = motivation * rate_factor * difficulty_factor
            
            # Get related domains for this skill
            related_domains = skill["related_domains"]
            
            # Create an opportunity for each related domain
            for domain in related_domains:
                opportunities.append({
                    "domain": domain,
                    "topic": skill["name"],
                    "score": adjusted_motivation,
                    "importance": 0.7,  # Competence building is important
                    "urgency": 0.5,
                    "knowledge_gap": 1.0 - level,
                    "reason": reason,
                    "skill_id": skill_id,
                    "skill_level": level
                })
        
        return opportunities
    
    async def _identify_mastery_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for mastery-based exploration"""
        opportunities = []
        
        # Get skills near mastery level
        near_mastery_skills = []
        mastery_threshold = self.competence_model["competence_thresholds"]["high"]
        
        for skill_id, skill in self.competence_model["skill_areas"].items():
            if 0.7 <= skill["level"] < mastery_threshold:
                near_mastery_skills.append((skill_id, skill))
        
        # For each near-mastery skill, find related knowledge gaps
        for skill_id, skill in near_mastery_skills:
            related_domains = skill["related_domains"]
            
            for domain in related_domains:
                # Find knowledge gaps in this domain
                domain_gaps = [gap for gap in await self.curiosity_system.identify_knowledge_gaps()
                             if gap["domain"] == domain]
                
                for gap in domain_gaps:
                    # Calculate how important this gap is for mastery
                    mastery_relevance = 1.0 - gap["gap_size"]  # Smaller gaps more relevant for mastery
                    
                    # Calculate motivation score
                    motivation_score = mastery_relevance * (1.0 - (skill["level"] - 0.7) / 0.3)
                    
                    if motivation_score > 0.5:  # Threshold for considering it a mastery opportunity
                        opportunities.append({
                            "domain": domain,
                            "topic": gap["topic"],
                            "score": motivation_score,
                            "importance": gap["importance"],
                            "urgency": 0.6,  # Mastery is somewhat urgent
                            "knowledge_gap": gap["gap_size"],
                            "reason": "Near mastery completion",
                            "skill_id": skill_id,
                            "skill_level": skill["level"]
                        })
        
        return opportunities
    
    async def apply_motivation_decay(self) -> Dict[str, Any]:
        """Apply decay to motivation levels over time"""
        decay_stats = {
            "areas_affected": 0,
            "total_decay": 0.0,
            "average_decay": 0.0
        }
        
        decay_rate = self.config["motivation_decay_rate"]
        
        # Apply decay to all motivation levels
        for area, level in self.motivation_levels.items():
            # Calculate decay amount
            decay_amount = level * decay_rate
            
            # Apply decay
            new_level = max(self.config["min_motivation_level"], level - decay_amount)
            self.motivation_levels[area] = new_level
            
            # Update statistics
            decay_stats["areas_affected"] += 1
            decay_stats["total_decay"] += decay_amount
        
        # Calculate average decay
        if decay_stats["areas_affected"] > 0:
            decay_stats["average_decay"] = decay_stats["total_decay"] / decay_stats["areas_affected"]
        
        return decay_stats
    
    async def boost_motivation(self, area: str, boost_amount: float) -> Dict[str, Any]:
        """Boost motivation level for a specific area"""
        if area not in self.motivation_levels:
            # Initialize if not exists
            self.motivation_levels[area] = 0.5
        
        # Apply boost
        old_level = self.motivation_levels[area]
        new_level = min(1.0, old_level + boost_amount * self.config["motivation_boost_rate"])
        self.motivation_levels[area] = new_level
        
        # Store motivation history
        self.motivation_history.append({
            "timestamp": datetime.now().isoformat(),
            "area": area,
            "motivation_level": new_level,
            "cause": "manual_boost",
            "change": new_level - old_level
        })
        
        return {
            "area": area,
            "old_level": old_level,
            "new_level": new_level,
            "change": new_level - old_level
        }
    
    async def get_motivation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the motivation system"""
        # Calculate averages and distributions
        avg_motivation = 0.0
        if self.motivation_levels:
            avg_motivation = sum(self.motivation_levels.values()) / len(self.motivation_levels)
        
        # Calculate distribution
        motivation_distribution = {
            "low": 0,
            "medium": 0,
            "high": 0
        }
        
        for level in self.motivation_levels.values():
            if level < 0.4:
                motivation_distribution["low"] += 1
            elif level < 0.7:
                motivation_distribution["medium"] += 1
            else:
                motivation_distribution["high"] += 1
        
        # Calculate most motivated areas
        sorted_motivations = sorted(
            [(area, level) for area, level in self.motivation_levels.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        top_areas = [{"area": area, "level": level} for area, level in sorted_motivations[:5]]
        
        # Calculate curiosity target statistics
        target_types = {}
        for target in self.curiosity_targets.values():
            motivation_type = target.get("motivation_type", "unknown")
            target_types[motivation_type] = target_types.get(motivation_type, 0) + 1
        
        # Create statistics dictionary
        statistics = {
            "average_motivation": avg_motivation,
            "motivation_distribution": motivation_distribution,
            "top_motivated_areas": top_areas,
            "total_areas": len(self.motivation_levels),
            "curiosity_targets": {
                "total": len(self.curiosity_targets),
                "by_type": target_types
            },
            "configuration": {
                "weights": {
                    "novelty": self.config["novelty_weight"],
                    "competence": self.config["competence_weight"],
                    "mastery": self.config["mastery_weight"]
                }
            }
        }
        
        return statistics
    
    async def save_state(self, file_path: str) -> bool:
        """Save current state to file"""
        try:
            # Save curiosity system state
            await self.curiosity_system.save_state(f"{file_path}_curiosity")
            
            # Save motivation system state
            motivation_state = {
                "novelty_model": self.novelty_model,
                "competence_model": {
                    "skill_areas": self.competence_model["skill_areas"],
                    "learning_curves": self.competence_model["learning_curves"],
                    "competence_thresholds": self.competence_model["competence_thresholds"]
                },
                "curiosity_targets": self.curiosity_targets,
                "motivation_levels": self.motivation_levels,
                "motivation_history": self.motivation_history,
                "config": self.config,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(motivation_state, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving motivation state: {e}")
            return False
        
    async def load_state(self, file_path: str) -> bool:
        """Load state from file"""
        try:
            # Load curiosity system state
            await self.curiosity_system.load_state(f"{file_path}_curiosity")
            
            # Load motivation system state
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Load motivation system components
            self.novelty_model = state["novelty_model"]
            self.competence_model = state["competence_model"]
            self.curiosity_targets = state["curiosity_targets"]
            self.motivation_levels = state["motivation_levels"]
            self.motivation_history = state["motivation_history"]
            self.config = state["config"]
            
            return True
        except Exception as e:
            logger.error(f"Error loading motivation state: {e}")
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
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified knowledge integration system that merges functionality from both:
1) knowledge_integration.py (KnowledgeIntegrationSystem, etc.)
2) knowledge_core.py (KnowledgeCore, CuriositySystem, etc.)

Everything has been combined so that no methods or logic are missing.
"""

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


###############################################################################
#                           Knowledge Node & Relation
###############################################################################

class KnowledgeNode:
    """
    Represents a node in the knowledge graph, storing a piece of knowledge.
    Combines the functionality from both modules, including 'from_dict' methods,
    'access', 'update', etc.
    """
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

        # Arbitrary metadata (e.g. merged_from info, domain tags, etc.)
        self.metadata: Dict[str, Any] = {}

    def update(self,
               new_content: Dict[str, Any],
               new_confidence: Optional[float] = None,
               source: Optional[str] = None) -> None:
        """
        Merge new content into the node's existing content, optionally adjusting confidence and source.
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
        Create a KnowledgeNode from a dictionary (e.g., loading from JSON).
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
    Also merges the logic from both modules (including 'from_dict').
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
        Create a KnowledgeRelation from a dictionary (e.g., loading from JSON).
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


###############################################################################
#                    Exploration & Curiosity / Reasoning Classes
###############################################################################

class ExplorationTarget:
    """
    Represents a target for exploration (from the second module).
    """
    def __init__(self, target_id: str, domain: str, topic: str,
                 importance: float = 0.5, urgency: float = 0.5,
                 knowledge_gap: float = 0.5):
        self.id = target_id
        self.domain = domain
        self.topic = topic
        self.importance = importance   # 0.0-1.0
        self.urgency = urgency         # 0.0-1.0
        self.knowledge_gap = knowledge_gap  # 0.0-1.0
        self.created_at = datetime.now()
        self.last_explored = None
        self.exploration_count = 0
        self.exploration_results = []
        self.related_questions = []
        self.priority_score = self._calculate_priority()

    def _calculate_priority(self) -> float:
        """
        Basic priority formula from domain, using weighted factors.
        """
        weights = {"importance": 0.4, "urgency": 0.3, "knowledge_gap": 0.3}
        priority = (
            self.importance * weights["importance"] +
            self.urgency * weights["urgency"] +
            self.knowledge_gap * weights["knowledge_gap"]
        )
        return priority

    def update_priority(self) -> float:
        """
        Update priority based on exploration count, recency, etc.
        """
        if self.exploration_count > 0:
            exploration_factor = 1.0 / (1.0 + self.exploration_count * 0.5)
            if self.last_explored:
                days_since = (datetime.now() - self.last_explored).total_seconds() / (24 * 3600)
                recency_factor = min(1.0, days_since / 30)
            else:
                recency_factor = 1.0
            self.priority_score = self._calculate_priority() * exploration_factor * recency_factor
        else:
            self.priority_score = self._calculate_priority()

        return self.priority_score

    def record_exploration(self, result: Dict[str, Any]) -> None:
        """
        Record that we explored this target and optionally reduce knowledge gap.
        """
        self.exploration_count += 1
        self.last_explored = datetime.now()
        self.exploration_results.append({
            "timestamp": self.last_explored.isoformat(),
            "result": result
        })
        if "knowledge_gained" in result:
            knowledge_gained = result["knowledge_gained"]
            self.knowledge_gap = max(0.0, self.knowledge_gap - knowledge_gained)
        self.update_priority()

    def add_related_question(self, question: str) -> None:
        if question not in self.related_questions:
            self.related_questions.append(question)

    def to_dict(self) -> Dict[str, Any]:
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
    """
    Maps knowledge domains and gaps (from the second module).
    """
    def __init__(self):
        self.domains = {}  # domain -> {topic -> level}
        self.connections = {}  # (domain1, topic1) -> [(domain2, topic2, strength), ...]
        self.importance_levels = {}  # (domain, topic) -> importance
        self.last_updated = {}  # (domain, topic) -> datetime

    def add_knowledge(self, domain: str, topic: str,
                      level: float = 0.0, importance: float = 0.5) -> None:
        if domain not in self.domains:
            self.domains[domain] = {}
        self.domains[domain][topic] = level
        key = (domain, topic)
        self.importance_levels[key] = importance
        self.last_updated[key] = datetime.now()

    def add_connection(self, domain1: str, topic1: str,
                       domain2: str, topic2: str,
                       strength: float = 0.5) -> None:
        key1 = (domain1, topic1)
        if key1 not in self.connections:
            self.connections[key1] = []
        conn = (domain2, topic2, strength)
        if conn not in self.connections[key1]:
            self.connections[key1].append(conn)
        # Reverse
        key2 = (domain2, topic2)
        if key2 not in self.connections:
            self.connections[key2] = []
        rev_conn = (domain1, topic1, strength)
        if rev_conn not in self.connections[key2]:
            self.connections[key2].append(rev_conn)

    def get_knowledge_level(self, domain: str, topic: str) -> float:
        if domain in self.domains and topic in self.domains[domain]:
            return self.domains[domain][topic]
        return 0.0

    def get_importance(self, domain: str, topic: str) -> float:
        return self.importance_levels.get((domain, topic), 0.5)

    def get_knowledge_gaps(self) -> List[Tuple[str, str, float, float]]:
        # returns list of (domain, topic, gap_size, importance)
        gaps = []
        for domain, topics in self.domains.items():
            for topic, level in topics.items():
                if level < 0.7:
                    gap_size = 1.0 - level
                    importance = self.get_importance(domain, topic)
                    gaps.append((domain, topic, gap_size, importance))
        gaps.sort(key=lambda x: x[2] * x[3], reverse=True)
        return gaps

    def get_related_topics(self, domain: str, topic: str) -> List[Tuple[str, str, float]]:
        key = (domain, topic)
        return self.connections.get(key, [])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domains": {domain: dict(topics) for domain, topics in self.domains.items()},
            "connections": {f"{k[0]}|{k[1]}": v for k, v in self.connections.items()},
            "importance_levels": {f"{k[0]}|{k[1]}": v for k, v in self.importance_levels.items()},
            "last_updated": {f"{k[0]}|{k[1]}": v.isoformat() for k, v in self.last_updated.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeMap':
        km = cls()
        for domain, topics in data["domains"].items():
            km.domains[domain] = topics
        for key_str, conns in data["connections"].items():
            domain, topic = key_str.split("|")
            km.connections[(domain, topic)] = conns
        for key_str, importance in data["importance_levels"].items():
            domain, topic = key_str.split("|")
            km.importance_levels[(domain, topic)] = importance
        for key_str, timestamp in data["last_updated"].items():
            domain, topic = key_str.split("|")
            km.last_updated[(domain, topic)] = datetime.fromisoformat(timestamp)
        return km


class SimpleReasoningSystem:
    """
    Minimal "reasoning" component that attempts to resolve contradictions by
    preferring the higher confidence node. From the second module.
    """
    async def resolve_contradiction(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> Dict[str, Any]:
        confidence1 = node1["confidence"]
        confidence2 = node2["confidence"]
        if abs(confidence1 - confidence2) < 0.05:
            return {"resolved": False, "preferred_node": None}
        else:
            if confidence1 >= confidence2:
                return {"resolved": True, "preferred_node": node1["id"]}
            else:
                return {"resolved": True, "preferred_node": node2["id"]}


###############################################################################
#              Curiosity System & Intrinsic Motivation (Module 2)
###############################################################################

class CuriositySystem:
    """
    System for curiosity-driven exploration and knowledge gap identification.
    (From the second module)
    """
    def __init__(self):
        self.knowledge_map = KnowledgeMap()
        self.exploration_targets = {}  # id -> ExplorationTarget
        self.exploration_history = []
        self.next_target_id = 1

        # Configuration
        self.config = {
            "max_active_targets": 20,
            "exploration_budget": 0.5,
            "novelty_bias": 0.7,
            "importance_threshold": 0.3,
            "knowledge_decay_rate": 0.01
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
        self.knowledge_map.add_knowledge(domain, topic, level, importance)
        # Update related exploration targets
        for tid, target in self.exploration_targets.items():
            if target.domain == domain and target.topic == topic:
                original_gap = target.knowledge_gap
                target.knowledge_gap = max(0.0, 1.0 - level)
                target.update_priority()
                if target.knowledge_gap < 0.2 and not target.last_explored:
                    target.record_exploration({
                        "source": "external_knowledge",
                        "knowledge_gained": original_gap - target.knowledge_gap
                    })

    async def add_knowledge_connection(self, domain1: str, topic1: str,
                                       domain2: str, topic2: str,
                                       strength: float = 0.5) -> None:
        self.knowledge_map.add_connection(domain1, topic1, domain2, topic2, strength)

    async def identify_knowledge_gaps(self) -> List[Dict[str, Any]]:
        gaps = self.knowledge_map.get_knowledge_gaps()
        gap_dicts = []
        for (dom, top, gap_size, importance) in gaps:
            if importance >= self.config["importance_threshold"]:
                gap_dicts.append({
                    "domain": dom,
                    "topic": top,
                    "gap_size": gap_size,
                    "importance": importance,
                    "priority": gap_size * importance
                })
        gap_dicts.sort(key=lambda x: x["priority"], reverse=True)
        return gap_dicts

    async def create_exploration_target(self, domain: str, topic: str,
                                        importance: float = 0.5, urgency: float = 0.5,
                                        knowledge_gap: Optional[float] = None) -> str:
        target_id = f"target_{self.next_target_id}"
        self.next_target_id += 1
        if knowledge_gap is None:
            level = self.knowledge_map.get_knowledge_level(domain, topic)
            knowledge_gap = 1.0 - level
        target = ExplorationTarget(
            target_id=target_id,
            domain=domain,
            topic=topic,
            importance=importance,
            urgency=urgency,
            knowledge_gap=knowledge_gap
        )
        self.exploration_targets[target_id] = target
        return target_id

    async def get_exploration_targets(self, limit: int = 10, min_priority: float = 0.0) -> List[Dict[str, Any]]:
        for t in self.exploration_targets.values():
            t.update_priority()
        sorted_targets = sorted(
            [t for t in self.exploration_targets.values() if t.priority_score >= min_priority],
            key=lambda x: x.priority_score, reverse=True
        )
        return [st.to_dict() for st in sorted_targets[:limit]]

    async def record_exploration(self, target_id: str,
                                 result: Dict[str, Any]) -> Dict[str, Any]:
        if target_id not in self.exploration_targets:
            return {"error": f"Target {target_id} not found"}
        target = self.exploration_targets[target_id]
        target.record_exploration(result)

        if "knowledge_gained" in result and result["knowledge_gained"] > 0:
            level = 1.0 - target.knowledge_gap
            await self.add_knowledge(target.domain, target.topic, level, target.importance)

        self.stats["total_explorations"] += 1
        if result.get("success", False):
            self.stats["successful_explorations"] += 1
        self.stats["knowledge_gained"] += result.get("knowledge_gained", 0.0)

        total_importance = sum(t.importance for t in self.exploration_targets.values())
        if self.exploration_targets:
            self.stats["avg_importance"] = total_importance / len(self.exploration_targets)

        self.exploration_history.append({
            "timestamp": datetime.now().isoformat(),
            "target_id": target_id,
            "domain": target.domain,
            "topic": target.topic,
            "result": result
        })
        if len(self.exploration_history) > 1000:
            self.exploration_history = self.exploration_history[-1000:]
        return {
            "target": target.to_dict(),
            "updated_knowledge_level": 1.0 - target.knowledge_gap
        }

    async def generate_questions(self, target_id: str, limit: int = 5) -> List[str]:
        if target_id not in self.exploration_targets:
            return []
        target = self.exploration_targets[target_id]
        if len(target.related_questions) >= limit:
            return target.related_questions[:limit]

        questions = [
            f"What is {target.topic} in the context of {target.domain}?",
            f"Why is {target.topic} important in {target.domain}?",
            f"How does {target.topic} relate to other topics in {target.domain}?",
            f"What are the key components or aspects of {target.topic}?",
            f"What are common misconceptions about {target.topic}?"
        ]
        related_topics = self.knowledge_map.get_related_topics(target.domain, target.topic)
        for (dom, top, strength) in related_topics[:3]:
            questions.append(f"How does {target.topic} relate to {top}?")

        for q in questions:
            target.add_related_question(q)
        return questions[:limit]

    async def apply_knowledge_decay(self) -> Dict[str, Any]:
        """
        Apply knowledge decay to older knowledge in the knowledge map.
        Called during integration cycle in KnowledgeCore.
        """
        decay_stats = {"domains_affected": 0, "topics_affected": 0, "total_decay": 0.0, "average_decay": 0.0}
        now = datetime.now()
        decay_factor = self.config["knowledge_decay_rate"]

        for domain, topics in self.knowledge_map.domains.items():
            domain_affected = False
            for topic, level in list(topics.items()):
                key = (domain, topic)
                last_updated = self.knowledge_map.last_updated.get(key)
                if last_updated:
                    days_since = (now - last_updated).total_seconds() / (24 * 3600)
                    if days_since > 30:
                        # Decay approach
                        decay_amount = level * decay_factor * (days_since / 30)
                        new_level = max(0.0, level - decay_amount)
                        self.knowledge_map.domains[domain][topic] = new_level
                        decay_stats["topics_affected"] += 1
                        decay_stats["total_decay"] += decay_amount
                        domain_affected = True
            if domain_affected:
                decay_stats["domains_affected"] += 1
        if decay_stats["topics_affected"] > 0:
            decay_stats["average_decay"] = decay_stats["total_decay"] / decay_stats["topics_affected"]
        return decay_stats

    async def get_curiosity_statistics(self) -> Dict[str, Any]:
        domain_count = len(self.knowledge_map.domains)
        topic_count = sum(len(tops) for tops in self.knowledge_map.domains.values())
        knowledge_levels = []
        for dom, tops in self.knowledge_map.domains.items():
            for topic, level in tops.items():
                knowledge_levels.append(level)
        avg_knowledge = sum(knowledge_levels) / len(knowledge_levels) if knowledge_levels else 0.0
        gaps = self.knowledge_map.get_knowledge_gaps()
        avg_gap_size = sum(g[2] for g in gaps) / len(gaps) if gaps else 0.0
        active_targets = len(self.exploration_targets)
        explored_targets = sum(1 for t in self.exploration_targets.values() if t.exploration_count > 0)
        stats = {
            "knowledge_map": {
                "domain_count": domain_count,
                "topic_count": topic_count,
                "connection_count": sum(len(v) for v in self.knowledge_map.connections.values()),
                "average_knowledge_level": avg_knowledge,
                "average_gap_size": avg_gap_size,
                "knowledge_gap_count": len(gaps),
            },
            "exploration": {
                "active_targets": active_targets,
                "explored_targets": explored_targets,
                "exploration_ratio": explored_targets / active_targets if active_targets else 0.0,
                "success_rate": self.stats["successful_explorations"] / self.stats["total_explorations"]
                if self.stats["total_explorations"] > 0 else 0.0,
                "total_knowledge_gained": self.stats["knowledge_gained"],
                "average_target_importance": self.stats["avg_importance"]
            },
            "configuration": self.config
        }
        return stats

    async def prioritize_domains(self) -> List[Dict[str, Any]]:
        domain_stats = {}
        for dom, tops in self.knowledge_map.domains.items():
            domain_topics = len(tops)
            if domain_topics > 0:
                domain_knowledge = sum(tops.values()) / domain_topics
            else:
                domain_knowledge = 0.0
            domain_importance = 0.0
            for topic in tops:
                key = (dom, topic)
                imp = self.knowledge_map.importance_levels.get(key, 0.5)
                domain_importance += imp
            if domain_topics > 0:
                domain_importance /= domain_topics
            domain_gap = 1.0 - domain_knowledge
            domain_priority = domain_gap * domain_importance
            domain_stats[dom] = {
                "topic_count": domain_topics,
                "average_knowledge": domain_knowledge,
                "average_importance": domain_importance,
                "knowledge_gap": domain_gap,
                "priority": domain_priority
            }
        prioritized = [
            {"domain": d, **st} for d, st in domain_stats.items()
        ]
        prioritized.sort(key=lambda x: x["priority"], reverse=True)
        return prioritized

    async def save_state(self, file_path: str) -> bool:
        try:
            state = {
                "knowledge_map": self.knowledge_map.to_dict(),
                "exploration_targets": {tid: t.to_dict() for tid, t in self.exploration_targets.items()},
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
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            self.knowledge_map = KnowledgeMap.from_dict(state["knowledge_map"])
            self.exploration_targets = {}
            for tid, tdata in state["exploration_targets"].items():
                self.exploration_targets[tid] = ExplorationTarget.from_dict(tdata)
            self.exploration_history = state["exploration_history"]
            self.next_target_id = state["next_target_id"]
            self.config = state["config"]
            self.stats = state["stats"]
            return True
        except Exception as e:
            logger.error(f"Error loading curiosity state: {e}")
            return False


class IntrinsicMotivationSystem:
    """
    Extends the CuriositySystem with competence-based, novelty-based, mastery-based motivation.
    (From the second module.)
    """
    def __init__(self, curiosity_system: Optional[CuriositySystem] = None):
        self.curiosity_system = curiosity_system or CuriositySystem()
        self.novelty_model = {
            "recent_experiences": [],
            "novelty_threshold": 0.7,
            "novelty_decay_rate": 0.1
        }
        self.competence_model = {
            "skill_areas": {},
            "learning_curves": {},
            "competence_thresholds": {"low": 0.3, "medium": 0.6, "high": 0.9}
        }
        self.curiosity_targets = {}
        self.config = {
            "novelty_weight": 0.4,
            "competence_weight": 0.3,
            "mastery_weight": 0.3,
            "min_motivation_level": 0.2,
            "max_active_targets": 10,
            "motivation_boost_rate": 0.2,
            "motivation_decay_rate": 0.1
        }
        self.motivation_levels = {}
        self.exploration_history = []
        self.motivation_history = []

    async def initialize(self, skill_areas: List[Dict[str, Any]] = None) -> None:
        if skill_areas:
            for skill in skill_areas:
                sid = skill.get("id", f"skill_{len(self.competence_model['skill_areas'])}")
                self.competence_model["skill_areas"][sid] = {
                    "name": skill.get("name", sid),
                    "level": skill.get("level", 0.1),
                    "experience": skill.get("experience", 0),
                    "last_practiced": datetime.now().isoformat(),
                    "related_domains": skill.get("related_domains", [])
                }
                self.competence_model["learning_curves"][sid] = {
                    "learning_rate": skill.get("learning_rate", 0.1),
                    "plateau": skill.get("plateau", 0.9),
                    "difficulty": skill.get("difficulty", 0.5)
                }
                self.motivation_levels[sid] = 0.5

    async def update_competence(self, skill_id: str, experience_gained: float, success_rate: float) -> Dict[str, Any]:
        if skill_id not in self.competence_model["skill_areas"]:
            return {"error": f"Skill {skill_id} not found"}

        skill = self.competence_model["skill_areas"][skill_id]
        learning_curve = self.competence_model["learning_curves"][skill_id]
        effective_experience = experience_gained * (0.5 + 0.5 * success_rate)
        old_level = skill["level"]
        skill["experience"] += effective_experience

        lr = learning_curve["learning_rate"]
        difficulty = learning_curve["difficulty"]
        plateau = learning_curve["plateau"]
        progress_factor = 1.0 - (skill["level"] / plateau)
        level_gain = effective_experience * lr * progress_factor * (1.0 - difficulty)
        skill["level"] = min(plateau, skill["level"] + level_gain)
        skill["last_practiced"] = datetime.now().isoformat()

        motivation_change = await self._calculate_competence_motivation_change(skill_id, old_level, skill["level"])
        self.motivation_levels[skill_id] = max(0.0, min(1.0, self.motivation_levels[skill_id] + motivation_change))

        self.motivation_history.append({
            "timestamp": datetime.now().isoformat(),
            "skill_id": skill_id,
            "motivation_level": self.motivation_levels[skill_id],
            "cause": "competence_update",
            "change": motivation_change
        })

        return {
            "skill_id": skill_id,
            "old_level": old_level,
            "new_level": skill["level"],
            "experience_gained": effective_experience,
            "motivation_change": motivation_change,
            "current_motivation": self.motivation_levels[skill_id]
        }

    async def _calculate_competence_motivation_change(self, skill_id: str, old_level: float, new_level: float) -> float:
        if old_level == new_level:
            return 0.0
        level_change = new_level - old_level
        thresholds = self.competence_model["competence_thresholds"]
        if new_level < thresholds["low"]:
            motivation_factor = 0.5
        elif new_level < thresholds["medium"]:
            motivation_factor = 1.0
        elif new_level < thresholds["high"]:
            motivation_factor = 1.2
        else:
            motivation_factor = 0.3
        motivation_change = level_change * motivation_factor * self.config["competence_weight"]
        if 0 < motivation_change < 0.01:
            motivation_change = 0.01
        return motivation_change

    async def track_novelty(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        features = self._extract_experience_features(experience)
        novelty_score = await self._calculate_novelty_score(features)
        self.novelty_model["recent_experiences"].append({
            "timestamp": datetime.now().isoformat(),
            "features": features,
            "novelty_score": novelty_score,
            "experience_id": experience.get("id", f"exp_{len(self.novelty_model['recent_experiences'])}")
        })
        if len(self.novelty_model["recent_experiences"]) > 100:
            self.novelty_model["recent_experiences"] = self.novelty_model["recent_experiences"][-100:]

        domains = experience.get("domains", [])
        topics = experience.get("topics", [])
        areas = list(set(domains + topics))
        motivation_changes = {}
        for area in areas:
            if area not in self.motivation_levels:
                self.motivation_levels[area] = 0.5
            if novelty_score > self.novelty_model["novelty_threshold"]:
                motivation_change = novelty_score * self.config["novelty_weight"]
            else:
                motivation_change = -0.05 * self.config["novelty_weight"]
            self.motivation_levels[area] += motivation_change
            self.motivation_levels[area] = max(0.0, min(1.0, self.motivation_levels[area]))
            motivation_changes[area] = {"change": motivation_change, "level": self.motivation_levels[area]}
            self.motivation_history.append({
                "timestamp": datetime.now().isoformat(),
                "area": area,
                "motivation_level": self.motivation_levels[area],
                "cause": "novelty_update",
                "change": motivation_change,
                "novelty_score": novelty_score
            })
        return {
            "novelty_score": novelty_score,
            "is_novel": novelty_score > self.novelty_model["novelty_threshold"],
            "motivation_changes": motivation_changes
        }

    def _extract_experience_features(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        features = {}
        for key in ["type", "domain", "topic", "source", "difficulty"]:
            if key in experience:
                features[key] = experience[key]
        if experience.get("type") == "interaction":
            features["entities"] = experience.get("entities", [])
            features["actions"] = experience.get("actions", [])
            features["outcome"] = experience.get("outcome")
        elif experience.get("type") == "observation":
            features["entities"] = experience.get("entities", [])
            features["environment"] = experience.get("environment")
            features["attributes"] = experience.get("attributes", [])
        elif experience.get("type") == "information":
            features["concepts"] = experience.get("concepts", [])
            features["facts"] = experience.get("facts", [])
            features["connections"] = experience.get("connections", [])
        features["complexity"] = experience.get("complexity", 0.5)
        features["emotional_impact"] = experience.get("emotional_impact", 0.0)
        return features

    async def _calculate_novelty_score(self, features: Dict[str, Any]) -> float:
        if not self.novelty_model["recent_experiences"]:
            return 1.0
        similarities = []
        for rexp in self.novelty_model["recent_experiences"]:
            sim = self._calculate_feature_similarity(features, rexp["features"])
            similarities.append(sim)
        max_similarity = max(similarities) if similarities else 0.0
        novelty = 1.0 - max_similarity
        return novelty

    def _calculate_feature_similarity(self, f1: Dict[str, Any], f2: Dict[str, Any]) -> float:
        common_keys = set(f1.keys()) & set(f2.keys())
        if not common_keys:
            return 0.0
        sims = []
        for k in common_keys:
            if isinstance(f1[k], list) and isinstance(f2[k], list):
                set1 = set(f1[k])
                set2 = set(f2[k])
                union_size = len(set1 | set2)
                intersection_size = len(set1 & set2)
                sims.append((intersection_size / union_size) if union_size else 1.0)
            elif isinstance(f1[k], (int, float)) and isinstance(f2[k], (int, float)):
                distance = abs(f1[k] - f2[k])
                max_range = 1.0
                sims.append(1.0 - min(1.0, distance / max_range))
            elif f1[k] == f2[k]:
                sims.append(1.0)
            else:
                sims.append(0.0)
        return sum(sims) / len(sims)

    async def generate_exploration_targets(self, limit: int = 5) -> List[Dict[str, Any]]:
        novelty_opps = await self._identify_novelty_opportunities()
        competence_opps = await self._identify_competence_opportunities()
        mastery_opps = await self._identify_mastery_opportunities()
        all_ops = []
        for opp in novelty_opps:
            all_ops.append({**opp, "score": opp["score"] * self.config["novelty_weight"],
                            "motivation_type": "novelty"})
        for opp in competence_opps:
            all_ops.append({**opp, "score": opp["score"] * self.config["competence_weight"],
                            "motivation_type": "competence"})
        for opp in mastery_opps:
            all_ops.append({**opp, "score": opp["score"] * self.config["mastery_weight"],
                            "motivation_type": "mastery"})
        all_ops.sort(key=lambda x: x["score"], reverse=True)
        targets = []
        for opp in all_ops[:limit]:
            tid = await self.curiosity_system.create_exploration_target(
                domain=opp["domain"],
                topic=opp["topic"],
                importance=opp["importance"],
                urgency=opp["urgency"],
                knowledge_gap=opp.get("knowledge_gap", 0.5)
            )
            questions = await self.curiosity_system.generate_questions(tid)
            targets.append({
                "target_id": tid,
                "domain": opp["domain"],
                "topic": opp["topic"],
                "motivation_type": opp["motivation_type"],
                "score": opp["score"],
                "importance": opp["importance"],
                "urgency": opp["urgency"],
                "questions": questions,
                "created_at": datetime.now().isoformat()
            })
            self.curiosity_targets[tid] = {
                "domain": opp["domain"],
                "topic": opp["topic"],
                "interest_level": opp["score"],
                "created_at": datetime.now().isoformat(),
                "motivation_type": opp["motivation_type"]
            }
        return targets

    async def _identify_novelty_opportunities(self) -> List[Dict[str, Any]]:
        ops = []
        gaps = await self.curiosity_system.identify_knowledge_gaps()
        for g in gaps:
            dom, top = g["domain"], g["topic"]
            nov = await self._calculate_novelty_potential(dom, top)
            if nov > 0.3:
                ops.append({
                    "domain": dom,
                    "topic": top,
                    "score": nov,
                    "importance": g["importance"],
                    "urgency": nov,
                    "knowledge_gap": g["gap_size"],
                    "reason": "High novelty potential"
                })
        # Also for unexplored domains
        all_domains = list(self.curiosity_system.knowledge_map.domains.keys())
        for d in all_domains:
            d_exps = [x for x in self.novelty_model["recent_experiences"]
                      if x["features"].get("domain") == d]
            if not d_exps:
                ops.append({
                    "domain": d,
                    "topic": "exploration",
                    "score": 0.9,
                    "importance": 0.7,
                    "urgency": 0.8,
                    "knowledge_gap": 0.8,
                    "reason": "Unexplored domain"
                })
        return ops

    async def _calculate_novelty_potential(self, domain: str, topic: str) -> float:
        d_exps = [x for x in self.novelty_model["recent_experiences"] if x["features"].get("domain") == domain]
        t_exps = [x for x in self.novelty_model["recent_experiences"] if x["features"].get("topic") == topic]
        if not d_exps and not t_exps:
            return 1.0
        times = []
        if d_exps or t_exps:
            combo = d_exps + t_exps
            stamps = [datetime.fromisoformat(xx["timestamp"]) for xx in combo]
            now = datetime.now()
            diffs = [(now - s).total_seconds() / (24 * 3600) for s in stamps]
            avg_time = sum(diffs) / len(diffs) if diffs else 0.0
        else:
            avg_time = 30.0
        recency_factor = min(1.0, avg_time / 30)  # 30 days
        exposure_count = len(d_exps) + len(t_exps)
        exposure_factor = 1.0 / (1.0 + 0.2 * exposure_count)
        novelty_potential = 0.7 * recency_factor + 0.3 * exposure_factor
        return novelty_potential

    async def _identify_competence_opportunities(self) -> List[Dict[str, Any]]:
        ops = []
        for sid, skill in self.competence_model["skill_areas"].items():
            level = skill["level"]
            thresholds = self.competence_model["competence_thresholds"]
            if level < thresholds["low"]:
                base_motivation = 0.4
                reason = "Foundational skill building"
            elif level < thresholds["medium"]:
                base_motivation = 0.9
                reason = "Rapid skill development"
            elif level < thresholds["high"]:
                base_motivation = 0.7
                reason = "Skill mastery"
            else:
                base_motivation = 0.2
                reason = "Skill maintenance"
            lc = self.competence_model["learning_curves"][sid]
            lr = lc["learning_rate"]
            diff = lc["difficulty"]
            rate_factor = 1.0 + (lr - 0.5)
            diff_factor = 1.0 - abs(diff - 0.5) * 2
            adjusted = base_motivation * rate_factor * diff_factor
            for dom in skill["related_domains"]:
                ops.append({
                    "domain": dom,
                    "topic": skill["name"],
                    "score": adjusted,
                    "importance": 0.7,
                    "urgency": 0.5,
                    "knowledge_gap": 1.0 - level,
                    "reason": reason,
                    "skill_id": sid,
                    "skill_level": level
                })
        return ops

    async def _identify_mastery_opportunities(self) -> List[Dict[str, Any]]:
        ops = []
        thr = self.competence_model["competence_thresholds"]["high"]
        near_mastery = [(sid, sk) for sid, sk in self.competence_model["skill_areas"].items()
                        if 0.7 <= sk["level"] < thr]
        for sid, sk in near_mastery:
            rel = sk["related_domains"]
            for dom in rel:
                domain_gaps = [g for g in await self.curiosity_system.identify_knowledge_gaps() if g["domain"] == dom]
                for gg in domain_gaps:
                    mastery_relevance = 1.0 - gg["gap_size"]
                    motivation_score = mastery_relevance * (1.0 - (sk["level"] - 0.7) / 0.3)
                    if motivation_score > 0.5:
                        ops.append({
                            "domain": dom,
                            "topic": gg["topic"],
                            "score": motivation_score,
                            "importance": gg["importance"],
                            "urgency": 0.6,
                            "knowledge_gap": gg["gap_size"],
                            "reason": "Near mastery completion",
                            "skill_id": sid,
                            "skill_level": sk["level"]
                        })
        return ops

    async def apply_motivation_decay(self) -> Dict[str, Any]:
        dec_stats = {"areas_affected": 0, "total_decay": 0.0, "average_decay": 0.0}
        dr = self.config["motivation_decay_rate"]
        for area, lvl in self.motivation_levels.items():
            decay_amount = lvl * dr
            new_lvl = max(self.config["min_motivation_level"], lvl - decay_amount)
            self.motivation_levels[area] = new_lvl
            dec_stats["areas_affected"] += 1
            dec_stats["total_decay"] += decay_amount
        if dec_stats["areas_affected"] > 0:
            dec_stats["average_decay"] = dec_stats["total_decay"] / dec_stats["areas_affected"]
        return dec_stats

    async def boost_motivation(self, area: str, boost_amount: float) -> Dict[str, Any]:
        if area not in self.motivation_levels:
            self.motivation_levels[area] = 0.5
        old_lvl = self.motivation_levels[area]
        new_lvl = min(1.0, old_lvl + boost_amount * self.config["motivation_boost_rate"])
        self.motivation_levels[area] = new_lvl
        self.motivation_history.append({
            "timestamp": datetime.now().isoformat(),
            "area": area,
            "motivation_level": new_lvl,
            "cause": "manual_boost",
            "change": new_lvl - old_lvl
        })
        return {"area": area, "old_level": old_lvl, "new_level": new_lvl, "change": new_lvl - old_lvl}

    async def get_motivation_statistics(self) -> Dict[str, Any]:
        avg_mot = 0.0
        if self.motivation_levels:
            avg_mot = sum(self.motivation_levels.values()) / len(self.motivation_levels)
        dist = {"low": 0, "medium": 0, "high": 0}
        for lvl in self.motivation_levels.values():
            if lvl < 0.4:
                dist["low"] += 1
            elif lvl < 0.7:
                dist["medium"] += 1
            else:
                dist["high"] += 1
        sorted_mot = sorted(self.motivation_levels.items(), key=lambda x: x[1], reverse=True)
        top_areas = [{"area": a, "level": l} for (a, l) in sorted_mot[:5]]
        ttypes = {}
        for ct in self.curiosity_targets.values():
            mt = ct.get("motivation_type", "unknown")
            ttypes[mt] = ttypes.get(mt, 0) + 1
        stats = {
            "average_motivation": avg_mot,
            "motivation_distribution": dist,
            "top_motivated_areas": top_areas,
            "total_areas": len(self.motivation_levels),
            "curiosity_targets": {"total": len(self.curiosity_targets), "by_type": ttypes},
            "configuration": {
                "weights": {
                    "novelty": self.config["novelty_weight"],
                    "competence": self.config["competence_weight"],
                    "mastery": self.config["mastery_weight"]
                }
            }
        }
        return stats

    async def save_state(self, file_path: str) -> bool:
        try:
            # Save curiosity system first
            await self.curiosity_system.save_state(f"{file_path}_curiosity")
            motivation_state = {
                "novelty_model": self.novelty_model,
                "competence_model": {
                    "skill_areas": self.competence_model["skill_areas"],
                    "learning_curves": self.competence_model["learning_curves"],
                    "competence_thresholds": self.competence_model["competence_thresholds"]
                },
                "curiosity_targets": self.curiosity_targets,
                "motivation_levels": self.motivation_levels,
                "motivation_history": self.motivation_history,
                "config": self.config,
                "timestamp": datetime.now().isoformat()
            }
            with open(file_path, 'w') as f:
                json.dump(motivation_state, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving motivation state: {e}")
            return False

    async def load_state(self, file_path: str) -> bool:
        try:
            await self.curiosity_system.load_state(f"{file_path}_curiosity")
            with open(file_path, 'r') as f:
                state = json.load(f)
            self.novelty_model = state["novelty_model"]
            self.competence_model = state["competence_model"]
            self.curiosity_targets = state["curiosity_targets"]
            self.motivation_levels = state["motivation_levels"]
            self.motivation_history = state["motivation_history"]
            self.config = state["config"]
            return True
        except Exception as e:
            logger.error(f"Error loading motivation state: {e}")
            return False


###############################################################################
#                            Knowledge Core (Unified)
###############################################################################

class KnowledgeCore:
    """
    Main system that integrates knowledge across different components, storing data
    in a directed graph and using embeddings for similarity detection, plus a
    reasoning system for contradiction handling, and a built-in CuriositySystem.
    This merges all logic from module 1 (KnowledgeIntegrationSystem) and
    module 2 (KnowledgeCore, etc.).
    """

    def __init__(self, knowledge_store_file: str = "knowledge_store.json"):
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

        # Default config
        self.config = {
            "conflict_threshold": 0.7,
            "support_threshold": 0.7,
            "similarity_threshold": 0.8,  # merges
            "decay_rate": 0.01,
            "integration_cycle_interval": 10,  # minutes
            "enable_embeddings": True,
            "max_node_age_days": 30,
            "pruning_confidence_threshold": 0.3
        }

        self.last_integration_cycle = datetime.now()
        self.next_node_id = 1

        self.integration_cache: Dict[str, Any] = {}
        self.query_cache: Dict[str, Any] = {}

        self.memory_system = None
        self.reasoning_system = None
        self.knowledge_store_file = knowledge_store_file

        # Our curiosity system for knowledge-gap-based exploration
        self.curiosity_system = CuriositySystem()

        # Attempt to load an embedding model if configured
        self._embedding_model = None
        self._try_load_embedding_model()

    def _try_load_embedding_model(self) -> None:
        if self.config["enable_embeddings"]:
            try:
                # We try sentence_transformers first
                try:
                    from sentence_transformers import SentenceTransformer
                    self._embedding_model = SentenceTransformer('all-mpnet-base-v2')
                    logger.info("Loaded embedding model (SentenceTransformer).")
                except ImportError:
                    # Then try huggingface transformers
                    try:
                        from transformers import AutoTokenizer, AutoModel
                        import torch
                        class SimpleEmbedder:
                            def __init__(self, model_name):
                                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                                self.model = AutoModel.from_pretrained(model_name)
                            def encode(self, texts, show_progress_bar=False):
                                def mean_pooling(model_output, attention_mask):
                                    token_embeddings = model_output[0]
                                    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                                    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

                                encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
                                with torch.no_grad():
                                    output = self.model(**encoded)
                                pooled = mean_pooling(output, encoded['attention_mask'])
                                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                                return pooled.numpy()
                        self._embedding_model = SimpleEmbedder('sentence-transformers/all-mpnet-base-v2')
                        logger.info("Loaded embedding model (Transformers).")
                    except ImportError:
                        logger.warning("No suitable embedding library found.")
                        self._embedding_model = None
            except Exception as e:
                logger.warning(f"Error loading embedding model: {e}")
                self._embedding_model = None

    async def initialize(self, system_references: Dict[str, Any]) -> None:
        if "memory_system" in system_references:
            self.memory_system = system_references["memory_system"]
        if "reasoning_system" in system_references:
            self.reasoning_system = system_references["reasoning_system"]
        else:
            self.reasoning_system = SimpleReasoningSystem()
        await self._load_knowledge()
        logger.info("Knowledge Core initialized.")

    async def _load_knowledge(self) -> None:
        if not os.path.exists(self.knowledge_store_file):
            logger.info("No existing knowledge store found; starting fresh.")
            return
        try:
            with open(self.knowledge_store_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for nd in data.get("nodes", []):
                node = KnowledgeNode.from_dict(nd)
                self.nodes[node.id] = node
                self.graph.add_node(node.id, **node.to_dict())
                try:
                    numeric_id = int(node.id.split("_")[-1])
                    if numeric_id >= self.next_node_id:
                        self.next_node_id = numeric_id + 1
                except:
                    pass
            for rd in data.get("relations", []):
                rel = KnowledgeRelation.from_dict(rd)
                self.graph.add_edge(rel.source_id, rel.target_id, **rel.to_dict())
                if rel.type == "supports":
                    self.nodes[rel.source_id].supporting_nodes.append(rel.target_id)
                elif rel.type == "contradicts":
                    self.nodes[rel.source_id].conflicting_nodes.append(rel.target_id)
            if self.config["enable_embeddings"] and self._embedding_model:
                await self._rebuild_embeddings()
            logger.info(f"Knowledge store loaded with {len(self.nodes)} nodes and {self.graph.number_of_edges()} edges.")
        except Exception as e:
            logger.error(f"Error loading knowledge: {str(e)}")

    async def _rebuild_embeddings(self) -> None:
        for node in self.nodes.values():
            await self._update_node_embedding(node)

    async def save_knowledge(self) -> None:
        try:
            nodes_data = [n.to_dict() for n in self.nodes.values()]
            relations_data = []
            for s, t, d in self.graph.edges(data=True):
                relation_dict = {
                    "source_id": s,
                    "target_id": t,
                    "type": d.get("type"),
                    "weight": d.get("weight", 1.0),
                    "timestamp": d.get("timestamp"),
                    "metadata": d.get("metadata", {})
                }
                if isinstance(relation_dict["timestamp"], datetime):
                    relation_dict["timestamp"] = relation_dict["timestamp"].isoformat()
                relations_data.append(relation_dict)
            store_data = {"nodes": nodes_data, "relations": relations_data}
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
        node_id = f"node_{self.next_node_id}"
        self.next_node_id += 1
        node = KnowledgeNode(node_id, type, content, source, confidence)
        # Check for existing similar nodes
        similar_nodes = await self._find_similar_nodes(node)
        if similar_nodes:
            most_similar_id, sim_val = similar_nodes[0]
            if sim_val > self.config["similarity_threshold"]:
                return await self._integrate_with_existing(node, most_similar_id, sim_val)
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.to_dict())
        if self.config["enable_embeddings"] and self._embedding_model:
            await self._add_node_embedding(node)
        if relations:
            for r in relations:
                await self.add_relation(
                    source_id=node_id,
                    target_id=r["target_id"],
                    type=r["type"],
                    weight=r.get("weight", 1.0),
                    metadata=r.get("metadata", {})
                )
        for sid, sval in similar_nodes:
            if sval > self.config["similarity_threshold"] * 0.6:
                await self.add_relation(
                    node_id, sid, "similar_to", sval, {"auto_generated": True}
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
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.warning(f"Cannot add relation: {source_id} or {target_id} not found.")
            return False
        rel = KnowledgeRelation(source_id, target_id, type, weight, metadata or {})
        self.graph.add_edge(source_id, target_id, **rel.to_dict())
        if type == "supports":
            if target_id not in self.nodes[source_id].supporting_nodes:
                self.nodes[source_id].supporting_nodes.append(target_id)
        elif type == "contradicts":
            if target_id not in self.nodes[source_id].conflicting_nodes:
                self.nodes[source_id].conflicting_nodes.append(target_id)
            await self._handle_contradiction(source_id, target_id)
        self.integration_stats["relations_added"] += 1
        # Also reflect in curiosity system if relevant
        if type in ["related", "similar_to", "specializes"]:
            s_node = self.nodes[source_id]
            t_node = self.nodes[target_id]
            s_dom = s_node.content.get("domain", s_node.type)
            s_top = s_node.content.get("topic", list(s_node.content.keys())[0] if s_node.content else "unknown")
            t_dom = t_node.content.get("domain", t_node.type)
            t_top = t_node.content.get("topic", list(t_node.content.keys())[0] if t_node.content else "unknown")
            await self.curiosity_system.add_knowledge_connection(s_dom, s_top, t_dom, t_top, weight)
        return True

    async def _integrate_with_existing(self, new_node: KnowledgeNode,
                                       existing_id: str, similarity: float) -> str:
        existing_node = self.nodes[existing_id]
        existing_node.access()
        if new_node.confidence > existing_node.confidence:
            updated_content = new_node.content.copy()
            for k, v in existing_node.content.items():
                if k not in updated_content:
                    updated_content[k] = v
            updated_conf = new_node.confidence * 0.7 + existing_node.confidence * 0.3
            existing_node.update(updated_content, updated_conf, f"{existing_node.source}+{new_node.source}")
        else:
            for k, v in new_node.content.items():
                if k not in existing_node.content:
                    existing_node.content[k] = v
            updated_conf = existing_node.confidence * 0.7 + new_node.confidence * 0.3
            existing_node.confidence = updated_conf
            existing_node.last_accessed = datetime.now()
        self.graph.nodes[existing_id].update(existing_node.to_dict())
        if self.config["enable_embeddings"] and self._embedding_model:
            await self._update_node_embedding(existing_node)
        logger.debug(f"Integrated node {new_node.id} into existing node {existing_id}.")
        return existing_id

    async def _handle_contradiction(self, node1_id: str, node2_id: str) -> None:
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        if node1.confidence > 0.7 and node2.confidence > 0.7:
            logger.info(f"Detected high-confidence conflict between {node1_id} and {node2_id}")
            if self.reasoning_system:
                try:
                    resolution = await self.reasoning_system.resolve_contradiction(node1.to_dict(), node2.to_dict())
                    if resolution["resolved"]:
                        pref = resolution["preferred_node"]
                        if pref == node1_id:
                            node2.confidence *= 0.7
                        else:
                            node1.confidence *= 0.7
                        self.integration_stats["conflicts_resolved"] += 1
                        logger.info(f"Resolved conflict. Preferred node: {pref}")
                except Exception as e:
                    logger.error(f"Error resolving contradiction: {e}")
        elif abs(node1.confidence - node2.confidence) > 0.3:
            if node1.confidence < node2.confidence:
                node1.confidence *= 0.9
            else:
                node2.confidence *= 0.9
            logger.debug(f"Adjusted confidence in conflict: {node1_id} vs {node2_id}.")

    async def _find_similar_nodes(self, node: KnowledgeNode) -> List[Tuple[str, float]]:
        if self.config["enable_embeddings"] and self._embedding_model:
            embedding = await self._generate_embedding(node)
            sim_list = []
            for nid, emb in self.node_embeddings.items():
                sim = self._calculate_embedding_similarity(embedding, emb)
                if sim > self.config["similarity_threshold"] * 0.5:
                    sim_list.append((nid, sim))
            sim_list.sort(key=lambda x: x[1], reverse=True)
            return sim_list
        else:
            # Fallback
            results = []
            for nid, ex_node in self.nodes.items():
                if ex_node.type == node.type:
                    c_sim = self._calculate_content_similarity(node.content, ex_node.content)
                    if c_sim > self.config["similarity_threshold"] * 0.5:
                        results.append((nid, c_sim))
            results.sort(key=lambda x: x[1], reverse=True)
            return results

    async def _add_node_embedding(self, node: KnowledgeNode) -> None:
        emb = await self._generate_embedding(node)
        self.node_embeddings[node.id] = emb

    async def _update_node_embedding(self, node: KnowledgeNode) -> None:
        emb = await self._generate_embedding(node)
        self.node_embeddings[node.id] = emb

    async def _generate_embedding(self, node: KnowledgeNode) -> np.ndarray:
        if self._embedding_model:
            content_str = json.dumps(node.content, ensure_ascii=False)
            vecs = self._embedding_model.encode([content_str], show_progress_bar=False)
            return vecs[0]
        else:
            # Fallback
            fallback = np.zeros(128, dtype=np.float32)
            c_str = json.dumps(node.content)
            for i, ch in enumerate(c_str):
                idx = i % 128
                fallback[idx] += (ord(ch) % 96) / 100.0
            nrm = np.linalg.norm(fallback)
            if nrm > 0:
                fallback = fallback / nrm
            return fallback

    def _calculate_embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        dot = np.dot(emb1, emb2)
        n1 = np.linalg.norm(emb1)
        n2 = np.linalg.norm(emb2)
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return float(dot / (n1 * n2))

    def _calculate_content_similarity(self, c1: Dict[str, Any], c2: Dict[str, Any]) -> float:
        all_keys = set(c1.keys()) | set(c2.keys())
        if not all_keys:
            return 0.0
        matching_keys = set(c1.keys()) & set(c2.keys())
        matching_vals = sum(1 for k in matching_keys if c1[k] == c2[k])
        key_sim = len(matching_keys) / len(all_keys) if all_keys else 0.0
        val_sim = matching_vals / len(matching_keys) if matching_keys else 0.0
        return 0.7 * key_sim + 0.3 * val_sim

    async def _check_integration_cycle(self) -> None:
        now = datetime.now()
        elapsed = (now - self.last_integration_cycle).total_seconds()
        if elapsed > self.config["integration_cycle_interval"] * 60:
            await self._run_integration_cycle()

    async def _run_integration_cycle(self) -> None:
        logger.info("Running knowledge integration cycle.")
        self.last_integration_cycle = datetime.now()
        self.integration_stats["integration_cycles"] += 1

        await self._prune_nodes()
        await self._resolve_conflicts()
        await self._consolidate_similar_nodes()
        await self._infer_relations()
        self._apply_confidence_decay()
        await self._update_knowledge_stats()

        # Also apply knowledge decay in the curiosity system
        await self.curiosity_system.apply_knowledge_decay()

        logger.info("Integration cycle completed.")
        await self.save_knowledge()

    async def _prune_nodes(self) -> None:
        now = datetime.now()
        to_remove = []
        for nid, node in self.nodes.items():
            age_days = (now - node.timestamp).total_seconds() / (24 * 3600)
            if (age_days > self.config["max_node_age_days"] and
                node.confidence < self.config["pruning_confidence_threshold"] and
                    node.access_count < 3):
                to_remove.append(nid)
        for rid in to_remove:
            await self._remove_node(rid)
        if to_remove:
            logger.info(f"Pruned {len(to_remove)} node(s).")

    async def _remove_node(self, node_id: str) -> None:
        if node_id in self.nodes:
            del self.nodes[node_id]
        if node_id in self.graph:
            self.graph.remove_node(node_id)
        if node_id in self.node_embeddings:
            del self.node_embeddings[node_id]
        for n in self.nodes.values():
            if node_id in n.supporting_nodes:
                n.supporting_nodes.remove(node_id)
            if node_id in n.conflicting_nodes:
                n.conflicting_nodes.remove(node_id)

    async def _resolve_conflicts(self) -> None:
        contradictions = []
        for s, t, d in self.graph.edges(data=True):
            if d.get("type") == "contradicts":
                contradictions.append((s, t))
        for (n1, n2) in contradictions:
            if n1 in self.nodes and n2 in self.nodes:
                await self._handle_contradiction(n1, n2)

    async def _consolidate_similar_nodes(self) -> None:
        if not self.config["enable_embeddings"] or not self._embedding_model:
            return
        nids = list(self.nodes.keys())
        if len(nids) < 2:
            return
        pairs = []
        max_comps = min(1000, len(nids) * (len(nids) - 1) // 2)
        count = 0
        for i in range(len(nids)):
            for j in range(i + 1, len(nids)):
                count += 1
                if count > max_comps:
                    break
                id1 = nids[i]
                id2 = nids[j]
                if id1 not in self.node_embeddings or id2 not in self.node_embeddings:
                    continue
                if id2 in self.nodes[id1].conflicting_nodes:
                    continue
                sim = self._calculate_embedding_similarity(self.node_embeddings[id1], self.node_embeddings[id2])
                if sim > self.config["similarity_threshold"]:
                    pairs.append((id1, id2, sim))
        pairs.sort(key=lambda x: x[2], reverse=True)
        merged = set()
        merges_done = 0
        for (id1, id2, s_val) in pairs:
            if id1 in merged or id2 in merged:
                continue
            if id1 not in self.nodes or id2 not in self.nodes:
                continue
            if self.nodes[id1].confidence >= self.nodes[id2].confidence:
                tgt, src = id1, id2
            else:
                tgt, src = id2, id1
            await self._merge_nodes(tgt, src)
            merged.add(src)
            merges_done += 1
        if merges_done > 0:
            logger.info(f"Consolidated {merges_done} similar node pairs.")

    async def _merge_nodes(self, target_id: str, source_id: str) -> None:
        if target_id not in self.nodes or source_id not in self.nodes:
            return
        tgt_node = self.nodes[target_id]
        src_node = self.nodes[source_id]
        for k, v in src_node.content.items():
            if k not in tgt_node.content:
                tgt_node.content[k] = v
        merged_from = tgt_node.metadata.get("merged_from", [])
        merged_from.append(source_id)
        tgt_node.metadata["merged_from"] = merged_from
        tgt_node.metadata["merged_timestamp"] = datetime.now().isoformat()
        new_conf = tgt_node.confidence * 0.7 + src_node.confidence * 0.3
        tgt_node.confidence = min(1.0, new_conf)
        if src_node.source not in tgt_node.source:
            tgt_node.source = f"{tgt_node.source}+{src_node.source}"
        for succ in list(self.graph.successors(source_id)):
            if succ != target_id:
                e_data = self.graph.get_edge_data(source_id, succ)
                self.graph.add_edge(target_id, succ, **e_data)
        for pred in list(self.graph.predecessors(source_id)):
            if pred != target_id:
                e_data = self.graph.get_edge_data(pred, source_id)
                self.graph.add_edge(pred, target_id, **e_data)
        for n_id in src_node.supporting_nodes:
            if n_id != target_id and n_id not in tgt_node.supporting_nodes:
                tgt_node.supporting_nodes.append(n_id)
        for n_id in src_node.conflicting_nodes:
            if n_id != target_id and n_id not in tgt_node.conflicting_nodes:
                tgt_node.conflicting_nodes.append(n_id)
        self.graph.nodes[target_id].update(tgt_node.to_dict())
        await self._remove_node(source_id)
        if self.config["enable_embeddings"] and self._embedding_model:
            await self._update_node_embedding(tgt_node)

    async def _infer_relations(self) -> None:
        new_rels = []
        for n1 in self.nodes:
            for n2 in self.graph.successors(n1):
                d12 = self.graph.get_edge_data(n1, n2)
                rel1 = d12.get("type")
                for n3 in self.graph.successors(n2):
                    if n3 == n1:
                        continue
                    d23 = self.graph.get_edge_data(n2, n3)
                    rel2 = d23.get("type")
                    if self.graph.has_edge(n1, n3):
                        continue
                    inferred = self._infer_relation_type(rel1, rel2)
                    if inferred:
                        new_rels.append((n1, n3, inferred))
        for (src, trg, rtype) in new_rels:
            if src in self.nodes and trg in self.nodes:
                await self.add_relation(src, trg, rtype, weight=0.7, metadata={"inferred": True})
        if new_rels:
            logger.info(f"Inferred {len(new_rels)} new relations.")

    def _infer_relation_type(self, t1: str, t2: str) -> Optional[str]:
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
        now = datetime.now()
        for n in self.nodes.values():
            days_inactive = (now - n.last_accessed).total_seconds() / (24 * 3600)
            if days_inactive > 7:
                factor = 1.0 - (self.config["decay_rate"] * min(days_inactive / 30.0, 1.0))
                n.confidence *= factor
                self.graph.nodes[n.id].update(n.to_dict())

    async def _update_knowledge_stats(self) -> None:
        self.integration_stats["node_count"] = len(self.nodes)
        self.integration_stats["relation_count"] = self.graph.number_of_edges()
        ntypes = {}
        for no in self.nodes.values():
            ntypes[no.type] = ntypes.get(no.type, 0) + 1
        self.integration_stats["node_types"] = ntypes
        rtypes = {}
        for s, t, d in self.graph.edges(data=True):
            rt = d.get("type")
            if rt:
                rtypes[rt] = rtypes.get(rt, 0) + 1
        self.integration_stats["relation_types"] = rtypes

    async def query_knowledge(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.integration_stats["knowledge_queries"] += 1
        node_type = query.get("type")
        content_filter = query.get("content_filter", {})
        relation_filter = query.get("relation_filter", {})
        limit = query.get("limit", 10)
        cache_key = json.dumps({
            "type": node_type,
            "content_filter": content_filter,
            "relation_filter": relation_filter,
            "limit": limit
        }, sort_keys=True)
        now = datetime.now()
        if cache_key in self.query_cache:
            entry = self.query_cache[cache_key]
            cached_time = datetime.fromisoformat(entry["timestamp"])
            if (now - cached_time).total_seconds() < 60:
                return entry["results"]
        matching = []
        for nid, node in self.nodes.items():
            if node_type and node.type != node_type:
                continue
            c_ok = True
            for k, v in content_filter.items():
                if k not in node.content or node.content[k] != v:
                    c_ok = False
                    break
            if not c_ok:
                continue
            r_ok = True
            if relation_filter:
                rtype = relation_filter.get("type")
                other_id = relation_filter.get("node_id")
                direct = relation_filter.get("direction", "outgoing")
                if rtype and other_id:
                    if direct == "outgoing":
                        if not self.graph.has_edge(nid, other_id):
                            r_ok = False
                        else:
                            edata = self.graph.get_edge_data(nid, other_id)
                            if edata["type"] != rtype:
                                r_ok = False
                    else:
                        if not self.graph.has_edge(other_id, nid):
                            r_ok = False
                        else:
                            edata = self.graph.get_edge_data(other_id, nid)
                            if edata["type"] != rtype:
                                r_ok = False
            if not r_ok:
                continue
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

    async def get_related_knowledge(self, node_id: str,
                                    relation_type: Optional[str] = None,
                                    direction: str = "both",
                                    limit: int = 10) -> List[Dict[str, Any]]:
        if node_id not in self.nodes:
            return []
        self.nodes[node_id].access()
        neighbors = []
        if direction in ["outgoing", "both"]:
            for tgt in self.graph.successors(node_id):
                ed = self.graph.get_edge_data(node_id, tgt)
                if relation_type is None or ed["type"] == relation_type:
                    if tgt in self.nodes:
                        neighbors.append({
                            "node": self.nodes[tgt].to_dict(),
                            "relation": {
                                "type": ed["type"],
                                "direction": "outgoing",
                                "weight": ed.get("weight", 1.0)
                            }
                        })
        if direction in ["incoming", "both"]:
            for src in self.graph.predecessors(node_id):
                ed = self.graph.get_edge_data(src, node_id)
                if relation_type is None or ed["type"] == relation_type:
                    if src in self.nodes:
                        neighbors.append({
                            "node": self.nodes[src].to_dict(),
                            "relation": {
                                "type": ed["type"],
                                "direction": "incoming",
                                "weight": ed.get("weight", 1.0)
                            }
                        })
        neighbors.sort(key=lambda x: x["relation"]["weight"], reverse=True)
        return neighbors[:limit]

    async def get_knowledge_statistics(self) -> Dict[str, Any]:
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
        curiosity_stats = await self.curiosity_system.get_curiosity_statistics()
        stats["curiosity_system"] = curiosity_stats
        return stats

    # Convenience methods mirroring the CuriositySystem

    async def identify_knowledge_gaps(self) -> List[Dict[str, Any]]:
        return await self.curiosity_system.identify_knowledge_gaps()

    async def create_exploration_target(self, domain: str, topic: str,
                                        importance: float = 0.5, urgency: float = 0.5,
                                        knowledge_gap: Optional[float] = None) -> str:
        return await self.curiosity_system.create_exploration_target(domain, topic, importance, urgency, knowledge_gap)

    async def get_exploration_targets(self, limit: int = 10, min_priority: float = 0.0) -> List[Dict[str, Any]]:
        return await self.curiosity_system.get_exploration_targets(limit, min_priority)

    async def record_exploration(self, target_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        return await self.curiosity_system.record_exploration(target_id, result)

    async def generate_questions(self, target_id: str, limit: int = 5) -> List[str]:
        return await self.curiosity_system.generate_questions(target_id, limit)

    async def prioritize_domains(self) -> List[Dict[str, Any]]:
        return await self.curiosity_system.prioritize_domains()
