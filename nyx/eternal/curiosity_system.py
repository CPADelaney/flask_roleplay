# nyx/eternal/curiosity_system.py

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import math
import time
import random
import heapq
from collections import Counter

logger = logging.getLogger(__name__)

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

class Curios
