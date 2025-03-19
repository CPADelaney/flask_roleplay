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
