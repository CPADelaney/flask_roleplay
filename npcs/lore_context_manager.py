# npcs/lore_context_manager.py

"""
Lore Context Manager for NPCs

This module provides enhanced lore context management for NPCs, including:
- Sophisticated caching of lore elements
- Lore impact analysis
- Lore propagation through NPC networks
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json

from data.npc_dal import NPCDataAccess
from memory.wrapper import MemorySystem
from nyx.nyx_governance import AgentType, DirectiveType
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from db.connection import get_db_connection_context
from lore.core import canon
from lore.core.lore_system import LoreSystem

logger = logging.getLogger(__name__)

class LoreContextCache:
    """Enhanced cache for lore context with TTL and invalidation."""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}
        self.ttl_seconds = ttl_seconds
        self.last_access = {}
        self.invalidation_triggers = defaultdict(set)
        
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get item from cache if it exists and is not expired."""
        if key not in self.cache:
            return None
            
        item, timestamp = self.cache[key]
        if (datetime.now() - timestamp).total_seconds() > self.ttl_seconds:
            del self.cache[key]
            return None
            
        self.last_access[key] = datetime.now()
        return item
        
    def set(self, key: str, value: Dict[str, Any], invalidation_triggers: Optional[List[str]] = None):
        """Set cache item with current timestamp and invalidation triggers."""
        self.cache[key] = (value, datetime.now())
        if invalidation_triggers:
            for trigger in invalidation_triggers:
                self.invalidation_triggers[trigger].add(key)
                
    def invalidate(self, trigger: str):
        """Invalidate all cache entries triggered by the given event."""
        if trigger in self.invalidation_triggers:
            for key in self.invalidation_triggers[trigger]:
                if key in self.cache:
                    del self.cache[key]
            del self.invalidation_triggers[trigger]
            
    def clear(self):
        """Clear all cache and invalidation triggers."""
        self.cache.clear()
        self.invalidation_triggers.clear()
        self.last_access.clear()

class LoreImpactAnalyzer:
    """Analyzes how lore changes affect NPC behavior."""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.impact_history = []
        self.npc_behavior_changes = defaultdict(list)
        
    async def analyze_lore_impact(self, lore_change: Dict[str, Any], affected_npcs: List[int]) -> Dict[str, Any]:
        """Analyze how a lore change affects NPC behavior."""
        impact_analysis = {
            "lore_change": lore_change,
            "timestamp": datetime.now().isoformat(),
            "affected_npcs": [],
            "behavior_changes": [],
            "propagation_path": []
        }
        
        for npc_id in affected_npcs:
            npc_impact = await self._analyze_npc_impact(npc_id, lore_change)
            impact_analysis["affected_npcs"].append(npc_impact)
            
            # Track behavior changes
            if npc_impact["behavior_changes"]:
                self.npc_behavior_changes[npc_id].append({
                    "timestamp": datetime.now().isoformat(),
                    "changes": npc_impact["behavior_changes"]
                })
        
        self.impact_history.append(impact_analysis)
        return impact_analysis
        
    async def _analyze_npc_impact(self, npc_id: int, lore_change: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze impact on a specific NPC."""
        # Get NPC's current state and beliefs
        npc_state = await self._get_npc_state(npc_id)
        npc_beliefs = await self._get_npc_beliefs(npc_id)
        
        # Analyze potential behavior changes
        behavior_changes = await self._predict_behavior_changes(npc_state, npc_beliefs, lore_change)
        
        return {
            "npc_id": npc_id,
            "behavior_changes": behavior_changes,
            "belief_updates": await self._predict_belief_updates(npc_beliefs, lore_change),
            "relationship_impacts": await self._analyze_relationship_impacts(npc_id, lore_change)
        }
        
    async def _get_npc_state(self, npc_id: int) -> Dict[str, Any]:
        """Get current state of an NPC - READ ONLY operation."""
        try:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT npc_name, dominance, cruelty, personality_traits,
                           current_location, mask_integrity
                    FROM NPCStats
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                    """,
                    npc_id, self.user_id, self.conversation_id
                )
                
                if not row:
                    return {}
                
                # Parse personality traits if needed
                personality_traits = []
                if row['personality_traits'] and isinstance(row['personality_traits'], str):
                    try:
                        personality_traits = json.loads(row['personality_traits'])
                    except json.JSONDecodeError:
                        pass
                elif row['personality_traits']:
                    personality_traits = row['personality_traits']
                
                return {
                    "npc_name": row['npc_name'],
                    "dominance": row['dominance'],
                    "cruelty": row['cruelty'],
                    "personality_traits": personality_traits,
                    "current_location": row['current_location'],
                    "mask_integrity": row['mask_integrity'] if row['mask_integrity'] is not None else 100
                }
        except Exception as e:
            logger.error(f"Error getting NPC state: {e}")
            return {}
        
    async def _get_npc_beliefs(self, npc_id: int) -> Dict[str, Any]:
        """Get current beliefs of an NPC - READ ONLY operation."""
        try:
            memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
            beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=npc_id
            )
            
            return {
                "beliefs": beliefs,
                "count": len(beliefs) if beliefs else 0
            }
        except Exception as e:
            logger.error(f"Error getting NPC beliefs: {e}")
            return {"beliefs": [], "count": 0}
        
    async def _predict_behavior_changes(self, npc_state: Dict[str, Any], 
                                      npc_beliefs: Dict[str, Any], 
                                      lore_change: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict how NPC behavior might change based on lore change."""
        # Check if the lore change aligns with or contradicts existing beliefs
        behavior_changes = []
        
        # Get lore impact level
        impact_level = self._calculate_lore_impact_level(lore_change)
        
        # Skip if too low impact
        if impact_level < 0.2:
            return behavior_changes
            
        # Get relevant beliefs
        relevant_beliefs = []
        for belief in npc_beliefs.get("beliefs", []):
            belief_text = belief.get("belief", "").lower()
            lore_name = lore_change.get("name", "").lower()
            
            # Check if belief relates to the lore change
            if any(term in belief_text for term in lore_name.split()):
                relevant_beliefs.append(belief)
                
        # If no relevant beliefs, create a behavior change based on personality
        if not relevant_beliefs:
            # Use personality to determine reaction
            dominance = npc_state.get("dominance", 50) / 100.0
            cruelty = npc_state.get("cruelty", 50) / 100.0
            
            # Basic reaction based on personality
            behavior_changes.append({
                "type": "general_reaction",
                "description": self._generate_personality_based_reaction(
                    dominance, cruelty, lore_change, impact_level
                ),
                "likelihood": impact_level * 0.8
            })
            
            return behavior_changes
            
        # Otherwise, create behavior changes based on beliefs
        for belief in relevant_beliefs:
            belief_text = belief.get("belief", "").lower()
            confidence = belief.get("confidence", 0.5)
            
            # Check if lore change confirms or contradicts belief
            contradicts = await self._check_lore_contradicts_belief(lore_change, belief_text)
            
            if contradicts:
                # Lore contradicts belief - may cause distress or rejection
                behavior_changes.append({
                    "type": "belief_contradiction",
                    "description": f"May reject or resist information that contradicts belief: '{belief_text}'",
                    "likelihood": confidence * impact_level
                })
                
                # Possibility of belief change
                if impact_level > 0.6 and confidence < 0.7:
                    behavior_changes.append({
                        "type": "belief_update",
                        "description": f"May update belief based on new information",
                        "likelihood": (1 - confidence) * impact_level
                    })
            else:
                # Lore confirms belief - reinforcement
                behavior_changes.append({
                    "type": "belief_reinforcement",
                    "description": f"Will feel vindicated about belief: '{belief_text}'",
                    "likelihood": confidence * impact_level
                })
                
                # May act more confidently
                behavior_changes.append({
                    "type": "confidence_boost",
                    "description": f"May act more confidently due to belief reinforcement",
                    "likelihood": confidence * impact_level * 0.8
                })
            
        return behavior_changes
        
    def _calculate_lore_impact_level(self, lore_change: Dict[str, Any]) -> float:
        """Calculate how impactful a lore change is (0.0-1.0)."""
        # Base impact
        impact = 0.5
        
        # Major lore revelations have higher impact
        if lore_change.get("is_major_revelation", False):
            impact += 0.3
            
        # Personal relevance increases impact
        if lore_change.get("personal_relevance", False):
            impact += 0.2
            
        # Emotional content increases impact
        if lore_change.get("emotional_content", False):
            impact += 0.1
            
        return min(1.0, impact)
        
    def _generate_personality_based_reaction(
        self, dominance: float, cruelty: float, 
        lore_change: Dict[str, Any], impact_level: float
    ) -> str:
        """Generate a personality-based reaction description."""
        lore_name = lore_change.get("name", "this information")
        
        # High dominance reaction
        if dominance > 0.7:
            return f"Will seek to control or leverage {lore_name} to maintain power"
            
        # High cruelty reaction
        elif cruelty > 0.7:
            return f"May exploit {lore_name} to harm rivals or gain advantage"
            
        # Balanced reaction
        elif 0.4 <= dominance <= 0.6 and 0.4 <= cruelty <= 0.6:
            return f"Will consider how {lore_name} affects their interests and relationships"
            
        # Low dominance, low cruelty
        else:
            return f"May be cautious about {lore_name} and how others might use it"
        
    async def _check_lore_contradicts_belief(self, lore_change: Dict[str, Any], belief_text: str) -> bool:
        """Check if lore change contradicts a belief."""
        # Basic keyword contradiction check
        lore_desc = lore_change.get("description", "").lower()
        
        # Simple negation patterns
        contradiction_pairs = [
            ("always", "never"),
            ("all", "none"),
            ("must", "must not"),
            ("is", "is not"),
            ("can", "cannot")
        ]
        
        # Check for direct contradictions
        for pos, neg in contradiction_pairs:
            if pos in belief_text and neg in lore_desc:
                return True
            if neg in belief_text and pos in lore_desc:
                return True
                
        # If no direct contradiction, assume it doesn't contradict
        return False
        
    async def _predict_belief_updates(self, npc_beliefs: Dict[str, Any], 
                                    lore_change: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict how NPC beliefs might update based on lore change."""
        # Implementation would use AI to predict belief updates
        # Placeholder implementation
        return []
        
    async def _analyze_relationship_impacts(self, npc_id: int, 
                                          lore_change: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze how lore change affects NPC's relationships."""
        # Implementation would analyze relationship impacts
        # Placeholder implementation
        return []

    async def _get_relationship(self, npc1_id: int, npc2_id: int) -> Dict[str, Any]:
        """Get the relationship between two NPCs - READ ONLY operation."""
        try:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow("""
                    SELECT link_type, link_level 
                    FROM SocialLinks
                    WHERE user_id = $1 AND conversation_id = $2
                    AND (
                        (entity1_type = 'npc' AND entity1_id = $3 AND entity2_type = 'npc' AND entity2_id = $4)
                        OR
                        (entity1_type = 'npc' AND entity1_id = $4 AND entity2_type = 'npc' AND entity2_id = $3)
                    )
                """, self.user_id, self.conversation_id, npc1_id, npc2_id, npc2_id, npc1_id)
                
                if row:
                    return {
                        "link_type": row['link_type'],
                        "trust": row['link_level']
                    }
                else:
                    # Default relationship
                    return {
                        "link_type": "neutral",
                        "trust": 50
                    }
        except Exception as e:
            logger.error(f"Error getting relationship: {e}")
            return {"link_type": "neutral", "trust": 50}

class LorePropagationSystem:
    """Manages lore propagation through NPC networks."""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.propagation_history = []
        self.network_graph = defaultdict(set)
        self.gossip_effectiveness = defaultdict(float)
        
    async def propagate_lore_change(self, lore_change: Dict[str, Any], 
                                  source_npc_id: int, 
                                  target_npcs: List[int]) -> Dict[str, Any]:
        """Propagate a lore change through the NPC network."""
        propagation_result = {
            "lore_change": lore_change,
            "timestamp": datetime.now().isoformat(),
            "source_npc": source_npc_id,
            "propagation_path": [],
            "effectiveness": {}
        }
        
        # Update network graph
        self._update_network_graph(source_npc_id, target_npcs)
        
        # Propagate through network
        for target_id in target_npcs:
            path = await self._find_propagation_path(source_npc_id, target_id)
            effectiveness = await self._calculate_propagation_effectiveness(path, lore_change)
            
            propagation_result["propagation_path"].append({
                "target_npc": target_id,
                "path": path,
                "effectiveness": effectiveness
            })
            
            self.gossip_effectiveness[f"{source_npc_id}-{target_id}"] = effectiveness
            
        self.propagation_history.append(propagation_result)
        return propagation_result
        
    def _update_network_graph(self, source_id: int, target_ids: List[int]):
        """Update the NPC network graph with new connections."""
        for target_id in target_ids:
            self.network_graph[source_id].add(target_id)
            self.network_graph[target_id].add(source_id)
            
    async def _find_propagation_path(self, source_id: int, target_id: int) -> List[int]:
        """Find the optimal path for lore propagation between NPCs."""
        # Basic implementation using breadth-first search
        if source_id == target_id:
            return [source_id]
            
        queue = [[source_id]]
        visited = {source_id}
        
        while queue:
            path = queue.pop(0)
            current = path[-1]
            
            for neighbor in self.network_graph[current]:
                if neighbor == target_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
                    
        # If no path found, return direct path as fallback
        return [source_id, target_id]
        
    async def _calculate_propagation_effectiveness(self, path: List[int], 
                                                 lore_change: Dict[str, Any]) -> float:
        """Calculate how effectively lore propagates through a given path."""
        # Baseline effectiveness
        effectiveness = 1.0
        
        # Each hop reduces effectiveness
        path_length = len(path)
        if path_length > 1:
            effectiveness *= (0.9 ** (path_length - 1))
            
        # Get relationships between path nodes to further adjust effectiveness
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]
            
            # Get relationship between these NPCs - READ ONLY operation
            relationship = await self._get_relationship(source_id, target_id)
            trust_level = relationship.get("trust", 50) / 100.0
            
            # Trust affects propagation
            effectiveness *= (0.5 + (trust_level * 0.5))
            
        # lore complexity reduces effectiveness
        lore_complexity = lore_change.get("complexity", 0.5)
        effectiveness *= (1 - (lore_complexity * 0.3))
        
        # Cap at sensible range
        return max(0.1, min(1.0, effectiveness))
        
    async def _get_relationship(self, npc1_id: int, npc2_id: int) -> Dict[str, Any]:
        """Get the relationship between two NPCs - READ ONLY operation."""
        try:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow("""
                    SELECT link_type, link_level 
                    FROM SocialLinks
                    WHERE user_id = $1 AND conversation_id = $2
                    AND (
                        (entity1_type = 'npc' AND entity1_id = $3 AND entity2_type = 'npc' AND entity2_id = $4)
                        OR
                        (entity1_type = 'npc' AND entity1_id = $4 AND entity2_type = 'npc' AND entity2_id = $3)
                    )
                """, self.user_id, self.conversation_id, npc1_id, npc2_id, npc2_id, npc1_id)
                
                if row:
                    return {
                        "link_type": row['link_type'],
                        "trust": row['link_level']
                    }
                else:
                    # Default relationship
                    return {
                        "link_type": "neutral",
                        "trust": 50
                    }
        except Exception as e:
            logger.error(f"Error getting relationship: {e}")
            return {"link_type": "neutral", "trust": 50}

class LoreContextManager:
    """Main manager for lore context in NPC system."""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.context_cache = LoreContextCache()
        self.impact_analyzer = LoreImpactAnalyzer(user_id, conversation_id)
        self.propagation_system = LorePropagationSystem(user_id, conversation_id)
        
    async def get_lore_context(self, npc_id: int, context_type: str) -> Dict[str, Any]:
        """Get lore context for an NPC, using cache if available."""
        cache_key = f"lore_context_{npc_id}_{context_type}"
        
        # Try to get from cache
        cached_context = self.context_cache.get(cache_key)
        if cached_context:
            return cached_context
            
        # If not in cache, fetch and cache
        context = await self._fetch_lore_context(npc_id, context_type)
        self.context_cache.set(cache_key, context)
        return context
        
    async def _fetch_lore_context(self, npc_id: int, context_type: str) -> Dict[str, Any]:
        """Fetch lore context from the lore system - READ ONLY operation."""
        try:
            async with get_db_connection_context() as conn:
                # Get basic NPC info
                npc_row = await conn.fetchrow("""
                    SELECT npc_name, dominance, cruelty
                    FROM NPCStats
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                """, npc_id, self.user_id, self.conversation_id)
                
                if not npc_row:
                    return {}
                    
                npc_name = npc_row['npc_name']
                
                # Get lore knowledge for this NPC
                knowledge_rows = await conn.fetch("""
                    SELECT lore_type, lore_id, knowledge_level
                    FROM LoreKnowledge
                    WHERE entity_type = 'npc' AND entity_id = $1
                    AND user_id = $2 AND conversation_id = $3
                """, npc_id, self.user_id, self.conversation_id)
                
                # Build context object
                context = {
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "context_type": context_type,
                    "knowledge": []
                }
                
                # Add knowledge items
                for row in knowledge_rows:
                    lore_type, lore_id, knowledge_level = row['lore_type'], row['lore_id'], row['knowledge_level']
                    
                    # Get lore details
                    lore_row = await conn.fetchrow("""
                        SELECT name, description 
                        FROM Lore
                        WHERE lore_type = $1 AND lore_id = $2
                        AND user_id = $3 AND conversation_id = $4
                    """, lore_type, lore_id, self.user_id, self.conversation_id)
                    
                    if lore_row:
                        context["knowledge"].append({
                            "lore_type": lore_type,
                            "lore_id": lore_id,
                            "name": lore_row['name'],
                            "description": lore_row['description'],
                            "knowledge_level": knowledge_level
                        })
                
                # Add additional context based on context_type
                if context_type == "change_impact":
                    # Add relationships
                    relationship_rows = await conn.fetch("""
                        SELECT entity2_id, link_type, link_level
                        FROM SocialLinks
                        WHERE user_id = $1 AND conversation_id = $2
                        AND entity1_type = 'npc' AND entity1_id = $3
                        AND entity2_type = 'npc'
                    """, self.user_id, self.conversation_id, npc_id)
                    
                    context["relationships"] = [
                        {
                            "npc_id": row['entity2_id'],
                            "link_type": row['link_type'],
                            "link_level": row['link_level']
                        }
                        for row in relationship_rows
                    ]
                
                return context
        except Exception as e:
            logger.error(f"Error fetching lore context: {e}")
            return {}
        
    async def handle_lore_change(self, lore_change: Dict[str, Any], 
                                source_npc_id: int, 
                                affected_npcs: List[int]) -> Dict[str, Any]:
        """Handle a lore change, including impact analysis and propagation."""
        # Analyze impact
        impact_analysis = await self.impact_analyzer.analyze_lore_impact(
            lore_change, affected_npcs
        )
        
        # Propagate change
        propagation_result = await self.propagation_system.propagate_lore_change(
            lore_change, source_npc_id, affected_npcs
        )
        
        # Invalidate affected cache entries
        self.context_cache.invalidate(f"lore_change_{lore_change['id']}")
        
        return {
            "impact_analysis": impact_analysis,
            "propagation_result": propagation_result
        }
