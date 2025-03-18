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
        """Get current state of an NPC."""
        # Implementation would fetch from NPC data access layer
        pass
        
    async def _get_npc_beliefs(self, npc_id: int) -> Dict[str, Any]:
        """Get current beliefs of an NPC."""
        # Implementation would fetch from belief system
        pass
        
    async def _predict_behavior_changes(self, npc_state: Dict[str, Any], 
                                      npc_beliefs: Dict[str, Any], 
                                      lore_change: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict how NPC behavior might change based on lore change."""
        # Implementation would use AI to predict behavior changes
        pass
        
    async def _predict_belief_updates(self, npc_beliefs: Dict[str, Any], 
                                    lore_change: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict how NPC beliefs might update based on lore change."""
        # Implementation would use AI to predict belief updates
        pass
        
    async def _analyze_relationship_impacts(self, npc_id: int, 
                                          lore_change: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze how lore change affects NPC's relationships."""
        # Implementation would analyze relationship impacts
        pass

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
        # Implementation would use pathfinding algorithm
        pass
        
    async def _calculate_propagation_effectiveness(self, path: List[int], 
                                                 lore_change: Dict[str, Any]) -> float:
        """Calculate how effectively lore propagates through a given path."""
        # Implementation would consider:
        # - Path length
        # - NPC relationships
        # - Lore complexity
        # - NPC knowledge levels
        pass

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
        """Fetch lore context from the lore system."""
        # Implementation would fetch from lore system
        pass
        
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