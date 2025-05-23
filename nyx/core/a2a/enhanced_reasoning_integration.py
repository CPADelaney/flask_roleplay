# nyx/core/a2a/enhanced_reasoning_integration.py
"""
Enhanced Context-Aware Reasoning Core - Part 3
Integration of all components with caching and performance optimizations.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Union
from datetime import datetime, timedelta
import asyncio
from functools import lru_cache, wraps
import hashlib
import pickle
from collections import OrderedDict
import psutil
import gc

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

# Import all the enhanced components
from .enhanced_reasoning_core import (
    ReasoningConfiguration, ReasoningState, EmotionReasoningIntegrator,
    GoalDirectedReasoningEngine, MemoryInformedReasoningEngine,
    CreativeInterventionGenerator, ReasoningError, CausalDiscoveryError
)
from .enhanced_reasoning_meta import (
    MetaReasoningModule, ReasoningStrategy, ExplanationGenerator,
    UncertaintyManager, ReasoningTemplateSystem
)

logger = logging.getLogger(__name__)

# ========================================================================================
# CACHING SYSTEM
# ========================================================================================

class ReasoningCache:
    """Sophisticated caching system for expensive calculations"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.similarity_cache: OrderedDict[str, Tuple[float, datetime]] = OrderedDict()
        self.path_cache: OrderedDict[str, Tuple[List[List[str]], datetime]] = OrderedDict()
        self.relevance_cache: OrderedDict[str, Tuple[float, datetime]] = OrderedDict()
        self.pattern_cache: OrderedDict[str, Tuple[Any, datetime]] = OrderedDict()
        
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        self.hit_count = 0
        self.miss_count = 0
        
    def _make_key(self, *args) -> str:
        """Create cache key from arguments"""
        # Create a stable hash from arguments
        key_data = pickle.dumps(args)
        return hashlib.md5(key_data).hexdigest()
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() - timestamp > self.ttl
    
    def _evict_if_needed(self, cache: OrderedDict):
        """Evict oldest entries if cache is full"""
        while len(cache) > self.max_size:
            cache.popitem(last=False)
    
    def get_similarity(self, item1: str, item2: str) -> Optional[float]:
        """Get cached similarity score"""
        key = self._make_key(item1, item2)
        
        if key in self.similarity_cache:
            value, timestamp = self.similarity_cache[key]
            if not self._is_expired(timestamp):
                self.hit_count += 1
                # Move to end (LRU)
                self.similarity_cache.move_to_end(key)
                return value
            else:
                del self.similarity_cache[key]
        
        self.miss_count += 1
        return None
    
    def set_similarity(self, item1: str, item2: str, similarity: float):
        """Cache similarity score"""
        key = self._make_key(item1, item2)
        self.similarity_cache[key] = (similarity, datetime.now())
        self._evict_if_needed(self.similarity_cache)
    
    def get_paths(self, start: str, end: str, graph_id: str) -> Optional[List[List[str]]]:
        """Get cached paths"""
        key = self._make_key(start, end, graph_id)
        
        if key in self.path_cache:
            value, timestamp = self.path_cache[key]
            if not self._is_expired(timestamp):
                self.hit_count += 1
                self.path_cache.move_to_end(key)
                return value
            else:
                del self.path_cache[key]
        
        self.miss_count += 1
        return None
    
    def set_paths(self, start: str, end: str, graph_id: str, paths: List[List[str]]):
        """Cache paths"""
        key = self._make_key(start, end, graph_id)
        self.path_cache[key] = (paths, datetime.now())
        self._evict_if_needed(self.path_cache)
    
    def get_relevance(self, source: str, target: str, method: str) -> Optional[float]:
        """Get cached relevance score"""
        key = self._make_key(source, target, method)
        
        if key in self.relevance_cache:
            value, timestamp = self.relevance_cache[key]
            if not self._is_expired(timestamp):
                self.hit_count += 1
                self.relevance_cache.move_to_end(key)
                return value
            else:
                del self.relevance_cache[key]
        
        self.miss_count += 1
        return None
    
    def set_relevance(self, source: str, target: str, method: str, relevance: float):
        """Cache relevance score"""
        key = self._make_key(source, target, method)
        self.relevance_cache[key] = (relevance, datetime.now())
        self._evict_if_needed(self.relevance_cache)
    
    def get_pattern(self, pattern_type: str, context_hash: str) -> Optional[Any]:
        """Get cached pattern analysis"""
        key = self._make_key(pattern_type, context_hash)
        
        if key in self.pattern_cache:
            value, timestamp = self.pattern_cache[key]
            if not self._is_expired(timestamp):
                self.hit_count += 1
                self.pattern_cache.move_to_end(key)
                return value
            else:
                del self.pattern_cache[key]
        
        self.miss_count += 1
        return None
    
    def set_pattern(self, pattern_type: str, context_hash: str, pattern: Any):
        """Cache pattern analysis"""
        key = self._make_key(pattern_type, context_hash)
        self.pattern_cache[key] = (pattern, datetime.now())
        self._evict_if_needed(self.pattern_cache)
    
    def clear_expired(self):
        """Remove all expired entries"""
        for cache in [self.similarity_cache, self.path_cache, 
                     self.relevance_cache, self.pattern_cache]:
            expired_keys = []
            for key, (_, timestamp) in cache.items():
                if self._is_expired(timestamp):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = (len(self.similarity_cache) + len(self.path_cache) + 
                        len(self.relevance_cache) + len(self.pattern_cache))
        
        hit_rate = self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0
        
        return {
            "total_entries": total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "similarity_entries": len(self.similarity_cache),
            "path_entries": len(self.path_cache),
            "relevance_entries": len(self.relevance_cache),
            "pattern_entries": len(self.pattern_cache)
        }

# ========================================================================================
# LAZY EVALUATION
# ========================================================================================

class LazyReasoningResult:
    """Compute results only when needed"""
    
    def __init__(self, computation_func: Callable, *args, **kwargs):
        self._computation = computation_func
        self._args = args
        self._kwargs = kwargs
        self._result = None
        self._computed = False
        self._error = None
    
    @property
    def value(self):
        """Get the computed value"""
        if not self._computed:
            try:
                self._result = asyncio.run(self._computation(*self._args, **self._kwargs))
            except Exception as e:
                self._error = e
                raise
            finally:
                self._computed = True
        
        if self._error:
            raise self._error
            
        return self._result
    
    @property
    def is_computed(self) -> bool:
        """Check if value has been computed"""
        return self._computed
    
    def compute_if_needed(self, condition: Callable[[], bool]):
        """Compute only if condition is true"""
        if condition() and not self._computed:
            return self.value
        return None

# ========================================================================================
# UNIFIED RELEVANCE FRAMEWORK
# ========================================================================================

class RelevanceCalculator:
    """Consolidated relevance calculations with caching"""
    
    def __init__(self, cache: ReasoningCache):
        self.cache = cache
        self.methods = {
            "semantic": self._semantic_relevance,
            "structural": self._structural_relevance,
            "temporal": self._temporal_relevance,
            "causal": self._causal_relevance,
            "contextual": self._contextual_relevance
        }
    
    async def calculate_relevance(self,
                                source: Any,
                                target: Any,
                                method: str = "semantic",
                                context: Optional[Dict[str, Any]] = None) -> float:
        """Single entry point for all relevance calculations"""
        # Try cache first
        cached = self.cache.get_relevance(str(source), str(target), method)
        if cached is not None:
            return cached
        
        # Compute relevance
        if method not in self.methods:
            method = "semantic"
        
        relevance_func = self.methods[method]
        relevance = await relevance_func(source, target, context)
        
        # Cache result
        self.cache.set_relevance(str(source), str(target), method, relevance)
        
        return relevance
    
    async def _semantic_relevance(self, source: Any, target: Any, context: Optional[Dict[str, Any]]) -> float:
        """Calculate semantic relevance"""
        # Convert to string representations
        source_text = self._to_text(source).lower()
        target_text = self._to_text(target).lower()
        
        # Simple word overlap (could use embeddings in production)
        source_words = set(source_text.split())
        target_words = set(target_text.split())
        
        if not source_words or not target_words:
            return 0.0
        
        overlap = len(source_words.intersection(target_words))
        total = len(source_words.union(target_words))
        
        relevance = overlap / total if total > 0 else 0.0
        
        # Context boost
        if context and context.get("user_input"):
            context_words = set(context["user_input"].lower().split())
            if source_words.intersection(context_words) and target_words.intersection(context_words):
                relevance *= 1.2
        
        return min(1.0, relevance)
    
    async def _structural_relevance(self, source: Any, target: Any, context: Optional[Dict[str, Any]]) -> float:
        """Calculate structural relevance"""
        # Check if nodes are in same graph structure
        relevance = 0.0
        
        # If both have graph positions
        if hasattr(source, 'graph_id') and hasattr(target, 'graph_id'):
            if source.graph_id == target.graph_id:
                relevance += 0.3
        
        # Check connectivity
        if hasattr(source, 'connections') and hasattr(target, 'id'):
            if target.id in source.connections:
                relevance += 0.5
        
        # Check hierarchical relationship
        if hasattr(source, 'level') and hasattr(target, 'level'):
            level_diff = abs(source.level - target.level)
            if level_diff == 1:
                relevance += 0.3
            elif level_diff == 0:
                relevance += 0.2
        
        return min(1.0, relevance)
    
    async def _temporal_relevance(self, source: Any, target: Any, context: Optional[Dict[str, Any]]) -> float:
        """Calculate temporal relevance"""
        relevance = 0.0
        
        # Check temporal ordering
        if hasattr(source, 'timestamp') and hasattr(target, 'timestamp'):
            time_diff = abs((source.timestamp - target.timestamp).total_seconds())
            
            # Closer in time = more relevant
            if time_diff < 60:  # Within a minute
                relevance = 0.9
            elif time_diff < 3600:  # Within an hour
                relevance = 0.7
            elif time_diff < 86400:  # Within a day
                relevance = 0.5
            else:
                relevance = 0.3
        
        # Check if temporal keywords present
        if context:
            temporal_keywords = ["before", "after", "during", "when", "while"]
            user_input = context.get("user_input", "").lower()
            if any(kw in user_input for kw in temporal_keywords):
                relevance *= 1.2
        
        return min(1.0, relevance)
    
    async def _causal_relevance(self, source: Any, target: Any, context: Optional[Dict[str, Any]]) -> float:
        """Calculate causal relevance"""
        relevance = 0.0
        
        # Direct causal relationship
        if hasattr(source, 'causes') and hasattr(target, 'id'):
            if target.id in source.causes:
                relevance = 0.9
        
        # Indirect causal relationship
        elif hasattr(source, 'causal_descendants') and hasattr(target, 'id'):
            if target.id in source.causal_descendants:
                relevance = 0.6
        
        # Same causal chain
        elif hasattr(source, 'causal_chain_id') and hasattr(target, 'causal_chain_id'):
            if source.causal_chain_id == target.causal_chain_id:
                relevance = 0.4
        
        return relevance
    
    async def _contextual_relevance(self, source: Any, target: Any, context: Optional[Dict[str, Any]]) -> float:
        """Calculate relevance based on current context"""
        if not context:
            return 0.5  # Neutral relevance
        
        relevance = 0.0
        
        # Combine multiple relevance types weighted by context
        semantic = await self._semantic_relevance(source, target, context)
        
        # Weight based on query type
        user_input = context.get("user_input", "").lower()
        
        if "why" in user_input or "cause" in user_input:
            causal = await self._causal_relevance(source, target, context)
            relevance = semantic * 0.3 + causal * 0.7
        elif "when" in user_input:
            temporal = await self._temporal_relevance(source, target, context)
            relevance = semantic * 0.4 + temporal * 0.6
        elif "structure" in user_input or "organize" in user_input:
            structural = await self._structural_relevance(source, target, context)
            relevance = semantic * 0.4 + structural * 0.6
        else:
            relevance = semantic
        
        return min(1.0, relevance)
    
    def _to_text(self, obj: Any) -> str:
        """Convert object to text representation"""
        if isinstance(obj, str):
            return obj
        elif hasattr(obj, 'name'):
            return obj.name
        elif hasattr(obj, 'description'):
            return obj.description
        elif hasattr(obj, '__str__'):
            return str(obj)
        else:
            return ""

# ========================================================================================
# UNIFIED PATH FINDER
# ========================================================================================

class PathFinder:
    """Consolidated path-finding logic with caching"""
    
    def __init__(self, cache: ReasoningCache):
        self.cache = cache
        self.strategies = {
            "causal": self._find_causal_paths,
            "conceptual": self._find_conceptual_paths,
            "shortest": self._find_shortest_paths,
            "strongest": self._find_strongest_paths
        }
    
    async def find_paths(self,
                        start: str,
                        end: str,
                        graph: Dict[str, Any],
                        graph_type: str = "causal",
                        constraints: Optional[Dict[str, Any]] = None,
                        max_paths: int = 5) -> List[List[str]]:
        """Unified path finding for causal and conceptual graphs"""
        # Check cache
        graph_id = graph.get("id", "default")
        cached_paths = self.cache.get_paths(start, end, graph_id)
        if cached_paths is not None:
            return cached_paths[:max_paths]
        
        # Find paths using appropriate strategy
        strategy = self.strategies.get(graph_type, self._find_causal_paths)
        paths = await strategy(start, end, graph, constraints)
        
        # Apply constraints
        if constraints:
            paths = self._apply_constraints(paths, constraints, graph)
        
        # Sort and limit
        paths = self._rank_paths(paths, graph)[:max_paths]
        
        # Cache result
        self.cache.set_paths(start, end, graph_id, paths)
        
        return paths
    
    async def _find_causal_paths(self, 
                                start: str, 
                                end: str, 
                                graph: Dict[str, Any],
                                constraints: Optional[Dict[str, Any]]) -> List[List[str]]:
        """Find causal paths in graph"""
        paths = []
        visited = set()
        
        # DFS with path tracking
        async def dfs(current: str, target: str, path: List[str]):
            if current == target:
                paths.append(path.copy())
                return
            
            if current in visited:
                return
            
            visited.add(current)
            
            # Get successors
            edges = graph.get("edges", {})
            for edge_key, edge_data in edges.items():
                if edge_key.startswith(f"{current}->"):
                    successor = edge_key.split("->")[1]
                    if successor not in path:  # Avoid cycles
                        path.append(successor)
                        await dfs(successor, target, path)
                        path.pop()
            
            visited.remove(current)
        
        # Start search
        await dfs(start, end, [start])
        
        return paths
    
    async def _find_conceptual_paths(self,
                                   start: str,
                                   end: str,
                                   graph: Dict[str, Any],
                                   constraints: Optional[Dict[str, Any]]) -> List[List[str]]:
        """Find conceptual paths (may include similarity-based connections)"""
        # Similar to causal but considers conceptual relations
        paths = await self._find_causal_paths(start, end, graph, constraints)
        
        # Also consider similarity-based paths
        if not paths and "concepts" in graph:
            # Try to find path through similar concepts
            similarity_paths = await self._find_similarity_paths(start, end, graph)
            paths.extend(similarity_paths)
        
        return paths
    
    async def _find_shortest_paths(self,
                                  start: str,
                                  end: str,
                                  graph: Dict[str, Any],
                                  constraints: Optional[Dict[str, Any]]) -> List[List[str]]:
        """Find shortest paths using BFS"""
        paths = []
        queue = [(start, [start])]
        visited = {start}
        shortest_length = float('inf')
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > shortest_length:
                continue
            
            if current == end:
                shortest_length = len(path)
                paths.append(path)
                continue
            
            # Get neighbors
            edges = graph.get("edges", {})
            for edge_key, edge_data in edges.items():
                if edge_key.startswith(f"{current}->"):
                    neighbor = edge_key.split("->")[1]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        
        return paths
    
    async def _find_strongest_paths(self,
                                  start: str,
                                  end: str,
                                  graph: Dict[str, Any],
                                  constraints: Optional[Dict[str, Any]]) -> List[List[str]]:
        """Find paths with highest combined strength"""
        # Use modified Dijkstra's algorithm
        import heapq
        
        # Distance = negative log of strength (to maximize strength)
        distances = {node: float('inf') for node in graph.get("nodes", [])}
        distances[start] = 0
        
        # Previous node for path reconstruction
        previous = {}
        
        # Priority queue: (distance, node)
        pq = [(0, start)]
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current == end:
                break
            
            if current_dist > distances[current]:
                continue
            
            # Check neighbors
            edges = graph.get("edges", {})
            for edge_key, edge_data in edges.items():
                if edge_key.startswith(f"{current}->"):
                    neighbor = edge_key.split("->")[1]
                    strength = edge_data.get("strength", 0.5)
                    
                    # Convert strength to distance
                    edge_distance = -np.log(strength) if strength > 0 else float('inf')
                    distance = distances[current] + edge_distance
                    
                    if distance < distances.get(neighbor, float('inf')):
                        distances[neighbor] = distance
                        previous[neighbor] = current
                        heapq.heappush(pq, (distance, neighbor))
        
        # Reconstruct path
        if end in previous:
            path = []
            current = end
            while current is not None:
                path.append(current)
                current = previous.get(current)
            path.reverse()
            return [path]
        
        return []
    
    async def _find_similarity_paths(self, start: str, end: str, graph: Dict[str, Any]) -> List[List[str]]:
        """Find paths through similar concepts"""
        concepts = graph.get("concepts", {})
        if not concepts:
            return []
        
        # Build similarity graph
        similarity_threshold = 0.5
        paths = []
        
        # Simple 2-hop similarity path
        for intermediate in concepts:
            if intermediate != start and intermediate != end:
                start_sim = await self._calculate_concept_similarity(start, intermediate, concepts)
                end_sim = await self._calculate_concept_similarity(intermediate, end, concepts)
                
                if start_sim > similarity_threshold and end_sim > similarity_threshold:
                    paths.append([start, intermediate, end])
        
        return paths
    
    async def _calculate_concept_similarity(self, c1: str, c2: str, concepts: Dict[str, Any]) -> float:
        """Calculate similarity between concepts"""
        # Simplified - in production would use embeddings
        concept1 = concepts.get(c1, {})
        concept2 = concepts.get(c2, {})
        
        # Property overlap
        props1 = set(concept1.get("properties", {}).keys())
        props2 = set(concept2.get("properties", {}).keys())
        
        if not props1 or not props2:
            return 0.0
        
        overlap = len(props1.intersection(props2))
        total = len(props1.union(props2))
        
        return overlap / total if total > 0 else 0.0
    
    def _apply_constraints(self, 
                         paths: List[List[str]], 
                         constraints: Dict[str, Any],
                         graph: Dict[str, Any]) -> List[List[str]]:
        """Apply constraints to filter paths"""
        filtered_paths = []
        
        for path in paths:
            valid = True
            
            # Max length constraint
            if "max_length" in constraints and len(path) > constraints["max_length"]:
                valid = False
            
            # Min strength constraint
            if "min_strength" in constraints:
                path_strength = self._calculate_path_strength(path, graph)
                if path_strength < constraints["min_strength"]:
                    valid = False
            
            # Required nodes constraint
            if "must_include" in constraints:
                required = set(constraints["must_include"])
                if not required.issubset(set(path)):
                    valid = False
            
            # Excluded nodes constraint
            if "must_exclude" in constraints:
                excluded = set(constraints["must_exclude"])
                if excluded.intersection(set(path)):
                    valid = False
            
            if valid:
                filtered_paths.append(path)
        
        return filtered_paths
    
    def _calculate_path_strength(self, path: List[str], graph: Dict[str, Any]) -> float:
        """Calculate combined strength of path"""
        if len(path) < 2:
            return 1.0
        
        strength = 1.0
        edges = graph.get("edges", {})
        
        for i in range(len(path) - 1):
            edge_key = f"{path[i]}->{path[i+1]}"
            edge_strength = edges.get(edge_key, {}).get("strength", 0.5)
            strength *= edge_strength
        
        return strength
    
    def _rank_paths(self, paths: List[List[str]], graph: Dict[str, Any]) -> List[List[str]]:
        """Rank paths by multiple criteria"""
        scored_paths = []
        
        for path in paths:
            score = 0.0
            
            # Length score (shorter is better)
            length_score = 1.0 / len(path) if len(path) > 0 else 0
            score += length_score * 0.3
            
            # Strength score
            strength_score = self._calculate_path_strength(path, graph)
            score += strength_score * 0.5
            
            # Diversity score (unique nodes)
            diversity_score = len(set(path)) / len(path) if len(path) > 0 else 0
            score += diversity_score * 0.2
            
            scored_paths.append((score, path))
        
        # Sort by score
        scored_paths.sort(key=lambda x: x[0], reverse=True)
        
        return [path for _, path in scored_paths]

# ========================================================================================
# PROPERTY MANAGEMENT
# ========================================================================================

class PropertyManager:
    """Consolidated property handling across entities"""
    
    def __init__(self):
        self.property_extractors = {
            "node": self._extract_node_properties,
            "concept": self._extract_concept_properties,
            "relation": self._extract_relation_properties,
            "pattern": self._extract_pattern_properties
        }
    
    def extract_properties(self, 
                         entity: Any, 
                         property_type: str = "all",
                         include_derived: bool = True) -> Dict[str, Any]:
        """Extract properties from nodes, concepts, etc."""
        # Determine entity type
        entity_type = self._determine_entity_type(entity)
        
        # Get base properties
        if entity_type in self.property_extractors:
            properties = self.property_extractors[entity_type](entity)
        else:
            properties = self._extract_generic_properties(entity)
        
        # Filter by property type
        if property_type != "all":
            properties = self._filter_properties(properties, property_type)
        
        # Add derived properties if requested
        if include_derived:
            derived = self._derive_properties(entity, properties)
            properties.update(derived)
        
        return properties
    
    def compare_properties(self, 
                         props1: Dict[str, Any], 
                         props2: Dict[str, Any],
                         comparison_type: str = "similarity") -> Dict[str, Any]:
        """Unified property comparison"""
        if comparison_type == "similarity":
            return self._compare_similarity(props1, props2)
        elif comparison_type == "difference":
            return self._compare_difference(props1, props2)
        elif comparison_type == "compatibility":
            return self._compare_compatibility(props1, props2)
        else:
            return {"error": "Unknown comparison type"}
    
    def _determine_entity_type(self, entity: Any) -> str:
        """Determine the type of entity"""
        if hasattr(entity, 'node_type'):
            return "node"
        elif hasattr(entity, 'concept_type'):
            return "concept"
        elif hasattr(entity, 'relation_type'):
            return "relation"
        elif hasattr(entity, 'pattern_type'):
            return "pattern"
        else:
            return "generic"
    
    def _extract_node_properties(self, node: Any) -> Dict[str, Any]:
        """Extract properties from a causal node"""
        properties = {}
        
        # Standard properties
        for attr in ['name', 'description', 'domain', 'node_type', 'observable']:
            if hasattr(node, attr):
                properties[attr] = getattr(node, attr)
        
        # Causal properties
        if hasattr(node, 'causes'):
            properties['causes'] = list(node.causes)
        if hasattr(node, 'caused_by'):
            properties['caused_by'] = list(node.caused_by)
        
        # Metadata
        if hasattr(node, 'metadata'):
            properties['metadata'] = node.metadata
        
        return properties
    
    def _extract_concept_properties(self, concept: Any) -> Dict[str, Any]:
        """Extract properties from a concept"""
        properties = {}
        
        # Direct properties
        if hasattr(concept, 'properties'):
            properties.update(concept.properties)
        
        # Concept-specific attributes
        for attr in ['name', 'domain', 'abstraction_level', 'relations']:
            if hasattr(concept, attr):
                properties[attr] = getattr(concept, attr)
        
        return properties
    
    def _extract_relation_properties(self, relation: Any) -> Dict[str, Any]:
        """Extract properties from a relation"""
        properties = {}
        
        for attr in ['source', 'target', 'relation_type', 'strength', 'mechanism']:
            if hasattr(relation, attr):
                properties[attr] = getattr(relation, attr)
        
        return properties
    
    def _extract_pattern_properties(self, pattern: Any) -> Dict[str, Any]:
        """Extract properties from a pattern"""
        properties = {}
        
        for attr in ['pattern_type', 'frequency', 'confidence', 'domain', 'components']:
            if hasattr(pattern, attr):
                properties[attr] = getattr(pattern, attr)
        
        return properties
    
    def _extract_generic_properties(self, entity: Any) -> Dict[str, Any]:
        """Extract properties from generic entity"""
        properties = {}
        
        # Get all non-private attributes
        for attr in dir(entity):
            if not attr.startswith('_') and not callable(getattr(entity, attr)):
                try:
                    value = getattr(entity, attr)
                    # Only include serializable values
                    if isinstance(value, (str, int, float, bool, list, dict)):
                        properties[attr] = value
                except:
                    pass
        
        return properties
    
    def _filter_properties(self, properties: Dict[str, Any], property_type: str) -> Dict[str, Any]:
        """Filter properties by type"""
        filtered = {}
        
        type_patterns = {
            "causal": ["cause", "effect", "mechanism", "strength"],
            "semantic": ["meaning", "definition", "description", "domain"],
            "structural": ["type", "level", "position", "connections"],
            "metadata": ["created", "modified", "source", "confidence"]
        }
        
        patterns = type_patterns.get(property_type, [])
        
        for key, value in properties.items():
            if any(pattern in key.lower() for pattern in patterns):
                filtered[key] = value
        
        return filtered
    
    def _derive_properties(self, entity: Any, base_properties: Dict[str, Any]) -> Dict[str, Any]:
        """Derive additional properties from base properties"""
        derived = {}
        
        # Centrality metrics
        if "causes" in base_properties and "caused_by" in base_properties:
            derived["in_degree"] = len(base_properties["caused_by"])
            derived["out_degree"] = len(base_properties["causes"])
            derived["total_degree"] = derived["in_degree"] + derived["out_degree"]
        
        # Complexity metrics
        if "properties" in base_properties:
            derived["property_count"] = len(base_properties["properties"])
        
        # Type classification
        if "node_type" in base_properties:
            if base_properties["node_type"] == "observable":
                derived["measurable"] = True
            elif base_properties["node_type"] == "latent":
                derived["measurable"] = False
        
        return derived
    
    def _compare_similarity(self, props1: Dict[str, Any], props2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare properties for similarity"""
        comparison = {
            "overall_similarity": 0.0,
            "common_properties": [],
            "similar_values": [],
            "property_overlap": 0.0
        }
        
        # Common keys
        keys1 = set(props1.keys())
        keys2 = set(props2.keys())
        common_keys = keys1.intersection(keys2)
        
        comparison["common_properties"] = list(common_keys)
        comparison["property_overlap"] = len(common_keys) / max(len(keys1), len(keys2)) if keys1 or keys2 else 0
        
        # Value similarity for common properties
        value_similarities = []
        for key in common_keys:
            val1 = props1[key]
            val2 = props2[key]
            
            if val1 == val2:
                comparison["similar_values"].append({
                    "property": key,
                    "similarity": 1.0
                })
                value_similarities.append(1.0)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                diff = abs(val1 - val2)
                max_val = max(abs(val1), abs(val2))
                similarity = 1.0 - (diff / max_val) if max_val > 0 else 0
                comparison["similar_values"].append({
                    "property": key,
                    "similarity": similarity
                })
                value_similarities.append(similarity)
        
        # Overall similarity
        if value_similarities:
            comparison["overall_similarity"] = sum(value_similarities) / len(value_similarities)
        else:
            comparison["overall_similarity"] = comparison["property_overlap"]
        
        return comparison
    
    def _compare_difference(self, props1: Dict[str, Any], props2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare properties for differences"""
        comparison = {
            "unique_to_first": {},
            "unique_to_second": {},
            "different_values": []
        }
        
        keys1 = set(props1.keys())
        keys2 = set(props2.keys())
        
        # Unique properties
        for key in keys1 - keys2:
            comparison["unique_to_first"][key] = props1[key]
        
        for key in keys2 - keys1:
            comparison["unique_to_second"][key] = props2[key]
        
        # Different values for common properties
        for key in keys1.intersection(keys2):
            if props1[key] != props2[key]:
                comparison["different_values"].append({
                    "property": key,
                    "value1": props1[key],
                    "value2": props2[key]
                })
        
        return comparison
    
    def _compare_compatibility(self, props1: Dict[str, Any], props2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare properties for compatibility"""
        comparison = {
            "compatible": True,
            "compatibility_score": 1.0,
            "conflicts": [],
            "complementary": []
        }
        
        # Check for conflicts
        common_keys = set(props1.keys()).intersection(set(props2.keys()))
        
        for key in common_keys:
            val1 = props1[key]
            val2 = props2[key]
            
            # Type compatibility
            if type(val1) != type(val2):
                comparison["conflicts"].append({
                    "property": key,
                    "reason": "type_mismatch",
                    "types": [type(val1).__name__, type(val2).__name__]
                })
                comparison["compatible"] = False
            
            # Value compatibility for specific properties
            elif key in ["domain", "type", "category"]:
                if val1 != val2:
                    comparison["conflicts"].append({
                        "property": key,
                        "reason": "value_mismatch",
                        "values": [val1, val2]
                    })
                    comparison["compatible"] = False
        
        # Check for complementary properties
        unique1 = set(props1.keys()) - set(props2.keys())
        unique2 = set(props2.keys()) - set(props1.keys())
        
        if unique1 and unique2:
            comparison["complementary"] = {
                "from_first": list(unique1),
                "from_second": list(unique2)
            }
        
        # Calculate compatibility score
        conflict_penalty = len(comparison["conflicts"]) * 0.2
        comparison["compatibility_score"] = max(0, 1.0 - conflict_penalty)
        
        return comparison

# ========================================================================================
# ENHANCED CONTEXT-AWARE REASONING CORE
# ========================================================================================

class EnhancedContextAwareReasoningCore(ContextAwareModule):
    """
    Enhanced Context-Aware Reasoning Core with all improvements integrated
    """
    
    def __init__(self, original_reasoning_core):
        super().__init__("enhanced_reasoning_core")
        
        # Original core
        self.original_core = original_reasoning_core
        
        # Configuration
        self.config = ReasoningConfiguration()
        self.config.validate()
        
        # State management
        self.state = ReasoningState()
        
        # Enhanced components
        self.emotion_integrator = EmotionReasoningIntegrator(self.config)
        self.goal_engine = GoalDirectedReasoningEngine(self.config)
        self.memory_engine = MemoryInformedReasoningEngine(self.config)
        self.intervention_generator = CreativeInterventionGenerator()
        
        # Meta-reasoning
        self.meta_reasoning = MetaReasoningModule()
        
        # Explanation generation
        self.explanation_generator = ExplanationGenerator()
        
        # Uncertainty management
        self.uncertainty_manager = UncertaintyManager()
        
        # Template system
        self.template_system = ReasoningTemplateSystem()
        
        # Performance optimization
        self.cache = ReasoningCache()
        self.relevance_calculator = RelevanceCalculator(self.cache)
        self.path_finder = PathFinder(self.cache)
        self.property_manager = PropertyManager()
        
        # Context subscriptions
        self.context_subscriptions = [
            "emotional_state_update", "memory_retrieval_complete", 
            "goal_context_available", "knowledge_update",
            "perception_input", "multimodal_integration",
            "causal_discovery_request", "conceptual_blend_request",
            "intervention_request", "counterfactual_query"
        ]
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize
        self._initialize_reasoning_parameters()
        
    def _initialize_reasoning_parameters(self):
        """Initialize reasoning parameters with defaults"""
        self.reasoning_params = {
            "discovery_threshold": 0.4,
            "min_evidence_strength": 0.3,
            "exploration_depth": 3,
            "exploration_breadth": 10,
            "hypothesis_generation_rate": 0.5,
            "max_nodes_to_explore": 1000,
            "use_caching": True,
            "prune_frequency": 100,
            "max_retained_paths": 50
        }
    
    async def on_context_received(self, context: SharedContext):
        """Enhanced context reception with full integration"""
        logger.debug(f"Enhanced reasoning core received context for user: {context.user_id}")
        
        # Start performance monitoring
        self.performance_monitor.start_operation("context_processing")
        
        try:
            # Analyze input with all enhancements
            reasoning_implications = await self._enhanced_analyze_input(context)
            
            # Get contextually relevant resources
            relevant_models = await self._get_enhanced_relevant_models(context)
            relevant_spaces = await self._get_enhanced_relevant_spaces(context)
            
            # Apply template recommendations
            template_recommendations = self.template_system.get_template_recommendations({
                "user_input": context.user_input,
                "domain": self._extract_domain_from_context(context)
            })
            
            # Send enhanced reasoning context
            await self.send_context_update(
                update_type="enhanced_reasoning_context_available",
                data={
                    "reasoning_implications": reasoning_implications,
                    "available_models": relevant_models,
                    "available_spaces": relevant_spaces,
                    "template_recommendations": template_recommendations,
                    "active_reasoning_type": reasoning_implications.get("reasoning_type", "none"),
                    "confidence": reasoning_implications.get("confidence", 0.0),
                    "meta_assessment": await self.meta_reasoning.evaluate_reasoning_strategy(context.__dict__)
                },
                priority=ContextPriority.HIGH
            )
            
        finally:
            self.performance_monitor.end_operation("context_processing")
    
    async def _enhanced_analyze_input(self, context: SharedContext) -> Dict[str, Any]:
        """Enhanced input analysis with all components"""
        # Base analysis
        base_analysis = await self._analyze_input_for_reasoning(context.user_input)
        
        # Enhance with emotional context
        if context.emotional_state:
            adjusted_params = await self.emotion_integrator.adjust_reasoning_from_emotion(
                context.emotional_state,
                self.reasoning_params.copy()
            )
            base_analysis["emotion_adjusted_parameters"] = adjusted_params
        
        # Enhance with goal context
        if context.goal_context:
            goal_plan = await self.goal_engine.align_reasoning_with_goals(
                context.goal_context,
                list(self.original_core.causal_models.keys()),
                list(self.original_core.concept_spaces.keys())
            )
            base_analysis["goal_reasoning_plan"] = goal_plan
        
        # Enhance with memory patterns
        if context.memory_context:
            memory_guidance = await self.memory_engine.inform_reasoning_from_memory(
                context.memory_context,
                context.__dict__
            )
            base_analysis["memory_guidance"] = memory_guidance
        
        return base_analysis
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Enhanced input processing with full integration"""
        start_time = datetime.now()
        
        # Select reasoning strategy
        strategy = await self._select_reasoning_strategy(context)
        self.meta_reasoning.start_reasoning_session(strategy)
        
        try:
            # Get enhanced analysis
            enhanced_analysis = await self._enhanced_analyze_input(context)
            
            # Check for applicable templates
            template_results = None
            if enhanced_analysis.get("template_recommendations"):
                best_template = enhanced_analysis["template_recommendations"][0]
                if best_template["score"] > 0.7:
                    template_results = await self.template_system.apply_template(
                        best_template["template_id"],
                        context.__dict__
                    )
            
            # Execute reasoning with enhancements
            if template_results and template_results.get("success"):
                reasoning_results = template_results
            else:
                reasoning_results = await self._execute_enhanced_reasoning(
                    context, enhanced_analysis, strategy
                )
            
            # Generate explanations
            if reasoning_results.get("causal_paths"):
                explanations = await self.explanation_generator.generate_causal_explanation(
                    reasoning_results["causal_paths"],
                    target_audience="general"
                )
                reasoning_results["explanations"] = explanations
            
            # Propagate uncertainties
            if reasoning_results.get("uncertainties"):
                propagated = await self.uncertainty_manager.propagate_uncertainty(
                    reasoning_results["uncertainties"],
                    reasoning_results.get("causal_graph", {})
                )
                reasoning_results["propagated_uncertainties"] = propagated
            
            # Record attempt
            duration = (datetime.now() - start_time).total_seconds()
            self.state.record_reasoning_attempt(
                ReasoningAttempt(
                    timestamp=start_time,
                    reasoning_type=enhanced_analysis.get("reasoning_type", "unknown"),
                    input_context=context.__dict__,
                    approach=enhanced_analysis,
                    results=reasoning_results,
                    success=True,
                    duration=duration,
                    models_used=list(self.state.current_models),
                    spaces_used=list(self.state.current_spaces),
                    cross_module_integration=True
                )
            )
            
            # End meta-reasoning session
            self.meta_reasoning.end_reasoning_session(
                success=True,
                insights_generated=len(reasoning_results.get("insights", []))
            )
            
            return reasoning_results
            
        except Exception as e:
            logger.error(f"Enhanced reasoning failed: {e}")
            self.meta_reasoning.end_reasoning_session(success=False)
            raise
    
    async def _select_reasoning_strategy(self, context: SharedContext) -> ReasoningStrategy:
        """Select optimal reasoning strategy based on context"""
        # Get meta-reasoning evaluation
        evaluation = await self.meta_reasoning.evaluate_reasoning_strategy(context.__dict__)
        
        # Check for recommendations
        if evaluation["recommendations"]:
            for rec in evaluation["recommendations"]:
                if rec.get("action") == "switch_strategy":
                    return ReasoningStrategy(rec["target_strategy"])
        
        # Default selection based on context
        user_input = context.user_input.lower()
        
        if "explore" in user_input or "discover" in user_input:
            return ReasoningStrategy.BREADTH_FIRST
        elif "specific" in user_input or "exact" in user_input:
            return ReasoningStrategy.DEPTH_FIRST
        elif len(self.state.reasoning_history) > 10:
            # Use best performing strategy from history
            return ReasoningStrategy.BEST_FIRST
        else:
            return ReasoningStrategy.HYBRID
    
    async def _execute_enhanced_reasoning(self,
                                        context: SharedContext,
                                        analysis: Dict[str, Any],
                                        strategy: ReasoningStrategy) -> Dict[str, Any]:
        """Execute reasoning with all enhancements"""
        results = {
            "strategy_used": strategy.value,
            "insights": [],
            "models_analyzed": [],
            "spaces_analyzed": [],
            "interventions_suggested": [],
            "uncertainties": {},
            "performance_metrics": {}
        }
        
        # Apply adjusted parameters
        adjusted_params = analysis.get("emotion_adjusted_parameters", self.reasoning_params)
        
        # Execute based on reasoning type
        reasoning_type = analysis.get("reasoning_type", "general")
        
        if reasoning_type == "causal":
            causal_results = await self._execute_enhanced_causal_reasoning(
                context, analysis, adjusted_params
            )
            results.update(causal_results)
            
        elif reasoning_type == "conceptual":
            conceptual_results = await self._execute_enhanced_conceptual_reasoning(
                context, analysis, adjusted_params
            )
            results.update(conceptual_results)
            
        elif reasoning_type == "integrated":
            integrated_results = await self._execute_enhanced_integrated_reasoning(
                context, analysis, adjusted_params
            )
            results.update(integrated_results)
            
        elif reasoning_type == "counterfactual":
            counterfactual_results = await self._execute_enhanced_counterfactual_reasoning(
                context, analysis, adjusted_params
            )
            results.update(counterfactual_results)
        
        # Apply memory shortcuts if available
        if analysis.get("memory_guidance", {}).get("shortcuts"):
            shortcuts = analysis["memory_guidance"]["shortcuts"]
            for shortcut in shortcuts:
                if shortcut["confidence"] > 0.8:
                    # Apply shortcut
                    shortcut_results = await self._apply_reasoning_shortcut(shortcut, context)
                    results["insights"].extend(shortcut_results.get("insights", []))
        
        # Generate creative interventions if needed
        if analysis.get("suggests_intervention") or "intervention" in results:
            for model_id in results.get("models_analyzed", []):
                intervention = await self.intervention_generator.suggest_creative_intervention(
                    model_id["model_id"],
                    context.__dict__,
                    {"intervention_points": results.get("intervention_points", [])}
                )
                if intervention:
                    results["interventions_suggested"].append(intervention)
        
        # Cache performance metrics
        results["performance_metrics"] = {
            "cache_stats": self.cache.get_stats(),
            "reasoning_duration": results.get("duration", 0),
            "nodes_explored": self.meta_reasoning.current_performance.nodes_explored if self.meta_reasoning.current_performance else 0
        }
        
        return results
    
    async def _execute_enhanced_causal_reasoning(self,
                                               context: SharedContext,
                                               analysis: Dict[str, Any],
                                               params: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced causal reasoning with all improvements"""
        results = {
            "reasoning_type": "causal",
            "models_analyzed": [],
            "causal_paths": [],
            "intervention_points": [],
            "uncertainties": {}
        }
        
        # Get goal-aligned models
        if analysis.get("goal_reasoning_plan"):
            model_ids = analysis["goal_reasoning_plan"]["selected_models"]
        else:
            model_ids = list(self.original_core.causal_models.keys())[:3]
        
        for model_id in model_ids:
            if model_id not in self.original_core.causal_models:
                continue
                
            model = self.original_core.causal_models[model_id]
            self.state.current_models.add(model_id)
            
            # Find causal paths with caching
            relevant_paths = await self._find_causal_paths_cached(model, context)
            results["causal_paths"].extend(relevant_paths)
            
            # Identify intervention points with uncertainty
            intervention_analysis = await self._analyze_interventions_with_uncertainty(
                model, context, params
            )
            results["intervention_points"].extend(intervention_analysis["points"])
            results["uncertainties"].update(intervention_analysis["uncertainties"])
            
            # Record exploration
            self.meta_reasoning.record_exploration(len(model.nodes))
            
            results["models_analyzed"].append({
                "model_id": model_id,
                "paths_found": len(relevant_paths),
                "interventions_identified": len(intervention_analysis["points"])
            })
        
        return results
    
    async def _find_causal_paths_cached(self, model, context: SharedContext) -> List[Dict[str, Any]]:
        """Find causal paths with caching"""
        # Extract relevant nodes from context
        input_keywords = set(context.user_input.lower().split())
        
        relevant_start_nodes = []
        relevant_end_nodes = []
        
        for node_id, node in model.nodes.items():
            node_name_lower = node.name.lower()
            node_relevance = await self.relevance_calculator.calculate_relevance(
                node, context.user_input, "contextual", context.__dict__
            )
            
            if node_relevance > 0.5:
                if any(kw in ["cause", "why", "because"] for kw in input_keywords):
                    relevant_end_nodes.append(node_id)
                else:
                    relevant_start_nodes.append(node_id)
        
        # Find paths between relevant nodes
        all_paths = []
        graph_data = self._model_to_graph(model)
        
        for start in relevant_start_nodes[:2]:  # Limit for performance
            for end in relevant_end_nodes[:2]:
                if start != end:
                    paths = await self.path_finder.find_paths(
                        start, end, graph_data, "causal",
                        constraints={"max_length": 5, "min_strength": 0.2}
                    )
                    
                    for path in paths:
                        path_dict = self._path_to_dict(path, model)
                        all_paths.append(path_dict)
        
        return all_paths
    
    def _model_to_graph(self, model) -> Dict[str, Any]:
        """Convert causal model to graph format for path finder"""
        graph = {
            "id": model.id if hasattr(model, 'id') else "model",
            "nodes": list(model.nodes.keys()),
            "edges": {},
            "parents": {},
            "children": {}
        }
        
        for node_id in model.nodes:
            graph["parents"][node_id] = []
            graph["children"][node_id] = []
        
        for relation in model.relations:
            edge_key = f"{relation.source}->{relation.target}"
            graph["edges"][edge_key] = {
                "strength": relation.strength,
                "mechanism": relation.mechanism
            }
            
            graph["children"][relation.source].append(relation.target)
            graph["parents"][relation.target].append(relation.source)
        
        return graph
    
    def _path_to_dict(self, path: List[str], model) -> Dict[str, Any]:
        """Convert path to dictionary format"""
        path_dict = {
            "nodes": [],
            "total_strength": 1.0,
            "mechanisms": []
        }
        
        for i, node_id in enumerate(path):
            node = model.nodes.get(node_id)
            if node:
                path_dict["nodes"].append({
                    "node_id": node_id,
                    "node_name": node.name,
                    "node_type": getattr(node, 'node_type', 'standard')
                })
                
                if i > 0:
                    # Find relation
                    prev_id = path[i-1]
                    for relation in model.relations:
                        if relation.source == prev_id and relation.target == node_id:
                            path_dict["total_strength"] *= relation.strength
                            path_dict["mechanisms"].append(relation.mechanism)
                            break
        
        return path_dict
    
    async def _analyze_interventions_with_uncertainty(self,
                                                    model,
                                                    context: SharedContext,
                                                    params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze intervention points with uncertainty propagation"""
        analysis = {
            "points": [],
            "uncertainties": {}
        }
        
        # Initial uncertainties
        initial_uncertainties = {}
        for node_id, node in model.nodes.items():
            if hasattr(node, 'uncertainty'):
                initial_uncertainties[node_id] = node.uncertainty
            else:
                # Default uncertainty
                initial_uncertainties[node_id] = UncertaintyEstimate(
                    value=0.5,
                    lower_bound=0.2,
                    upper_bound=0.8,
                    uncertainty_type=UncertaintyType.MODEL
                )
        
        # Propagate uncertainties
        graph_data = self._model_to_graph(model)
        propagated = await self.uncertainty_manager.propagate_uncertainty(
            initial_uncertainties, graph_data
        )
        analysis["uncertainties"] = propagated
        
        # Find critical uncertainties
        impact_scores = self._calculate_impact_scores(model)
        critical = await self.uncertainty_manager.identify_critical_uncertainties(
            propagated, impact_scores
        )
        
        # Convert to intervention points
        for critical_point in critical:
            node = model.nodes.get(critical_point["node"])
            if node:
                analysis["points"].append({
                    "node_id": critical_point["node"],
                    "node_name": node.name,
                    "intervention_score": critical_point["criticality"],
                    "uncertainty": critical_point["uncertainty"],
                    "reduction_priority": critical_point["reduction_priority"]
                })
        
        return analysis
    
    def _calculate_impact_scores(self, model) -> Dict[str, float]:
        """Calculate impact scores for nodes"""
        scores = {}
        
        for node_id, node in model.nodes.items():
            # Simple impact based on connectivity
            out_degree = sum(1 for r in model.relations if r.source == node_id)
            in_degree = sum(1 for r in model.relations if r.target == node_id)
            
            # Normalize
            total_relations = len(model.relations)
            if total_relations > 0:
                scores[node_id] = (out_degree + in_degree * 0.5) / total_relations
            else:
                scores[node_id] = 0.5
        
        return scores
    
    async def _apply_reasoning_shortcut(self, 
                                      shortcut: Dict[str, Any],
                                      context: SharedContext) -> Dict[str, Any]:
        """Apply a reasoning shortcut from memory"""
        results = {
            "insights": [],
            "shortcut_applied": shortcut["name"]
        }
        
        for step in shortcut["steps"]:
            if step["action"] == "apply_known_relations":
                # Apply pre-discovered relations
                for relation in step["data"]:
                    results["insights"].append({
                        "type": "causal_relation",
                        "content": f"Known: {relation}",
                        "confidence": shortcut["confidence"]
                    })
            
            elif step["action"] == "focus_on_key_factors":
                # Focus reasoning on specific factors
                for factor in step["data"]:
                    results["insights"].append({
                        "type": "key_factor",
                        "content": f"Key factor: {factor}",
                        "confidence": shortcut["confidence"]
                    })
        
        return results
    
    async def _execute_enhanced_conceptual_reasoning(self,
                                                   context: SharedContext,
                                                   analysis: Dict[str, Any],
                                                   params: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced conceptual reasoning"""
        results = {
            "reasoning_type": "conceptual",
            "spaces_analyzed": [],
            "concepts_explored": [],
            "blends_created": [],
            "patterns_identified": []
        }
        
        # Get relevant spaces
        space_ids = analysis.get("goal_reasoning_plan", {}).get("selected_spaces", 
                                                               list(self.original_core.concept_spaces.keys())[:3])
        
        for space_id in space_ids:
            if space_id not in self.original_core.concept_spaces:
                continue
                
            space = self.original_core.concept_spaces[space_id]
            self.state.current_spaces.add(space_id)
            
            # Explore concepts with enhanced relevance
            relevant_concepts = await self._explore_concepts_enhanced(space, context, params)
            results["concepts_explored"].extend(relevant_concepts)
            
            # Identify patterns
            patterns = await self._identify_patterns_enhanced(space, context)
            results["patterns_identified"].extend(patterns)
            
            results["spaces_analyzed"].append({
                "space_id": space_id,
                "concepts_found": len(relevant_concepts),
                "patterns_found": len(patterns)
            })
        
        # Try creative blending if multiple spaces
        if len(space_ids) >= 2:
            blend_results = await self._try_creative_blending(space_ids[:2], context, params)
            if blend_results:
                results["blends_created"].append(blend_results)
        
        return results
    
    async def _explore_concepts_enhanced(self, 
                                       space, 
                                       context: SharedContext,
                                       params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Explore concepts with enhanced relevance and caching"""
        concepts = []
        
        for concept_id, concept in space.concepts.items():
            # Calculate relevance with caching
            relevance = await self.relevance_calculator.calculate_relevance(
                concept, context.user_input, "contextual", context.__dict__
            )
            
            if relevance > params.get("relevance_threshold", 0.3):
                # Extract properties
                properties = self.property_manager.extract_properties(concept, "all")
                
                concepts.append({
                    "concept_id": concept_id,
                    "name": concept.get("name"),
                    "relevance": relevance,
                    "properties": properties,
                    "semantic_role": self._determine_semantic_role(concept, space)
                })
        
        # Sort by relevance
        concepts.sort(key=lambda c: c["relevance"], reverse=True)
        
        return concepts[:params.get("max_concepts", 10)]
    
    async def _identify_patterns_enhanced(self, space, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify patterns with caching"""
        # Check cache
        context_hash = hashlib.md5(context.user_input.encode()).hexdigest()
        cached_patterns = self.cache.get_pattern("conceptual", context_hash)
        
        if cached_patterns:
            return cached_patterns
        
        # Identify patterns (simplified - would be more complex in production)
        patterns = []
        
        # Hierarchical patterns
        hierarchy = self._find_hierarchical_patterns(space)
        if hierarchy:
            patterns.append({
                "type": "hierarchy",
                "description": f"Hierarchical organization with {len(hierarchy)} levels",
                "details": hierarchy
            })
        
        # Cluster patterns
        clusters = self._find_conceptual_clusters(space)
        if clusters:
            patterns.append({
                "type": "clusters",
                "description": f"Found {len(clusters)} conceptual clusters",
                "details": clusters
            })
        
        # Cache results
        self.cache.set_pattern("conceptual", context_hash, patterns)
        
        return patterns
    
    async def _try_creative_blending(self,
                                   space_ids: List[str],
                                   context: SharedContext,
                                   params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Try creative conceptual blending"""
        if len(space_ids) < 2:
            return None
        
        space1 = self.original_core.concept_spaces.get(space_ids[0])
        space2 = self.original_core.concept_spaces.get(space_ids[1])
        
        if not space1 or not space2:
            return None
        
        # Determine blend type based on context
        blend_type = self._determine_blend_type_from_context(context)
        
        # Find mappings
        mappings = await self._find_contextual_mappings(space_ids[0], space_ids[1], context)
        
        if not mappings:
            return None
        
        # Create blend
        blend_result = {
            "blend_type": blend_type,
            "input_spaces": [space1.name, space2.name],
            "mappings_found": len(mappings),
            "novel_concepts": [],
            "emergent_properties": []
        }
        
        # Generate novel concepts (simplified)
        for mapping in mappings[:3]:
            concept1 = space1.concepts.get(mapping["concept1"])
            concept2 = space2.concepts.get(mapping["concept2"])
            
            if concept1 and concept2:
                novel_concept = {
                    "name": f"{concept1['name']}_{concept2['name']}_blend",
                    "source_concepts": [concept1["name"], concept2["name"]],
                    "mapping_strength": mapping["similarity"],
                    "properties": self._blend_properties(concept1, concept2, blend_type)
                }
                blend_result["novel_concepts"].append(novel_concept)
        
        # Identify emergent properties
        if len(blend_result["novel_concepts"]) > 2:
            blend_result["emergent_properties"] = ["cross_domain_synthesis", "novel_combinations"]
        
        return blend_result
    
    def _blend_properties(self, concept1: Dict, concept2: Dict, blend_type: str) -> Dict[str, Any]:
        """Blend properties based on blend type"""
        props1 = concept1.get("properties", {})
        props2 = concept2.get("properties", {})
        
        if blend_type == "fusion":
            # Deep fusion - merge all properties
            blended = {**props1, **props2}
            # Resolve conflicts by creating composite values
            for key in set(props1.keys()).intersection(set(props2.keys())):
                blended[key] = f"fused({props1[key]}, {props2[key]})"
        
        elif blend_type == "composition":
            # Standard composition
            blended = {**props1}
            # Add unique properties from concept2
            for key, value in props2.items():
                if key not in blended:
                    blended[key] = value
        
        else:
            # Default - simple merge
            blended = {**props1, **props2}
        
        return blended
    
    async def _execute_enhanced_integrated_reasoning(self,
                                                   context: SharedContext,
                                                   analysis: Dict[str, Any],
                                                   params: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced integrated causal and conceptual reasoning"""
        # Execute both types
        causal_results = await self._execute_enhanced_causal_reasoning(context, analysis, params)
        conceptual_results = await self._execute_enhanced_conceptual_reasoning(context, analysis, params)
        
        # Integrate results
        integrated_results = {
            "reasoning_type": "integrated",
            "causal_results": causal_results,
            "conceptual_results": conceptual_results,
            "integration_insights": [],
            "cross_domain_connections": []
        }
        
        # Find integration opportunities
        for causal_path in causal_results.get("causal_paths", []):
            for concept in conceptual_results.get("concepts_explored", []):
                # Check for connections
                connection_strength = await self._assess_causal_conceptual_connection(
                    causal_path, concept
                )
                
                if connection_strength > 0.5:
                    integrated_results["cross_domain_connections"].append({
                        "causal_element": causal_path["nodes"][-1]["node_name"],
                        "concept": concept["name"],
                        "connection_strength": connection_strength,
                        "insight": "Causal mechanism maps to conceptual structure"
                    })
        
        # Generate integration insights
        if integrated_results["cross_domain_connections"]:
            integrated_results["integration_insights"].append({
                "type": "causal_conceptual_mapping",
                "description": f"Found {len(integrated_results['cross_domain_connections'])} connections between causal and conceptual domains",
                "implications": "Suggests unified underlying structure"
            })
        
        return integrated_results
    
    async def _assess_causal_conceptual_connection(self, 
                                                 causal_path: Dict[str, Any],
                                                 concept: Dict[str, Any]) -> float:
        """Assess connection between causal path and concept"""
        # Extract key terms from causal path
        causal_terms = set()
        for node in causal_path["nodes"]:
            causal_terms.update(node["node_name"].lower().split())
        
        # Extract concept terms
        concept_terms = set(concept["name"].lower().split())
        concept_terms.update(concept.get("properties", {}).keys())
        
        # Calculate overlap
        overlap = len(causal_terms.intersection(concept_terms))
        total = len(causal_terms.union(concept_terms))
        
        return overlap / total if total > 0 else 0.0
    
    async def _execute_enhanced_counterfactual_reasoning(self,
                                                       context: SharedContext,
                                                       analysis: Dict[str, Any],
                                                       params: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced counterfactual reasoning"""
        results = {
            "reasoning_type": "counterfactual",
            "scenarios_analyzed": [],
            "alternative_paths": [],
            "robustness_analysis": {}
        }
        
        # Find intervention point from query
        intervention_point = self._extract_intervention_point(context.user_input)
        
        if not intervention_point:
            return results
        
        # Analyze across relevant models
        model_ids = list(self.original_core.causal_models.keys())[:2]
        
        for model_id in model_ids:
            model = self.original_core.causal_models.get(model_id)
            if not model:
                continue
            
            # Generate counterfactual scenario
            scenario = await self._generate_counterfactual_scenario(model_id, context)
            if scenario:
                results["scenarios_analyzed"].append(scenario)
                
                # Find alternative paths
                alt_paths = await self._analyze_alternative_paths(model_id, scenario)
                results["alternative_paths"].extend(alt_paths)
        
        # Analyze robustness
        if results["scenarios_analyzed"]:
            results["robustness_analysis"] = self._analyze_decision_robustness(
                results["scenarios_analyzed"]
            )
        
        # Generate explanations
        if results["alternative_paths"]:
            explanations = await self.explanation_generator.generate_counterfactual_explanations(
                [], results["alternative_paths"], intervention_point
            )
            results["counterfactual_explanations"] = explanations
        
        return results
    
    def _extract_intervention_point(self, user_input: str) -> Optional[str]:
        """Extract intervention point from user query"""
        input_lower = user_input.lower()
        
        # Look for "what if X" pattern
        if "what if" in input_lower:
            after_what_if = input_lower.split("what if")[1].strip()
            # Extract first noun phrase (simplified)
            words = after_what_if.split()
            if words:
                return " ".join(words[:3])
        
        return None
    
    def _analyze_decision_robustness(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze robustness of decisions across scenarios"""
        robustness = {
            "overall_robustness": 0.0,
            "scenario_variance": 0.0,
            "critical_factors": []
        }
        
        if not scenarios:
            return robustness
        
        # Extract outcomes across scenarios
        outcomes = []
        for scenario in scenarios:
            # Get final state value (simplified)
            final_state = scenario.get("counterfactual_state", {})
            outcome_value = len(final_state)  # Simplified metric
            outcomes.append(outcome_value)
        
        # Calculate variance
        if len(outcomes) > 1:
            mean_outcome = sum(outcomes) / len(outcomes)
            variance = sum((x - mean_outcome) ** 2 for x in outcomes) / len(outcomes)
            robustness["scenario_variance"] = variance
            
            # Lower variance = higher robustness
            robustness["overall_robustness"] = 1.0 / (1.0 + variance)
        
        # Identify critical factors
        intervention_nodes = set()
        for scenario in scenarios:
            intervention_nodes.update(scenario.get("intervention_nodes", []))
        
        robustness["critical_factors"] = list(intervention_nodes)
        
        return robustness
    
    # Performance monitoring
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            "cache_performance": self.cache.get_stats(),
            "state_metrics": self.state.performance_metrics,
            "meta_reasoning": {
                "strategy_effectiveness": dict(self.meta_reasoning.strategy_effectiveness),
                "current_performance": self.meta_reasoning.current_performance.__dict__ if self.meta_reasoning.current_performance else None
            },
            "memory_usage": self.performance_monitor.get_memory_usage(),
            "operation_timings": self.performance_monitor.get_operation_timings()
        }
    
    # Delegate missing methods to original core
    def __getattr__(self, name):
        """Delegate any missing methods to the original reasoning core"""
        return getattr(self.original_core, name)

# ========================================================================================
# PERFORMANCE MONITORING
# ========================================================================================

class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self):
        self.operation_timings: Dict[str, List[float]] = {}
        self.operation_starts: Dict[str, datetime] = {}
    
    def start_operation(self, operation_name: str):
        """Start timing an operation"""
        self.operation_starts[operation_name] = datetime.now()
    
    def end_operation(self, operation_name: str):
        """End timing an operation"""
        if operation_name in self.operation_starts:
            duration = (datetime.now() - self.operation_starts[operation_name]).total_seconds()
            
            if operation_name not in self.operation_timings:
                self.operation_timings[operation_name] = []
            
            self.operation_timings[operation_name].append(duration)
            
            # Keep only recent timings
            if len(self.operation_timings[operation_name]) > 100:
                self.operation_timings[operation_name] = self.operation_timings[operation_name][-100:]
            
            del self.operation_starts[operation_name]
    
    def get_operation_timings(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics for operations"""
        stats = {}
        
        for operation, timings in self.operation_timings.items():
            if timings:
                stats[operation] = {
                    "avg": sum(timings) / len(timings),
                    "min": min(timings),
                    "max": max(timings),
                    "count": len(timings)
                }
        
        return stats
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }

# ========================================================================================
# INTEGRATION FUNCTION
# ========================================================================================

def create_enhanced_reasoning_core(original_core):
    """Create enhanced reasoning core with all improvements"""
    return EnhancedContextAwareReasoningCore(original_core)
