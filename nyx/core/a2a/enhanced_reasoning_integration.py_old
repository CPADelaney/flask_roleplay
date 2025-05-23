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
from collections import OrderedDict, defaultdict, Counter
import psutil
import gc
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

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
    UncertaintyManager, ReasoningTemplateSystem,
    UncertaintyType, UncertaintyEstimate
)

logger = logging.getLogger(__name__)

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

class NodeVulnerability(Enum):
    """Types of node vulnerabilities"""
    SINGLE_POINT_FAILURE = "single_point_failure"
    HIGH_CENTRALITY = "high_centrality" 
    RESOURCE_CONSTRAINED = "resource_constrained"
    EXTERNAL_DEPENDENCY = "external_dependency"

from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class TarjanState:
    """State for Tarjan's algorithm"""
    discovery_time: Dict[str, int] = field(default_factory=dict)
    low: Dict[str, int] = field(default_factory=dict)
    parent: Dict[str, Optional[str]] = field(default_factory=dict)
    visited: Set[str] = field(default_factory=set)
    time: int = 0
    articulation_points: Set[str] = field(default_factory=set)
    bridges: Set[Tuple[str, str]] = field(default_factory=set)

class CutVertexDetector:
    """Complete implementation of cut vertex (articulation point) detection"""
    
    def __init__(self):
        self.state = None
        
    def is_cut_vertex(self, node: str, graph: Dict[str, Any]) -> bool:
        """
        Check if a node is a cut vertex (articulation point) using Tarjan's algorithm.
        
        A node is a cut vertex if its removal increases the number of connected components.
        
        Args:
            node: The node to check
            graph: Graph representation with 'adjacency' and 'reverse_adjacency'
            
        Returns:
            True if the node is a cut vertex, False otherwise
        """
        # Build undirected adjacency for articulation point detection
        undirected_adj = self._build_undirected_adjacency(graph)
        
        # Find all articulation points in the graph
        articulation_points = self.find_all_articulation_points(undirected_adj)
        
        return node in articulation_points
    
    def find_all_articulation_points(self, adjacency: Dict[str, Set[str]]) -> Set[str]:
        """
        Find all articulation points in the graph using Tarjan's algorithm.
        
        Time complexity: O(V + E) where V is vertices and E is edges
        """
        self.state = TarjanState()
        articulation_points = set()
        
        # Run DFS from each unvisited node (handles disconnected graphs)
        for node in adjacency:
            if node not in self.state.visited:
                self._tarjan_dfs(node, adjacency, articulation_points)
        
        return articulation_points
    
    def _tarjan_dfs(self, u: str, adjacency: Dict[str, Set[str]], 
                    articulation_points: Set[str]) -> None:
        """
        DFS for Tarjan's articulation point algorithm.
        
        Key insights:
        1. Root of DFS tree is articulation point if it has >1 children
        2. Non-root node u is articulation point if it has child v where
           no vertex in subtree rooted at v has back edge to ancestor of u
        """
        # Initialize discovery time and low value
        self.state.visited.add(u)
        self.state.discovery_time[u] = self.state.low[u] = self.state.time
        self.state.time += 1
        
        # Count children in DFS tree
        children = 0
        
        # Explore all adjacent vertices
        for v in adjacency.get(u, set()):
            if v not in self.state.visited:
                # v is not visited, so it's a child in DFS tree
                children += 1
                self.state.parent[v] = u
                
                # Recurse for child
                self._tarjan_dfs(v, adjacency, articulation_points)
                
                # Update low value of u
                self.state.low[u] = min(self.state.low[u], self.state.low[v])
                
                # Check articulation point conditions
                # Case 1: u is root and has more than one child
                if self.state.parent.get(u) is None and children > 1:
                    articulation_points.add(u)
                
                # Case 2: u is not root and low value of child is greater than or equal
                # to discovery value of u (no back edge from subtree rooted at v to ancestors of u)
                if self.state.parent.get(u) is not None and self.state.low[v] >= self.state.discovery_time[u]:
                    articulation_points.add(u)
                    
            elif v != self.state.parent.get(u):
                # Update low value of u for back edge
                self.state.low[u] = min(self.state.low[u], self.state.discovery_time[v])
    
    def _build_undirected_adjacency(self, graph: Dict[str, Any]) -> Dict[str, Set[str]]:
        """Convert directed graph to undirected for articulation point detection"""
        undirected = defaultdict(set)
        
        # Add edges from adjacency list
        for source, targets in graph.get('adjacency', {}).items():
            for target in targets:
                undirected[source].add(target)
                undirected[target].add(source)
        
        # Ensure all nodes are included
        for node in graph.get('node_properties', {}):
            if node not in undirected:
                undirected[node] = set()
                
        return dict(undirected)
    
    def find_bridges(self, adjacency: Dict[str, Set[str]]) -> Set[Tuple[str, str]]:
        """
        Find all bridges (cut edges) in the graph.
        
        A bridge is an edge whose removal increases the number of connected components.
        """
        self.state = TarjanState()
        
        for node in adjacency:
            if node not in self.state.visited:
                self._find_bridges_dfs(node, adjacency)
        
        return self.state.bridges
    
    def _find_bridges_dfs(self, u: str, adjacency: Dict[str, Set[str]]) -> None:
        """DFS for finding bridges"""
        self.state.visited.add(u)
        self.state.discovery_time[u] = self.state.low[u] = self.state.time
        self.state.time += 1
        
        for v in adjacency.get(u, set()):
            if v not in self.state.visited:
                self.state.parent[v] = u
                self._find_bridges_dfs(v, adjacency)
                
                self.state.low[u] = min(self.state.low[u], self.state.low[v])
                
                # If low value of v is greater than discovery value of u,
                # then u-v is a bridge
                if self.state.low[v] > self.state.discovery_time[u]:
                    self.state.bridges.add((u, v))
                    
            elif v != self.state.parent.get(u):
                self.state.low[u] = min(self.state.low[u], self.state.discovery_time[v])
    
    def get_biconnected_components(self, adjacency: Dict[str, Set[str]]) -> List[Set[str]]:
        """
        Find all biconnected components in the graph.
        
        A biconnected component is a maximal subgraph with no cut vertices.
        """
        # Find all bridges first
        bridges = self.find_bridges(adjacency)
        
        # Remove bridges to get biconnected components
        modified_adj = defaultdict(set)
        for node, neighbors in adjacency.items():
            for neighbor in neighbors:
                if (node, neighbor) not in bridges and (neighbor, node) not in bridges:
                    modified_adj[node].add(neighbor)
        
        # Find connected components in the modified graph
        visited = set()
        components = []
        
        for node in adjacency:
            if node not in visited:
                component = set()
                self._dfs_component(node, modified_adj, visited, component)
                if component:
                    components.append(component)
        
        return components
    
    def _dfs_component(self, node: str, adjacency: Dict[str, Set[str]], 
                      visited: Set[str], component: Set[str]) -> None:
        """DFS to find connected components"""
        visited.add(node)
        component.add(node)
        
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                self._dfs_component(neighbor, adjacency, visited, component)

    # Enhanced version integrated with the path robustness calculation
    
    def analyze_graph_vulnerabilities(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive analysis of graph vulnerabilities including cut vertices and bridges.
        
        Returns detailed information about critical points in the graph structure.
        """
        detector = CutVertexDetector()
        
        # Build undirected adjacency
        undirected_adj = detector._build_undirected_adjacency(graph)
        
        # Find all articulation points
        articulation_points = detector.find_all_articulation_points(undirected_adj)
        
        # Find all bridges
        bridges = detector.find_bridges(undirected_adj)
        
        # Find biconnected components
        biconnected_components = detector.get_biconnected_components(undirected_adj)
        
        # Analyze impact of each articulation point
        ap_impact = {}
        for ap in articulation_points:
            impact = self._analyze_articulation_point_impact(ap, graph, undirected_adj)
            ap_impact[ap] = impact
        
        # Analyze bridge criticality
        bridge_criticality = {}
        for bridge in bridges:
            criticality = self._analyze_bridge_criticality(bridge, graph)
            bridge_criticality[bridge] = criticality
        
        # Calculate overall graph robustness metrics
        total_nodes = len(graph.get('node_properties', {}))
        total_edges = sum(len(targets) for targets in graph.get('adjacency', {}).values())
        
        vulnerability_metrics = {
            'articulation_point_ratio': len(articulation_points) / max(total_nodes, 1),
            'bridge_ratio': len(bridges) / max(total_edges, 1),
            'largest_biconnected_component': max(len(c) for c in biconnected_components) if biconnected_components else 0,
            'component_count': len(biconnected_components),
            'is_biconnected': len(articulation_points) == 0,
            'is_bridge_connected': len(bridges) == 0
        }
        
        # Identify most critical vulnerabilities
        critical_nodes = [
            node for node, impact in ap_impact.items()
            if impact['severity'] == 'critical'
        ]
        
        critical_edges = [
            edge for edge, criticality in bridge_criticality.items()
            if criticality['severity'] == 'critical'
        ]
        
        return {
            'articulation_points': list(articulation_points),
            'articulation_point_impacts': ap_impact,
            'bridges': [list(b) for b in bridges],
            'bridge_criticality': {f"{b[0]}->{b[1]}": crit for b, crit in bridge_criticality.items()},
            'biconnected_components': [list(c) for c in biconnected_components],
            'vulnerability_metrics': vulnerability_metrics,
            'critical_vulnerabilities': {
                'nodes': critical_nodes,
                'edges': critical_edges
            },
            'recommendations': self._generate_vulnerability_recommendations(
                articulation_points, bridges, vulnerability_metrics
            )
        }
    
    def _analyze_articulation_point_impact(self, node: str, graph: Dict[str, Any], 
                                         undirected_adj: Dict[str, Set[str]]) -> Dict[str, Any]:
        """Analyze the impact of removing an articulation point"""
        # Count components before removal
        components_before = self._count_connected_components(undirected_adj)
        
        # Simulate removal
        modified_adj = {k: v.copy() for k, v in undirected_adj.items()}
        if node in modified_adj:
            # Remove node and all its edges
            neighbors = modified_adj[node].copy()
            del modified_adj[node]
            for neighbor in neighbors:
                if neighbor in modified_adj:
                    modified_adj[neighbor].discard(node)
        
        # Count components after removal
        components_after = self._count_connected_components(modified_adj)
        
        # Calculate impact metrics
        components_created = components_after - components_before
        nodes_isolated = self._count_isolated_nodes(modified_adj)
        
        # Determine severity
        if components_created >= 3:
            severity = 'critical'
        elif components_created == 2:
            severity = 'high'
        else:
            severity = 'medium'
        
        # Check if critical paths go through this node
        node_props = graph.get('node_properties', {}).get(node, {})
        if node_props.get('criticality', 0) > 0.8:
            severity = 'critical'
        
        return {
            'components_created': components_created,
            'nodes_isolated': nodes_isolated,
            'neighbor_count': len(undirected_adj.get(node, set())),
            'severity': severity,
            'node_criticality': node_props.get('criticality', 0.5),
            'current_load': node_props.get('current_load', 0),
            'capacity': node_props.get('capacity', 1.0)
        }
    
    def _analyze_bridge_criticality(self, bridge: Tuple[str, str], 
                                   graph: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the criticality of a bridge (cut edge)"""
        u, v = bridge
        
        # Get edge properties
        edge_strength = graph.get('edge_strengths', {}).get((u, v), 0.5)
        
        # Check if this is the only path between important nodes
        u_props = graph.get('node_properties', {}).get(u, {})
        v_props = graph.get('node_properties', {}).get(v, {})
        
        # Calculate criticality based on multiple factors
        criticality_score = 0.0
        
        # Factor 1: Node importance
        u_importance = u_props.get('criticality', 0.5)
        v_importance = v_props.get('criticality', 0.5)
        criticality_score += (u_importance + v_importance) / 2 * 0.3
        
        # Factor 2: Traffic/load through edge
        if 'edge_loads' in graph:
            edge_load = graph['edge_loads'].get((u, v), 0)
            max_load = max(graph['edge_loads'].values()) if graph['edge_loads'] else 1
            criticality_score += (edge_load / max_load) * 0.3
        
        # Factor 3: Edge strength (weak edges are more critical as bridges)
        criticality_score += (1.0 - edge_strength) * 0.2
        
        # Factor 4: Component sizes that would be separated
        # (This is approximate - full calculation would be expensive)
        u_degree = len(graph.get('adjacency', {}).get(u, []))
        v_degree = len(graph.get('adjacency', {}).get(v, []))
        degree_factor = min(u_degree, v_degree) / max(u_degree, v_degree)
        criticality_score += (1.0 - degree_factor) * 0.2
        
        # Determine severity
        if criticality_score > 0.7:
            severity = 'critical'
        elif criticality_score > 0.5:
            severity = 'high'
        elif criticality_score > 0.3:
            severity = 'medium'
        else:
            severity = 'low'
        
        return {
            'criticality_score': criticality_score,
            'severity': severity,
            'edge_strength': edge_strength,
            'connects_critical_nodes': u_importance > 0.7 or v_importance > 0.7,
            'is_weak_link': edge_strength < 0.3
        }
    
    def _count_connected_components(self, adjacency: Dict[str, Set[str]]) -> int:
        """Count the number of connected components in the graph"""
        visited = set()
        component_count = 0
        
        for node in adjacency:
            if node not in visited:
                # Start DFS from this node
                stack = [node]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        stack.extend(adjacency.get(current, set()) - visited)
                component_count += 1
        
        return component_count
    
    def _count_isolated_nodes(self, adjacency: Dict[str, Set[str]]) -> int:
        """Count nodes with no connections"""
        return sum(1 for node, neighbors in adjacency.items() if not neighbors)
    
    def _generate_vulnerability_recommendations(self, articulation_points: Set[str], 
                                              bridges: Set[Tuple[str, str]], 
                                              metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations based on vulnerability analysis"""
        recommendations = []
        
        # High ratio of articulation points
        if metrics['articulation_point_ratio'] > 0.2:
            recommendations.append({
                'type': 'add_redundancy',
                'priority': 'high',
                'description': f"{len(articulation_points)} nodes are single points of failure. Add redundant connections to create alternate paths.",
                'specific_nodes': list(articulation_points)[:5]  # Top 5
            })
        
        # Many bridges
        if metrics['bridge_ratio'] > 0.3:
            recommendations.append({
                'type': 'add_parallel_edges',
                'priority': 'high',
                'description': f"{len(bridges)} edges are bridges. Add parallel connections for critical edges.",
                'specific_edges': [list(b) for b in list(bridges)[:5]]
            })
        
        # Not biconnected
        if not metrics['is_biconnected']:
            recommendations.append({
                'type': 'improve_connectivity',
                'priority': 'medium',
                'description': "Graph is not biconnected. Consider adding edges to eliminate articulation points.",
                'target_state': 'biconnected'
            })
        
        # Many small components
        if metrics['component_count'] > 5:
            recommendations.append({
                'type': 'merge_components',
                'priority': 'medium',
                'description': f"Graph has {metrics['component_count']} biconnected components. Consider connecting smaller components.",
                'benefit': 'improved_fault_tolerance'
            })
        
        return recommendations

@dataclass
class RobustnessFactors:
    """Factors contributing to path robustness"""
    edge_reliability: float
    path_redundancy: float
    node_reliability: float
    bottleneck_penalty: float
    length_penalty: float
    feedback_bonus: float
    
    @property
    def overall_robustness(self) -> float:
        """Calculate overall robustness from all factors"""
        return (self.edge_reliability * 
                self.path_redundancy * 
                self.node_reliability * 
                self.bottleneck_penalty * 
                self.length_penalty * 
                self.feedback_bonus)

    def _calculate_path_robustness(self, path: List[str], model) -> float:
        """
        Calculate how robust a path is to perturbations.
        
        This comprehensive implementation considers:
        1. Edge reliability along the path
        2. Alternative paths between nodes
        3. Node vulnerabilities
        4. Critical bottlenecks
        5. Path length effects
        6. Feedback loops and cycles
        7. Cascade failure risks
        """
        if len(path) < 2:
            return 1.0
        
        # Build graph representation for analysis
        graph = self._build_graph_from_model(model)
        
        # Calculate various robustness factors
        factors = RobustnessFactors(
            edge_reliability=self._calculate_edge_reliability(path, model),
            path_redundancy=self._calculate_path_redundancy(path, graph),
            node_reliability=self._calculate_node_reliability(path, model, graph),
            bottleneck_penalty=self._calculate_bottleneck_penalty(path, graph),
            length_penalty=self._calculate_length_penalty(path),
            feedback_bonus=self._calculate_feedback_bonus(path, graph)
        )
        
        # Consider cascade failure risk
        cascade_risk = self._assess_cascade_failure_risk(path, model, graph)
        
        # Calculate final robustness
        base_robustness = factors.overall_robustness
        cascade_adjusted = base_robustness * (1.0 - cascade_risk)
        
        # Ensure robustness is in [0, 1] range
        return max(0.0, min(1.0, cascade_adjusted))
    
    def _build_graph_from_model(self, model) -> Dict[str, Any]:
        """Build graph representation from model for analysis"""
        graph = {
            'adjacency': defaultdict(list),
            'reverse_adjacency': defaultdict(list),
            'edge_strengths': {},
            'node_properties': {}
        }
        
        # Build adjacency lists
        for relation in model.relations:
            graph['adjacency'][relation.source].append(relation.target)
            graph['reverse_adjacency'][relation.target].append(relation.source)
            graph['edge_strengths'][(relation.source, relation.target)] = relation.strength
        
        # Store node properties
        for node_id, node in model.nodes.items():
            graph['node_properties'][node_id] = {
                'type': getattr(node, 'node_type', 'standard'),
                'criticality': getattr(node, 'criticality', 0.5),
                'reliability': getattr(node, 'reliability', 0.9),
                'capacity': getattr(node, 'capacity', 1.0),
                'current_load': getattr(node, 'current_load', 0.5)
            }
        
        return graph
    
    def _calculate_edge_reliability(self, path: List[str], model) -> float:
        """Calculate reliability based on edge strengths along path"""
        if len(path) < 2:
            return 1.0
        
        edge_strengths = []
        
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            # Find edge strength
            edge_strength = 0.0
            for relation in model.relations:
                if relation.source == source and relation.target == target:
                    edge_strength = relation.strength
                    
                    # Adjust for edge properties
                    if hasattr(relation, 'reliability'):
                        edge_strength *= relation.reliability
                    if hasattr(relation, 'variability'):
                        # High variability reduces effective strength
                        edge_strength *= (1.0 - relation.variability * 0.5)
                    break
            
            edge_strengths.append(edge_strength)
        
        # Use geometric mean for overall edge reliability
        # This penalizes weak links more than arithmetic mean
        if edge_strengths:
            return np.power(np.prod(edge_strengths), 1.0 / len(edge_strengths))
        return 0.0
    
    def _calculate_path_redundancy(self, path: List[str], graph: Dict[str, Any]) -> float:
        """Calculate redundancy based on alternative paths"""
        if len(path) < 2:
            return 1.0
        
        redundancy_scores = []
        
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            # Find all alternative paths between consecutive nodes
            alternatives = self._find_alternative_paths(
                source, target, graph, 
                exclude_edge=(source, target),
                max_length=5  # Don't consider very long alternatives
            )
            
            # Calculate redundancy score for this segment
            if alternatives:
                # Consider both quantity and quality of alternatives
                alt_count = len(alternatives)
                alt_quality = np.mean([self._evaluate_path_quality(alt, graph) 
                                      for alt in alternatives])
                
                # Redundancy increases with more high-quality alternatives
                # Using log to prevent excessive bonus from many weak alternatives
                segment_redundancy = 1.0 + np.log1p(alt_count) * alt_quality * 0.3
            else:
                # No alternatives - this segment is critical
                segment_redundancy = 0.5
            
            redundancy_scores.append(segment_redundancy)
        
        # Overall redundancy is limited by weakest segment
        return np.mean(redundancy_scores) * np.min(redundancy_scores) / np.max(redundancy_scores)
    
    def _find_alternative_paths(self, source: str, target: str, graph: Dict[str, Any],
                               exclude_edge: Optional[Tuple[str, str]] = None,
                               max_length: int = 5) -> List[List[str]]:
        """Find alternative paths between source and target using BFS"""
        alternatives = []
        
        # BFS to find paths
        queue = deque([(source, [source])])
        visited_paths = set()
        
        while queue and len(alternatives) < 10:  # Limit to 10 alternatives
            current, path = queue.popleft()
            
            # Skip if path is too long
            if len(path) > max_length:
                continue
            
            # Check if we reached target
            if current == target and len(path) > 1:
                path_tuple = tuple(path)
                if path_tuple not in visited_paths:
                    visited_paths.add(path_tuple)
                    alternatives.append(path)
                continue
            
            # Explore neighbors
            for neighbor in graph['adjacency'].get(current, []):
                # Skip excluded edge
                if exclude_edge and (current, neighbor) == exclude_edge:
                    continue
                
                # Avoid cycles
                if neighbor not in path:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
        
        return alternatives
    
    def _evaluate_path_quality(self, path: List[str], graph: Dict[str, Any]) -> float:
        """Evaluate quality of an alternative path"""
        if len(path) < 2:
            return 0.0
        
        quality = 1.0
        
        # Factor 1: Path strength (product of edge strengths)
        for i in range(len(path) - 1):
            edge_key = (path[i], path[i + 1])
            edge_strength = graph['edge_strengths'].get(edge_key, 0.5)
            quality *= edge_strength
        
        # Factor 2: Length penalty (shorter is better)
        length_penalty = 1.0 / (1.0 + 0.2 * (len(path) - 2))
        quality *= length_penalty
        
        # Factor 3: Node reliability along path
        node_reliability = 1.0
        for node in path[1:-1]:  # Exclude source and target
            node_props = graph['node_properties'].get(node, {})
            node_reliability *= node_props.get('reliability', 0.9)
        quality *= node_reliability
        
        return quality
    
    def _calculate_node_reliability(self, path: List[str], model, graph: Dict[str, Any]) -> float:
        """Calculate reliability based on node properties and vulnerabilities"""
        if not path:
            return 1.0
        
        node_scores = []
        
        for node in path:
            node_data = graph['node_properties'].get(node, {})
            base_reliability = node_data.get('reliability', 0.9)
            
            # Check for vulnerabilities
            vulnerabilities = self._identify_node_vulnerabilities(node, model, graph)
            
            # Adjust reliability based on vulnerabilities
            vulnerability_penalty = 1.0
            for vuln in vulnerabilities:
                if vuln == NodeVulnerability.SINGLE_POINT_FAILURE:
                    vulnerability_penalty *= 0.6
                elif vuln == NodeVulnerability.HIGH_CENTRALITY:
                    vulnerability_penalty *= 0.8
                elif vuln == NodeVulnerability.RESOURCE_CONSTRAINED:
                    # Check load vs capacity
                    load_ratio = node_data.get('current_load', 0.5) / node_data.get('capacity', 1.0)
                    if load_ratio > 0.8:
                        vulnerability_penalty *= 0.7
                elif vuln == NodeVulnerability.EXTERNAL_DEPENDENCY:
                    vulnerability_penalty *= 0.85
            
            node_reliability = base_reliability * vulnerability_penalty
            node_scores.append(node_reliability)
        
        # Use harmonic mean to emphasize weak nodes
        if node_scores:
            return len(node_scores) / sum(1.0 / score for score in node_scores)
        return 0.0
    
    def _identify_node_vulnerabilities(self, node: str, model, graph: Dict[str, Any]) -> List[NodeVulnerability]:
        """Identify vulnerabilities of a specific node"""
        vulnerabilities = []
        
        # Check if it's a single point of failure
        in_degree = len(graph['reverse_adjacency'].get(node, []))
        out_degree = len(graph['adjacency'].get(node, []))
        
        # High centrality - many connections
        total_degree = in_degree + out_degree
        if total_degree > len(graph['node_properties']) * 0.3:  # Connected to >30% of nodes
            vulnerabilities.append(NodeVulnerability.HIGH_CENTRALITY)
        
        # Single point of failure - uses the integrated CutVertexDetector
        if self._is_cut_vertex(node, graph):
            vulnerabilities.append(NodeVulnerability.SINGLE_POINT_FAILURE)
        
        # Resource constrained
        node_data = graph['node_properties'].get(node, {})
        if node_data.get('current_load', 0) / node_data.get('capacity', 1.0) > 0.8:
            vulnerabilities.append(NodeVulnerability.RESOURCE_CONSTRAINED)
        
        # External dependency (simplified check)
        if node_data.get('type') == 'external' or 'external' in node.lower():
            vulnerabilities.append(NodeVulnerability.EXTERNAL_DEPENDENCY)
        
        return vulnerabilities
    
    def _path_exists_excluding(self, source: str, target: str, exclude: str, 
                              graph: Dict[str, Any]) -> bool:
        """Check if path exists between source and target excluding a node"""
        visited = set([exclude])  # Exclude the node
        queue = deque([source])
        
        while queue:
            current = queue.popleft()
            if current == target:
                return True
            
            if current in visited:
                continue
                
            visited.add(current)
            
            for neighbor in graph['adjacency'].get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        
        return False
    
    def _calculate_bottleneck_penalty(self, path: List[str], graph: Dict[str, Any]) -> float:
        """Calculate penalty for bottlenecks in the path"""
        if len(path) < 2:
            return 1.0
        
        penalties = []
        
        for i, node in enumerate(path):
            # Skip source and target
            if i == 0 or i == len(path) - 1:
                continue
            
            # Check if this node is a bottleneck
            bottleneck_score = self._calculate_bottleneck_score(node, graph)
            
            # Convert to penalty (1.0 = no penalty, 0.0 = complete bottleneck)
            penalty = 1.0 - bottleneck_score
            penalties.append(penalty)
        
        # If no intermediate nodes, no bottleneck penalty
        if not penalties:
            return 1.0
        
        # Overall penalty is product (all bottlenecks matter)
        return np.prod(penalties)
    
    def _calculate_bottleneck_score(self, node: str, graph: Dict[str, Any]) -> float:
        """Calculate how much of a bottleneck a node is (0 = not bottleneck, 1 = severe)"""
        # Factor 1: Betweenness centrality approximation
        in_degree = len(graph['reverse_adjacency'].get(node, []))
        out_degree = len(graph['adjacency'].get(node, []))
        total_nodes = len(graph['node_properties'])
        
        # High in/out degree relative to graph size indicates bottleneck
        centrality_score = (in_degree + out_degree) / (2 * total_nodes)
        
        # Factor 2: Load vs capacity
        node_data = graph['node_properties'].get(node, {})
        load_ratio = node_data.get('current_load', 0.5) / node_data.get('capacity', 1.0)
        
        # Factor 3: Number of paths going through this node (simplified)
        # In a full implementation, we'd count actual paths
        path_concentration = min(1.0, centrality_score * 2)
        
        # Combine factors
        bottleneck_score = (centrality_score * 0.4 + 
                           load_ratio * 0.4 + 
                           path_concentration * 0.2)
        
        return min(1.0, bottleneck_score)
    
    def _calculate_length_penalty(self, path: List[str]) -> float:
        """Calculate penalty based on path length"""
        if not path:
            return 1.0
        
        length = len(path)
        
        # Short paths are more robust (fewer failure points)
        # Using exponential decay
        optimal_length = 3  # Paths of length 3 are considered optimal
        
        if length <= optimal_length:
            return 1.0
        else:
            # Decay factor of 0.9 per additional hop
            excess_length = length - optimal_length
            return 0.9 ** excess_length
    
    def _calculate_feedback_bonus(self, path: List[str], graph: Dict[str, Any]) -> float:
        """Calculate bonus for positive feedback loops that strengthen the path"""
        if len(path) < 3:
            return 1.0
        
        feedback_strength = 0.0
        
        # Check for feedback loops involving path nodes
        for i, node in enumerate(path):
            # Look for cycles that include this node and other path nodes
            for j, other_node in enumerate(path):
                if i != j and abs(i - j) > 1:  # Not adjacent in path
                    # Check if there's a connection from other_node back to node
                    if node in graph['adjacency'].get(other_node, []):
                        # Found a feedback loop
                        edge_strength = graph['edge_strengths'].get((other_node, node), 0.5)
                        
                        # Positive feedback if it reinforces the path direction
                        if j > i:  # Forward feedback
                            feedback_strength += edge_strength * 0.1
        
        # Convert to bonus factor (capped at 1.2 for 20% maximum bonus)
        return min(1.2, 1.0 + feedback_strength)
    
    def _assess_cascade_failure_risk(self, path: List[str], model, graph: Dict[str, Any]) -> float:
        """Assess risk of cascade failures along the path"""
        if len(path) < 2:
            return 0.0
        
        cascade_risks = []
        
        for i, node in enumerate(path):
            # Skip endpoints
            if i == 0 or i == len(path) - 1:
                continue
            
            # Calculate cascade risk for this node
            node_risk = 0.0
            
            # Factor 1: High connectivity increases cascade risk
            total_connections = (len(graph['adjacency'].get(node, [])) + 
                               len(graph['reverse_adjacency'].get(node, [])))
            connectivity_risk = min(1.0, total_connections / (len(graph['node_properties']) * 0.5))
            node_risk += connectivity_risk * 0.3
            
            # Factor 2: Load near capacity increases risk
            node_data = graph['node_properties'].get(node, {})
            load_ratio = node_data.get('current_load', 0.5) / node_data.get('capacity', 1.0)
            if load_ratio > 0.7:
                capacity_risk = (load_ratio - 0.7) / 0.3  # Linear increase from 0.7 to 1.0
                node_risk += capacity_risk * 0.4
            
            # Factor 3: Dependency chains
            downstream_nodes = self._count_downstream_dependencies(node, graph, max_depth=3)
            dependency_risk = min(1.0, downstream_nodes / len(graph['node_properties']))
            node_risk += dependency_risk * 0.3
            
            cascade_risks.append(node_risk)
        
        # Overall cascade risk is the maximum risk among nodes
        # (cascade can start from the weakest point)
        return max(cascade_risks) if cascade_risks else 0.0
    
    def _count_downstream_dependencies(self, start_node: str, graph: Dict[str, Any], 
                                      max_depth: int = 3) -> int:
        """Count nodes that depend on the given node (downstream)"""
        visited = set()
        queue = deque([(start_node, 0)])
        count = 0
        
        while queue:
            node, depth = queue.popleft()
            
            if depth > max_depth or node in visited:
                continue
                
            visited.add(node)
            if node != start_node:
                count += 1
            
            # Add downstream nodes
            for neighbor in graph['adjacency'].get(node, []):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        
        return count
    
    # Additional analysis methods for comprehensive robustness
    
    def analyze_path_robustness_detailed(self, path: List[str], model) -> Dict[str, Any]:
        """Provide detailed robustness analysis with actionable insights"""
        robustness = self._calculate_path_robustness(path, model)
        graph = self._build_graph_from_model(model)
        
        # Detailed factor analysis
        factors = {
            'overall_robustness': robustness,
            'edge_reliability': self._calculate_edge_reliability(path, model),
            'path_redundancy': self._calculate_path_redundancy(path, graph),
            'node_reliability': self._calculate_node_reliability(path, model, graph),
            'bottleneck_penalty': self._calculate_bottleneck_penalty(path, graph),
            'length_penalty': self._calculate_length_penalty(path),
            'feedback_bonus': self._calculate_feedback_bonus(path, graph),
            'cascade_risk': self._assess_cascade_failure_risk(path, model, graph)
        }
        
        # Identify weak points
        weak_points = []
        for i in range(len(path) - 1):
            segment_robustness = self._calculate_segment_robustness(
                path[i], path[i + 1], model, graph
            )
            if segment_robustness < 0.5:
                weak_points.append({
                    'from': path[i],
                    'to': path[i + 1],
                    'robustness': segment_robustness,
                    'issues': self._identify_segment_issues(path[i], path[i + 1], model, graph)
                })
        
        # Improvement recommendations
        recommendations = self._generate_robustness_recommendations(
            path, factors, weak_points, model, graph
        )
        
        return {
            'robustness_score': robustness,
            'factors': factors,
            'weak_points': weak_points,
            'recommendations': recommendations,
            'risk_assessment': self._assess_overall_risk(factors)
        }
    
    def _calculate_segment_robustness(self, source: str, target: str, 
                                    model, graph: Dict[str, Any]) -> float:
        """Calculate robustness of a single path segment"""
        # Edge strength
        edge_strength = graph['edge_strengths'].get((source, target), 0.5)
        
        # Alternative paths
        alternatives = self._find_alternative_paths(source, target, graph, 
                                                  exclude_edge=(source, target))
        redundancy = 1.0 if alternatives else 0.5
        
        # Node reliabilities
        source_reliability = graph['node_properties'].get(source, {}).get('reliability', 0.9)
        target_reliability = graph['node_properties'].get(target, {}).get('reliability', 0.9)
        
        return edge_strength * redundancy * source_reliability * target_reliability
    
    def _identify_segment_issues(self, source: str, target: str, 
                               model, graph: Dict[str, Any]) -> List[str]:
        """Identify specific issues with a path segment"""
        issues = []
        
        edge_strength = graph['edge_strengths'].get((source, target), 0.5)
        if edge_strength < 0.3:
            issues.append("weak_connection")
        
        alternatives = self._find_alternative_paths(source, target, graph, 
                                                  exclude_edge=(source, target))
        if not alternatives:
            issues.append("no_redundancy")
        
        source_load = graph['node_properties'].get(source, {}).get('current_load', 0.5)
        source_capacity = graph['node_properties'].get(source, {}).get('capacity', 1.0)
        if source_load / source_capacity > 0.8:
            issues.append("source_overloaded")
        
        if self._is_cut_vertex(source, graph):
            issues.append("source_is_critical_point")
        
        return issues
    
    def _generate_robustness_recommendations(self, path: List[str], factors: Dict[str, float],
                                           weak_points: List[Dict], model, 
                                           graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations to improve path robustness"""
        recommendations = []
        
        # Edge reliability improvements
        if factors['edge_reliability'] < 0.6:
            recommendations.append({
                'type': 'strengthen_edges',
                'priority': 'high',
                'description': 'Strengthen weak connections along the path',
                'specific_edges': [(wp['from'], wp['to']) for wp in weak_points 
                                 if 'weak_connection' in wp.get('issues', [])]
            })
        
        # Redundancy improvements
        if factors['path_redundancy'] < 0.7:
            recommendations.append({
                'type': 'add_redundancy',
                'priority': 'high',
                'description': 'Add alternative paths between critical segments',
                'specific_segments': [(wp['from'], wp['to']) for wp in weak_points 
                                    if 'no_redundancy' in wp.get('issues', [])]
            })
        
        # Node reliability improvements
        if factors['node_reliability'] < 0.7:
            critical_nodes = [node for node in path if 
                             self._is_cut_vertex(node, graph) or
                             graph['node_properties'].get(node, {}).get('reliability', 1.0) < 0.7]
            recommendations.append({
                'type': 'improve_node_reliability',
                'priority': 'medium',
                'description': 'Improve reliability of critical nodes',
                'specific_nodes': critical_nodes
            })
        
        # Capacity improvements
        overloaded_nodes = [node for node in path if 
                           graph['node_properties'].get(node, {}).get('current_load', 0) /
                           graph['node_properties'].get(node, {}).get('capacity', 1) > 0.8]
        if overloaded_nodes:
            recommendations.append({
                'type': 'increase_capacity',
                'priority': 'high',
                'description': 'Increase capacity of overloaded nodes',
                'specific_nodes': overloaded_nodes
            })
        
        # Path length optimization
        if factors['length_penalty'] < 0.7 and len(path) > 5:
            recommendations.append({
                'type': 'shorten_path',
                'priority': 'low',
                'description': 'Consider shorter alternative paths to reduce failure points',
                'current_length': len(path),
                'recommended_max': 5
            })
        
        return recommendations
    
    def _assess_overall_risk(self, factors: Dict[str, float]) -> Dict[str, Any]:
        """Assess overall risk level based on robustness factors"""
        robustness = factors['overall_robustness']
        
        if robustness > 0.8:
            risk_level = 'low'
            description = 'Path is highly robust with good redundancy and reliability'
        elif robustness > 0.6:
            risk_level = 'medium'
            description = 'Path has moderate robustness but some vulnerabilities exist'
        elif robustness > 0.4:
            risk_level = 'high'
            description = 'Path has significant vulnerabilities and limited redundancy'
        else:
            risk_level = 'critical'
            description = 'Path is highly vulnerable with multiple critical failure points'
        
        # Identify primary risk factors
        risk_factors = []
        if factors['edge_reliability'] < 0.6:
            risk_factors.append('weak_connections')
        if factors['path_redundancy'] < 0.6:
            risk_factors.append('lack_of_alternatives')
        if factors['node_reliability'] < 0.6:
            risk_factors.append('unreliable_nodes')
        if factors['cascade_risk'] > 0.7:
            risk_factors.append('cascade_failure_risk')
        
        return {
            'risk_level': risk_level,
            'description': description,
            'primary_risks': risk_factors,
            'mitigation_priority': 'immediate' if risk_level == 'critical' else 
                                 'high' if risk_level == 'high' else 
                                 'medium' if risk_level == 'medium' else 'low'
        }

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

        self.cut_vertex_detector = CutVertexDetector()
        
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

    def _is_cut_vertex(self, node: str, graph: Dict[str, Any]) -> bool:
        """
        Check if a node is a cut vertex (articulation point) using the CutVertexDetector.
        
        This method properly integrates the CutVertexDetector with the reasoning core.
        
        Args:
            node: The node to check
            graph: Graph representation with 'adjacency' and 'reverse_adjacency'
            
        Returns:
            True if the node is a cut vertex, False otherwise
        """
        return self.cut_vertex_detector.is_cut_vertex(node, graph)
    
    def analyze_graph_vulnerabilities(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive analysis of graph vulnerabilities including cut vertices and bridges.
        
        This method uses the CutVertexDetector to provide detailed vulnerability analysis.
        
        Returns detailed information about critical points in the graph structure.
        """
        # Build undirected adjacency
        undirected_adj = self.cut_vertex_detector._build_undirected_adjacency(graph)
        
        # Find all articulation points
        articulation_points = self.cut_vertex_detector.find_all_articulation_points(undirected_adj)
        
        # Find all bridges
        bridges = self.cut_vertex_detector.find_bridges(undirected_adj)
        
        # Find biconnected components
        biconnected_components = self.cut_vertex_detector.get_biconnected_components(undirected_adj)
        
        # Analyze impact of each articulation point
        ap_impact = {}
        for ap in articulation_points:
            impact = self._analyze_articulation_point_impact(ap, graph, undirected_adj)
            ap_impact[ap] = impact
        
        # Analyze bridge criticality
        bridge_criticality = {}
        for bridge in bridges:
            criticality = self._analyze_bridge_criticality(bridge, graph)
            bridge_criticality[bridge] = criticality
        
        # Calculate overall graph robustness metrics
        total_nodes = len(graph.get('node_properties', {}))
        total_edges = sum(len(targets) for targets in graph.get('adjacency', {}).values())
        
        vulnerability_metrics = {
            'articulation_point_ratio': len(articulation_points) / max(total_nodes, 1),
            'bridge_ratio': len(bridges) / max(total_edges, 1),
            'largest_biconnected_component': max(len(c) for c in biconnected_components) if biconnected_components else 0,
            'component_count': len(biconnected_components),
            'is_biconnected': len(articulation_points) == 0,
            'is_bridge_connected': len(bridges) == 0
        }
        
        # Identify most critical vulnerabilities
        critical_nodes = [
            node for node, impact in ap_impact.items()
            if impact['severity'] == 'critical'
        ]
        
        critical_edges = [
            edge for edge, criticality in bridge_criticality.items()
            if criticality['severity'] == 'critical'
        ]
        
        return {
            'articulation_points': list(articulation_points),
            'articulation_point_impacts': ap_impact,
            'bridges': [list(b) for b in bridges],
            'bridge_criticality': {f"{b[0]}->{b[1]}": crit for b, crit in bridge_criticality.items()},
            'biconnected_components': [list(c) for c in biconnected_components],
            'vulnerability_metrics': vulnerability_metrics,
            'critical_vulnerabilities': {
                'nodes': critical_nodes,
                'edges': critical_edges
            },
            'recommendations': self._generate_vulnerability_recommendations(
                articulation_points, bridges, vulnerability_metrics
            )
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

    async def _find_hierarchical_patterns(self, space) -> Dict[str, Any]:
        """Find hierarchical patterns in concept space"""
        hierarchy = {
            "levels": defaultdict(list),
            "parent_child_relations": [],
            "root_concepts": [],
            "leaf_concepts": [],
            "max_depth": 0
        }
        
        # Build parent-child relationships
        parent_map = defaultdict(list)
        child_map = defaultdict(list)
        
        for concept_id, concept in space.concepts.items():
            # Check for hierarchical relations
            relations = concept.get("relations", {})
            
            for rel_type, targets in relations.items():
                if rel_type in ["is_a", "subclass_of", "part_of", "instance_of"]:
                    for target in targets:
                        parent_map[concept_id].append(target)
                        child_map[target].append(concept_id)
                        hierarchy["parent_child_relations"].append({
                            "parent": target,
                            "child": concept_id,
                            "relation": rel_type
                        })
        
        # Find roots (concepts with no parents)
        for concept_id in space.concepts:
            if concept_id not in parent_map:
                hierarchy["root_concepts"].append(concept_id)
        
        # Find leaves (concepts with no children)
        for concept_id in space.concepts:
            if concept_id not in child_map:
                hierarchy["leaf_concepts"].append(concept_id)
        
        # Build levels through BFS from roots
        visited = set()
        for root in hierarchy["root_concepts"]:
            queue = [(root, 0)]
            
            while queue:
                current, level = queue.pop(0)
                if current in visited:
                    continue
                    
                visited.add(current)
                hierarchy["levels"][level].append(current)
                hierarchy["max_depth"] = max(hierarchy["max_depth"], level)
                
                # Add children to queue
                for child in child_map.get(current, []):
                    if child not in visited:
                        queue.append((child, level + 1))
        
        # Add any disconnected concepts
        for concept_id in space.concepts:
            if concept_id not in visited:
                hierarchy["levels"][0].append(concept_id)
        
        # Calculate hierarchy metrics
        hierarchy["metrics"] = {
            "total_levels": len(hierarchy["levels"]),
            "branching_factor": len(child_map) / max(len(parent_map), 1),
            "hierarchy_ratio": len(hierarchy["parent_child_relations"]) / max(len(space.concepts), 1)
        }
        
        return hierarchy

    
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

    def _calculate_cluster_coherence(self, members: List[str], space) -> float:
        """Calculate how coherent a cluster is"""
        if len(members) < 2:
            return 1.0
        
        total_similarity = 0
        comparisons = 0
        
        for i, concept1_id in enumerate(members):
            for concept2_id in members[i+1:]:
                concept1 = space.concepts.get(concept1_id, {})
                concept2 = space.concepts.get(concept2_id, {})
                
                # Property similarity
                props1 = set(concept1.get("properties", {}).keys())
                props2 = set(concept2.get("properties", {}).keys())
                
                if props1 or props2:
                    prop_similarity = len(props1.intersection(props2)) / len(props1.union(props2))
                else:
                    prop_similarity = 0
                
                # Relation similarity
                rels1 = set(concept1.get("relations", {}).keys())
                rels2 = set(concept2.get("relations", {}).keys())
                
                if rels1 or rels2:
                    rel_similarity = len(rels1.intersection(rels2)) / len(rels1.union(rels2))
                else:
                    rel_similarity = 0
                
                total_similarity += (prop_similarity + rel_similarity) / 2
                comparisons += 1
        
        return total_similarity / comparisons if comparisons > 0 else 0

    def _determine_semantic_role(self, concept: Dict[str, Any], space) -> str:
        """Determine the semantic role of a concept within its space"""
        roles = []
        
        # Check abstraction level
        abstraction = concept.get("abstraction_level", "concrete")
        if abstraction == "abstract":
            roles.append("abstraction")
        elif abstraction == "meta":
            roles.append("meta-concept")
        
        # Check relation patterns
        relations = concept.get("relations", {})
        
        # Hub role - many outgoing connections
        if len(relations) > 5:
            roles.append("hub")
        
        # Bridge role - connects different domains
        connected_domains = set()
        for rel_type, targets in relations.items():
            for target in targets:
                target_concept = space.concepts.get(target, {})
                domain = target_concept.get("domain", "general")
                connected_domains.add(domain)
        
        if len(connected_domains) > 2:
            roles.append("bridge")
        
        # Foundational role - many concepts depend on it
        incoming_count = 0
        for other_id, other_concept in space.concepts.items():
            other_relations = other_concept.get("relations", {})
            for rel_type, targets in other_relations.items():
                if concept.get("id", concept.get("name")) in targets:
                    incoming_count += 1
        
        if incoming_count > 5:
            roles.append("foundation")
        
        # Specialized role - few but specific connections
        if len(relations) == 1 and list(relations.values())[0]:
            roles.append("specialized")
        
        # Terminal role - no outgoing connections
        if not relations:
            roles.append("terminal")
        
        # Return primary role or default
        if roles:
            return roles[0]
        return "standard"

    
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

    def _find_central_concept(self, members: List[str], space) -> str:
        """Find the most central concept in a cluster"""
        if not members:
            return None
        if len(members) == 1:
            return members[0]
        
        centrality_scores = {}
        
        for concept_id in members:
            concept = space.concepts.get(concept_id, {})
            
            # Calculate centrality based on connections to other members
            connections = 0
            relations = concept.get("relations", {})
            
            for rel_type, targets in relations.items():
                for target in targets:
                    if target in members:
                        connections += 1
            
            # Also consider being targeted by others
            for other_id in members:
                if other_id != concept_id:
                    other_concept = space.concepts.get(other_id, {})
                    other_relations = other_concept.get("relations", {})
                    
                    for rel_type, targets in other_relations.items():
                        if concept_id in targets:
                            connections += 1
            
            centrality_scores[concept_id] = connections
        
        # Return concept with highest centrality
        return max(centrality_scores.items(), key=lambda x: x[1])[0]

    def _find_shared_properties(self, members: List[str], space) -> Dict[str, Any]:
        """Find properties shared by cluster members"""
        if not members:
            return {}
        
        # Count property occurrences
        property_counts = defaultdict(lambda: defaultdict(int))
        
        for concept_id in members:
            concept = space.concepts.get(concept_id, {})
            properties = concept.get("properties", {})
            
            for prop_name, prop_value in properties.items():
                property_counts[prop_name][str(prop_value)] += 1
        
        # Find shared properties (present in >50% of members)
        shared = {}
        threshold = len(members) / 2
        
        for prop_name, value_counts in property_counts.items():
            total_with_prop = sum(value_counts.values())
            
            if total_with_prop >= threshold:
                # Find most common value
                most_common_value = max(value_counts.items(), key=lambda x: x[1])
                shared[prop_name] = {
                    "coverage": total_with_prop / len(members),
                    "most_common_value": most_common_value[0],
                    "value_frequency": most_common_value[1] / total_with_prop
                }
        
        return shared

    def _find_common_relations(self, members: List[str], space) -> List[Dict[str, Any]]:
        """Find relations common to cluster members"""
        if not members:
            return []
        
        # Count relation types
        relation_counts = defaultdict(int)
        relation_targets = defaultdict(set)
        
        for concept_id in members:
            concept = space.concepts.get(concept_id, {})
            relations = concept.get("relations", {})
            
            for rel_type, targets in relations.items():
                relation_counts[rel_type] += 1
                relation_targets[rel_type].update(targets)
        
        # Find common relations
        common_relations = []
        threshold = len(members) / 3  # Present in at least 1/3 of members
        
        for rel_type, count in relation_counts.items():
            if count >= threshold:
                common_relations.append({
                    "relation_type": rel_type,
                    "frequency": count / len(members),
                    "unique_targets": len(relation_targets[rel_type]),
                    "internal_targets": len([t for t in relation_targets[rel_type] if t in members])
                })
        
        # Sort by frequency
        common_relations.sort(key=lambda r: r["frequency"], reverse=True)
        
        return common_relations

    async def _find_conceptual_clusters(self, space) -> List[Dict[str, Any]]:
        """Find clusters of related concepts using similarity and connections"""
        clusters = []
        
        # Build similarity matrix
        concepts = list(space.concepts.keys())
        n_concepts = len(concepts)
        
        if n_concepts < 2:
            return []
        
        # Create feature vectors for concepts
        feature_vectors = []
        for concept_id in concepts:
            concept = space.concepts[concept_id]
            
            # Extract features
            features = []
            
            # Property-based features
            properties = concept.get("properties", {})
            prop_values = [hash(str(v)) % 100 for v in properties.values()]
            features.extend(prop_values[:10])  # Limit to 10 properties
            
            # Pad if needed
            while len(features) < 10:
                features.append(0)
            
            # Relation-based features
            relations = concept.get("relations", {})
            features.append(len(relations))
            features.append(sum(len(targets) for targets in relations.values()))
            
            # Domain feature
            domain_hash = hash(concept.get("domain", "general")) % 100
            features.append(domain_hash)
            
            feature_vectors.append(features)
        
        # Normalize features
        if len(feature_vectors) > 1:
            scaler = StandardScaler()
            feature_vectors = scaler.fit_transform(feature_vectors)
            
            # Cluster using DBSCAN
            clustering = DBSCAN(eps=1.5, min_samples=2)
            cluster_labels = clustering.fit_predict(feature_vectors)
            
            # Group concepts by cluster
            cluster_groups = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                if label != -1:  # Not noise
                    cluster_groups[label].append(concepts[i])
            
            # Create cluster descriptions
            for cluster_id, members in cluster_groups.items():
                cluster_info = {
                    "cluster_id": f"cluster_{cluster_id}",
                    "members": members,
                    "size": len(members),
                    "coherence": self._calculate_cluster_coherence(members, space),
                    "central_concept": self._find_central_concept(members, space),
                    "shared_properties": self._find_shared_properties(members, space),
                    "common_relations": self._find_common_relations(members, space)
                }
                clusters.append(cluster_info)
        
        # Sort by size
        clusters.sort(key=lambda c: c["size"], reverse=True)
        
        return clusters
    
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

    def _determine_blend_type_from_context(self, context: SharedContext) -> str:
        """Determine the appropriate conceptual blending type based on context"""
        user_input = context.user_input.lower()
        
        # Analyze query for blend type indicators
        blend_indicators = {
            "fusion": ["combine", "merge", "fuse", "integrate", "unify"],
            "composition": ["compose", "build", "construct", "assemble"],
            "completion": ["complete", "fill", "extend", "expand"],
            "elaboration": ["elaborate", "detail", "specify", "refine"],
            "analogy": ["like", "similar", "analogous", "compare"]
        }
        
        scores = defaultdict(int)
        
        for blend_type, keywords in blend_indicators.items():
            for keyword in keywords:
                if keyword in user_input:
                    scores[blend_type] += 1
        
        # Consider context factors
        if context.emotional_state:
            emotion = context.emotional_state.get("dominant_emotion", ("neutral", 0))[0]
            if emotion == "Curiosity":
                scores["elaboration"] += 1
            elif emotion == "Excitement":
                scores["fusion"] += 1
        
        if context.goal_context:
            goals = context.goal_context.get("active_goals", [])
            for goal in goals:
                if "create" in goal.get("description", "").lower():
                    scores["composition"] += 1
                elif "understand" in goal.get("description", "").lower():
                    scores["analogy"] += 1
        
        # Return highest scoring type or default
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return "composition"  # Default

    async def _find_contextual_mappings(self, space1_id: str, space2_id: str, 
                                       context: SharedContext) -> List[Dict[str, Any]]:
        """Find mappings between concept spaces based on context"""
        space1 = self.original_core.concept_spaces.get(space1_id)
        space2 = self.original_core.concept_spaces.get(space2_id)
        
        if not space1 or not space2:
            return []
        
        mappings = []
        user_input = context.user_input.lower()
        
        # Extract key terms from context
        key_terms = set(user_input.split())
        
        # Score all possible mappings
        mapping_candidates = []
        
        for c1_id, c1 in space1.concepts.items():
            for c2_id, c2 in space2.concepts.items():
                # Calculate mapping score
                score = 0.0
                
                # Name similarity
                name_sim = self._calculate_concept_name_similarity(
                    c1.get("name", ""), c2.get("name", "")
                )
                score += name_sim * 0.3
                
                # Property overlap
                props1 = set(c1.get("properties", {}).keys())
                props2 = set(c2.get("properties", {}).keys())
                if props1 and props2:
                    prop_overlap = len(props1.intersection(props2)) / len(props1.union(props2))
                    score += prop_overlap * 0.3
                
                # Relation type overlap
                rels1 = set(c1.get("relations", {}).keys())
                rels2 = set(c2.get("relations", {}).keys())
                if rels1 and rels2:
                    rel_overlap = len(rels1.intersection(rels2)) / len(rels1.union(rels2))
                    score += rel_overlap * 0.2
                
                # Context relevance
                c1_relevance = sum(1 for term in key_terms if term in c1.get("name", "").lower())
                c2_relevance = sum(1 for term in key_terms if term in c2.get("name", "").lower())
                context_relevance = (c1_relevance + c2_relevance) / max(len(key_terms), 1)
                score += context_relevance * 0.2
                
                if score > 0.3:  # Threshold
                    mapping_candidates.append({
                        "concept1": c1_id,
                        "concept2": c2_id,
                        "similarity": score,
                        "mapping_type": self._determine_mapping_type(c1, c2)
                    })
        
        # Sort by score and take top mappings
        mapping_candidates.sort(key=lambda m: m["similarity"], reverse=True)
        
        # Filter for diversity - don't map same concept multiple times
        used_concepts1 = set()
        used_concepts2 = set()
        
        for candidate in mapping_candidates[:20]:  # Consider top 20
            c1 = candidate["concept1"]
            c2 = candidate["concept2"]
            
            if c1 not in used_concepts1 and c2 not in used_concepts2:
                mappings.append(candidate)
                used_concepts1.add(c1)
                used_concepts2.add(c2)
                
                if len(mappings) >= 5:  # Limit mappings
                    break
        
        return mappings
    
    def _calculate_concept_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between concept names"""
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Exact match
        if name1_lower == name2_lower:
            return 1.0
        
        # Substring match
        if name1_lower in name2_lower or name2_lower in name1_lower:
            return 0.8
        
        # Word overlap
        words1 = set(name1_lower.split())
        words2 = set(name2_lower.split())
        
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return overlap / union if union > 0 else 0.0
        
        # Character n-gram similarity
        ngrams1 = set(name1_lower[i:i+3] for i in range(len(name1_lower)-2))
        ngrams2 = set(name2_lower[i:i+3] for i in range(len(name2_lower)-2))
        
        if ngrams1 and ngrams2:
            ngram_overlap = len(ngrams1.intersection(ngrams2))
            ngram_union = len(ngrams1.union(ngrams2))
            return ngram_overlap / ngram_union * 0.5 if ngram_union > 0 else 0.0
        
        return 0.0
    
    def _determine_mapping_type(self, concept1: Dict[str, Any], 
                               concept2: Dict[str, Any]) -> str:
        """Determine the type of mapping between two concepts"""
        # Check abstraction levels
        abs1 = concept1.get("abstraction_level", "concrete")
        abs2 = concept2.get("abstraction_level", "concrete")
        
        if abs1 == abs2:
            return "horizontal"  # Same level mapping
        elif abs1 == "abstract" and abs2 == "concrete":
            return "instantiation"  # Abstract to concrete
        elif abs1 == "concrete" and abs2 == "abstract":
            return "abstraction"  # Concrete to abstract
        
        # Check domains
        domain1 = concept1.get("domain", "general")
        domain2 = concept2.get("domain", "general")
        
        if domain1 != domain2:
            return "cross_domain"
        
        # Check relation patterns
        rels1 = set(concept1.get("relations", {}).keys())
        rels2 = set(concept2.get("relations", {}).keys())
        
        if rels1 == rels2 and rels1:
            return "structural"  # Same relational structure
        
        return "similarity"  # Default
    
    async def _generate_counterfactual_scenario(self, model_id: str, 
                                              context: SharedContext) -> Dict[str, Any]:
        """Generate a counterfactual scenario for analysis"""
        model = self.original_core.causal_models.get(model_id)
        if not model:
            return None
        
        scenario = {
            "model_id": model_id,
            "intervention_point": None,
            "intervention_type": None,
            "original_state": {},
            "counterfactual_state": {},
            "changes_propagated": [],
            "intervention_nodes": []
        }
        
        # Extract intervention from context
        user_input = context.user_input.lower()
        
        # Find intervention target
        for node_id, node in model.nodes.items():
            node_name_lower = node.name.lower()
            if node_name_lower in user_input:
                scenario["intervention_point"] = node_id
                scenario["intervention_nodes"].append(node_id)
                break
        
        if not scenario["intervention_point"]:
            return None
        
        # Determine intervention type
        if "increase" in user_input or "more" in user_input:
            scenario["intervention_type"] = "increase"
            intervention_value = 1.5  # 50% increase
        elif "decrease" in user_input or "less" in user_input:
            scenario["intervention_type"] = "decrease"
            intervention_value = 0.5  # 50% decrease
        elif "remove" in user_input or "eliminate" in user_input:
            scenario["intervention_type"] = "remove"
            intervention_value = 0.0
        else:
            scenario["intervention_type"] = "change"
            intervention_value = 1.2  # 20% increase default
        
        # Create original state snapshot
        for node_id, node in model.nodes.items():
            scenario["original_state"][node_id] = {
                "value": getattr(node, "value", 1.0),
                "active": getattr(node, "active", True)
            }
        
        # Create counterfactual state
        scenario["counterfactual_state"] = scenario["original_state"].copy()
        
        # Apply intervention
        intervention_node = scenario["intervention_point"]
        original_value = scenario["original_state"][intervention_node]["value"]
        scenario["counterfactual_state"][intervention_node]["value"] = original_value * intervention_value
        
        # Propagate changes through causal network
        changes_to_propagate = [(intervention_node, intervention_value)]
        visited = set()
        
        while changes_to_propagate:
            current_node, change_factor = changes_to_propagate.pop(0)
            
            if current_node in visited:
                continue
            visited.add(current_node)
            
            # Find downstream effects
            for relation in model.relations:
                if relation.source == current_node:
                    target_node = relation.target
                    effect_strength = relation.strength
                    
                    # Calculate propagated change
                    propagated_factor = 1.0 + (change_factor - 1.0) * effect_strength
                    
                    # Update counterfactual state
                    original_target_value = scenario["original_state"][target_node]["value"]
                    scenario["counterfactual_state"][target_node]["value"] = (
                        original_target_value * propagated_factor
                    )
                    
                    # Record change
                    scenario["changes_propagated"].append({
                        "from": current_node,
                        "to": target_node,
                        "mechanism": relation.mechanism,
                        "strength": effect_strength,
                        "change_factor": propagated_factor
                    })
                    
                    # Add to propagation queue if significant change
                    if abs(propagated_factor - 1.0) > 0.05:
                        changes_to_propagate.append((target_node, propagated_factor))
        
        # Calculate scenario metrics
        scenario["metrics"] = {
            "nodes_affected": len([n for n, state in scenario["counterfactual_state"].items()
                                  if state["value"] != scenario["original_state"][n]["value"]]),
            "max_change": max(abs(scenario["counterfactual_state"][n]["value"] - 
                                 scenario["original_state"][n]["value"])
                             for n in scenario["original_state"]),
            "propagation_depth": len(visited)
        }
        
        return scenario
    
    async def _analyze_alternative_paths(self, model_id: str, 
                                       scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze alternative causal paths in counterfactual scenario"""
        model = self.original_core.causal_models.get(model_id)
        if not model:
            return []
        
        alternative_paths = []
        intervention_point = scenario["intervention_point"]
        
        # Find key outcome nodes (nodes with no outgoing edges)
        outcome_nodes = []
        for node_id in model.nodes:
            has_outgoing = any(r.source == node_id for r in model.relations)
            if not has_outgoing:
                outcome_nodes.append(node_id)
        
        # For each outcome, find alternative paths
        graph_data = self._model_to_graph(model)
        
        for outcome in outcome_nodes:
            # Find all paths from intervention to outcome
            all_paths = await self.path_finder.find_paths(
                intervention_point, outcome, graph_data, "causal",
                constraints={"max_length": 6}, max_paths=10
            )
            
            if len(all_paths) > 1:
                # Compare paths
                for i, path in enumerate(all_paths):
                    path_info = {
                        "path_id": f"path_{i}",
                        "nodes": path,
                        "length": len(path),
                        "original_strength": self._calculate_path_strength(path, graph_data),
                        "counterfactual_strength": self._calculate_counterfactual_strength(
                            path, scenario
                        ),
                        "robustness": self._calculate_path_robustness(path, model),
                        "bottlenecks": self._identify_path_bottlenecks(path, model)
                    }
                    
                    # Determine if path is enhanced or diminished
                    strength_change = (path_info["counterfactual_strength"] - 
                                     path_info["original_strength"])
                    
                    if abs(strength_change) > 0.1:
                        path_info["change_type"] = "enhanced" if strength_change > 0 else "diminished"
                        path_info["change_magnitude"] = abs(strength_change)
                        alternative_paths.append(path_info)
        
        # Sort by change magnitude
        alternative_paths.sort(key=lambda p: p.get("change_magnitude", 0), reverse=True)
        
        return alternative_paths
    
    def _calculate_counterfactual_strength(self, path: List[str], 
                                         scenario: Dict[str, Any]) -> float:
        """Calculate path strength in counterfactual scenario"""
        strength = 1.0
        
        for i in range(len(path) - 1):
            node = path[i]
            
            # Get counterfactual value
            cf_value = scenario["counterfactual_state"].get(node, {}).get("value", 1.0)
            orig_value = scenario["original_state"].get(node, {}).get("value", 1.0)
            
            # Adjust strength based on value change
            if orig_value != 0:
                value_ratio = cf_value / orig_value
                strength *= value_ratio
        
        return min(1.0, strength)

    def _identify_path_bottlenecks(self, path: List[str], model) -> List[Dict[str, Any]]:
        """Identify bottlenecks in a causal path"""
        bottlenecks = []
        
        for i in range(len(path) - 1):
            node = path[i]
            next_node = path[i + 1]
            
            # Check if this is a bottleneck
            is_bottleneck = False
            bottleneck_type = None
            severity = 0.0
            
            # Type 1: Weak connection
            for relation in model.relations:
                if relation.source == node and relation.target == next_node:
                    if relation.strength < 0.3:
                        is_bottleneck = True
                        bottleneck_type = "weak_connection"
                        severity = 1.0 - relation.strength
                    break
            
            # Type 2: High fan-out (many outgoing connections dilute effect)
            outgoing_count = sum(1 for r in model.relations if r.source == node)
            if outgoing_count > 5:
                is_bottleneck = True
                bottleneck_type = "high_fanout"
                severity = min(1.0, outgoing_count / 10)
            
            # Type 3: Contested node (multiple strong inputs)
            incoming_count = sum(1 for r in model.relations if r.target == node and r.strength > 0.5)
            if incoming_count > 3:
                is_bottleneck = True
                bottleneck_type = "contested"
                severity = min(1.0, incoming_count / 5)
            
            if is_bottleneck:
                bottlenecks.append({
                    "node": node,
                    "position": i,
                    "type": bottleneck_type,
                    "severity": severity,
                    "description": self._describe_bottleneck(bottleneck_type, node, model)
                })
        
        return bottlenecks
    
    def _describe_bottleneck(self, bottleneck_type: str, node: str, model) -> str:
        """Generate description for a bottleneck"""
        node_name = model.nodes.get(node, {}).name if hasattr(model.nodes.get(node), 'name') else node
        
        descriptions = {
            "weak_connection": f"{node_name} has weak causal influence on its targets",
            "high_fanout": f"{node_name} disperses its effect across many targets",
            "contested": f"{node_name} receives conflicting influences from multiple sources"
        }
        
        return descriptions.get(bottleneck_type, f"{node_name} is a bottleneck")
    
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
