# nyx/core/procedural_memory/temporal.py

import datetime
import asyncio
import threading
import functools
import heapq
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Iterator
from pydantic import BaseModel, Field

# OpenAI Agents SDK imports
from agents.tracing import custom_span, trace as agents_trace

from .models import Procedure

class TemporalNode(BaseModel):
    """Node in a temporal procedure graph"""
    id: str
    action: Dict[str, Any]
    temporal_constraints: List[Dict[str, Any]] = Field(default_factory=list)
    duration: Optional[Tuple[float, float]] = None  # (min, max) duration
    next_nodes: List[str] = Field(default_factory=list)
    prev_nodes: List[str] = Field(default_factory=list)
    
    def add_constraint(self, constraint: Dict[str, Any]) -> None:
        """Add a temporal constraint to this node"""
        self.temporal_constraints.append(constraint)
    
    def is_valid(self, execution_history: List[Dict[str, Any]]) -> bool:
        """Check if this node's temporal constraints are valid"""
        with custom_span("temporal_node_validation", 
                      {"node_id": self.id, "constraints_count": len(self.temporal_constraints)}):
            for constraint in self.temporal_constraints:
                constraint_type = constraint.get("type")
                
                if constraint_type == "after":
                    # Must occur after another action
                    ref_action = constraint.get("action")
                    if not any(h["action"] == ref_action for h in execution_history):
                        return False
                elif constraint_type == "before":
                    # Must occur before another action
                    ref_action = constraint.get("action")
                    if any(h["action"] == ref_action for h in execution_history):
                        return False
                elif constraint_type == "delay":
                    # Must wait minimum time from last action
                    if execution_history:
                        last_time = execution_history[-1].get("timestamp")
                        min_delay = constraint.get("min_delay", 0)
                        if last_time:
                            last_time = datetime.datetime.fromisoformat(last_time)
                            elapsed = (datetime.datetime.now() - last_time).total_seconds()
                            if elapsed < min_delay:
                                return False
            
            return True

    def estimate_memory_usage(self) -> int:
        """Estimate memory usage of this node in bytes"""
        # Base size
        memory = 500  # Base object overhead
        
        # Add size for action data
        memory += len(str(self.action)) * 2  # Rough estimate based on string representation
        
        # Add size for lists
        memory += len(self.temporal_constraints) * 200
        memory += len(self.next_nodes) * 20
        memory += len(self.prev_nodes) * 20
        
        # Add size for duration if present
        if self.duration:
            memory += 16  # Two float values
        
        return memory
    
    def is_executable_within(self, 
                            execution_history: List[Dict[str, Any]], 
                            timeout: float = 0.0) -> Tuple[bool, Optional[str]]:
        """
        Check if this node can be executed within the given timeout
        
        Args:
            execution_history: Previous execution history
            timeout: Maximum wait time in seconds (0 means no wait)
            
        Returns:
            Tuple of (can_execute, reason)
        """
        # First check immediate executability
        if self.is_valid(execution_history):
            return True, None
        
        # If no timeout specified, return current status
        if timeout <= 0:
            return False, "Temporal constraints not satisfied and no wait requested"
        
        # Check which constraints might be satisfied after waiting
        waiting_required = False
        wait_constraints = []
        
        for constraint in self.temporal_constraints:
            constraint_type = constraint.get("type")
            
            if constraint_type == "delay":
                # Check if waiting would satisfy this constraint
                if execution_history:
                    last_time = execution_history[-1].get("timestamp")
                    min_delay = constraint.get("min_delay", 0)
                    
                    if last_time:
                        last_time = datetime.datetime.fromisoformat(last_time)
                        elapsed = (datetime.datetime.now() - last_time).total_seconds()
                        
                        if elapsed < min_delay:
                            # Not enough time has passed
                            wait_needed = min_delay - elapsed
                            
                            if wait_needed <= timeout:
                                # We can wait for this constraint
                                waiting_required = True
                                wait_constraints.append({
                                    "type": "delay",
                                    "wait_needed": wait_needed,
                                    "constraint": constraint
                                })
                            else:
                                # Wait time exceeds our timeout
                                return False, f"Required delay {wait_needed:.2f}s exceeds timeout {timeout:.2f}s"
        
        # If we need to wait but can do so within the timeout, return that info
        if waiting_required:
            # Calculate max wait needed across all constraints
            max_wait = max(c["wait_needed"] for c in wait_constraints)
            return True, f"Can execute after waiting {max_wait:.2f}s"
        
        # Otherwise, there are constraints we can't satisfy by waiting
        return False, "Some temporal constraints cannot be satisfied by waiting"

class TemporalProcedureGraph(BaseModel):
    """Graph representation of a temporal procedure"""
    id: str
    name: str
    nodes: Dict[str, TemporalNode] = Field(default_factory=dict)
    edges: List[Tuple[str, str, Dict[str, Any]]] = Field(default_factory=list)
    start_nodes: List[str] = Field(default_factory=list)
    end_nodes: List[str] = Field(default_factory=list)
    domain: str
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    _graph_lock = None  # Will be initialized in __post_init__
    
    def __post_init__(self):
        """Initialize asyncio lock"""
        self._graph_lock = asyncio.Lock()
    
    async def add_node(self, node: TemporalNode) -> None:
        """Add a node to the graph (thread-safe)"""
        async with self._graph_lock:
            self.nodes[node.id] = node
            self.last_updated = datetime.datetime.now().isoformat()
    
    async def add_edge(self, from_id: str, to_id: str, properties: Dict[str, Any] = None) -> None:
        """Add an edge between nodes (thread-safe)"""
        async with self._graph_lock:
            if from_id in self.nodes and to_id in self.nodes:
                self.edges.append((from_id, to_id, properties or {}))
                
                # Update node connections
                self.nodes[from_id].next_nodes.append(to_id)
                self.nodes[to_id].prev_nodes.append(from_id)
                
                self.last_updated = datetime.datetime.now().isoformat()
    
    def validate_temporal_constraints(self) -> bool:
        """Validate that temporal constraints are consistent"""
        with custom_span("validate_temporal_constraints", {"graph_id": self.id}):
            # Check for cycles with minimum durations
            visited = set()
            path = set()
            
            # Check each start node
            for start in self.start_nodes:
                if not self._check_for_negative_cycles(start, visited, path, 0):
                    return False
            
            return True
    
    def _check_for_negative_cycles(self, 
                                node_id: str, 
                                visited: Set[str], 
                                path: Set[str], 
                                current_duration: float) -> bool:
        """Check for negative cycles in the graph (would make it impossible to satisfy)"""
        if node_id in path:
            # Found a cycle, check if the total duration is negative
            return current_duration >= 0
        
        if node_id in visited:
            return True
        
        visited.add(node_id)
        path.add(node_id)
        
        # Check outgoing edges
        for source, target, props in self.edges:
            if source == node_id:
                # Get edge duration
                min_duration = props.get("min_duration", 0)
                
                # Recurse
                if not self._check_for_negative_cycles(target, visited, path, 
                                                    current_duration + min_duration):
                    return False
        
        path.remove(node_id)
        return True
    
    def get_next_executable_nodes(self, execution_history: List[Dict[str, Any]]) -> List[str]:
        """Get nodes that can be executed next based on history"""
        with custom_span("get_next_executable_nodes", 
                      {"history_length": len(execution_history)}):
            # Start with nodes that have no predecessors if no history
            if not execution_history:
                return self.start_nodes
            
            # Get last executed node
            last_action = execution_history[-1].get("node_id")
            if not last_action or last_action not in self.nodes:
                # Can't determine next actions
                return []
            
            # Get possible next nodes
            next_nodes = self.nodes[last_action].next_nodes
            
            # Filter by temporal constraints
            valid_nodes = []
            for node_id in next_nodes:
                if node_id in self.nodes and self.nodes[node_id].is_valid(execution_history):
                    valid_nodes.append(node_id)
            
            return valid_nodes
    
    @classmethod
    def from_procedure(cls, procedure: Procedure) -> 'TemporalProcedureGraph':
        """Convert a standard procedure to a temporal procedure graph"""
        with custom_span("create_temporal_graph_from_procedure", 
                      {"procedure_id": procedure.id}):
            graph = cls(
                id=f"temporal_{procedure.id}",
                name=f"Temporal graph for {procedure.name}",
                domain=procedure.domain
            )
            
            # Create nodes for each step
            for i, step in enumerate(procedure.steps):
                node = TemporalNode(
                    id=f"node_{step['id']}",
                    action={
                        "function": step["function"],
                        "parameters": step.get("parameters", {}),
                        "description": step.get("description", f"Step {i+1}"),
                        "step_id": step["id"]
                    }
                )
                
                graph.nodes[node.id] = node
                
                # First step is a start node
                if i == 0:
                    graph.start_nodes.append(node.id)
                
                # Last step is an end node
                if i == len(procedure.steps) - 1:
                    graph.end_nodes.append(node.id)
            
            # Create edges for sequential execution
            for i in range(len(procedure.steps) - 1):
                current_id = f"node_{procedure.steps[i]['id']}"
                next_id = f"node_{procedure.steps[i+1]['id']}"
                
                graph.edges.append((current_id, next_id, {}))
                
                # Update node connections
                graph.nodes[current_id].next_nodes.append(next_id)
                graph.nodes[next_id].prev_nodes.append(current_id)
            
            return graph


    def estimate_memory_usage(self) -> int:
        """Estimate memory usage of this graph in bytes"""
        # Base size
        memory = 1000  # Base object overhead
        
        # Add size for nodes
        for node_id, node in self.nodes.items():
            memory += len(node_id) * 2  # String size
            if hasattr(node, "estimate_memory_usage"):
                memory += node.estimate_memory_usage()
            else:
                memory += 500  # Default estimate per node
        
        # Add size for edges
        memory += len(self.edges) * 100  # Rough estimate per edge
        
        # Add size for other lists
        memory += len(self.start_nodes) * 20
        memory += len(self.end_nodes) * 20
        
        return memory
    
    def cleanup(self) -> int:
        """Clean up any unnecessary data, return bytes saved"""
        # Not much to clean in the base implementation
        return 0
    
    @functools.lru_cache(maxsize=32)
    def get_critical_path(self) -> List[str]:
        """
        Find the critical path through the graph (longest path)
        Uses caching for performance
        
        Returns:
            List of node IDs representing the critical path
        """
        # Create distance map
        distances = {node_id: float('-inf') for node_id in self.nodes}
        predecessors = {node_id: None for node_id in self.nodes}
        
        # Set start nodes to distance 0
        for start in self.start_nodes:
            distances[start] = 0
        
        # Topological sort (if graph is acyclic)
        topo_order = self._topological_sort()
        if not topo_order:
            # Graph has cycles, use fallback algorithm
            logger.warning("Graph contains cycles, critical path may not be accurate")
            topo_order = list(self.nodes.keys())
        
        # Find longest path
        for node in topo_order:
            # Skip if not reachable
            if distances[node] == float('-inf'):
                continue
                
            # Check outgoing edges
            for source, target, props in self.edges:
                if source == node:
                    # Get edge duration
                    duration = props.get("min_duration", 0)
                    
                    # Update distance if better
                    if distances[node] + duration > distances[target]:
                        distances[target] = distances[node] + duration
                        predecessors[target] = node
        
        # Find end node with largest distance
        max_distance = float('-inf')
        end_node = None
        
        for end in self.end_nodes:
            if distances[end] > max_distance:
                max_distance = distances[end]
                end_node = end
        
        # No path found
        if end_node is None or max_distance == float('-inf'):
            return []
        
        # Reconstruct path
        path = []
        current = end_node
        
        while current is not None:
            path.append(current)
            current = predecessors[current]
        
        # Reverse to get start-to-end order
        path.reverse()
        
        return path
    
    def _topological_sort(self) -> Optional[List[str]]:
        """Perform topological sort on the graph, return None if cycles exist"""
        # Create adjacency list
        graph = {node_id: [] for node_id in self.nodes}
        for source, target, _ in self.edges:
            graph[source].append(target)
        
        # Count incoming edges
        incoming = {node_id: 0 for node_id in self.nodes}
        for source, target, _ in self.edges:
            incoming[target] += 1
        
        # Find nodes with no incoming edges
        queue = [node_id for node_id, count in incoming.items() if count == 0]
        result = []
        
        # Process queue
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # Reduce incoming count for neighbors
            for neighbor in graph[node]:
                incoming[neighbor] -= 1
                
                # If no more incoming edges, add to queue
                if incoming[neighbor] == 0:
                    queue.append(neighbor)
        
        # If not all nodes processed, graph has cycles
        if len(result) != len(self.nodes):
            return None
        
        return result
    
    def get_executable_nodes_with_timeout(self, 
                                        execution_history: List[Dict[str, Any]],
                                        max_wait_time: float = 0.0) -> List[Tuple[str, float]]:
        """
        Get nodes that can be executed within a given timeout
        
        Args:
            execution_history: Previous execution history
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            List of tuples (node_id, wait_time)
        """
        executable_nodes = []
        
        # Start with nodes that have no predecessors if no history
        candidates = self.start_nodes if not execution_history else []
        
        # If we have history, get possible next nodes from last execution
        if execution_history:
            last_action = execution_history[-1].get("node_id")
            if last_action and last_action in self.nodes:
                candidates = self.nodes[last_action].next_nodes
        
        # Check each candidate
        for node_id in candidates:
            if node_id not in self.nodes:
                continue
                
            node = self.nodes[node_id]
            can_execute, reason = node.is_executable_within(execution_history, max_wait_time)
            
            if can_execute:
                # Extract wait time from reason if available
                wait_time = 0.0
                if reason and "waiting" in reason.lower():
                    try:
                        wait_time = float(reason.split("waiting")[1].split("s")[0].strip())
                    except (ValueError, IndexError):
                        pass
                        
                executable_nodes.append((node_id, wait_time))
        
        # Sort by wait time (shorter waits first)
        executable_nodes.sort(key=lambda x: x[1])
        
        return executable_nodes
    
    def create_execution_plan(self, current_state: Dict[str, Any], goal_state: Dict[str, Any]) -> Optional[List[str]]:
        """
        Create an execution plan to reach a goal state from current state
        
        Args:
            current_state: Current state values
            goal_state: Desired end state values
            
        Returns:
            Ordered list of node IDs to execute, or None if no plan found
        """
        # Find end nodes that satisfy goal state
        suitable_end_nodes = []
        
        for node_id in self.end_nodes:
            node = self.nodes[node_id]
            # Skip nodes with no action (they can't change state)
            if not node.action:
                continue
                
            # Check if action helps achieve goal
            postconditions = node.action.get("postconditions", {})
            
            # Count how many goal state items this node satisfies
            matches = 0
            for key, goal_value in goal_state.items():
                if key in postconditions and postconditions[key] == goal_value:
                    matches += 1
            
            if matches > 0:
                suitable_end_nodes.append((node_id, matches))
        
        # Sort by most goal matches
        suitable_end_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Try to find a path to each suitable end node
        for end_node, _ in suitable_end_nodes:
            # Find paths from any start node to this end node
            for start_node in self.start_nodes:
                # Check if start node's preconditions are met by current state
                start = self.nodes[start_node]
                preconditions = start.action.get("preconditions", {})
                
                preconditions_met = True
                for key, value in preconditions.items():
                    if key not in current_state or current_state[key] != value:
                        preconditions_met = False
                        break
                
                if not preconditions_met:
                    continue
                    
                # Find shortest path from start to end
                path = self._find_shortest_path(start_node, end_node)
                if path:
                    return path
        
        return None
    
    def _find_shortest_path(self, start: str, end: str) -> Optional[List[str]]:
        """Find shortest path from start to end using A* algorithm"""
        # Check if nodes exist
        if start not in self.nodes or end not in self.nodes:
            return None
        
        # Priority queue for A*
        queue = [(0, start, [start])]  # (priority, node, path)
        visited = set()
        
        while queue:
            # Get node with lowest priority
            priority, node, path = heapq.heappop(queue)
            
            # If we reached the target, return path
            if node == end:
                return path
            
            # Skip if already visited
            if node in visited:
                continue
                
            visited.add(node)
            
            # Check neighbors
            for source, target, props in self.edges:
                if source == node and target not in visited:
                    # Calculate priority (negative of min_duration for shortest path)
                    duration = props.get("min_duration", 0)
                    new_priority = priority - duration
                    
                    # Add to queue
                    heapq.heappush(queue, (new_priority, target, path + [target]))
        
        # No path found
        return None
    
    def add_edge(self, from_id: str, to_id: str, properties: Dict[str, Any] = None) -> None:
        """Add an edge between nodes"""
        if from_id in self.nodes and to_id in self.nodes:
            self.edges.append((from_id, to_id, properties or {}))
            
            # Update node connections
            self.nodes[from_id].next_nodes.append(to_id)
            self.nodes[to_id].prev_nodes.append(from_id)
            
            self.last_updated = datetime.datetime.now().isoformat()
    
    def get_next_executable_nodes(self, execution_history: List[Dict[str, Any]]) -> List[str]:
        """Get nodes that can be executed next based on history"""
        # Start with nodes that have no predecessors if no history
        if not execution_history:
            return self.start_nodes
        
        # Get last executed node
        last_action = execution_history[-1].get("node_id")
        if not last_action or last_action not in self.nodes:
            # Can't determine next actions
            return []
        
        # Get possible next nodes
        next_nodes = self.nodes[last_action].next_nodes
        
        # Filter by temporal constraints
        valid_nodes = []
        for node_id in next_nodes:
            if node_id in self.nodes and self.nodes[node_id].is_valid(execution_history):
                valid_nodes.append(node_id)
        
        return valid_nodes
    
    def validate_temporal_constraints(self) -> bool:
        """Validate that temporal constraints are consistent"""
        # Check for cycles with minimum durations
        visited = set()
        path = set()
        
        # Check each start node
        for start in self.start_nodes:
            if not self._check_for_negative_cycles(start, visited, path, 0):
                return False
        
        return True
    
    def _check_for_negative_cycles(self, 
                                 node_id: str, 
                                 visited: Set[str], 
                                 path: Set[str], 
                                 current_duration: float) -> bool:
        """Check for negative cycles in the graph (would make it impossible to satisfy)"""
        if node_id in path:
            # Found a cycle, check if the total duration is negative
            return current_duration >= 0
        
        if node_id in visited:
            return True
        
        visited.add(node_id)
        path.add(node_id)
        
        # Check outgoing edges
        for source, target, props in self.edges:
            if source == node_id:
                # Get edge duration
                min_duration = props.get("min_duration", 0)
                
                # Recurse
                if not self._check_for_negative_cycles(target, visited, path, 
                                                     current_duration + min_duration):
                    return False
        
        path.remove(node_id)
        return True
    
    @classmethod
    def from_procedure(cls, procedure: Procedure) -> 'TemporalProcedureGraph':
        """Convert a standard procedure to a temporal procedure graph"""
        graph = cls(
            id=f"temporal_{procedure.id}",
            name=f"Temporal graph for {procedure.name}",
            domain=procedure.domain
        )
        
        # Create nodes for each step
        for i, step in enumerate(procedure.steps):
            node = TemporalNode(
                id=f"node_{step['id']}",
                action={
                    "function": step["function"],
                    "parameters": step.get("parameters", {}),
                    "description": step.get("description", f"Step {i+1}")
                }
            )
            
            graph.add_node(node)
            
            # First step is a start node
            if i == 0:
                graph.start_nodes.append(node.id)
            
            # Last step is an end node
            if i == len(procedure.steps) - 1:
                graph.end_nodes.append(node.id)
        
        # Create edges for sequential execution
        for i in range(len(procedure.steps) - 1):
            current_id = f"node_{procedure.steps[i]['id']}"
            next_id = f"node_{procedure.steps[i+1]['id']}"
            
            graph.add_edge(current_id, next_id)
        
        return graph

class ProcedureGraph(BaseModel):
    """Graph representation of a procedure for flexible execution"""
    nodes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    entry_points: List[str] = Field(default_factory=list)
    exit_points: List[str] = Field(default_factory=list)
    _graph_lock = None  # Will be initialized in __post_init__
    
    def __post_init__(self):
        """Initialize asyncio lock"""
        self._graph_lock = asyncio.Lock()
    
    async def add_node(self, node_id: str, data: Dict[str, Any]) -> None:
        """Add a node to the graph (thread-safe)"""
        async with self._graph_lock:
            self.nodes[node_id] = data
    
    async def add_edge(self, from_id: str, to_id: str, properties: Dict[str, Any] = None) -> None:
        """Add an edge to the graph (thread-safe)"""
        async with self._graph_lock:
            self.edges.append({
                "from": from_id,
                "to": to_id,
                "properties": properties or {}
            })
    
    def find_execution_path(
        self, 
        context: Dict[str, Any],
        goal: Dict[str, Any]
    ) -> List[str]:
        """Find execution path through the graph given context and goal"""
        with custom_span("find_execution_path", 
                      {"entry_points": len(self.entry_points), 
                       "exit_points": len(self.exit_points)}):
            if not self.entry_points:
                return []
            
            # Find all paths from entry to exit points
            all_paths = []
            
            for entry in self.entry_points:
                for exit_point in self.exit_points:
                    paths = self._find_all_paths(entry, exit_point)
                    all_paths.extend(paths)
            
            if not all_paths:
                return []
            
            # Score each path based on context and goal
            scored_paths = []
            
            for path in all_paths:
                score = self._score_path(path, context, goal)
                scored_paths.append((path, score))
            
            # Return highest scoring path
            best_path, _ = max(scored_paths, key=lambda x: x[1])
            return best_path

    _procedure_graph_lock = threading.RLock()  # Class level lock for thread safety

    def add_node(self, node_id: str, data: Dict[str, Any]) -> None:
        """Add a node to the graph (thread-safe)"""
        with self._procedure_graph_lock:
            self.nodes[node_id] = data
    
    def add_edge(self, from_id: str, to_id: str, properties: Dict[str, Any] = None) -> None:
        """Add an edge to the graph (thread-safe)"""
        with self._procedure_graph_lock:
            self.edges.append({
                "from": from_id,
                "to": to_id,
                "properties": properties or {}
            })
    
    def find_execution_path(
        self, 
        context: Dict[str, Any],
        goal: Dict[str, Any]
    ) -> List[str]:
        """
        Find execution path through the graph given context and goal
        Optimized version with caching
        """
        # Generate cache key
        context_key = frozenset((k, str(v)) for k, v in context.items() if not isinstance(v, (dict, list, set)))
        goal_key = frozenset((k, str(v)) for k, v in goal.items())
        cache_key = (context_key, goal_key)
        
        # Check cache using function-level attribute
        if not hasattr(self.find_execution_path, "cache"):
            self.find_execution_path.cache = {}
        
        if cache_key in self.find_execution_path.cache:
            cached_result, timestamp = self.find_execution_path.cache[cache_key]
            # Cache results for 5 minutes
            if (datetime.datetime.now() - timestamp).total_seconds() < 300:
                return cached_result
        
        # Find paths as before
        if not self.entry_points:
            return []
        
        all_paths = []
        
        # Use optimized path finding for better performance
        for entry in self.entry_points:
            for exit_point in self.exit_points:
                paths = self._find_paths_optimized(entry, exit_point)
                all_paths.extend(paths)
        
        if not all_paths:
            # Cache empty result
            self.find_execution_path.cache[cache_key] = ([], datetime.datetime.now())
            return []
        
        # Score paths in parallel
        scored_paths = []
        
        # Score each path (sequential for now)
        for path in all_paths:
            score = self._score_path(path, context, goal)
            scored_paths.append((path, score))
        
        # Return highest scoring path
        best_path, _ = max(scored_paths, key=lambda x: x[1])
        
        # Cache result with timestamp
        self.find_execution_path.cache[cache_key] = (best_path, datetime.datetime.now())
        
        # Clean cache if too large (keep only most recent 50 entries)
        if len(self.find_execution_path.cache) > 50:
            # Sort by timestamp
            sorted_cache = sorted(self.find_execution_path.cache.items(), 
                                key=lambda x: x[1][1], reverse=True)
            # Keep only the most recent 50
            self.find_execution_path.cache = dict(sorted_cache[:50])
        
        return best_path
    
    def _find_paths_optimized(self, start: str, end: str, max_paths: int = 5) -> List[List[str]]:
        """
        Find up to max_paths between two nodes using BFS
        More efficient than the recursive approach for larger graphs
        """
        if start not in self.nodes or end not in self.nodes:
            return []
        
        # Use BFS to find paths
        queue = [(start, [start])]  # (current, path)
        paths = []
        
        while queue and len(paths) < max_paths:
            current, path = queue.pop(0)
            
            # If reached end, save path
            if current == end:
                paths.append(path)
                continue
            
            # Avoid cycles
            if current != start and current in path[:-1]:
                continue
            
            # Find neighbors
            neighbors = []
            for edge in self.edges:
                if edge["from"] == current:
                    neighbors.append(edge["to"])
            
            # Add neighbors to queue
            for neighbor in neighbors:
                if neighbor not in path:  # Avoid cycles
                    queue.append((neighbor, path + [neighbor]))
        
        return paths
    
    def estimate_memory_usage(self) -> int:
        """Estimate memory usage of this graph in bytes"""
        # Base size
        memory = 1000  # Base object overhead
        
        # Add size for nodes
        for node_id, node_data in self.nodes.items():
            memory += len(node_id) * 2  # String size
            memory += 500  # Average node size
            memory += sum(len(str(k)) + len(str(v)) for k, v in node_data.items())
        
        # Add size for edges
        memory += len(self.edges) * 100  # Rough estimate per edge
        
        # Add size for other lists
        memory += len(self.entry_points) * 20
        memory += len(self.exit_points) * 20
        
        # Add size for cache if exists
        if hasattr(self.find_execution_path, "cache"):
            memory += len(self.find_execution_path.cache) * 500  # Rough estimate
        
        return memory
    
    def cleanup(self) -> int:
        """Clean up any unnecessary data, return bytes saved"""
        bytes_saved = 0
        
        # Clear cached path finding results
        if hasattr(self.find_execution_path, "cache"):
            cache_size = len(self.find_execution_path.cache) * 500  # Rough estimate
            self.find_execution_path.cache = {}
            bytes_saved += cache_size
        
        return bytes_saved
    
    def iter_nodes(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Iterate over nodes in a memory-efficient way"""
        for node_id, node_data in self.nodes.items():
            yield (node_id, node_data)
    
    def iter_edges(self) -> Iterator[Dict[str, Any]]:
        """Iterate over edges in a memory-efficient way"""
        for edge in self.edges:
            yield edge
    
    def validate_graph(self) -> Tuple[bool, Optional[str]]:
        """
        Verify that the graph is valid
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for disconnected nodes
        reachable = set()
        
        # Start from entry points
        for entry in self.entry_points:
            self._mark_reachable(entry, reachable)
        
        # Check if all nodes are reachable
        unreachable = set(self.nodes.keys()) - reachable
        if unreachable:
            return False, f"Graph contains unreachable nodes: {unreachable}"
        
        # Check if exit points are reachable
        for exit_point in self.exit_points:
            if exit_point not in reachable:
                return False, f"Exit point {exit_point} is not reachable"
        
        # Graph is valid
        return True, None
    
    def _mark_reachable(self, node_id: str, reachable: Set[str]) -> None:
        """Mark a node and all its descendants as reachable"""
        if node_id in reachable:
            return
            
        reachable.add(node_id)
        
        # Find neighbors
        for edge in self.edges:
            if edge["from"] == node_id:
                self._mark_reachable(edge["to"], reachable)
    
    def _find_all_paths(self, start: str, end: str, path: List[str] = None) -> List[List[str]]:
        """Find all paths between two nodes"""
        if path is None:
            path = []
        
        path = path + [start]
        
        if start == end:
            return [path]
        
        if start not in self.nodes:
            return []
        
        paths = []
        
        # Find outgoing edges
        for edge in self.edges:
            if edge["from"] == start and edge["to"] not in path:
                new_paths = self._find_all_paths(edge["to"], end, path)
                for new_path in new_paths:
                    paths.append(new_path)
        
        return paths
    
    def _score_path(self, path: List[str], context: Dict[str, Any], goal: Dict[str, Any]) -> float:
        """Score a path based on context and goal"""
        score = 0.5  # Base score
        
        # Check context match for each node
        for node_id in path:
            node_data = self.nodes.get(node_id, {})
            preconditions = node_data.get("preconditions", {})
            
            # Check if preconditions match context
            matches = 0
            total = len(preconditions)
            
            for key, value in preconditions.items():
                if key in context:
                    if isinstance(value, (list, tuple, set)):
                        if context[key] in value:
                            matches += 1
                    elif isinstance(value, dict) and "min" in value and "max" in value:
                        if value["min"] <= context[key] <= value["max"]:
                            matches += 1
                    elif context[key] == value:
                        matches += 1
            
            # Add to score based on precondition match percentage
            if total > 0:
                score += 0.1 * (matches / total)
        
        # Check if path achieves goal
        last_node = self.nodes.get(path[-1], {})
        postconditions = last_node.get("postconditions", {})
        
        goal_matches = 0
        goal_total = len(goal)
        
        for key, value in goal.items():
            if key in postconditions:
                if isinstance(value, (list, tuple, set)):
                    if postconditions[key] in value:
                        goal_matches += 1
                elif isinstance(value, dict) and "min" in value and "max" in value:
                    if value["min"] <= postconditions[key] <= value["max"]:
                        goal_matches += 1
                elif postconditions[key] == value:
                    goal_matches += 1
        
        # Add to score based on goal match percentage
        if goal_total > 0:
            goal_score = goal_matches / goal_total
            score += 0.4 * goal_score  # Goal achievement is important
        
        return score
    
    @classmethod
    def from_procedure(cls, procedure: Procedure) -> 'ProcedureGraph':
        """Convert a standard procedure to a graph representation"""
        with custom_span("create_procedure_graph", {"procedure_id": procedure.id}):
            graph = cls()
            
            # Create nodes for each step
            for i, step in enumerate(procedure.steps):
                node_id = f"node_{step['id']}"
                
                # Extract preconditions and postconditions
                preconditions = step.get("preconditions", {})
                postconditions = step.get("postconditions", {})
                
                # Create node
                graph.nodes[node_id] = {
                    "step_id": step["id"],
                    "function": step["function"],
                    "parameters": step.get("parameters", {}),
                    "description": step.get("description", f"Step {i+1}"),
                    "preconditions": preconditions,
                    "postconditions": postconditions
                }
                
                # First step is an entry point
                if i == 0:
                    graph.entry_points.append(node_id)
                
                # Last step is an exit point
                if i == len(procedure.steps) - 1:
                    graph.exit_points.append(node_id)
            
            # Create edges for sequential execution
            for i in range(len(procedure.steps) - 1):
                from_id = f"node_{procedure.steps[i]['id']}"
                to_id = f"node_{procedure.steps[i+1]['id']}"
                
                graph.edges.append({
                    "from": from_id,
                    "to": to_id,
                    "properties": {}
                })
            
            return graph
