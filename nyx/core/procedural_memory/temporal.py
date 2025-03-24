# nyx/core/procedural_memory/temporal.py

import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from pydantic import BaseModel, Field

from ..models import Procedure

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
    
    def add_node(self, node: TemporalNode) -> None:
        """Add a node to the graph"""
        self.nodes[node.id] = node
        self.last_updated = datetime.datetime.now().isoformat()
    
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
    
    def add_node(self, node_id: str, data: Dict[str, Any]) -> None:
        """Add a node to the graph"""
        self.nodes[node_id] = data
    
    def add_edge(self, from_id: str, to_id: str, properties: Dict[str, Any] = None) -> None:
        """Add an edge to the graph"""
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
        graph = cls()
        
        # Create nodes for each step
        for i, step in enumerate(procedure.steps):
            node_id = f"node_{step['id']}"
            
            # Extract preconditions and postconditions
            preconditions = step.get("preconditions", {})
            postconditions = step.get("postconditions", {})
            
            # Create node
            graph.add_node(node_id, {
                "step_id": step["id"],
                "function": step["function"],
                "parameters": step.get("parameters", {}),
                "description": step.get("description", f"Step {i+1}"),
                "preconditions": preconditions,
                "postconditions": postconditions
            })
            
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
            
            graph.add_edge(from_id, to_id)
        
        return graph
