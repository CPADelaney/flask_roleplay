# nyx/eternal/metacognitive_controller.py

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import math
import random
import psutil

logger = logging.getLogger(__name__)

class CognitiveProcess:
    """Represents a cognitive process being monitored"""
    
    def __init__(self, process_id: str, name: str, type: str,
               priority: float = 0.5, resource_allocation: float = 0.1):
        self.id = process_id
        self.name = name
        self.type = type  # reasoning, memory, perception, planning, etc.
        self.priority = priority  # 0.0 to 1.0
        self.resource_allocation = resource_allocation  # 0.0 to 1.0
        self.performance_metrics = {}
        self.bottlenecks = []
        self.dependencies = []
        self.start_time = datetime.now()
        self.last_activity = datetime.now()
        self.total_runtime = 0.0
        self.status = "idle"  # idle, running, blocked, completed, failed
    
    def update_performance(self, metrics: Dict[str, float]) -> None:
        """Update performance metrics for this process"""
        self.last_activity = datetime.now()
        
        # Update all provided metrics
        for key, value in metrics.items():
            self.performance_metrics[key] = value
        
        # Update derived metrics
        total_runtime = (datetime.now() - self.start_time).total_seconds()
        self.total_runtime = total_runtime
        
        if "throughput" in metrics and total_runtime > 0:
            self.performance_metrics["efficiency"] = metrics["throughput"] / total_runtime
    
    def add_bottleneck(self, description: str, severity: float,
                    resource_type: str = "general") -> None:
        """Add a bottleneck for this process"""
        self.bottlenecks.append({
            "description": description,
            "severity": severity,
            "resource_type": resource_type,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent bottlenecks
        if len(self.bottlenecks) > 10:
            self.bottlenecks = self.bottlenecks[-10:]
    
    def add_dependency(self, process_id: str, importance: float = 0.5) -> None:
        """Add a dependency on another process"""
        if process_id not in [d["process_id"] for d in self.dependencies]:
            self.dependencies.append({
                "process_id": process_id,
                "importance": importance
            })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "priority": self.priority,
            "resource_allocation": self.resource_allocation,
            "performance_metrics": self.performance_metrics,
            "bottlenecks": self.bottlenecks,
            "dependencies": self.dependencies,
            "start_time": self.start_time.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "total_runtime": self.total_runtime,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CognitiveProcess':
        """Create from dictionary representation"""
        process = cls(
            process_id=data["id"],
            name=data["name"],
            type=data["type"],
            priority=data["priority"],
            resource_allocation=data["resource_allocation"]
        )
        
        process.performance_metrics = data["performance_metrics"]
        process.bottlenecks = data["bottlenecks"]
        process.dependencies = data["dependencies"]
        process.start_time = datetime.fromisoformat(data["start_time"])
        process.last_activity = datetime.fromisoformat(data["last_activity"])
        process.total_runtime = data["total_runtime"]
        process.status = data["status"]
        
        return process

class MentalModel:
    """Represents a mental model of a domain or concept"""
    
    def __init__(self, model_id: str, name: str, domain: str,
               confidence: float = 0.5, complexity: float = 0.5):
        self.id = model_id
        self.name = name
        self.domain = domain
        self.confidence = confidence  # 0.0 to 1.0
        self.complexity = complexity  # 0.0 to 1.0
        self.elements = {}  # Key concepts in this model
        self.relations = {}  # Relations between concepts
        self.last_updated = datetime.now()
        self.last_used = datetime.now()
        self.usage_count = 0
        self.accuracy_history = []
    
    def add_element(self, key: str, description: str,
                 importance: float = 0.5) -> None:
        """Add a key element to the mental model"""
        self.elements[key] = {
            "description": description,
            "importance": importance,
            "added": datetime.now().isoformat()
        }
        self.last_updated = datetime.now()
    
    def add_relation(self, source: str, target: str, 
                   type: str, strength: float = 0.5) -> None:
        """Add a relation between elements"""
        if source not in self.elements or target not in self.elements:
            return  # Can't add relation for non-existent elements
        
        relation_key = f"{source}_{target}"
        self.relations[relation_key] = {
            "source": source,
            "target": target,
            "type": type,
            "strength": strength,
            "added": datetime.now().isoformat()
        }
        self.last_updated = datetime.now()
    
    def record_usage(self, accuracy: Optional[float] = None) -> None:
        """Record usage of this mental model"""
        self.last_used = datetime.now()
        self.usage_count += 1
        
        if accuracy is not None:
            self.accuracy_history.append({
                "timestamp": datetime.now().isoformat(),
                "accuracy": accuracy
            })
            
            # Update overall confidence based on accuracy history
            if self.accuracy_history:
                avg_accuracy = sum(entry["accuracy"] for entry in self.accuracy_history) / len(self.accuracy_history)
                self.confidence = (self.confidence * 0.7) + (avg_accuracy * 0.3)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain,
            "confidence": self.confidence,
            "complexity": self.complexity,
            "elements": self.elements,
            "relations": self.relations,
            "last_updated": self.last_updated.isoformat(),
            "last_used": self.last_used.isoformat(),
            "usage_count": self.usage_count,
            "accuracy_history": self.accuracy_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MentalModel':
        """Create from dictionary representation"""
        model = cls(
            model_id=data["id"],
            name=data["name"],
            domain=data["domain"],
            confidence=data["confidence"],
            complexity=data["complexity"]
        )
        
        model.elements = data["elements"]
        model.relations = data["relations"]
        model.last_updated = datetime.fromisoformat(data["last_updated"])
        model.last_used = datetime.fromisoformat(data["last_used"])
        model.usage_count = data["usage_count"]
        model.accuracy_history = data["accuracy_history"]
        
        return model

class MetaCognitiveController:
    """Advanced meta-cognitive system that orchestrates all cognitive processes"""
    
    def __init__(self):
        self.cognitive_processes = {}  # Track all cognitive processes
        self.resource_allocation = {}  # Dynamic resource allocation
        self.process_performance = {}  # Track performance of each process
        self.mental_models = {}        # Store mental models of the world
        self.attention_focus = None    # Current attention focus
        
        # Internal state
        self.execution_context = {}    # Current execution context
        self.error_logs = []           # Logs of cognitive errors
        self.insights = []             # Meta-cognitive insights
        self.performance_history = []  # Historical performance data
        
        # System metrics
        self.system_metrics = {
            "start_time": datetime.now(),
            "total_runtime": 0.0,
            "cycles_completed": 0,
            "total_processes": 0,
            "resource_usage": {
                "cpu": 0.0,
                "memory": 0,
                "io": 0
            },
            "average_cycle_time": 0.0,
            "error_rate": 0.0
        }
        
        # Configuration
        self.config = {
            "evaluation_interval": 10,  # Cycles between evaluations
            "min_cycle_time": 0.1,      # Minimum seconds per cycle
            "max_cycle_time": 5.0,      # Maximum seconds per cycle
            "resource_reallocation_threshold": 0.2,  # Minimum change to trigger reallocation
            "bottleneck_severity_threshold": 0.7,    # Severity threshold for critical bottlenecks
            "attention_shift_threshold": 0.8,        # Threshold to shift attention focus
            "max_concurrent_processes": 10           # Maximum processes running concurrently
        }
        
        # Next ID counters
        self.next_process_id = 1
        self.next_model_id = 1
    
    async def evaluate_cognition(self) -> Dict[str, Any]:
        """Evaluate performance of all cognitive processes and reallocate resources"""
        cycle_start = time.time()
        
        # Collect performance metrics from all cognitive systems
        performance_data = await self._collect_performance_data()
        
        # Identify bottlenecks and underperforming processes
        bottlenecks = self._identify_bottlenecks(performance_data)
        
        # Analyze cognitive strategies effectiveness
        strategy_analysis = self._analyze_cognitive_strategies()
        
        # Reallocate resources based on needs and goals
        resource_changes = self._reallocate_resources(bottlenecks, strategy_analysis)
        
        # Generate meta-cognitive insights
        insights = self._generate_metacognitive_insights()
        
        # Update system metrics
        self._update_system_metrics(cycle_start)
        
        # Store performance history
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "cycle": self.system_metrics["cycles_completed"],
            "performance_data": performance_data,
            "bottlenecks": bottlenecks,
            "resource_changes": resource_changes
        })
        
        # Limit history size
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        return {
            "bottlenecks": bottlenecks,
            "strategy_analysis": strategy_analysis,
            "resource_changes": resource_changes,
            "insights": insights,
            "system_metrics": self.system_metrics
        }
    
    async def _collect_performance_data(self) -> Dict[str, Dict[str, Any]]:
        """Collect performance data from all cognitive processes"""
        performance_data = {}
        
        for process_id, process in self.cognitive_processes.items():
            # Skip inactive processes
            if process.status in ["completed", "failed"]:
                continue
                
            # Get current metrics
            metrics = process.performance_metrics.copy()
            
            # Add derived metrics
            metrics["runtime"] = process.total_runtime
            metrics["efficiency"] = self._calculate_process_efficiency(process)
            metrics["resource_utilization"] = self._calculate_resource_utilization(process)
            metrics["bottleneck_count"] = len(process.bottlenecks)
            
            # Store in performance data
            performance_data[process_id] = {
                "process": process.to_dict(),
                "metrics": metrics
            }
        
        return performance_data
    
    def _calculate_process_efficiency(self, process: CognitiveProcess) -> float:
        """Calculate efficiency of a cognitive process"""
        # Default efficiency
        efficiency = 0.5
        
        # If throughput and runtime are available, use them
        if "throughput" in process.performance_metrics and process.total_runtime > 0:
            throughput = process.performance_metrics["throughput"]
            runtime = process.total_runtime
            
            # Normalize by resource allocation
            resource_factor = max(0.1, process.resource_allocation)
            efficiency = throughput / (runtime * resource_factor)
            
            # Cap at reasonable values
            efficiency = min(1.0, max(0.0, efficiency))
            
        return efficiency
    
    def _calculate_resource_utilization(self, process: CognitiveProcess) -> float:
        """Calculate resource utilization of a cognitive process"""
        allocation = process.resource_allocation
        
        # Check if process reports its own utilization
        if "resource_utilization" in process.performance_metrics:
            return process.performance_metrics["resource_utilization"]
        
        # Otherwise, approximate based on metrics
        utilization = 0.5  # Default value
        
        if "cpu_usage" in process.performance_metrics:
            cpu_usage = process.performance_metrics["cpu_usage"]
            utilization = cpu_usage / allocation if allocation > 0 else 1.0
        
        # Cap at reasonable values
        utilization = min(1.0, max(0.0, utilization))
        
        return utilization
    
    def _identify_bottlenecks(self, performance_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify bottlenecks and underperforming processes"""
        bottlenecks = []
        
        for process_id, data in performance_data.items():
            process = data["process"]
            metrics = data["metrics"]
            
            # Check for high resource utilization
            if metrics.get("resource_utilization", 0) > 0.9:
                bottlenecks.append({
                    "process_id": process_id,
                    "type": "resource_utilization",
                    "severity": 0.8,
                    "description": f"Process {process['name']} has high resource utilization",
                    "metrics": {
                        "resource_utilization": metrics.get("resource_utilization", 0)
                    }
                })
            
            # Check for low efficiency
            if metrics.get("efficiency", 0) < 0.3:
                bottlenecks.append({
                    "process_id": process_id,
                    "type": "low_efficiency",
                    "severity": 0.7,
                    "description": f"Process {process['name']} has low efficiency",
                    "metrics": {
                        "efficiency": metrics.get("efficiency", 0)
                    }
                })
            
            # Check for process-specific bottlenecks
            if process["bottlenecks"]:
                # Get most recent bottleneck
                recent_bottleneck = process["bottlenecks"][-1]
                
                bottlenecks.append({
                    "process_id": process_id,
                    "type": "process_bottleneck",
                    "severity": recent_bottleneck["severity"],
                    "description": recent_bottleneck["description"],
                    "resource_type": recent_bottleneck["resource_type"]
                })
            
            # Check for blocked dependencies
            if process["status"] == "blocked":
                for dependency in process["dependencies"]:
                    dep_id = dependency["process_id"]
                    dep_importance = dependency["importance"]
                    
                    if dep_id in self.cognitive_processes:
                        dep_process = self.cognitive_processes[dep_id]
                        
                        if dep_process.status != "completed":
                            bottlenecks.append({
                                "process_id": process_id,
                                "type": "blocked_dependency",
                                "severity": dep_importance,
                                "description": f"Process {process['name']} blocked by dependency on {dep_process.name}",
                                "dependency_id": dep_id
                            })
        
        # Sort bottlenecks by severity
        bottlenecks.sort(key=lambda x: x["severity"], reverse=True)
        
        return bottlenecks
    
    def _analyze_cognitive_strategies(self) -> Dict[str, Any]:
        """Analyze effectiveness of current cognitive strategies"""
        analysis = {
            "overall_effectiveness": 0.0,
            "strategy_evaluations": {},
            "adaptation_rate": 0.0,
            "learning_effectiveness": 0.0,
            "recommended_changes": []
        }
        
        # Group processes by type
        processes_by_type = {}
        for process in self.cognitive_processes.values():
            if process.type not in processes_by_type:
                processes_by_type[process.type] = []
                
            processes_by_type[process.type].append(process)
        
        # Evaluate effectiveness by process type
        total_score = 0.0
        evaluated_types = 0
        
        for process_type, processes in processes_by_type.items():
            # Skip if no active processes
            active_processes = [p for p in processes if p.status not in ["completed", "failed"]]
            if not active_processes:
                continue
                
            # Calculate average performance metrics
            avg_metrics = {}
            for metric in ["efficiency", "throughput", "accuracy", "response_time"]:
                values = []
                
                for process in active_processes:
                    if metric in process.performance_metrics:
                        values.append(process.performance_metrics[metric])
                
                if values:
                    avg_metrics[metric] = sum(values) / len(values)
            
            # Calculate effectiveness score
            effectiveness_score = 0.5  # Default score
            
            if "efficiency" in avg_metrics:
                effectiveness_score = avg_metrics["efficiency"]
            elif "accuracy" in avg_metrics and "response_time" in avg_metrics:
                # Balance accuracy and speed
                norm_time = min(1.0, 1.0 / (1.0 + avg_metrics["response_time"]))
                effectiveness_score = 0.7 * avg_metrics["accuracy"] + 0.3 * norm_time
            
            # Add to analysis
            analysis["strategy_evaluations"][process_type] = {
                "effectiveness": effectiveness_score,
                "average_metrics": avg_metrics,
                "process_count": len(active_processes)
            }
            
            # Add to overall score
            total_score += effectiveness_score
            evaluated_types += 1
            
            # Generate recommendations
            if effectiveness_score < 0.4:
                analysis["recommended_changes"].append({
                    "process_type": process_type,
                    "current_effectiveness": effectiveness_score,
                    "recommendation": f"Improve {process_type} strategy - consider new algorithms or increasing resources"
                })
        
        # Calculate overall effectiveness
        if evaluated_types > 0:
            analysis["overall_effectiveness"] = total_score / evaluated_types
        
        # Calculate adaptation and learning metrics
        if len(self.performance_history) >= 5:
            # Check how metrics have improved over time
            improvement_rates = []
            
            for process_type in processes_by_type:
                if process_type in analysis["strategy_evaluations"]:
                    current_score = analysis["strategy_evaluations"][process_type]["effectiveness"]
                    
                    # Get score from 5 cycles ago if available
                    history_idx = max(0, len(self.performance_history) - 5)
                    
                    if history_idx < len(self.performance_history):
                        historical_data = self.performance_history[history_idx]
                        
                        # Find processes of this type in historical data
                        historical_processes = []
                        for p_id, p_data in historical_data["performance_data"].items():
                            if p_data["process"]["type"] == process_type:
                                historical_processes.append(p_data)
                        
                        if historical_processes:
                            # Calculate historical effectiveness
                            hist_effectiveness = 0.0
                            for p_data in historical_processes:
                                metrics = p_data["metrics"]
                                if "efficiency" in metrics:
                                    hist_effectiveness += metrics["efficiency"]
                                
                            hist_effectiveness /= len(historical_processes)
                            
                            # Calculate improvement rate
                            if hist_effectiveness > 0:
                                improvement = (current_score - hist_effectiveness) / hist_effectiveness
                                improvement_rates.append(improvement)
            
            if improvement_rates:
                avg_improvement = sum(improvement_rates) / len(improvement_rates)
                analysis["adaptation_rate"] = max(0.0, min(1.0, avg_improvement + 0.5))  # Normalize to 0-1
                analysis["learning_effectiveness"] = analysis["adaptation_rate"]
        
        return analysis
    
    def _reallocate_resources(self, bottlenecks: List[Dict[str, Any]], 
                          strategy_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Reallocate resources based on bottlenecks and strategy analysis"""
        # Initialize with current allocations
        new_allocations = {
            process_id: process.resource_allocation
            for process_id, process in self.cognitive_processes.items()
            if process.status in ["idle", "running", "blocked"]
        }
        
        # No reallocation needed if no active processes
        if not new_allocations:
            return {}
        
        # Handle critical bottlenecks first
        critical_bottlenecks = [b for b in bottlenecks 
                              if b["severity"] >= self.config["bottleneck_severity_threshold"]]
        
        for bottleneck in critical_bottlenecks:
            process_id = bottleneck["process_id"]
            if process_id in new_allocations:
                current = new_allocations[process_id]
                
                # Increase allocation for resource bottlenecks
                if bottleneck["type"] in ["resource_utilization", "process_bottleneck"]:
                    new_allocations[process_id] = min(0.8, current * 1.5)
        
        # Adjust based on strategy effectiveness
        for process_type, eval_data in strategy_analysis["strategy_evaluations"].items():
            effectiveness = eval_data["effectiveness"]
            
            # Get processes of this type
            type_processes = [p_id for p_id, p in self.cognitive_processes.items()
                           if p.type == process_type and p.status in ["idle", "running", "blocked"]]
            
            for process_id in type_processes:
                if process_id in new_allocations:
                    current = new_allocations[process_id]
                    
                    # Adjust based on effectiveness
                    if effectiveness < 0.3:
                        # Low effectiveness - increase resources
                        new_allocations[process_id] = min(0.7, current * 1.3)
                    elif effectiveness > 0.8:
                        # High effectiveness - may be able to reduce resources
                        new_allocations[process_id] = max(0.1, current * 0.9)
        
        # Normalize allocations to sum to 1.0
        total_allocation = sum(new_allocations.values())
        
        if total_allocation > 0:
            for process_id in new_allocations:
                new_allocations[process_id] /= total_allocation
        
        # Calculate changes from current allocation
        changes = {}
        
        for process_id, new_allocation in new_allocations.items():
            process = self.cognitive_processes.get(process_id)
            if process:
                current = process.resource_allocation
                change = new_allocation - current
                
                # Only record significant changes
                if abs(change) >= self.config["resource_reallocation_threshold"]:
                    changes[process_id] = change
                    
                    # Update process allocation
                    process.resource_allocation = new_allocation
        
        return changes
    
    def _generate_metacognitive_insights(self) -> List[Dict[str, Any]]:
        """Generate insights about cognitive processes and patterns"""
        insights = []
        
        # Check for recurring bottlenecks
        recurring_bottlenecks = self._identify_recurring_bottlenecks()
        
        for bottleneck in recurring_bottlenecks:
            insights.append({
                "type": "recurring_bottleneck",
                "process_id": bottleneck["process_id"],
                "description": f"Recurring bottleneck in {bottleneck['process_name']}: {bottleneck['description']}",
                "occurrences": bottleneck["occurrences"],
                "severity": bottleneck["avg_severity"],
                "recommendation": bottleneck["recommendation"]
            })
        
        # Check for inefficient process dependencies
        inefficient_dependencies = self._identify_inefficient_dependencies()
        
        for dependency in inefficient_dependencies:
            insights.append({
                "type": "inefficient_dependency",
                "source_id": dependency["source_id"],
                "target_id": dependency["target_id"],
                "description": f"Inefficient dependency from {dependency['source_name']} to {dependency['target_name']}",
                "impact": dependency["impact"],
                "recommendation": dependency["recommendation"]
            })
        
        # Check for resource utilization patterns
        resource_patterns = self._identify_resource_patterns()
        
        for pattern in resource_patterns:
            insights.append({
                "type": "resource_pattern",
                "resource_type": pattern["resource_type"],
                "description": pattern["description"],
                "trend": pattern["trend"],
                "impact": pattern["impact"],
                "recommendation": pattern["recommendation"]
            })
        
        # Generate insights about learning and adaptation
        if len(self.performance_history) >= 10:
            learning_insights = self._analyze_learning_trends()
            insights.extend(learning_insights)
        
        return insights
    
    def _identify_recurring_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify recurring bottlenecks across cycles"""
        bottleneck_counts = {}
        
        # Count bottleneck occurrences across history
        for history_entry in self.performance_history:
            for bottleneck in history_entry.get("bottlenecks", []):
                process_id = bottleneck.get("process_id")
                bottleneck_type = bottleneck.get("type")
                description = bottleneck.get("description", "")
                severity = bottleneck.get("severity", 0.5)
                
                # Create key for this bottleneck type
                key = f"{process_id}_{bottleneck_type}"
                
                if key not in bottleneck_counts:
                    bottleneck_counts[key] = {
                        "process_id": process_id,
                        "type": bottleneck_type,
                        "description": description,
                        "occurrences": 0,
                        "total_severity": 0.0,
                        "cycles": []
                    }
                
                bottleneck_counts[key]["occurrences"] += 1
                bottleneck_counts[key]["total_severity"] += severity
                bottleneck_counts[key]["cycles"].append(history_entry.get("cycle"))
        
        # Filter to significant recurring bottlenecks
        recurring = []
        threshold = max(3, len(self.performance_history) // 3)  # At least 3 occurrences or 1/3 of history
        
        for key, data in bottleneck_counts.items():
            if data["occurrences"] >= threshold:
                # Get process name
                process_name = "Unknown Process"
                if data["process_id"] in self.cognitive_processes:
                    process_name = self.cognitive_processes[data["process_id"]].name
                
                avg_severity = data["total_severity"] / data["occurrences"]
                
                # Generate recommendation based on bottleneck type
                recommendation = "Consider allocating more resources to this process"
                
                if data["type"] == "resource_utilization":
                    recommendation = "Optimize resource usage or increase allocation"
                elif data["type"] == "low_efficiency":
                    recommendation = "Review algorithm efficiency or simplify process"
                elif data["type"] == "blocked_dependency":
                    recommendation = "Restructure dependencies or prioritize blocking processes"
                
                recurring.append({
                    "process_id": data["process_id"],
                    "process_name": process_name,
                    "type": data["type"],
                    "description": data["description"],
                    "occurrences": data["occurrences"],
                    "avg_severity": avg_severity,
                    "recommendation": recommendation
                })
        
        # Sort by severity * occurrences
        recurring.sort(key=lambda x: x["avg_severity"] * x["occurrences"], reverse=True)
        
        return recurring
    
    def _identify_inefficient_dependencies(self) -> List[Dict[str, Any]]:
        """Identify inefficient dependencies between processes"""
        inefficient_dependencies = []
        
        # Build dependency graph
        dependency_graph = {}
        for process_id, process in self.cognitive_processes.items():
            for dependency in process.dependencies:
                dep_id = dependency["process_id"]
                importance = dependency["importance"]
                
                if process_id not in dependency_graph:
                    dependency_graph[process_id] = []
                
                dependency_graph[process_id].append({
                    "target_id": dep_id,
                    "importance": importance
                })
        
        # Check for inefficiencies
        for source_id, dependencies in dependency_graph.items():
            source_process = self.cognitive_processes.get(source_id)
            if not source_process:
                continue
                
            for dep in dependencies:
                target_id = dep["target_id"]
                importance = dep["importance"]
                
                target_process = self.cognitive_processes.get(target_id)
                if not target_process:
                    continue
                
                # Check for performance issues
                inefficiency_score = 0.0
                reasons = []
                
                # Check for frequent blocking
                if source_process.status == "blocked" and target_process.status != "completed":
                    inefficiency_score += 0.4
                    reasons.append("Frequent blocking")
                
                # Check for high latency in dependent process
                if "response_time" in target_process.performance_metrics:
                    if target_process.performance_metrics["response_time"] > 2.0:  # Arbitrary threshold
                        inefficiency_score += 0.3
                        reasons.append("High dependency latency")
                
                # Check for high importance but low performance
                if importance > 0.7:
                    efficiency = self._calculate_process_efficiency(target_process)
                    if efficiency < 0.4:  # Low efficiency
                        inefficiency_score += importance * 0.4
                        reasons.append("High importance but low performance")
                
                # Add if significant inefficiency found
                if inefficiency_score > 0.3:
                    recommendation = ""
                    
                    if "Frequent blocking" in reasons:
                        recommendation = "Consider making dependency asynchronous or preemptive"
                    elif "High dependency latency" in reasons:
                        recommendation = "Optimize dependent process or use caching"
                    else:
                        recommendation = "Reconsider dependency structure or improve dependent process"
                    
                    inefficient_dependencies.append({
                        "source_id": source_id,
                        "source_name": source_process.name,
                        "target_id": target_id,
                        "target_name": target_process.name,
                        "importance": importance,
                        "inefficiency_score": inefficiency_score,
                        "reasons": reasons,
                        "impact": inefficiency_score * importance,
                        "recommendation": recommendation
                    })
        
        # Sort by impact
        inefficient_dependencies.sort(key=lambda x: x["impact"], reverse=True)
        
        return inefficient_dependencies
    
    def _identify_resource_patterns(self) -> List[Dict[str, Any]]:
        """Identify patterns in resource utilization"""
        patterns = []
        
        # Skip if not enough history
        if len(self.performance_history) < 5:
            return patterns
        
        # Analyze CPU usage patterns
        cpu_usage = [entry["system_metrics"]["resource_usage"]["cpu"] 
                    for entry in self.performance_history if "system_metrics" in entry]
        
        if cpu_usage:
            cpu_trend = self._calculate_resource_trend(cpu_usage)
            
            if cpu_trend["direction"] != "stable":
                patterns.append({
                    "resource_type": "cpu",
                    "description": f"CPU usage is {cpu_trend['direction']}",
                    "trend": cpu_trend,
                    "impact": "High CPU usage may cause slowdowns or throttling",
                    "recommendation": "Optimize CPU-intensive processes" if cpu_trend["direction"] == "increasing" else "Current CPU optimizations are effective"
                })
        
        # Analyze memory usage patterns
        memory_usage = [entry["system_metrics"]["resource_usage"]["memory"] 
                      for entry in self.performance_history if "system_metrics" in entry]
        
        if memory_usage:
            memory_trend = self._calculate_resource_trend(memory_usage)
            
            if memory_trend["direction"] != "stable":
                patterns.append({
                    "resource_type": "memory",
                    "description": f"Memory usage is {memory_trend['direction']}",
                    "trend": memory_trend,
                    "impact": "Increasing memory usage may lead to resource exhaustion",
                    "recommendation": "Check for memory leaks or implement garbage collection" if memory_trend["direction"] == "increasing" else "Current memory optimizations are effective"
                })
        
        # Analyze I/O usage patterns
        io_usage = [entry["system_metrics"]["resource_usage"]["io"] 
                  for entry in self.performance_history if "system_metrics" in entry]
        
        if io_usage:
            io_trend = self._calculate_resource_trend(io_usage)
            
            if io_trend["direction"] != "stable":
                patterns.append({
                    "resource_type": "io",
                    "description": f"I/O usage is {io_trend['direction']}",
                    "trend": io_trend,
                    "impact": "High I/O usage may cause system bottlenecks",
                    "recommendation": "Implement caching or batching for I/O operations" if io_trend["direction"] == "increasing" else "Current I/O optimizations are effective"
                })
        
        return patterns
    
    def _calculate_resource_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend from a series of resource values"""
        if len(values) < 2:
            return {"direction": "stable", "magnitude": 0.0}
            
        # Calculate linear regression
        n = len(values)
        x = list(range(n))
        mean_x = sum(x) / n
        mean_y = sum(values) / n
        
        numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        
        if denominator == 0:
            return {"direction": "stable", "magnitude": 0.0}
            
        slope = numerator / denominator
        
        # Normalize slope based on mean value
        if mean_y != 0:
            normalized_slope = slope / abs(mean_y)
        else:
            normalized_slope = slope
            
        # Determine direction and magnitude
        if abs(normalized_slope) < 0.05:
            direction = "stable"
        elif normalized_slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
            
        return {
            "direction": direction,
            "magnitude": abs(normalized_slope),
            "slope": slope,
            "mean": mean_y
        }
    
    def _analyze_learning_trends(self) -> List[Dict[str, Any]]:
        """Analyze trends in learning and adaptation"""
        insights = []
        
        # Calculate improvement rates over time for different process types
        process_types = set(p.type for p in self.cognitive_processes.values())
        
        for process_type in process_types:
            # Extract efficiency values across history
            efficiency_values = []
            
            for entry in self.performance_history:
                # Find processes of this type
                type_processes = []
                
                for p_id, p_data in entry.get("performance_data", {}).items():
                    if p_data["process"]["type"] == process_type:
                        type_processes.append(p_data)
                
                if type_processes:
                    # Calculate average efficiency
                    avg_efficiency = 0.0
                    count = 0
                    
                    for p_data in type_processes:
                        if "efficiency" in p_data["metrics"]:
                            avg_efficiency += p_data["metrics"]["efficiency"]
                            count += 1
                    
                    if count > 0:
                        efficiency_values.append(avg_efficiency / count)
            
            # Calculate learning trend if we have enough data
            if len(efficiency_values) >= 5:
                trend = self._calculate_resource_trend(efficiency_values)
                
                if trend["direction"] == "increasing":
                    insights.append({
                        "type": "learning_improvement",
                        "process_type": process_type,
                        "description": f"{process_type} processes are showing learning improvements",
                        "trend": trend,
                        "recommendation": "Continue current learning approach"
                    })
                elif trend["direction"] == "decreasing":
                    insights.append({
                        "type": "learning_degradation",
                        "process_type": process_type,
                        "description": f"{process_type} processes are showing degrading performance",
                        "trend": trend,
                        "recommendation": "Review learning parameters or implement more exploration"
                    })
                elif trend["magnitude"] < 0.01:
                    insights.append({
                        "type": "learning_plateau",
                        "process_type": process_type,
                        "description": f"{process_type} processes have plateaued in learning",
                        "trend": trend,
                        "recommendation": "Introduce new learning challenges or adjust learning rate"
                    })
        
        return insights
    
    def _update_system_metrics(self, cycle_start: float) -> None:
        """Update system metrics based on current state"""
        now = time.time()
        cycle_time = now - cycle_start
        
        # Update system metrics
        self.system_metrics["cycles_completed"] += 1
        
        # Update total runtime
        runtime = (datetime.now() - self.system_metrics["start_time"]).total_seconds()
        self.system_metrics["total_runtime"] = runtime
        
        # Update process count
        self.system_metrics["total_processes"] = len(self.cognitive_processes)
        
        # Update average cycle time with exponential moving average
        alpha = 0.2  # Weight for current cycle
        current_avg = self.system_metrics["average_cycle_time"]
        new_avg = (1 - alpha) * current_avg + alpha * cycle_time
        self.system_metrics["average_cycle_time"] = new_avg
        
        # Update resource usage
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.Process().memory_info()
            io_counters = psutil.Process().io_counters() if hasattr(psutil.Process(), 'io_counters') else None
            
            self.system_metrics["resource_usage"]["cpu"] = cpu_percent
            self.system_metrics["resource_usage"]["memory"] = memory_info.rss
            
            if io_counters:
                # Calculate IO rate since last update
                if hasattr(self, "_last_io_counters"):
                    last_io = self._last_io_counters
                    io_rate = ((io_counters.read_bytes + io_counters.write_bytes) - 
                             (last_io.read_bytes + last_io.write_bytes)) / cycle_time
                    self.system_metrics["resource_usage"]["io"] = io_rate
                
                self._last_io_counters = io_counters
            
        except Exception as e:
            logger.warning(f"Error updating resource metrics: {e}")
        
        # Update error rate
        total_errors = len(self.error_logs)
        if runtime > 0:
            self.system_metrics["error_rate"] = total_errors / runtime
    
    async def register_cognitive_process(self, name: str, type: str,
                               priority: float = 0.5, 
                               resource_allocation: float = 0.1) -> str:
        """Register a new cognitive process for monitoring"""
        # Generate process ID
        process_id = f"process_{self.next_process_id}"
        self.next_process_id += 1
        
        # Create process
        process = CognitiveProcess(
            process_id=process_id,
            name=name,
            type=type,
            priority=priority,
            resource_allocation=resource_allocation
        )
        
        # Add to processes
        self.cognitive_processes[process_id] = process
        
        return process_id
    
    async def update_process_status(self, process_id: str, status: str,
                           metrics: Optional[Dict[str, float]] = None) -> bool:
        """Update status and metrics for a cognitive process"""
        if process_id not in self.cognitive_processes:
            return False
            
        process = self.cognitive_processes[process_id]
        
        # Update status
        process.status = status
        
        # Update metrics if provided
        if metrics:
            process.update_performance(metrics)
        
        return True
    
    async def report_bottleneck(self, process_id: str, description: str, 
                        severity: float = 0.5, 
                        resource_type: str = "general") -> bool:
        """Report a bottleneck for a cognitive process"""
        if process_id not in self.cognitive_processes:
            return False
            
        process = self.cognitive_processes[process_id]
        
        # Add bottleneck
        process.add_bottleneck(
            description=description,
            severity=severity,
            resource_type=resource_type
        )
        
        return True
    
    async def add_process_dependency(self, process_id: str, 
                           dependency_id: str,
                           importance: float = 0.5) -> bool:
        """Add a dependency between cognitive processes"""
        if process_id not in self.cognitive_processes or dependency_id not in self.cognitive_processes:
            return False
            
        process = self.cognitive_processes[process_id]
        
        # Add dependency
        process.add_dependency(dependency_id, importance)
        
        return True
    
    async def create_mental_model(self, name: str, domain: str,
                         confidence: float = 0.5,
                         complexity: float = 0.5) -> str:
        """Create a new mental model"""
        # Generate model ID
        model_id = f"model_{self.next_model_id}"
        self.next_model_id += 1
        
        # Create model
        model = MentalModel(
            model_id=model_id,
            name=name,
            domain=domain,
            confidence=confidence,
            complexity=complexity
        )
        
        # Add to models
        self.mental_models[model_id] = model
        
        return model_id
    
    async def add_mental_model_element(self, model_id: str, key: str,
                             description: str,
                             importance: float = 0.5) -> bool:
        """Add an element to a mental model"""
        if model_id not in self.mental_models:
            return False
            
        model = self.mental_models[model_id]
        
        # Add element
        model.add_element(key, description, importance)
        
        return True
    
    async def add_mental_model_relation(self, model_id: str, 
                              source: str, target: str,
                              type: str, strength: float = 0.5) -> bool:
        """Add a relation between elements in a mental model"""
        if model_id not in self.mental_models:
            return False
            
        model = self.mental_models[model_id]
        
        # Add relation
        model.add_relation(source, target, type, strength)
        
        return True
    
    async def record_mental_model_usage(self, model_id: str, 
                              accuracy: Optional[float] = None) -> bool:
        """Record usage of a mental model"""
        if model_id not in self.mental_models:
            return False
            
        model = self.mental_models[model_id]
        
        # Record usage
        model.record_usage(accuracy)
        
        return True
    
    async def set_attention_focus(self, focus: Dict[str, Any]) -> None:
        """Set the current attention focus"""
        self.attention_focus = {
            "target": focus.get("target"),
            "priority": focus.get("priority", 0.5),
            "timestamp": datetime.now().isoformat(),
            "reason": focus.get("reason", ""),
            "expiration": focus.get("expiration")
        }
    
    async def clear_attention_focus(self) -> None:
        """Clear the current attention focus"""
        self.attention_focus = None
    
    async def get_attention_focus(self) -> Optional[Dict[str, Any]]:
        """Get the current attention focus"""
        return self.attention_focus
    
    async def log_cognitive_error(self, error: Dict[str, Any]) -> None:
        """Log a cognitive error"""
        self.error_logs.append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "cycle": self.system_metrics["cycles_completed"]
        })
        
        # Limit error log size
        if len(self.error_logs) > 100:
            self.error_logs = self.error_logs[-100:]
    
    async def get_cognitive_processes(self) -> Dict[str, Dict[str, Any]]:
        """Get all cognitive processes"""
        return {
            process_id: process.to_dict()
            for process_id, process in self.cognitive_processes.items()
        }
    
    async def get_mental_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all mental models"""
        return {
            model_id: model.to_dict()
            for model_id, model in self.mental_models.items()
        }
    
    async def get_performance_history(self) -> List[Dict[str, Any]]:
        """Get performance history"""
        return self.performance_history
    
    async def get_error_logs(self) -> List[Dict[str, Any]]:
        """Get error logs"""
        return self.error_logs
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return self.system_metrics
    
    async def save_state(self, file_path: str) -> bool:
        """Save current state to file"""
        try:
            state = {
                "cognitive_processes": {
                    process_id: process.to_dict()
                    for process_id, process in self.cognitive_processes.items()
                },
                "mental_models": {
                    model_id: model.to_dict()
                    for model_id, model in self.mental_models.items()
                },
                "attention_focus": self.attention_focus,
                "performance_history": self.performance_history,
                "error_logs": self.error_logs,
                "insights": self.insights,
                "system_metrics": self.system_metrics,
                "config": self.config,
                "next_process_id": self.next_process_id,
                "next_model_id": self.next_model_id,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving metacognitive state: {e}")
            return False
        
    async def load_state(self, file_path: str) -> bool:
        """Load state from file"""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Load cognitive processes
            self.cognitive_processes = {}
            for process_id, process_data in state["cognitive_processes"].items():
                self.cognitive_processes[process_id] = CognitiveProcess.from_dict(process_data)
            
            # Load mental models
            self.mental_models = {}
            for model_id, model_data in state["mental_models"].items():
                self.mental_models[model_id] = MentalModel.from_dict(model_data)
            
            # Load other attributes
            self.attention_focus = state["attention_focus"]
            self.performance_history = state["performance_history"]
            self.error_logs = state["error_logs"]
            self.insights = state["insights"]
            self.system_metrics = state["system_metrics"]
            self.config = state["config"]
            self.next_process_id = state["next_process_id"]
            self.next_model_id = state["next_model_id"]
            
            return True
        except Exception as e:
            logger.error(f"Error loading metacognitive state: {e}")
            return False
    
    async def improve_meta_parameters(self) -> Dict[str, Any]:
        """Recursively improve the meta-parameters themselves"""
        # History of parameter configurations and their performance
        param_history = self._get_parameter_history()
        
        # Analyze which parameter changes led to improvements
        param_effectiveness = self._analyze_parameter_effectiveness(param_history)
        
        # Generate new parameter configurations to try
        new_params = self._generate_parameter_candidates(param_effectiveness)
        
        # Evaluate and select the most promising configuration
        selected_params = await self._evaluate_parameter_candidates(new_params)
        
        # Apply the selected parameters
        self._apply_meta_parameters(selected_params)
        
        return selected_params
    
    def _get_parameter_history(self) -> List[Dict[str, Any]]:
        """Get history of parameter configurations and their performance"""
        if not hasattr(self, '_parameter_history'):
            self._parameter_history = []
            
            # Add current configuration as baseline
            self._parameter_history.append({
                "parameters": self.config.copy(),
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "cycle_time": self.system_metrics["average_cycle_time"],
                    "error_rate": self.system_metrics["error_rate"],
                    # Add more relevant metrics
                }
            })
            
        return self._parameter_history
    
    def _analyze_parameter_effectiveness(self, param_history: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze which parameter changes led to improvements"""
        if len(param_history) < 2:
            # Not enough history to analyze
            return {param: {"effect": 0.0, "confidence": 0.0} for param in self.config}
            
        # Analyze each parameter
        effectiveness = {}
        
        for param in self.config:
            # Skip parameters without enough variation
            values = [entry["parameters"].get(param) for entry in param_history]
            unique_values = set(values)
            
            if len(unique_values) < 2:
                effectiveness[param] = {"effect": 0.0, "confidence": 0.0}
                continue
                
            # Calculate correlation with performance
            cycle_times = [entry["metrics"].get("cycle_time", 0) for entry in param_history]
            error_rates = [entry["metrics"].get("error_rate", 0) for entry in param_history]
            
            # Correlate parameter with cycle time
            time_correlation = self._calculate_correlation(values, cycle_times)
            # Correlate parameter with error rate
            error_correlation = self._calculate_correlation(values, error_rates)
            
            # Combine correlations (negative correlation with time and error is good)
            effect = -0.5 * time_correlation - 0.5 * error_correlation
            
            # Determine confidence based on sample size and consistency
            confidence = min(1.0, len(param_history) / 10) * min(1.0, abs(effect) * 2)
            
            effectiveness[param] = {
                "effect": effect,
                "confidence": confidence
            }
        
        return effectiveness
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation coefficient between two variables"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
            
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        denominator_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        if denominator_x == 0 or denominator_y == 0:
            return 0.0
            
        return numerator / (math.sqrt(denominator_x) * math.sqrt(denominator_y))
    
    def _generate_parameter_candidates(self, param_effectiveness: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Generate new parameter configurations to try"""
        candidates = []
        current_config = self.config.copy()
        
        # Generate individual parameter variations first
        for param, effect_data in param_effectiveness.items():
            effect = effect_data["effect"]
            confidence = effect_data["confidence"]
            
            # Skip parameters with low confidence
            if confidence < 0.2:
                continue
                
            # Adjust parameter based on effect direction and confidence
            if abs(effect) > 0.1:  # Only adjust if effect is significant
                candidate = current_config.copy()
                
                # Adjust in the direction of improvement
                adjustment = effect * confidence * 0.2  # Scale adjustment by effect and confidence
                
                # Apply adjustment based on parameter type
                if param == "evaluation_interval":
                    # Integer parameter
                    current_value = candidate[param]
                    new_value = max(1, round(current_value * (1 + adjustment)))
                    candidate[param] = new_value
                elif param in ["min_cycle_time", "max_cycle_time"]:
                    # Bounded float parameter
                    current_value = candidate[param]
                    new_value = max(0.01, current_value * (1 + adjustment))
                    candidate[param] = new_value
                elif param.endswith("threshold"):
                    # Threshold parameter (0-1)
                    current_value = candidate[param]
                    new_value = min(1.0, max(0.1, current_value + adjustment * 0.1))
                    candidate[param] = new_value
                else:
                    # Default case
                    current_value = candidate[param]
                    new_value = current_value * (1 + adjustment)
                    candidate[param] = new_value
                
                candidates.append(candidate)
        
        # Generate combinations of top parameters
        top_params = sorted(param_effectiveness.items(), 
                         key=lambda x: abs(x[1]["effect"]) * x[1]["confidence"], 
                         reverse=True)[:3]
        
        if len(top_params) >= 2:
            combo_candidate = current_config.copy()
            
            for param, effect_data in top_params:
                effect = effect_data["effect"]
                confidence = effect_data["confidence"]
                
                if abs(effect) > 0.1 and confidence > 0.2:
                    adjustment = effect * confidence * 0.1
                    
                    # Apply adjustment (similar logic as above)
                    if param == "evaluation_interval":
                        current_value = combo_candidate[param]
                        new_value = max(1, round(current_value * (1 + adjustment)))
                        combo_candidate[param] = new_value
                    elif param in ["min_cycle_time", "max_cycle_time"]:
                        current_value = combo_candidate[param]
                        new_value = max(0.01, current_value * (1 + adjustment))
                        combo_candidate[param] = new_value
                    elif param.endswith("threshold"):
                        current_value = combo_candidate[param]
                        new_value = min(1.0, max(0.1, current_value + adjustment * 0.1))
                        combo_candidate[param] = new_value
                    else:
                        current_value = combo_candidate[param]
                        new_value = current_value * (1 + adjustment)
                        combo_candidate[param] = new_value
            
            candidates.append(combo_candidate)
        
        # Add a random exploration candidate
        random_candidate = current_config.copy()
        
        for param in random_candidate:
            # Apply random adjustment
            if param == "evaluation_interval":
                random_candidate[param] = max(1, round(random_candidate[param] * random.uniform(0.8, 1.2)))
            elif param in ["min_cycle_time", "max_cycle_time"]:
                random_candidate[param] = max(0.01, random_candidate[param] * random.uniform(0.8, 1.2))
            elif param.endswith("threshold"):
                random_candidate[param] = min(1.0, max(0.1, random_candidate[param] + random.uniform(-0.1, 0.1)))
            else:
                random_candidate[param] = random_candidate[param] * random.uniform(0.8, 1.2)
        
        candidates.append(random_candidate)
        
        # Add current configuration as a safe fallback
        candidates.append(current_config)
        
        return candidates
    
    async def _evaluate_parameter_candidates(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate and select the most promising parameter configuration"""
        # This would typically involve running simulations or tests
        # For simplicity, we'll use a heuristic evaluation
        
        best_candidate = None
        best_score = float('-inf')
        
        for candidate in candidates:
            # Evaluate candidate
            score = self._evaluate_parameter_candidate(candidate)
            
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        # Log candidate evaluation
        if not hasattr(self, '_parameter_evaluations'):
            self._parameter_evaluations = []
            
        self._parameter_evaluations.append({
            "timestamp": datetime.now().isoformat(),
            "candidates": candidates,
            "best_candidate": best_candidate,
            "best_score": best_score
        })
        
        return best_candidate or self.config.copy()
    
    def _evaluate_parameter_candidate(self, candidate: Dict[str, Any]) -> float:
        """Evaluate a parameter candidate configuration"""
        # Start with a base score
        score = 0.0
        
        # Check for valid ranges and penalize invalid configurations
        if candidate["min_cycle_time"] > candidate["max_cycle_time"]:
            return float('-inf')  # Invalid configuration
        
        # Reward higher threshold values (typically means more selective)
        for param, value in candidate.items():
            if param.endswith("threshold"):
                score += value * 0.2
        
        # Reward lower evaluation interval (more frequent)
        if "evaluation_interval" in candidate:
            score += 1.0 / (1.0 + candidate["evaluation_interval"] * 0.1)
        
        # Reward balance between min and max cycle time
        if "min_cycle_time" in candidate and "max_cycle_time" in candidate:
            time_range = candidate["max_cycle_time"] - candidate["min_cycle_time"]
            score += 0.3 * (1.0 - math.exp(-time_range))  # Reward reasonable range
        
        # Add some noise for exploration
        score += random.uniform(-0.1, 0.1)
        
        return score
    
    def _apply_meta_parameters(self, selected_params: Dict[str, Any]) -> None:
        """Apply the selected parameters"""
        # Record the current state before applying changes
        if not hasattr(self, '_parameter_history'):
            self._parameter_history = []
            
        self._parameter_history.append({
            "parameters": self.config.copy(),
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "cycle_time": self.system_metrics["average_cycle_time"],
                "error_rate": self.system_metrics["error_rate"],
                # Add more relevant metrics
            }
        })
        
        # Apply new configuration
        self.config = selected_params.copy()
