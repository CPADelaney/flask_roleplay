# nyx/performance_monitor.py

from typing import Dict, Any, List
from datetime import datetime
import logging

# Configure logger
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitors and optimizes performance across autonomous systems"""
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls, user_id=None, conversation_id=None):
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    
    def __init__(self):
        self.metrics = {
            "memory": {
                "decision_times": [],
                "resource_usage": [],
                "error_rates": [],
                "success_rates": [],
                "cache_hits": 0,
                "cache_misses": 0
            },
            "npc": {
                "decision_times": [],
                "resource_usage": [],
                "error_rates": [],
                "success_rates": [],
                "behavior_changes": 0,
                "interaction_counts": 0
            },
            "lore": {
                "decision_times": [],
                "resource_usage": [],
                "error_rates": [],
                "success_rates": [],
                "generation_counts": 0,
                "pattern_matches": 0
            },
            "scene": {
                "decision_times": [],
                "resource_usage": [],
                "error_rates": [],
                "success_rates": [],
                "state_changes": 0,
                "event_counts": 0
            }
        }
        
        self.thresholds = {
            "decision_time": 1.0,  # seconds
            "resource_usage": 0.8,  # 80% max
            "error_rate": 0.1,     # 10% max
            "success_rate": 0.7    # 70% min
        }
        
        self.optimizations = {
            "caching": True,
            "batching": True,
            "pruning": True,
            "load_balancing": True
        }
        
        self.started = False
        self.start_time = None

    def start(self):
        """Start performance monitoring"""
        if not self.started:
            self.started = True
            self.start_time = datetime.now()
            logger.info("Performance monitoring started")

    def stop(self):
        """Stop performance monitoring"""
        if self.started:
            self.started = False
            logger.info("Performance monitoring stopped")

    def track_decision_impact(self, system: str, decision: Dict[str, Any]):
        """Track the performance impact of a decision"""
        if not self.started:
            return
            
        metrics = self.metrics[system]
        
        # Track decision time
        decision_time = decision.get("execution_time", 0)
        metrics["decision_times"].append(decision_time)
        
        # Track resource usage
        resource_usage = decision.get("resource_usage", 0)
        metrics["resource_usage"].append(resource_usage)
        
        # Track success/error
        if decision.get("error"):
            metrics["error_rates"].append(1)
            metrics["success_rates"].append(0)
        else:
            metrics["error_rates"].append(0)
            metrics["success_rates"].append(1)
            
        # Track system-specific metrics
        if system == "memory":
            self._track_memory_metrics(decision)
        elif system == "npc":
            self._track_npc_metrics(decision)
        elif system == "lore":
            self._track_lore_metrics(decision)
        elif system == "scene":
            self._track_scene_metrics(decision)
            
        # Check for optimization opportunities
        self._check_optimization_triggers(system)

    def _track_memory_metrics(self, decision: Dict[str, Any]):
        """Track memory-specific metrics"""
        metrics = self.metrics["memory"]
        
        # Track cache performance
        if decision.get("cache_hit"):
            metrics["cache_hits"] += 1
        else:
            metrics["cache_misses"] += 1
            
        # Calculate cache hit rate and adjust caching strategy
        cache_total = metrics["cache_hits"] + metrics["cache_misses"]
        if cache_total > 0:
            hit_rate = metrics["cache_hits"] / cache_total
            if hit_rate < 0.6:  # Below 60% hit rate
                self._adjust_cache_strategy()

    def _track_npc_metrics(self, decision: Dict[str, Any]):
        """Track NPC-specific metrics"""
        metrics = self.metrics["npc"]
        
        # Track behavior changes
        if decision.get("behavior_changed"):
            metrics["behavior_changes"] += 1
            
        # Track interactions
        if decision.get("interaction"):
            metrics["interaction_counts"] += 1
            
        # Analyze interaction patterns for optimization
        if metrics["interaction_counts"] > 100:
            self._analyze_interaction_patterns()

    def _track_lore_metrics(self, decision: Dict[str, Any]):
        """Track lore-specific metrics"""
        metrics = self.metrics["lore"]
        
        # Track generation counts
        if decision.get("generation"):
            metrics["generation_counts"] += 1
            
        # Track pattern matches
        if decision.get("pattern_match"):
            metrics["pattern_matches"] += 1
            
        # Analyze generation efficiency
        if metrics["generation_counts"] > 50:
            self._analyze_generation_efficiency()

    def _track_scene_metrics(self, decision: Dict[str, Any]):
        """Track scene-specific metrics"""
        metrics = self.metrics["scene"]
        
        # Track state changes
        if decision.get("state_changed"):
            metrics["state_changes"] += 1
            
        # Track events
        if decision.get("event"):
            metrics["event_counts"] += 1
            
        # Analyze state change patterns
        if metrics["state_changes"] > 100:
            self._analyze_state_change_patterns()

    def would_impact_performance(self, decision: Dict[str, Any], system: str) -> bool:
        """Predict if a decision would negatively impact performance"""
        if not self.started:
            return False
            
        metrics = self.metrics[system]
        
        # Check decision time impact
        avg_decision_time = sum(metrics["decision_times"]) / len(metrics["decision_times"]) if metrics["decision_times"] else 0
        if decision.get("estimated_time", 0) > avg_decision_time * 2:
            return True
            
        # Check resource usage impact
        avg_resource_usage = sum(metrics["resource_usage"]) / len(metrics["resource_usage"]) if metrics["resource_usage"] else 0
        if decision.get("estimated_resources", 0) > avg_resource_usage * 1.5:
            return True
            
        # Check system-specific impacts
        if system == "memory" and self._would_impact_memory(decision):
            return True
        elif system == "npc" and self._would_impact_npc(decision):
            return True
        elif system == "lore" and self._would_impact_lore(decision):
            return True
        elif system == "scene" and self._would_impact_scene(decision):
            return True
            
        return False

    def _would_impact_memory(self, decision: Dict[str, Any]) -> bool:
        """Check memory-specific performance impact"""
        metrics = self.metrics["memory"]
        
        # Check cache impact
        if decision.get("bypass_cache"):
            return True
            
        # Check memory operation complexity
        if decision.get("operation_complexity", 0) > 0.8:
            return True
            
        return False

    def _would_impact_npc(self, decision: Dict[str, Any]) -> bool:
        """Check NPC-specific performance impact"""
        metrics = self.metrics["npc"]
        
        # Check interaction complexity
        if decision.get("interaction_complexity", 0) > 0.7:
            return True
            
        # Check behavior change frequency
        recent_changes = sum(1 for t in metrics["decision_times"][-10:] if t > self.thresholds["decision_time"])
        if recent_changes > 5:  # More than 50% of recent decisions are slow
            return True
            
        return False

    def _would_impact_lore(self, decision: Dict[str, Any]) -> bool:
        """Check lore-specific performance impact"""
        metrics = self.metrics["lore"]
        
        # Check generation complexity
        if decision.get("generation_complexity", 0) > 0.9:
            return True
            
        # Check pattern matching load
        if metrics["pattern_matches"] > 100 and decision.get("requires_pattern_matching"):
            return True
            
        return False

    def _would_impact_scene(self, decision: Dict[str, Any]) -> bool:
        """Check scene-specific performance impact"""
        metrics = self.metrics["scene"]
        
        # Check state change complexity
        if decision.get("state_change_complexity", 0) > 0.8:
            return True
            
        # Check event processing load
        if metrics["event_counts"] > 200 and decision.get("generates_events"):
            return True
            
        return False

    def _check_optimization_triggers(self, system: str):
        """Check if optimization is needed"""
        metrics = self.metrics[system]
        
        # Check decision time threshold
        avg_decision_time = sum(metrics["decision_times"][-50:]) / 50 if len(metrics["decision_times"]) >= 50 else 0
        if avg_decision_time > self.thresholds["decision_time"]:
            self._optimize_decision_time(system)
            
        # Check resource usage threshold
        avg_resource_usage = sum(metrics["resource_usage"][-50:]) / 50 if len(metrics["resource_usage"]) >= 50 else 0
        if avg_resource_usage > self.thresholds["resource_usage"]:
            self._optimize_resource_usage(system)
            
        # Check error rate threshold
        error_rate = sum(metrics["error_rates"][-100:]) / 100 if len(metrics["error_rates"]) >= 100 else 0
        if error_rate > self.thresholds["error_rate"]:
            self._optimize_error_handling(system)
            
        # Check success rate threshold
        success_rate = sum(metrics["success_rates"][-100:]) / 100 if len(metrics["success_rates"]) >= 100 else 0
        if success_rate < self.thresholds["success_rate"]:
            self._optimize_success_rate(system)

    def _optimize_decision_time(self, system: str):
        """Optimize decision time performance"""
        if self.optimizations["caching"]:
            self._enhance_caching(system)
            
        if self.optimizations["batching"]:
            self._enhance_batching(system)
            
        if self.optimizations["pruning"]:
            self._prune_decision_tree(system)

    def _optimize_resource_usage(self, system: str):
        """Optimize resource usage"""
        if self.optimizations["load_balancing"]:
            self._balance_load(system)
            
        if self.optimizations["pruning"]:
            self._prune_resources(system)

    def _optimize_error_handling(self, system: str):
        """Optimize error handling"""
        metrics = self.metrics[system]
        
        # Analyze error patterns
        error_patterns = self._analyze_error_patterns(system)
        
        # Adjust error thresholds
        self._adjust_error_thresholds(system, error_patterns)
        
        # Enhance recovery strategies
        self._enhance_recovery_strategies(system, error_patterns)

    def _optimize_success_rate(self, system: str):
        """Optimize success rate"""
        metrics = self.metrics[system]
        
        # Analyze success patterns
        success_patterns = self._analyze_success_patterns(system)
        
        # Adjust decision strategies
        self._adjust_decision_strategies(system, success_patterns)
        
        # Enhance success criteria
        self._enhance_success_criteria(system, success_patterns)

    def get_decision_metrics(self) -> Dict[str, Any]:
        """Get decision-related metrics"""
        return {
            system: {
                "avg_decision_time": sum(metrics["decision_times"]) / len(metrics["decision_times"]) if metrics["decision_times"] else 0,
                "avg_resource_usage": sum(metrics["resource_usage"]) / len(metrics["resource_usage"]) if metrics["resource_usage"] else 0,
                "error_rate": sum(metrics["error_rates"]) / len(metrics["error_rates"]) if metrics["error_rates"] else 0,
                "success_rate": sum(metrics["success_rates"]) / len(metrics["success_rates"]) if metrics["success_rates"] else 0,
                "total_decisions": len(metrics["decision_times"]),
                "system_specific": self._get_system_specific_metrics(system)
            }
            for system, metrics in self.metrics.items()
        }

    def _get_system_specific_metrics(self, system: str) -> Dict[str, Any]:
        """Get system-specific metrics"""
        metrics = self.metrics[system]
        
        if system == "memory":
            return {
                "cache_hit_rate": metrics["cache_hits"] / (metrics["cache_hits"] + metrics["cache_misses"]) if (metrics["cache_hits"] + metrics["cache_misses"]) > 0 else 0
            }
        elif system == "npc":
            return {
                "behavior_change_rate": metrics["behavior_changes"] / len(metrics["decision_times"]) if metrics["decision_times"] else 0,
                "interaction_rate": metrics["interaction_counts"] / len(metrics["decision_times"]) if metrics["decision_times"] else 0
            }
        elif system == "lore":
            return {
                "generation_rate": metrics["generation_counts"] / len(metrics["decision_times"]) if metrics["decision_times"] else 0,
                "pattern_match_rate": metrics["pattern_matches"] / len(metrics["decision_times"]) if metrics["decision_times"] else 0
            }
        elif system == "scene":
            return {
                "state_change_rate": metrics["state_changes"] / len(metrics["decision_times"]) if metrics["decision_times"] else 0,
                "event_rate": metrics["event_counts"] / len(metrics["decision_times"]) if metrics["decision_times"] else 0
            }
        
        return {}
        
    # Placeholder methods for functions referenced but not implemented
    def _adjust_cache_strategy(self):
        """Placeholder for cache strategy adjustment"""
        logger.info("Adjusting cache strategy")
        
    def _analyze_interaction_patterns(self):
        """Placeholder for interaction pattern analysis"""
        logger.info("Analyzing interaction patterns")
        
    def _analyze_generation_efficiency(self):
        """Placeholder for generation efficiency analysis"""
        logger.info("Analyzing generation efficiency")
        
    def _analyze_state_change_patterns(self):
        """Placeholder for state change pattern analysis"""
        logger.info("Analyzing state change patterns")
        
    def _enhance_caching(self, system: str):
        """Placeholder for enhancing caching"""
        logger.info(f"Enhancing caching for {system}")
        
    def _enhance_batching(self, system: str):
        """Placeholder for enhancing batching"""
        logger.info(f"Enhancing batching for {system}")
        
    def _prune_decision_tree(self, system: str):
        """Placeholder for pruning decision tree"""
        logger.info(f"Pruning decision tree for {system}")
        
    def _balance_load(self, system: str):
        """Placeholder for balancing load"""
        logger.info(f"Balancing load for {system}")
        
    def _prune_resources(self, system: str):
        """Placeholder for pruning resources"""
        logger.info(f"Pruning resources for {system}")
        
    def _analyze_error_patterns(self, system: str):
        """Placeholder for analyzing error patterns"""
        logger.info(f"Analyzing error patterns for {system}")
        return {}
        
    def _adjust_error_thresholds(self, system: str, patterns: Dict[str, Any]):
        """Placeholder for adjusting error thresholds"""
        logger.info(f"Adjusting error thresholds for {system}")
        
    def _enhance_recovery_strategies(self, system: str, patterns: Dict[str, Any]):
        """Placeholder for enhancing recovery strategies"""
        logger.info(f"Enhancing recovery strategies for {system}")
        
    def _analyze_success_patterns(self, system: str):
        """Placeholder for analyzing success patterns"""
        logger.info(f"Analyzing success patterns for {system}")
        return {}
        
    def _adjust_decision_strategies(self, system: str, patterns: Dict[str, Any]):
        """Placeholder for adjusting decision strategies"""
        logger.info(f"Adjusting decision strategies for {system}")
        
    def _enhance_success_criteria(self, system: str, patterns: Dict[str, Any]):
        """Placeholder for enhancing success criteria"""
        logger.info(f"Enhancing success criteria for {system}")
