# nyx/core/a2a/enhanced_reasoning_meta.py
"""
Enhanced Context-Aware Reasoning Core - Part 2
Meta-reasoning, explanation generation, uncertainty propagation, and reasoning templates.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

# ========================================================================================
# META-REASONING CAPABILITIES
# ========================================================================================

class ReasoningStrategy(Enum):
    """Available reasoning strategies"""
    DEPTH_FIRST = "depth_first"
    BREADTH_FIRST = "breadth_first"
    BEST_FIRST = "best_first"
    MONTE_CARLO = "monte_carlo"
    HYBRID = "hybrid"

@dataclass
class ReasoningPerformance:
    """Track reasoning performance metrics"""
    strategy: ReasoningStrategy
    start_time: datetime
    end_time: Optional[datetime] = None
    nodes_explored: int = 0
    paths_found: int = 0
    insights_generated: int = 0
    memory_used: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    confidence: float = 0.5

class MetaReasoningModule:
    """Meta-reasoning capabilities for self-aware reasoning"""
    
    def __init__(self):
        self.performance_history: List[ReasoningPerformance] = []
        self.strategy_effectiveness: Dict[ReasoningStrategy, float] = {
            strategy: 0.5 for strategy in ReasoningStrategy
        }
        self.current_performance: Optional[ReasoningPerformance] = None
        self.reasoning_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.loop_detector = LoopDetector()
        
    async def evaluate_reasoning_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if current reasoning approach is effective"""
        evaluation = {
            "current_strategy": None,
            "effectiveness_score": 0.0,
            "bottlenecks": [],
            "recommendations": [],
            "confidence": 0.5
        }
        
        if not self.current_performance:
            return evaluation
        
        # Current strategy analysis
        current_strategy = self.current_performance.strategy
        evaluation["current_strategy"] = current_strategy.value
        
        # Calculate effectiveness
        effectiveness = await self._calculate_current_effectiveness()
        evaluation["effectiveness_score"] = effectiveness
        
        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks()
        evaluation["bottlenecks"] = bottlenecks
        
        # Generate recommendations
        if effectiveness < 0.4:  # Poor performance
            recommendations = await self._generate_strategy_recommendations(context)
            evaluation["recommendations"] = recommendations
        
        # Check for reasoning loops
        loop_detection = await self.loop_detector.detect_loops(self.reasoning_patterns)
        if loop_detection["has_loops"]:
            evaluation["bottlenecks"].append({
                "type": "reasoning_loop",
                "description": "Circular reasoning detected",
                "severity": "high",
                "patterns": loop_detection["loop_patterns"]
            })
            evaluation["recommendations"].append({
                "action": "break_loop",
                "strategy": "introduce_randomness",
                "priority": "immediate"
            })
        
        # Calculate confidence
        evaluation["confidence"] = self._calculate_meta_confidence()
        
        return evaluation
    
    async def _calculate_current_effectiveness(self) -> float:
        """Calculate effectiveness of current reasoning approach"""
        if not self.current_performance:
            return 0.5
        
        effectiveness = 0.0
        
        # Time efficiency
        if self.current_performance.start_time:
            elapsed = (datetime.now() - self.current_performance.start_time).total_seconds()
            if elapsed < 1.0:  # Very fast
                effectiveness += 0.3
            elif elapsed < 5.0:  # Reasonable
                effectiveness += 0.2
            elif elapsed > 30.0:  # Too slow
                effectiveness -= 0.2
        
        # Exploration efficiency
        nodes_per_insight = (self.current_performance.nodes_explored / 
                           max(1, self.current_performance.insights_generated))
        if nodes_per_insight < 5:  # Efficient
            effectiveness += 0.3
        elif nodes_per_insight > 20:  # Inefficient
            effectiveness -= 0.1
        
        # Path finding success
        if self.current_performance.paths_found > 0:
            effectiveness += 0.2
        
        # Historical performance of this strategy
        historical_effectiveness = self.strategy_effectiveness.get(
            self.current_performance.strategy, 0.5
        )
        effectiveness += historical_effectiveness * 0.2
        
        return max(0.0, min(1.0, effectiveness))
    
    def _detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks in reasoning"""
        bottlenecks = []
        
        if not self.current_performance:
            return bottlenecks
        
        # Memory bottleneck
        if self.current_performance.memory_used > 100:  # MB
            bottlenecks.append({
                "type": "memory_pressure",
                "description": f"High memory usage: {self.current_performance.memory_used:.1f}MB",
                "severity": "medium" if self.current_performance.memory_used < 500 else "high",
                "impact": "May cause slowdowns or failures"
            })
        
        # Exploration explosion
        if self.current_performance.nodes_explored > 1000:
            bottlenecks.append({
                "type": "exploration_explosion", 
                "description": f"Explored {self.current_performance.nodes_explored} nodes",
                "severity": "high",
                "impact": "Exponential growth in search space"
            })
        
        # Low insight generation
        insight_rate = (self.current_performance.insights_generated / 
                       max(1, self.current_performance.nodes_explored))
        if insight_rate < 0.01:  # Less than 1% of nodes generate insights
            bottlenecks.append({
                "type": "low_insight_rate",
                "description": f"Only {insight_rate:.1%} of explorations yield insights",
                "severity": "medium",
                "impact": "Inefficient exploration strategy"
            })
        
        return bottlenecks
    
    async def _generate_strategy_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations for improving reasoning strategy"""
        recommendations = []
        
        # Analyze context characteristics
        context_profile = self._profile_context(context)
        
        # Current performance issues
        if self.current_performance:
            current_strategy = self.current_performance.strategy
            
            # If depth-first is struggling with broad search space
            if (current_strategy == ReasoningStrategy.DEPTH_FIRST and 
                context_profile["breadth"] > 0.7):
                recommendations.append({
                    "action": "switch_strategy",
                    "target_strategy": ReasoningStrategy.BREADTH_FIRST,
                    "reason": "High breadth context suits breadth-first exploration",
                    "expected_improvement": 0.3
                })
            
            # If taking too long, try heuristic-based
            if self.current_performance.nodes_explored > 500:
                recommendations.append({
                    "action": "switch_strategy",
                    "target_strategy": ReasoningStrategy.BEST_FIRST,
                    "reason": "Large search space requires heuristic guidance",
                    "expected_improvement": 0.4
                })
            
            # If no clear path, try Monte Carlo
            if self.current_performance.paths_found == 0:
                recommendations.append({
                    "action": "switch_strategy",
                    "target_strategy": ReasoningStrategy.MONTE_CARLO,
                    "reason": "Random sampling might find hidden paths",
                    "expected_improvement": 0.2
                })
        
        # Context-based recommendations
        if context_profile["uncertainty"] > 0.7:
            recommendations.append({
                "action": "increase_evidence_threshold",
                "parameter": "min_evidence_strength",
                "factor": 1.5,
                "reason": "High uncertainty requires stronger evidence"
            })
        
        if context_profile["complexity"] > 0.8:
            recommendations.append({
                "action": "enable_hierarchical_reasoning",
                "parameter": "use_abstraction",
                "value": True,
                "reason": "Complex problems benefit from hierarchical decomposition"
            })
        
        return recommendations
    
    def _profile_context(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Profile context characteristics for strategy selection"""
        profile = {
            "breadth": 0.5,  # How broad vs deep is the problem
            "uncertainty": 0.5,  # Level of uncertainty
            "complexity": 0.5,  # Problem complexity
            "time_pressure": 0.5,  # Urgency
            "novelty": 0.5  # How novel/unique is the problem
        }
        
        # Analyze user input
        user_input = context.get("user_input", "").lower()
        
        # Breadth indicators
        if any(word in user_input for word in ["compare", "options", "alternatives", "various"]):
            profile["breadth"] = 0.8
        elif any(word in user_input for word in ["specific", "particular", "exact", "precise"]):
            profile["breadth"] = 0.2
        
        # Uncertainty indicators
        if any(word in user_input for word in ["might", "maybe", "possibly", "uncertain"]):
            profile["uncertainty"] = 0.8
        elif any(word in user_input for word in ["definitely", "certainly", "must", "always"]):
            profile["uncertainty"] = 0.2
        
        # Complexity indicators
        word_count = len(user_input.split())
        if word_count > 50:
            profile["complexity"] = 0.8
        elif word_count < 10:
            profile["complexity"] = 0.3
        
        # Time pressure (from constraints)
        if "time_sensitive" in context.get("constraints", []):
            profile["time_pressure"] = 0.9
        
        # Novelty (check against history)
        if hasattr(self, 'reasoning_patterns') and len(self.reasoning_patterns) > 0:
            # Simple novelty check - could be more sophisticated
            similar_patterns = 0
            for pattern_type, patterns in self.reasoning_patterns.items():
                for pattern in patterns[-10:]:  # Check recent patterns
                    if self._context_similarity(context, pattern.get("context", {})) > 0.7:
                        similar_patterns += 1
            
            profile["novelty"] = 1.0 - min(similar_patterns / 10, 1.0)
        
        return profile
    
    def _context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts"""
        # Simple implementation - could use more sophisticated methods
        similarity = 0.0
        
        # Compare user inputs
        input1 = context1.get("user_input", "").lower()
        input2 = context2.get("user_input", "").lower()
        
        words1 = set(input1.split())
        words2 = set(input2.split())
        
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            similarity = overlap / max(len(words1), len(words2))
        
        return similarity
    
    def _calculate_meta_confidence(self) -> float:
        """Calculate confidence in meta-reasoning assessment"""
        confidence = 0.5
        
        # More history = more confidence
        if len(self.performance_history) > 10:
            confidence += 0.2
        elif len(self.performance_history) > 50:
            confidence += 0.3
        
        # Consistent performance = more confidence
        if self.performance_history:
            recent_performances = self.performance_history[-10:]
            success_rate = sum(p.success for p in recent_performances) / len(recent_performances)
            confidence += success_rate * 0.2
        
        return min(1.0, confidence)
    
    async def adapt_reasoning_parameters(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamically adjust reasoning parameters based on performance"""
        adaptations = {
            "parameter_changes": {},
            "strategy_change": None,
            "explanation": []
        }
        
        # Analyze performance trends
        trends = self._analyze_performance_trends()
        
        # Adapt based on trends
        if trends["success_rate_declining"]:
            # Try more conservative parameters
            adaptations["parameter_changes"]["exploration_depth"] = -1
            adaptations["parameter_changes"]["evidence_threshold"] = 1.2
            adaptations["explanation"].append("Reducing exploration depth due to declining success rate")
        
        if trends["speed_declining"]:
            # Optimize for speed
            adaptations["parameter_changes"]["max_nodes_to_explore"] = 0.7
            adaptations["parameter_changes"]["use_caching"] = True
            adaptations["explanation"].append("Enabling optimizations due to speed concerns")
        
        if trends["memory_increasing"]:
            # Memory optimization
            adaptations["parameter_changes"]["prune_frequency"] = 2.0
            adaptations["parameter_changes"]["max_retained_paths"] = 0.5
            adaptations["explanation"].append("Increasing memory management due to growing usage")
        
        # Strategy adaptation
        if performance_metrics.get("consecutive_failures", 0) > 3:
            current_strategy = self.current_performance.strategy if self.current_performance else None
            new_strategy = self._select_alternative_strategy(current_strategy)
            adaptations["strategy_change"] = new_strategy.value
            adaptations["explanation"].append(f"Switching to {new_strategy.value} after repeated failures")
        
        # Update strategy effectiveness
        await self._update_strategy_effectiveness(performance_metrics)
        
        return adaptations
    
    def _analyze_performance_trends(self) -> Dict[str, bool]:
        """Analyze trends in performance history"""
        trends = {
            "success_rate_declining": False,
            "speed_declining": False,
            "memory_increasing": False,
            "insight_rate_declining": False
        }
        
        if len(self.performance_history) < 5:
            return trends
        
        recent = self.performance_history[-5:]
        older = self.performance_history[-10:-5] if len(self.performance_history) >= 10 else []
        
        if older:
            # Success rate trend
            recent_success = sum(p.success for p in recent) / len(recent)
            older_success = sum(p.success for p in older) / len(older)
            trends["success_rate_declining"] = recent_success < older_success * 0.8
            
            # Speed trend
            recent_durations = [
                (p.end_time - p.start_time).total_seconds() 
                for p in recent if p.end_time
            ]
            older_durations = [
                (p.end_time - p.start_time).total_seconds() 
                for p in older if p.end_time
            ]
            
            if recent_durations and older_durations:
                trends["speed_declining"] = (
                    sum(recent_durations) / len(recent_durations) > 
                    sum(older_durations) / len(older_durations) * 1.5
                )
            
            # Memory trend
            recent_memory = sum(p.memory_used for p in recent) / len(recent)
            older_memory = sum(p.memory_used for p in older) / len(older)
            trends["memory_increasing"] = recent_memory > older_memory * 1.5
        
        return trends
    
    def _select_alternative_strategy(self, current: Optional[ReasoningStrategy]) -> ReasoningStrategy:
        """Select an alternative strategy"""
        if not current:
            return ReasoningStrategy.HYBRID
        
        # Order strategies by effectiveness
        strategy_scores = sorted(
            self.strategy_effectiveness.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Select best strategy that's different from current
        for strategy, score in strategy_scores:
            if strategy != current:
                return strategy
        
        return ReasoningStrategy.HYBRID
    
    async def _update_strategy_effectiveness(self, metrics: Dict[str, Any]):
        """Update effectiveness scores for strategies"""
        if not self.current_performance:
            return
        
        strategy = self.current_performance.strategy
        
        # Calculate performance score
        score = 0.0
        if self.current_performance.success:
            score += 0.5
        
        # Efficiency component
        if self.current_performance.insights_generated > 0:
            efficiency = (self.current_performance.insights_generated / 
                         max(1, self.current_performance.nodes_explored))
            score += efficiency * 0.3
        
        # Speed component
        if self.current_performance.end_time:
            duration = (self.current_performance.end_time - 
                       self.current_performance.start_time).total_seconds()
            if duration < 5:
                score += 0.2
            elif duration < 20:
                score += 0.1
        
        # Update with exponential moving average
        alpha = 0.2  # Learning rate
        old_effectiveness = self.strategy_effectiveness[strategy]
        self.strategy_effectiveness[strategy] = (
            alpha * score + (1 - alpha) * old_effectiveness
        )
    
    async def detect_reasoning_loops(self) -> List[Dict[str, Any]]:
        """Detect when reasoning is stuck in loops"""
        return await self.loop_detector.detect_loops(self.reasoning_patterns)
    
    def start_reasoning_session(self, strategy: ReasoningStrategy):
        """Start tracking a new reasoning session"""
        self.current_performance = ReasoningPerformance(
            strategy=strategy,
            start_time=datetime.now()
        )
    
    def end_reasoning_session(self, success: bool, insights_generated: int = 0):
        """End current reasoning session"""
        if self.current_performance:
            self.current_performance.end_time = datetime.now()
            self.current_performance.success = success
            self.current_performance.insights_generated = insights_generated
            self.performance_history.append(self.current_performance)
            
            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
    
    def record_exploration(self, nodes_explored: int):
        """Record exploration progress"""
        if self.current_performance:
            self.current_performance.nodes_explored += nodes_explored
    
    def record_pattern(self, pattern_type: str, pattern: Dict[str, Any]):
        """Record reasoning pattern for loop detection"""
        self.reasoning_patterns[pattern_type].append({
            "pattern": pattern,
            "timestamp": datetime.now()
        })
        
        # Keep only recent patterns
        if len(self.reasoning_patterns[pattern_type]) > 50:
            self.reasoning_patterns[pattern_type] = self.reasoning_patterns[pattern_type][-50:]

class LoopDetector:
    """Detect reasoning loops and circular patterns"""
    
    def __init__(self):
        self.pattern_signatures: List[str] = []
        self.loop_threshold = 3  # Same pattern 3 times = loop
    
    async def detect_loops(self, reasoning_patterns: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Detect loops in reasoning patterns"""
        detection_result = {
            "has_loops": False,
            "loop_patterns": [],
            "recommendations": []
        }
        
        # Check each pattern type
        for pattern_type, patterns in reasoning_patterns.items():
            if len(patterns) < self.loop_threshold:
                continue
            
            # Generate signatures for recent patterns
            recent_patterns = patterns[-10:]
            signatures = [self._generate_signature(p["pattern"]) for p in recent_patterns]
            
            # Check for repeated signatures
            signature_counts = {}
            for sig in signatures:
                signature_counts[sig] = signature_counts.get(sig, 0) + 1
            
            # Identify loops
            for sig, count in signature_counts.items():
                if count >= self.loop_threshold:
                    detection_result["has_loops"] = True
                    detection_result["loop_patterns"].append({
                        "pattern_type": pattern_type,
                        "repetitions": count,
                        "signature": sig[:50] + "..."  # Truncate for readability
                    })
        
        # Generate recommendations
        if detection_result["has_loops"]:
            detection_result["recommendations"] = [
                "Introduce randomness to break deterministic patterns",
                "Expand search space with alternative approaches",
                "Use different initialization or parameters",
                "Apply loop-breaking heuristics"
            ]
        
        return detection_result
    
    def _generate_signature(self, pattern: Dict[str, Any]) -> str:
        """Generate a signature for a pattern"""
        # Simple signature generation - could be more sophisticated
        # Sort keys for consistency
        sorted_items = sorted(pattern.items())
        return str(sorted_items)

# ========================================================================================
# EXPLANATION GENERATION SYSTEM
# ========================================================================================

class ExplanationStyle(Enum):
    """Different explanation styles"""
    TECHNICAL = "technical"
    SIMPLE = "simple"
    NARRATIVE = "narrative"
    VISUAL = "visual"
    STEP_BY_STEP = "step_by_step"

class ExplanationGenerator:
    """Sophisticated explanation generation system"""
    
    def __init__(self):
        self.explanation_templates = self._load_explanation_templates()
        self.audience_profiles = self._load_audience_profiles()
        self.visual_schemas = self._load_visual_schemas()
    
    def _load_explanation_templates(self) -> Dict[str, Dict[str, str]]:
        """Load explanation templates for different contexts"""
        return {
            "causal_chain": {
                "technical": "The causal chain from {start} to {end} proceeds through {path} with mechanisms: {mechanisms}",
                "simple": "{start} leads to {end} because {simple_reason}",
                "narrative": "When {start} occurs, it triggers a series of events: {story}",
                "step_by_step": "Step 1: {start}\n{steps}\nFinal: {end}"
            },
            "intervention": {
                "technical": "Intervening at {node} with {method} yields {impact} through {mechanism}",
                "simple": "If you change {node}, you can expect {simple_impact}",
                "narrative": "Imagine {metaphor}. Similarly, changing {node} would {narrative_impact}",
                "step_by_step": "To implement:\n1. {step1}\n2. {step2}\n3. Monitor {metrics}"
            },
            "counterfactual": {
                "technical": "Under counterfactual {condition}, the state transitions from {original} to {alternative} with probability {prob}",
                "simple": "If {condition} had been different, then {simple_outcome}",
                "narrative": "In an alternate scenario where {condition}, the story unfolds differently: {alternate_story}",
                "step_by_step": "What would change:\n1. {change1}\n2. {change2}\nResult: {final_state}"
            },
            "pattern": {
                "technical": "Pattern {pattern_id} exhibits {properties} with frequency {freq} and confidence {conf}",
                "simple": "We often see that {simple_pattern}",
                "narrative": "There's a recurring theme: {pattern_story}",
                "step_by_step": "The pattern:\n1. {element1}\n2. {element2}\n3. {element3}"
            }
        }
    
    def _load_audience_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load audience profiles for adaptation"""
        return {
            "general": {
                "technical_level": 0.3,
                "preferred_style": ExplanationStyle.SIMPLE,
                "detail_preference": "moderate",
                "visual_preference": 0.5,
                "examples_needed": True
            },
            "expert": {
                "technical_level": 0.9,
                "preferred_style": ExplanationStyle.TECHNICAL,
                "detail_preference": "high",
                "visual_preference": 0.3,
                "examples_needed": False
            },
            "student": {
                "technical_level": 0.5,
                "preferred_style": ExplanationStyle.STEP_BY_STEP,
                "detail_preference": "high",
                "visual_preference": 0.7,
                "examples_needed": True
            },
            "executive": {
                "technical_level": 0.4,
                "preferred_style": ExplanationStyle.NARRATIVE,
                "detail_preference": "low",
                "visual_preference": 0.8,
                "examples_needed": True
            }
        }
    
    def _load_visual_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load schemas for visual representations"""
        return {
            "causal_diagram": {
                "type": "directed_graph",
                "elements": ["nodes", "edges", "labels"],
                "style": "hierarchical"
            },
            "process_flow": {
                "type": "flowchart",
                "elements": ["start", "process", "decision", "end"],
                "style": "sequential"
            },
            "concept_map": {
                "type": "network",
                "elements": ["concepts", "relationships", "properties"],
                "style": "radial"
            },
            "timeline": {
                "type": "linear",
                "elements": ["events", "durations", "milestones"],
                "style": "chronological"
            }
        }
    
    async def generate_causal_explanation(self,
                                        reasoning_path: List[Dict[str, Any]],
                                        target_audience: str = "general",
                                        include_confidence: bool = True) -> str:
        """Generate audience-appropriate causal explanations"""
        audience_profile = self.audience_profiles.get(target_audience, self.audience_profiles["general"])
        style = audience_profile["preferred_style"]
        
        # Extract key elements from reasoning path
        if not reasoning_path:
            return "No causal path found to explain."
        
        start_node = reasoning_path[0].get("node_name", "initial state")
        end_node = reasoning_path[-1].get("node_name", "final state")
        
        # Build explanation based on style
        if style == ExplanationStyle.TECHNICAL:
            explanation = self._generate_technical_causal_explanation(reasoning_path, include_confidence)
        elif style == ExplanationStyle.SIMPLE:
            explanation = self._generate_simple_causal_explanation(reasoning_path)
        elif style == ExplanationStyle.NARRATIVE:
            explanation = self._generate_narrative_causal_explanation(reasoning_path)
        elif style == ExplanationStyle.STEP_BY_STEP:
            explanation = self._generate_step_by_step_causal_explanation(reasoning_path)
        else:
            explanation = self._generate_simple_causal_explanation(reasoning_path)
        
        # Add visual representation if preferred
        if audience_profile["visual_preference"] > 0.6:
            visual = await self.create_visual_reasoning_map(reasoning_path)
            explanation += f"\n\n[Visual representation: {visual['description']}]"
        
        # Add examples if needed
        if audience_profile["examples_needed"]:
            example = self._generate_causal_example(start_node, end_node)
            explanation += f"\n\nExample: {example}"
        
        return explanation
    
    def _generate_technical_causal_explanation(self, 
                                             path: List[Dict[str, Any]], 
                                             include_confidence: bool) -> str:
        """Generate technical causal explanation"""
        explanation_parts = []
        
        explanation_parts.append("Causal Analysis:")
        
        for i, step in enumerate(path):
            if i == 0:
                explanation_parts.append(f"• Initial node: {step.get('node_name')} (Type: {step.get('node_type', 'standard')})")
            else:
                prev_step = path[i-1]
                mechanism = step.get('mechanism', 'direct causation')
                strength = step.get('strength', 0.5)
                
                explanation_parts.append(
                    f"• {prev_step.get('node_name')} → {step.get('node_name')} "
                    f"[Mechanism: {mechanism}, Strength: {strength:.2f}]"
                )
        
        if include_confidence:
            overall_confidence = self._calculate_path_confidence(path)
            explanation_parts.append(f"\nOverall confidence: {overall_confidence:.2f}")
        
        # Add technical details
        explanation_parts.append("\nTechnical details:")
        explanation_parts.append(f"• Path length: {len(path)}")
        explanation_parts.append(f"• Total effect size: {self._calculate_total_effect(path):.3f}")
        explanation_parts.append(f"• Weakest link: {self._find_weakest_link(path)}")
        
        return "\n".join(explanation_parts)
    
    def _generate_simple_causal_explanation(self, path: List[Dict[str, Any]]) -> str:
        """Generate simple causal explanation"""
        if len(path) < 2:
            return "The relationship is direct and straightforward."
        
        start = path[0].get('node_name', 'the beginning')
        end = path[-1].get('node_name', 'the result')
        
        # Create simple connecting phrase
        if len(path) == 2:
            return f"{start} directly causes {end}."
        elif len(path) == 3:
            middle = path[1].get('node_name', 'an intermediate factor')
            return f"{start} leads to {middle}, which in turn causes {end}."
        else:
            return f"{start} causes {end} through a series of {len(path)-1} steps."
    
    def _generate_narrative_causal_explanation(self, path: List[Dict[str, Any]]) -> str:
        """Generate narrative causal explanation"""
        story_parts = []
        
        story_parts.append(f"Our story begins with {path[0].get('node_name', 'an initial condition')}.")
        
        for i in range(1, len(path)):
            prev = path[i-1].get('node_name')
            curr = path[i].get('node_name')
            mechanism = path[i].get('mechanism', 'influence')
            
            # Create narrative transitions
            if mechanism == 'amplification':
                transition = f"This amplifies into {curr}"
            elif mechanism == 'inhibition':
                transition = f"However, this is dampened by {curr}"
            elif mechanism == 'trigger':
                transition = f"This triggers {curr}"
            else:
                transition = f"This leads to {curr}"
            
            story_parts.append(transition)
        
        story_parts.append(f"And so we arrive at our destination: {path[-1].get('node_name')}.")
        
        return " ".join(story_parts)
    
    def _generate_step_by_step_causal_explanation(self, path: List[Dict[str, Any]]) -> str:
        """Generate step-by-step causal explanation"""
        steps = []
        
        steps.append("Here's how it works step by step:\n")
        
        for i, node in enumerate(path):
            if i == 0:
                steps.append(f"Step 1: Start with {node.get('node_name')}")
            else:
                prev_node = path[i-1]
                mechanism = node.get('mechanism', 'causes')
                steps.append(
                    f"Step {i+1}: {prev_node.get('node_name')} {mechanism} {node.get('node_name')}"
                )
                
                # Add sub-points for clarity
                if node.get('strength', 1.0) < 0.5:
                    steps.append(f"   (Note: This is a weak effect)")
                if node.get('delay'):
                    steps.append(f"   (Takes approximately {node.get('delay')} to manifest)")
        
        steps.append(f"\nFinal result: {path[-1].get('node_name')}")
        
        return "\n".join(steps)
    
    def _calculate_path_confidence(self, path: List[Dict[str, Any]]) -> float:
        """Calculate confidence in a causal path"""
        if not path:
            return 0.0
        
        # Multiply individual link strengths (assuming independence)
        confidence = 1.0
        for step in path[1:]:  # Skip first node
            confidence *= step.get('strength', 0.5)
        
        # Penalize very long paths
        length_penalty = 0.95 ** max(0, len(path) - 3)
        confidence *= length_penalty
        
        return confidence
    
    def _calculate_total_effect(self, path: List[Dict[str, Any]]) -> float:
        """Calculate total effect size along path"""
        if len(path) < 2:
            return 0.0
        
        # Multiply effect sizes
        total_effect = 1.0
        for step in path[1:]:
            effect_size = step.get('effect_size', step.get('strength', 0.5))
            total_effect *= effect_size
        
        return total_effect
    
    def _find_weakest_link(self, path: List[Dict[str, Any]]) -> str:
        """Find the weakest link in the causal chain"""
        if len(path) < 2:
            return "N/A"
        
        weakest_strength = 1.0
        weakest_link = ""
        
        for i in range(1, len(path)):
            strength = path[i].get('strength', 0.5)
            if strength < weakest_strength:
                weakest_strength = strength
                weakest_link = f"{path[i-1].get('node_name')} → {path[i].get('node_name')}"
        
        return f"{weakest_link} (strength: {weakest_strength:.2f})"
    
    def _generate_causal_example(self, start: str, end: str) -> str:
        """Generate rich, context-aware examples for causal relationships"""
        # Expanded example database with categories
        example_database = {
            # Climate/Environment
            ("temperature", "ice"): "As temperature drops below 0°C, water molecules slow down and form crystalline structures, creating ice",
            ("greenhouse gases", "temperature"): "CO2 and methane trap heat in atmosphere, causing global temperatures to rise (greenhouse effect)",
            ("deforestation", "climate"): "Removing forests reduces CO2 absorption and alters local weather patterns, affecting regional climate",
            
            # Health/Medicine
            ("exercise", "health"): "Regular aerobic exercise strengthens heart muscle, improves circulation, and releases endorphins for mental wellbeing",
            ("stress", "immune system"): "Chronic stress elevates cortisol, suppressing T-cell function and making you more susceptible to illness",
            ("sleep", "memory"): "During REM sleep, the brain consolidates memories by strengthening neural connections formed during the day",
            ("diet", "energy"): "Balanced nutrition provides steady glucose levels, supporting consistent ATP production for cellular energy",
            
            # Psychology/Behavior
            ("practice", "skill"): "Repeated practice creates stronger neural pathways through myelination, making actions more automatic and efficient",
            ("reward", "motivation"): "Positive reinforcement triggers dopamine release, strengthening behavior-reward associations",
            ("trauma", "behavior"): "Traumatic experiences can alter amygdala responses, leading to hypervigilance or avoidance behaviors",
            
            # Economics/Business
            ("supply", "price"): "When supply decreases while demand remains constant, scarcity drives prices upward",
            ("innovation", "productivity"): "New technologies automate tasks and optimize processes, increasing output per worker hour",
            ("education", "income"): "Higher education provides specialized skills and networks, opening access to higher-paying positions",
            
            # Technology
            ("data", "insights"): "Large datasets reveal patterns invisible at smaller scales through statistical analysis and machine learning",
            ("automation", "efficiency"): "Automated systems eliminate human bottlenecks and operate 24/7, dramatically increasing throughput",
            ("connectivity", "collaboration"): "High-speed networks enable real-time communication, allowing distributed teams to work seamlessly",
            
            # Social/Relationships
            ("trust", "cooperation"): "When people trust each other, they're willing to take risks and share resources for mutual benefit",
            ("communication", "understanding"): "Clear, empathetic communication reduces misinterpretations and builds shared mental models",
            ("isolation", "depression"): "Social isolation reduces oxytocin and increases inflammation markers linked to depressive symptoms"
        }
        
        # Try exact match first
        start_lower = start.lower()
        end_lower = end.lower()
        
        for (s, e), example in example_database.items():
            if (s in start_lower and e in end_lower) or (e in start_lower and s in end_lower):
                return example
        
        # Try partial matches
        for (s, e), example in example_database.items():
            # Check if any key words match
            start_words = set(start_lower.split())
            end_words = set(end_lower.split())
            key_words = set(s.split() + e.split())
            
            if (start_words.intersection(key_words) and end_words.intersection(key_words)):
                return example
        
        # Generate contextual example based on patterns
        return self._generate_contextual_example(start, end)

    def _generate_contextual_example(self, start: str, end: str) -> str:
        """Generate a contextual example when no direct match exists"""
        # Identify the likely domain
        domains = {
            "physical": ["temperature", "pressure", "force", "energy", "matter", "speed"],
            "biological": ["cell", "organism", "gene", "protein", "growth", "evolution"],
            "psychological": ["mind", "emotion", "thought", "behavior", "perception", "memory"],
            "social": ["group", "society", "culture", "relationship", "community", "network"],
            "economic": ["money", "market", "trade", "value", "resource", "capital"],
            "technological": ["system", "data", "algorithm", "computer", "network", "software"]
        }
        
        start_domain = "general"
        end_domain = "general"
        
        for domain, keywords in domains.items():
            if any(kw in start.lower() for kw in keywords):
                start_domain = domain
            if any(kw in end.lower() for kw in keywords):
                end_domain = domain
        
        # Generate domain-appropriate example
        if start_domain == "physical" or end_domain == "physical":
            return f"Changes in {start} create measurable effects on {end} through physical mechanisms"
        elif start_domain == "biological" or end_domain == "biological":
            return f"{start} influences {end} through complex biological pathways and feedback loops"
        elif start_domain == "psychological" or end_domain == "psychological":
            return f"{start} shapes {end} through cognitive and emotional processing mechanisms"
        elif start_domain == "social" or end_domain == "social":
            return f"{start} affects {end} through social dynamics and interpersonal interactions"
        elif start_domain == "economic" or end_domain == "economic":
            return f"Changes in {start} drive {end} through market forces and economic incentives"
        elif start_domain == "technological" or end_domain == "technological":
            return f"{start} enables {end} through technological processes and system interactions"
        else:
            # Generic causal statement
            return f"When {start} changes, it influences {end} through a series of intermediate steps"
    
    async def create_visual_reasoning_map(self, reasoning_path: List[Any]) -> Dict[str, Any]:
        """Create visual representation of reasoning process"""
        visual_map = {
            "type": "causal_diagram",
            "nodes": [],
            "edges": [],
            "layout": "hierarchical",
            "description": "",
            "metadata": {}
        }
        
        # Build nodes
        for i, step in enumerate(reasoning_path):
            node = {
                "id": f"node_{i}",
                "label": step.get('node_name', f'Step {i}'),
                "type": step.get('node_type', 'standard'),
                "level": i,
                "properties": {
                    "strength": step.get('strength', 1.0),
                    "confidence": step.get('confidence', 0.5)
                }
            }
            visual_map["nodes"].append(node)
        
        # Build edges
        for i in range(1, len(reasoning_path)):
            edge = {
                "source": f"node_{i-1}",
                "target": f"node_{i}",
                "label": reasoning_path[i].get('mechanism', ''),
                "strength": reasoning_path[i].get('strength', 0.5),
                "style": self._get_edge_style(reasoning_path[i].get('strength', 0.5))
            }
            visual_map["edges"].append(edge)
        
        # Generate description
        visual_map["description"] = self._generate_visual_description(visual_map)
        
        # Add metadata
        visual_map["metadata"] = {
            "total_nodes": len(visual_map["nodes"]),
            "total_edges": len(visual_map["edges"]),
            "max_path_length": len(reasoning_path),
            "complexity": self._calculate_visual_complexity(visual_map)
        }
        
        return visual_map
    
    def _get_edge_style(self, strength: float) -> str:
        """Determine edge style based on strength"""
        if strength > 0.8:
            return "thick_solid"
        elif strength > 0.5:
            return "solid"
        elif strength > 0.3:
            return "dashed"
        else:
            return "dotted"
    
    def _generate_visual_description(self, visual_map: Dict[str, Any]) -> str:
        """Generate textual description of visual"""
        num_nodes = len(visual_map["nodes"])
        num_edges = len(visual_map["edges"])
        
        description = f"A {visual_map['layout']} diagram showing {num_nodes} nodes connected by {num_edges} edges. "
        
        # Describe flow
        if visual_map["nodes"]:
            start = visual_map["nodes"][0]["label"]
            end = visual_map["nodes"][-1]["label"]
            description += f"The flow proceeds from '{start}' to '{end}'. "
        
        # Describe key features
        strong_edges = [e for e in visual_map["edges"] if e["strength"] > 0.7]
        if strong_edges:
            description += f"There are {len(strong_edges)} strong connections. "
        
        weak_edges = [e for e in visual_map["edges"] if e["strength"] < 0.3]
        if weak_edges:
            description += f"Note {len(weak_edges)} weak links that may be bottlenecks."
        
        return description
    
    def _calculate_visual_complexity(self, visual_map: Dict[str, Any]) -> float:
        """Calculate visual complexity score"""
        nodes = len(visual_map["nodes"])
        edges = len(visual_map["edges"])
        
        if nodes == 0:
            return 0.0
        
        # Edge-to-node ratio
        density = edges / nodes
        
        # Complexity increases with size and density
        complexity = min(1.0, (nodes / 20) * 0.5 + density * 0.5)
        
        return complexity
    
    async def generate_counterfactual_explanations(self, 
                                                 original_path: List[Dict[str, Any]],
                                                 counterfactual_paths: List[List[Dict[str, Any]]],
                                                 intervention_point: str) -> List[str]:
        """Generate explanations for counterfactual scenarios"""
        explanations = []
        
        # Explain the intervention
        explanations.append(
            f"What if we changed {intervention_point}? Here are the alternative scenarios:"
        )
        
        for i, cf_path in enumerate(counterfactual_paths):
            # Find divergence point
            divergence_index = self._find_divergence_point(original_path, cf_path)
            
            if divergence_index >= 0:
                explanation = f"\nScenario {i+1}: "
                
                # Describe the change
                if divergence_index < len(cf_path):
                    new_direction = cf_path[divergence_index].get('node_name', 'unknown')
                    explanation += f"Instead of the original path, we would see progression to {new_direction}"
                
                # Describe the outcome
                original_outcome = original_path[-1].get('node_name', 'original outcome')
                new_outcome = cf_path[-1].get('node_name', 'alternative outcome')
                
                if original_outcome != new_outcome:
                    explanation += f", ultimately leading to {new_outcome} instead of {original_outcome}."
                else:
                    explanation += f", but still arriving at {original_outcome} through a different route."
                
                # Add probability if available
                probability = cf_path[-1].get('probability', 0)
                if probability > 0:
                    explanation += f" (Probability: {probability:.1%})"
                
                explanations.append(explanation)
        
        # Add summary
        if len(counterfactual_paths) > 1:
            explanations.append(
                f"\nIn summary, changing {intervention_point} could lead to "
                f"{len(set(p[-1].get('node_name') for p in counterfactual_paths))} "
                f"different possible outcomes."
            )
        
        return explanations
    
    def _find_divergence_point(self, 
                              original_path: List[Dict[str, Any]], 
                              cf_path: List[Dict[str, Any]]) -> int:
        """Find where paths diverge"""
        min_length = min(len(original_path), len(cf_path))
        
        for i in range(min_length):
            if original_path[i].get('node_name') != cf_path[i].get('node_name'):
                return i
        
        return min_length
    
    def adapt_explanation_to_emotion(self, 
                                   explanation: str, 
                                   emotional_state: Dict[str, Any]) -> str:
        """Adapt explanation based on emotional state"""
        dominant_emotion = emotional_state.get("dominant_emotion")
        if not dominant_emotion:
            return explanation
        
        emotion_name, strength = dominant_emotion
        
        # Emotional adaptations
        if emotion_name == "Anxiety" and strength > 0.6:
            # Add reassuring elements
            explanation = "I understand this might seem complex, but let me break it down clearly. " + explanation
            explanation += "\n\nThe key point to remember is that these are understandable patterns, not random occurrences."
        
        elif emotion_name == "Curiosity" and strength > 0.7:
            # Add more detail and exploration
            explanation += "\n\nInterestingly, this connects to broader patterns in the domain. Would you like to explore related phenomena?"
        
        elif emotion_name == "Frustration" and strength > 0.6:
            # Simplify and focus
            lines = explanation.split('\n')
            if len(lines) > 5:
                explanation = '\n'.join(lines[:3]) + "\n\n[Simplified for clarity - key points only]"
        
        elif emotion_name == "Confusion" and strength > 0.5:
            # Add more structure
            explanation = "Let me organize this step-by-step:\n\n" + explanation
            explanation = explanation.replace(". ", ".\n\n")  # Add more breaks
        
        return explanation

# ========================================================================================
# UNCERTAINTY PROPAGATION
# ========================================================================================

class UncertaintyType(Enum):
    """Types of uncertainty"""
    MEASUREMENT = "measurement"
    MODEL = "model"
    PARAMETER = "parameter"
    STRUCTURAL = "structural"

@dataclass
class UncertaintyEstimate:
    """Representation of uncertainty"""
    value: float
    lower_bound: float
    upper_bound: float
    confidence_level: float = 0.95
    uncertainty_type: UncertaintyType = UncertaintyType.MODEL
    source: str = ""

class UncertaintyManager:
    """Sophisticated uncertainty handling throughout reasoning"""
    
    def __init__(self):
        self.uncertainty_models = {
            "gaussian": self._gaussian_propagation,
            "monte_carlo": self._monte_carlo_propagation,
            "interval": self._interval_propagation,
            "fuzzy": self._fuzzy_propagation
        }
        self.critical_threshold = 0.3  # Uncertainty above this is critical
        
    async def propagate_uncertainty(self,
                                  initial_uncertainties: Dict[str, UncertaintyEstimate],
                                  causal_graph: Dict[str, Any],
                                  method: str = "gaussian") -> Dict[str, UncertaintyEstimate]:
        """Propagate uncertainty through causal chains"""
        if method not in self.uncertainty_models:
            method = "gaussian"
        
        propagation_func = self.uncertainty_models[method]
        return await propagation_func(initial_uncertainties, causal_graph)
    
    async def _gaussian_propagation(self,
                                  initial: Dict[str, UncertaintyEstimate],
                                  graph: Dict[str, Any]) -> Dict[str, UncertaintyEstimate]:
        """Propagate uncertainty using Gaussian approximation"""
        propagated = initial.copy()
        
        # Topological sort for propagation order
        sorted_nodes = self._topological_sort(graph)
        
        for node in sorted_nodes:
            if node in propagated:
                continue  # Already has uncertainty
            
            # Get parent uncertainties
            parents = graph.get("parents", {}).get(node, [])
            if not parents:
                continue
            
            # Combine parent uncertainties
            combined_variance = 0.0
            combined_value = 0.0
            
            for parent in parents:
                if parent in propagated:
                    parent_unc = propagated[parent]
                    edge_strength = graph.get("edges", {}).get(f"{parent}->{node}", {}).get("strength", 0.5)
                    
                    # Propagate value
                    combined_value += parent_unc.value * edge_strength
                    
                    # Propagate variance (assuming independence)
                    parent_std = (parent_unc.upper_bound - parent_unc.lower_bound) / 4  # 2 sigma
                    combined_variance += (edge_strength * parent_std) ** 2
            
            # Create propagated uncertainty
            combined_std = np.sqrt(combined_variance)
            propagated[node] = UncertaintyEstimate(
                value=combined_value,
                lower_bound=combined_value - 2 * combined_std,
                upper_bound=combined_value + 2 * combined_std,
                confidence_level=0.95,
                uncertainty_type=UncertaintyType.MODEL,
                source="propagated"
            )
        
        return propagated
    
    async def _monte_carlo_propagation(self,
                                     initial: Dict[str, UncertaintyEstimate],
                                     graph: Dict[str, Any]) -> Dict[str, UncertaintyEstimate]:
        """Propagate uncertainty using Monte Carlo simulation"""
        n_samples = 1000
        propagated = {}
        
        # Generate samples for initial uncertainties
        samples = {}
        for node, unc in initial.items():
            # Assume normal distribution
            std = (unc.upper_bound - unc.lower_bound) / 4
            samples[node] = np.random.normal(unc.value, std, n_samples)
        
        # Propagate through graph
        sorted_nodes = self._topological_sort(graph)
        
        for node in sorted_nodes:
            if node in samples:
                continue
            
            parents = graph.get("parents", {}).get(node, [])
            if not parents:
                continue
            
            # Calculate samples for this node
            node_samples = np.zeros(n_samples)
            
            for parent in parents:
                if parent in samples:
                    edge_strength = graph.get("edges", {}).get(f"{parent}->{node}", {}).get("strength", 0.5)
                    node_samples += samples[parent] * edge_strength
            
            samples[node] = node_samples
            
            # Convert samples to uncertainty estimate
            propagated[node] = UncertaintyEstimate(
                value=np.mean(node_samples),
                lower_bound=np.percentile(node_samples, 2.5),
                upper_bound=np.percentile(node_samples, 97.5),
                confidence_level=0.95,
                uncertainty_type=UncertaintyType.MODEL,
                source="monte_carlo"
            )
        
        return propagated
    
    async def _interval_propagation(self,
                                  initial: Dict[str, UncertaintyEstimate],
                                  graph: Dict[str, Any]) -> Dict[str, UncertaintyEstimate]:
        """Propagate uncertainty using interval arithmetic"""
        propagated = initial.copy()
        sorted_nodes = self._topological_sort(graph)
        
        for node in sorted_nodes:
            if node in propagated:
                continue
            
            parents = graph.get("parents", {}).get(node, [])
            if not parents:
                continue
            
            # Interval arithmetic
            lower_sum = 0.0
            upper_sum = 0.0
            
            for parent in parents:
                if parent in propagated:
                    parent_unc = propagated[parent]
                    edge_strength = graph.get("edges", {}).get(f"{parent}->{node}", {}).get("strength", 0.5)
                    
                    # Multiply intervals
                    if edge_strength >= 0:
                        lower_sum += parent_unc.lower_bound * edge_strength
                        upper_sum += parent_unc.upper_bound * edge_strength
                    else:
                        lower_sum += parent_unc.upper_bound * edge_strength
                        upper_sum += parent_unc.lower_bound * edge_strength
            
            propagated[node] = UncertaintyEstimate(
                value=(lower_sum + upper_sum) / 2,
                lower_bound=lower_sum,
                upper_bound=upper_sum,
                confidence_level=1.0,  # Interval arithmetic gives guaranteed bounds
                uncertainty_type=UncertaintyType.MODEL,
                source="interval"
            )
        
        return propagated
    
    async def _fuzzy_propagation(self,
                               initial: Dict[str, UncertaintyEstimate],
                               graph: Dict[str, Any]) -> Dict[str, UncertaintyEstimate]:
        """Propagate uncertainty using fuzzy logic"""
        # Simplified fuzzy propagation
        propagated = initial.copy()
        sorted_nodes = self._topological_sort(graph)
        
        for node in sorted_nodes:
            if node in propagated:
                continue
            
            parents = graph.get("parents", {}).get(node, [])
            if not parents:
                continue
            
            # Fuzzy combination
            combined_possibility = 0.0
            combined_necessity = 1.0
            
            for parent in parents:
                if parent in propagated:
                    parent_unc = propagated[parent]
                    edge_strength = graph.get("edges", {}).get(f"{parent}->{node}", {}).get("strength", 0.5)
                    
                    # Fuzzy implication
                    parent_membership = 1.0 - (parent_unc.upper_bound - parent_unc.value) / (parent_unc.upper_bound - parent_unc.lower_bound)
                    implied_membership = min(1.0, parent_membership * edge_strength)
                    
                    combined_possibility = max(combined_possibility, implied_membership)
                    combined_necessity = min(combined_necessity, implied_membership)
            
            # Convert back to uncertainty estimate
            fuzzy_value = (combined_possibility + combined_necessity) / 2
            fuzzy_width = combined_possibility - combined_necessity
            
            propagated[node] = UncertaintyEstimate(
                value=fuzzy_value,
                lower_bound=fuzzy_value - fuzzy_width / 2,
                upper_bound=fuzzy_value + fuzzy_width / 2,
                confidence_level=0.9,
                uncertainty_type=UncertaintyType.MODEL,
                source="fuzzy"
            )
        
        return propagated
    
    def _topological_sort(self, graph: Dict[str, Any]) -> List[str]:
        """Topological sort of graph nodes"""
        nodes = set(graph.get("nodes", []))
        in_degree = {node: 0 for node in nodes}
        
        # Calculate in-degrees
        for node in nodes:
            parents = graph.get("parents", {}).get(node, [])
            in_degree[node] = len(parents)
        
        # Kahn's algorithm
        queue = [node for node in nodes if in_degree[node] == 0]
        sorted_nodes = []
        
        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)
            
            # Update children
            children = graph.get("children", {}).get(node, [])
            for child in children:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return sorted_nodes
    
    async def identify_critical_uncertainties(self,
                                            uncertainties: Dict[str, UncertaintyEstimate],
                                            impact_analysis: Dict[str, float]) -> List[Dict[str, Any]]:
        """Find uncertainties that most impact conclusions"""
        critical = []
        
        for node, uncertainty in uncertainties.items():
            # Calculate uncertainty magnitude
            unc_magnitude = (uncertainty.upper_bound - uncertainty.lower_bound) / max(abs(uncertainty.value), 1.0)
            
            # Get impact score
            impact = impact_analysis.get(node, 0.5)
            
            # Calculate criticality
            criticality = unc_magnitude * impact
            
            if criticality > self.critical_threshold:
                critical.append({
                    "node": node,
                    "uncertainty": uncertainty,
                    "impact": impact,
                    "criticality": criticality,
                    "reduction_priority": self._calculate_reduction_priority(uncertainty, impact)
                })
        
        # Sort by criticality
        critical.sort(key=lambda x: x["criticality"], reverse=True)
        
        return critical
    
    def _calculate_reduction_priority(self, 
                                    uncertainty: UncertaintyEstimate, 
                                    impact: float) -> str:
        """Calculate priority for uncertainty reduction"""
        unc_magnitude = (uncertainty.upper_bound - uncertainty.lower_bound) / max(abs(uncertainty.value), 1.0)
        
        if impact > 0.8 and unc_magnitude > 0.5:
            return "critical"
        elif impact > 0.6 or unc_magnitude > 0.7:
            return "high"
        elif impact > 0.4 and unc_magnitude > 0.3:
            return "medium"
        else:
            return "low"
    
    def calculate_decision_robustness(self,
                                    decision_outcomes: Dict[str, float],
                                    uncertainties: Dict[str, UncertaintyEstimate]) -> float:
        """Calculate how robust a decision is to uncertainties"""
        # Simple robustness metric
        robustness_scores = []
        
        for outcome, value in decision_outcomes.items():
            if outcome in uncertainties:
                unc = uncertainties[outcome]
                
                # Check if decision changes within uncertainty bounds
                if value > 0:  # Positive outcome expected
                    robustness = unc.lower_bound / value if value != 0 else 0
                else:  # Negative outcome expected
                    robustness = unc.upper_bound / value if value != 0 else 0
                
                robustness_scores.append(min(1.0, max(0.0, robustness)))
        
        return sum(robustness_scores) / len(robustness_scores) if robustness_scores else 0.5
    
    def generate_uncertainty_report(self, 
                                  uncertainties: Dict[str, UncertaintyEstimate]) -> Dict[str, Any]:
        """Generate comprehensive uncertainty report"""
        report = {
            "summary": {},
            "by_type": defaultdict(list),
            "by_magnitude": defaultdict(list),
            "recommendations": []
        }
        
        # Overall statistics
        all_magnitudes = []
        for node, unc in uncertainties.items():
            magnitude = (unc.upper_bound - unc.lower_bound) / max(abs(unc.value), 1.0)
            all_magnitudes.append(magnitude)
            
            # Categorize by type
            report["by_type"][unc.uncertainty_type.value].append({
                "node": node,
                "magnitude": magnitude
            })
            
            # Categorize by magnitude
            if magnitude > 0.5:
                report["by_magnitude"]["high"].append(node)
            elif magnitude > 0.2:
                report["by_magnitude"]["medium"].append(node)
            else:
                report["by_magnitude"]["low"].append(node)
        
        # Summary statistics
        report["summary"] = {
            "total_uncertain_nodes": len(uncertainties),
            "average_uncertainty": np.mean(all_magnitudes) if all_magnitudes else 0,
            "max_uncertainty": max(all_magnitudes) if all_magnitudes else 0,
            "high_uncertainty_nodes": len(report["by_magnitude"]["high"])
        }
        
        # Recommendations
        if report["summary"]["average_uncertainty"] > 0.4:
            report["recommendations"].append("High overall uncertainty - consider gathering more data")
        
        if report["summary"]["high_uncertainty_nodes"] > 3:
            report["recommendations"].append("Multiple high-uncertainty nodes - focus on reducing uncertainty in critical paths")
        
        measurement_uncertainties = len(report["by_type"].get(UncertaintyType.MEASUREMENT.value, []))
        if measurement_uncertainties > 0:
            report["recommendations"].append(f"Address {measurement_uncertainties} measurement uncertainties through better data collection")
        
        return report

# ========================================================================================
# REASONING TEMPLATES
# ========================================================================================

@dataclass
class ReasoningTemplate:
    """Template for reusable reasoning patterns"""
    id: str
    name: str
    description: str
    pattern_type: str
    trigger_conditions: List[str]
    steps: List[Dict[str, Any]]
    expected_outputs: List[str]
    success_metrics: Dict[str, float]
    domain: str = "general"
    complexity: str = "medium"
    typical_duration: float = 1.0

class ReasoningTemplateSystem:
    """System for managing and applying reasoning templates"""
    
    def __init__(self):
        self.templates: Dict[str, ReasoningTemplate] = {}
        self.template_applications: List[Dict[str, Any]] = []
        self.success_history: Dict[str, List[float]] = defaultdict(list)
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize with common reasoning templates"""
        # Causal discovery template
        self.templates["causal_discovery_basic"] = ReasoningTemplate(
            id="causal_discovery_basic",
            name="Basic Causal Discovery",
            description="Standard approach to discovering causal relationships",
            pattern_type="causal",
            trigger_conditions=[
                "query_contains:why",
                "query_contains:cause",
                "query_contains:because"
            ],
            steps=[
                {"action": "identify_variables", "params": {"min_relevance": 0.5}},
                {"action": "test_correlations", "params": {"threshold": 0.3}},
                {"action": "apply_causal_criteria", "params": {"criteria": ["temporal", "mechanism"]}},
                {"action": "build_causal_graph", "params": {"prune_weak": True}}
            ],
            expected_outputs=["causal_graph", "key_relationships", "confidence_scores"],
            success_metrics={"relations_found": 3, "confidence": 0.6},
            domain="general",
            complexity="medium"
        )
        
        # Problem decomposition template
        self.templates["problem_decomposition"] = ReasoningTemplate(
            id="problem_decomposition",
            name="Hierarchical Problem Decomposition",
            description="Break complex problems into manageable sub-problems",
            pattern_type="analytical",
            trigger_conditions=[
                "high_complexity",
                "query_contains:how_to",
                "multiple_goals"
            ],
            steps=[
                {"action": "identify_main_goal", "params": {}},
                {"action": "decompose_recursively", "params": {"max_depth": 3}},
                {"action": "identify_dependencies", "params": {}},
                {"action": "order_subproblems", "params": {"strategy": "dependency_first"}}
            ],
            expected_outputs=["problem_tree", "subtask_list", "dependency_graph"],
            success_metrics={"decomposition_depth": 2, "subtasks_identified": 5},
            domain="general",
            complexity="high"
        )
        
        # Analogical reasoning template
        self.templates["analogical_reasoning"] = ReasoningTemplate(
            id="analogical_reasoning",
            name="Cross-Domain Analogical Reasoning",
            description="Apply patterns from one domain to another",
            pattern_type="creative",
            trigger_conditions=[
                "query_contains:like",
                "query_contains:similar",
                "cross_domain_request"
            ],
            steps=[
                {"action": "identify_source_pattern", "params": {"abstraction_level": "high"}},
                {"action": "find_target_mapping", "params": {"min_similarity": 0.4}},
                {"action": "validate_analogy", "params": {"check_constraints": True}},
                {"action": "generate_insights", "params": {"novelty_threshold": 0.6}}
            ],
            expected_outputs=["analogy_mapping", "transferred_insights", "validation_score"],
            success_metrics={"mapping_quality": 0.7, "insights_generated": 2},
            domain="general",
            complexity="medium"
        )
        
        # Hypothesis testing template
        self.templates["hypothesis_testing"] = ReasoningTemplate(
            id="hypothesis_testing",
            name="Scientific Hypothesis Testing",
            description="Test hypotheses using available evidence",
            pattern_type="empirical",
            trigger_conditions=[
                "query_contains:test",
                "query_contains:verify",
                "hypothesis_present"
            ],
            steps=[
                {"action": "formulate_hypothesis", "params": {"falsifiable": True}},
                {"action": "identify_predictions", "params": {"specific": True}},
                {"action": "gather_evidence", "params": {"unbiased": True}},
                {"action": "evaluate_hypothesis", "params": {"statistical_test": True}}
            ],
            expected_outputs=["test_results", "confidence_level", "alternative_explanations"],
            success_metrics={"evidence_quality": 0.7, "conclusion_strength": 0.6},
            domain="general",
            complexity="high"
        )
    
    async def extract_reasoning_template(self,
                                       successful_reasoning: Dict[str, Any]) -> Optional[ReasoningTemplate]:
        """Extract reusable pattern from successful reasoning"""
        # Analyze reasoning structure
        structure = self._analyze_reasoning_structure(successful_reasoning)
        
        if not self._is_pattern_extractable(structure):
            return None
        
        # Create new template
        template = ReasoningTemplate(
            id=f"extracted_{len(self.templates)}",
            name=f"Extracted Pattern: {structure['pattern_type']}",
            description=f"Pattern extracted from successful {structure['domain']} reasoning",
            pattern_type=structure['pattern_type'],
            trigger_conditions=self._extract_trigger_conditions(successful_reasoning),
            steps=self._extract_reasoning_steps(successful_reasoning),
            expected_outputs=structure['outputs'],
            success_metrics=self._extract_success_metrics(successful_reasoning),
            domain=structure['domain'],
            complexity=structure['complexity'],
            typical_duration=successful_reasoning.get('duration', 1.0)
        )
        
        # Validate template
        if self._validate_template(template):
            self.templates[template.id] = template
            return template
        
        return None
    
    def _analyze_reasoning_structure(self, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the structure of a reasoning process"""
        structure = {
            "pattern_type": "unknown",
            "domain": "general",
            "complexity": "medium",
            "outputs": [],
            "key_features": []
        }
        
        # Determine pattern type
        if reasoning.get("causal_paths"):
            structure["pattern_type"] = "causal"
        elif reasoning.get("decomposition"):
            structure["pattern_type"] = "analytical"
        elif reasoning.get("analogies"):
            structure["pattern_type"] = "creative"
        else:
            structure["pattern_type"] = "general"
        
        # Extract domain
        structure["domain"] = reasoning.get("domain", "general")
        
        # Assess complexity
        steps = reasoning.get("steps", [])
        if len(steps) > 10:
            structure["complexity"] = "high"
        elif len(steps) < 5:
            structure["complexity"] = "low"
        
        # Extract outputs
        structure["outputs"] = list(reasoning.get("results", {}).keys())
        
        return structure
    
    def _is_pattern_extractable(self, structure: Dict[str, Any]) -> bool:
        """Check if pattern is worth extracting"""
        # Must have clear structure
        if structure["pattern_type"] == "unknown":
            return False
        
        # Must have outputs
        if not structure["outputs"]:
            return False
        
        # Should be reasonably complex
        if structure["complexity"] == "low":
            return False
        
        return True
    
    def _extract_trigger_conditions(self, reasoning: Dict[str, Any]) -> List[str]:
        """Extract conditions that trigger this pattern"""
        conditions = []
        
        # Extract from input context
        input_context = reasoning.get("input_context", {})
        user_input = input_context.get("user_input", "").lower()
        
        # Keywords that triggered this reasoning
        trigger_keywords = ["why", "how", "what if", "test", "compare", "analyze"]
        for keyword in trigger_keywords:
            if keyword in user_input:
                conditions.append(f"query_contains:{keyword}")
        
        # Context conditions
        if input_context.get("high_uncertainty"):
            conditions.append("high_uncertainty")
        
        if input_context.get("multiple_options"):
            conditions.append("multiple_options")
        
        return conditions
    
    def _extract_reasoning_steps(self, reasoning: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract reasoning steps from successful reasoning"""
        steps = []
        
        reasoning_steps = reasoning.get("steps", [])
        for step in reasoning_steps:
            extracted_step = {
                "action": step.get("action", "unknown"),
                "params": step.get("parameters", {})
            }
            
            # Generalize parameters
            for key, value in extracted_step["params"].items():
                if isinstance(value, (int, float)):
                    # Keep numeric parameters as is
                    pass
                elif isinstance(value, str):
                    # Generalize specific strings
                    if value in ["specific_value", "context_dependent"]:
                        extracted_step["params"][key] = "TEMPLATE_VAR"
                else:
                    # Complex values become template variables
                    extracted_step["params"][key] = "TEMPLATE_VAR"
            
            steps.append(extracted_step)
        
        return steps
    
    def _extract_success_metrics(self, reasoning: Dict[str, Any]) -> Dict[str, float]:
        """Extract success metrics from reasoning"""
        metrics = {}
        
        results = reasoning.get("results", {})
        
        # Common metrics
        if "confidence" in results:
            metrics["confidence"] = results["confidence"]
        
        if "insights_count" in results:
            metrics["insights_generated"] = results["insights_count"]
        
        if "accuracy" in results:
            metrics["accuracy"] = results["accuracy"]
        
        # Pattern-specific metrics
        if reasoning.get("causal_paths"):
            metrics["relations_found"] = len(reasoning["causal_paths"])
        
        return metrics
    
    def _validate_template(self, template: ReasoningTemplate) -> bool:
        """Validate that template is well-formed"""
        # Must have steps
        if not template.steps:
            return False
        
        # Must have expected outputs
        if not template.expected_outputs:
            return False
        
        # Steps must have valid actions
        valid_actions = {
            "identify_variables", "test_correlations", "apply_criteria",
            "build_graph", "decompose", "map_concepts", "evaluate"
        }
        
        for step in template.steps:
            if step["action"] not in valid_actions and not step["action"].startswith("custom_"):
                return False
        
        return True
    
    async def apply_template(self, 
                           template_id: str, 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a reasoning template to current context"""
        if template_id not in self.templates:
            return {"error": "Template not found", "success": False}
        
        template = self.templates[template_id]
        
        # Check if template is applicable
        if not self._is_template_applicable(template, context):
            return {"error": "Template not applicable to context", "success": False}
        
        # Execute template steps
        results = {
            "template_id": template_id,
            "template_name": template.name,
            "steps_completed": [],
            "outputs": {},
            "success": False
        }
        
        try:
            for i, step in enumerate(template.steps):
                # Execute step (simplified - in reality would call actual reasoning functions)
                step_result = await self._execute_template_step(step, context, results)
                
                results["steps_completed"].append({
                    "step": i + 1,
                    "action": step["action"],
                    "result": step_result
                })
                
                # Check for early termination
                if step_result.get("terminate"):
                    break
            
            # Evaluate success
            success_evaluation = self._evaluate_template_success(results, template)
            results["success"] = success_evaluation["success"]
            results["success_metrics"] = success_evaluation["metrics"]
            
            # Record application
            self._record_template_application(template_id, results)
            
        except Exception as e:
            results["error"] = str(e)
            results["success"] = False
        
        return results
    
    def _is_template_applicable(self, template: ReasoningTemplate, context: Dict[str, Any]) -> bool:
        """Check if template conditions are met"""
        user_input = context.get("user_input", "").lower()
        
        # Check trigger conditions
        condition_met = False
        for condition in template.trigger_conditions:
            if condition.startswith("query_contains:"):
                keyword = condition.split(":")[1]
                if keyword in user_input:
                    condition_met = True
                    break
            elif condition == "high_complexity":
                if len(user_input.split()) > 30:
                    condition_met = True
                    break
            # Add other condition checks as needed
        
        return condition_met
    
    async def _execute_template_step(self, 
                                   step: Dict[str, Any], 
                                   context: Dict[str, Any],
                                   results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a template step with full functionality"""
        action = step["action"]
        params = step["params"].copy()
        
        # Replace template variables
        for key, value in params.items():
            if value == "TEMPLATE_VAR":
                params[key] = context.get(key, self._infer_param_value(key, context))
        
        step_result = {
            "action": action,
            "status": "pending",
            "outputs": {},
            "metadata": {}
        }
        
        try:
            # Route to appropriate execution method
            if action == "identify_variables":
                step_result = await self._execute_identify_variables(params, context, results)
                
            elif action == "test_correlations":
                step_result = await self._execute_test_correlations(params, context, results)
                
            elif action == "apply_causal_criteria":
                step_result = await self._execute_apply_causal_criteria(params, context, results)
                
            elif action == "build_causal_graph":
                step_result = await self._execute_build_causal_graph(params, context, results)
                
            elif action == "decompose_recursively":
                step_result = await self._execute_decompose_recursively(params, context, results)
                
            elif action == "identify_dependencies":
                step_result = await self._execute_identify_dependencies(params, context, results)
                
            elif action == "identify_source_pattern":
                step_result = await self._execute_identify_source_pattern(params, context, results)
                
            elif action == "find_target_mapping":
                step_result = await self._execute_find_target_mapping(params, context, results)
                
            elif action.startswith("custom_"):
                step_result = await self._execute_custom_action(action, params, context, results)
                
            else:
                step_result["status"] = "unsupported"
                step_result["error"] = f"Unknown action: {action}"
            
            # Record performance metrics
            step_result["metadata"]["execution_time"] = datetime.now().isoformat()
            
        except Exception as e:
            step_result["status"] = "failed"
            step_result["error"] = str(e)
            logger.error(f"Template step execution failed: {e}")
        
        return step_result
    
    async def _execute_identify_variables(self, params: Dict, context: Dict, results: Dict) -> Dict[str, Any]:
        """Execute variable identification step"""
        min_relevance = params.get("min_relevance", 0.5)
        
        # Extract variables from context
        user_input = context.get("user_input", "")
        variables = []
        
        # Use NLP-like analysis
        tokens = user_input.split()
        
        # Identify noun phrases as potential variables
        for i, token in enumerate(tokens):
            # Simple heuristics for variable detection
            if token[0].isupper() or token.endswith("ing") or token.endswith("ion"):
                relevance = self._calculate_token_relevance(token, context)
                
                if relevance >= min_relevance:
                    variables.append({
                        "name": token,
                        "type": self._infer_variable_type(token, context),
                        "relevance": relevance,
                        "position": i
                    })
        
        # Check against known concepts
        if hasattr(self, 'original_core'):
            for space in self.original_core.concept_spaces.values():
                for concept_id, concept in space.concepts.items():
                    concept_relevance = await self.relevance_calculator.calculate_relevance(
                        concept["name"], user_input, "semantic"
                    )
                    
                    if concept_relevance >= min_relevance:
                        variables.append({
                            "name": concept["name"],
                            "type": "concept",
                            "relevance": concept_relevance,
                            "source": "concept_space",
                            "space_id": space.id if hasattr(space, 'id') else "unknown"
                        })
        
        return {
            "action": "identify_variables",
            "status": "completed",
            "outputs": {
                "variables": variables,
                "variable_count": len(variables)
            },
            "metadata": {
                "min_relevance_used": min_relevance,
                "analysis_method": "hybrid"
            }
        }
    
    async def _execute_test_correlations(self, params: Dict, context: Dict, results: Dict) -> Dict[str, Any]:
        """Execute correlation testing step"""
        threshold = params.get("threshold", 0.3)
        
        # Get variables from previous step
        variables = results.get("outputs", {}).get("variables", [])
        
        if len(variables) < 2:
            return {
                "action": "test_correlations",
                "status": "skipped",
                "outputs": {"correlations": {}},
                "metadata": {"reason": "insufficient_variables"}
            }
        
        correlations = {}
        
        # Test pairwise correlations
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables[i+1:], i+1):
                # Calculate correlation based on co-occurrence and relationships
                correlation = await self._calculate_variable_correlation(var1, var2, context)
                
                if abs(correlation) >= threshold:
                    key = f"{var1['name']}-{var2['name']}"
                    correlations[key] = {
                        "value": correlation,
                        "significance": self._calculate_correlation_significance(correlation, context),
                        "type": "positive" if correlation > 0 else "negative"
                    }
        
        return {
            "action": "test_correlations",
            "status": "completed",
            "outputs": {
                "correlations": correlations,
                "significant_pairs": len(correlations),
                "strongest_correlation": max(correlations.items(), 
                                           key=lambda x: abs(x[1]["value"]))[0] if correlations else None
            }
        }
    
    async def _execute_apply_causal_criteria(self, params: Dict, context: Dict, results: Dict) -> Dict[str, Any]:
        """Execute causal criteria application"""
        criteria = params.get("criteria", ["temporal", "mechanism", "correlation"])
        correlations = results.get("outputs", {}).get("correlations", {})
        
        causal_relations = []
        
        for pair_key, correlation_data in correlations.items():
            var1_name, var2_name = pair_key.split("-")
            
            # Apply each criterion
            criteria_scores = {}
            
            if "temporal" in criteria:
                criteria_scores["temporal"] = await self._check_temporal_criterion(
                    var1_name, var2_name, context
                )
            
            if "mechanism" in criteria:
                criteria_scores["mechanism"] = await self._check_mechanism_criterion(
                    var1_name, var2_name, context
                )
            
            if "correlation" in criteria:
                criteria_scores["correlation"] = min(1.0, abs(correlation_data["value"]))
            
            # Calculate overall causal score
            if criteria_scores:
                causal_score = np.mean(list(criteria_scores.values()))
                
                if causal_score > 0.5:
                    causal_relations.append({
                        "cause": var1_name,
                        "effect": var2_name,
                        "strength": causal_score,
                        "criteria_scores": criteria_scores,
                        "confidence": self._calculate_causal_confidence(criteria_scores)
                    })
        
        return {
            "action": "apply_causal_criteria",
            "status": "completed",
            "outputs": {
                "causal_relations": causal_relations,
                "relations_found": len(causal_relations),
                "criteria_used": criteria
            }
        }
    
    async def _execute_build_causal_graph(self, params: Dict, context: Dict, results: Dict) -> Dict[str, Any]:
        """Execute causal graph building"""
        prune_weak = params.get("prune_weak", True)
        min_strength = params.get("min_strength", 0.3)
        
        causal_relations = results.get("outputs", {}).get("causal_relations", [])
        
        if not causal_relations:
            return {
                "action": "build_causal_graph",
                "status": "completed",
                "outputs": {"graph": None, "message": "No causal relations to build graph"},
            }
        
        # Build graph structure
        graph = {
            "nodes": {},
            "edges": [],
            "metadata": {
                "created": datetime.now().isoformat(),
                "context": context.get("user_input", "")[:100]
            }
        }
        
        # Add nodes
        all_variables = set()
        for relation in causal_relations:
            all_variables.add(relation["cause"])
            all_variables.add(relation["effect"])
        
        for var in all_variables:
            graph["nodes"][var] = {
                "id": var,
                "name": var,
                "type": "inferred",
                "properties": {}
            }
        
        # Add edges
        for relation in causal_relations:
            if not prune_weak or relation["strength"] >= min_strength:
                graph["edges"].append({
                    "source": relation["cause"],
                    "target": relation["effect"],
                    "strength": relation["strength"],
                    "confidence": relation["confidence"],
                    "criteria_scores": relation["criteria_scores"]
                })
        
        # Calculate graph metrics
        graph["metadata"]["metrics"] = {
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "density": len(graph["edges"]) / (len(graph["nodes"]) * (len(graph["nodes"]) - 1))
                      if len(graph["nodes"]) > 1 else 0
        }
        
        return {
            "action": "build_causal_graph",
            "status": "completed",
            "outputs": {
                "graph": graph,
                "graph_id": f"causal_graph_{hash(str(graph))}"
            }
        }
    
    def _evaluate_template_success(self, 
                                 results: Dict[str, Any], 
                                 template: ReasoningTemplate) -> Dict[str, Any]:
        """Evaluate if template application was successful"""
        evaluation = {
            "success": True,
            "metrics": {}
        }
        
        # Check if expected outputs were generated
        for expected_output in template.expected_outputs:
            if expected_output not in results.get("outputs", {}):
                evaluation["success"] = False
                evaluation["metrics"][f"missing_{expected_output}"] = True
        
        # Check success metrics
        for metric, threshold in template.success_metrics.items():
            actual_value = results.get("outputs", {}).get(metric, 0)
            evaluation["metrics"][metric] = actual_value
            
            if actual_value < threshold:
                evaluation["success"] = False
        
        return evaluation
    
    def _record_template_application(self, template_id: str, results: Dict[str, Any]):
        """Record template application for learning"""
        application = {
            "template_id": template_id,
            "timestamp": datetime.now(),
            "success": results["success"],
            "context_hash": hash(str(results.get("context", {}))),
            "metrics": results.get("success_metrics", {})
        }
        
        self.template_applications.append(application)
        
        # Update success history
        self.success_history[template_id].append(1.0 if results["success"] else 0.0)
        
        # Keep only recent history
        if len(self.success_history[template_id]) > 50:
            self.success_history[template_id] = self.success_history[template_id][-50:]
    
    def get_template_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend templates for given context"""
        recommendations = []
        
        for template_id, template in self.templates.items():
            if self._is_template_applicable(template, context):
                # Calculate recommendation score
                score = self._calculate_recommendation_score(template, context)
                
                recommendations.append({
                    "template_id": template_id,
                    "template_name": template.name,
                    "description": template.description,
                    "score": score,
                    "expected_duration": template.typical_duration,
                    "complexity": template.complexity,
                    "historical_success_rate": self._get_template_success_rate(template_id)
                })
        
        # Sort by score
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _calculate_recommendation_score(self, 
                                      template: ReasoningTemplate, 
                                      context: Dict[str, Any]) -> float:
        """Calculate how well template fits context"""
        score = 0.5  # Base score
        
        # Historical success rate
        success_rate = self._get_template_success_rate(template.id)
        score += success_rate * 0.3
        
        # Domain match
        context_domain = context.get("domain", "general")
        if template.domain == context_domain:
            score += 0.2
        elif template.domain == "general":
            score += 0.1
        
        # Complexity match
        context_complexity = self._assess_context_complexity(context)
        if template.complexity == context_complexity:
            score += 0.2
        
        return min(1.0, score)
    
    def _get_template_success_rate(self, template_id: str) -> float:
        """Get historical success rate of template"""
        history = self.success_history.get(template_id, [])
        if not history:
            return 0.5  # No history, assume average
        
        return sum(history) / len(history)
    
    def _assess_context_complexity(self, context: Dict[str, Any]) -> str:
        """Assess complexity of context"""
        user_input = context.get("user_input", "")
        
        # Simple heuristics
        word_count = len(user_input.split())
        unique_concepts = len(set(user_input.lower().split()))
        
        if word_count > 50 or unique_concepts > 20:
            return "high"
        elif word_count < 20 and unique_concepts < 10:
            return "low"
        else:
            return "medium"
    
    def _infer_param_value(self, param_name: str, context: Dict[str, Any]) -> Any:
        """Infer parameter value from context when not explicitly provided"""
        # Common parameter inference rules
        param_defaults = {
            "min_relevance": 0.5,
            "threshold": 0.3,
            "max_depth": 3,
            "min_strength": 0.3,
            "min_similarity": 0.4,
            "abstraction_level": "medium",
            "confidence_threshold": 0.6,
            "max_iterations": 100
        }
        
        # Check if parameter can be inferred from context
        if param_name == "domain":
            # Extract domain from user input
            user_input = context.get("user_input", "").lower()
            domains = ["health", "technology", "economics", "psychology", "environment", "education"]
            for domain in domains:
                if domain in user_input:
                    return domain
            return "general"
        
        elif param_name == "complexity":
            # Infer complexity from input length and structure
            user_input = context.get("user_input", "")
            word_count = len(user_input.split())
            if word_count > 50:
                return "high"
            elif word_count < 20:
                return "low"
            return "medium"
        
        elif param_name == "time_constraint":
            # Check if context indicates urgency
            constraints = context.get("constraints", [])
            if "time_sensitive" in constraints:
                return "urgent"
            return "normal"
        
        elif param_name == "evidence_requirement":
            # Based on criticality or confidence needs
            if context.get("high_stakes", False):
                return "high"
            return "moderate"
        
        # Return default if available
        return param_defaults.get(param_name, None)
    
    def _calculate_token_relevance(self, token: str, context: Dict[str, Any]) -> float:
        """Calculate relevance of a token to the context"""
        relevance = 0.0
        token_lower = token.lower()
        
        # Length-based relevance (longer tokens often more specific)
        if len(token) > 3:
            relevance += 0.1
        if len(token) > 6:
            relevance += 0.1
        
        # Capitalization suggests importance
        if token[0].isupper():
            relevance += 0.2
        
        # Check against domain keywords
        domain_keywords = context.get("domain_keywords", [])
        if token_lower in domain_keywords:
            relevance += 0.3
        
        # Check frequency in context
        user_input = context.get("user_input", "").lower()
        frequency = user_input.count(token_lower)
        if frequency > 1:
            relevance += min(0.2, frequency * 0.05)
        
        # Check if it's a known technical term
        technical_terms = {
            "algorithm", "process", "system", "function", "variable",
            "parameter", "model", "analysis", "hypothesis", "correlation",
            "causation", "effect", "factor", "component", "element"
        }
        if token_lower in technical_terms:
            relevance += 0.2
        
        # Check position in sentence (earlier often more important)
        words = context.get("user_input", "").split()
        if token in words[:5]:
            relevance += 0.1
        
        # Goal alignment
        if context.get("goal_context"):
            goals = context["goal_context"].get("active_goals", [])
            for goal in goals:
                if token_lower in goal.get("description", "").lower():
                    relevance += 0.2
                    break
        
        return min(1.0, relevance)
    
    def _infer_variable_type(self, token: str, context: Dict[str, Any]) -> str:
        """Infer the type of a variable from token and context"""
        token_lower = token.lower()
        
        # Check for common type patterns
        type_patterns = {
            "quantity": ["amount", "number", "count", "quantity", "volume", "size", "length"],
            "rate": ["rate", "speed", "velocity", "frequency", "per"],
            "state": ["state", "status", "condition", "mode", "phase"],
            "process": ["process", "procedure", "method", "algorithm", "workflow"],
            "entity": ["person", "user", "system", "object", "item", "thing"],
            "property": ["color", "shape", "texture", "quality", "attribute"],
            "temporal": ["time", "date", "duration", "period", "when", "timeline"],
            "spatial": ["location", "position", "place", "where", "coordinate"],
            "boolean": ["is", "has", "can", "should", "exists"],
            "categorical": ["type", "category", "class", "group", "kind"]
        }
        
        # Check token against patterns
        for var_type, patterns in type_patterns.items():
            if any(pattern in token_lower for pattern in patterns):
                return var_type
        
        # Check suffix patterns
        if token.endswith("ing"):
            return "process"
        elif token.endswith("ion") or token.endswith("ment"):
            return "state"
        elif token.endswith("er") or token.endswith("or"):
            return "entity"
        elif token.endswith("ity") or token.endswith("ness"):
            return "property"
        
        # Check context clues
        user_input = context.get("user_input", "")
        token_position = user_input.lower().find(token_lower)
        
        if token_position > 0:
            # Look at surrounding words
            before_text = user_input[:token_position].split()
            if before_text:
                last_word = before_text[-1].lower()
                if last_word in ["the", "a", "an"]:
                    return "entity"
                elif last_word in ["more", "less", "increase", "decrease"]:
                    return "quantity"
                elif last_word in ["is", "was", "becomes"]:
                    return "state"
        
        # Default to generic variable
        return "variable"
    
    async def _calculate_variable_correlation(self, var1: Dict[str, Any], var2: Dict[str, Any], 
                                            context: Dict[str, Any]) -> float:
        """Calculate correlation between two variables"""
        correlation = 0.0
        
        # Semantic similarity between variable names
        name_similarity = self._calculate_name_similarity(var1["name"], var2["name"])
        correlation += name_similarity * 0.3
        
        # Co-occurrence in context
        user_input = context.get("user_input", "").lower()
        var1_positions = [i for i, word in enumerate(user_input.split()) 
                         if var1["name"].lower() in word]
        var2_positions = [i for i, word in enumerate(user_input.split()) 
                         if var2["name"].lower() in word]
        
        if var1_positions and var2_positions:
            # Check proximity
            min_distance = min(abs(p1 - p2) for p1 in var1_positions for p2 in var2_positions)
            if min_distance <= 3:  # Within 3 words
                correlation += 0.4
            elif min_distance <= 7:  # Within 7 words
                correlation += 0.2
        
        # Type compatibility
        type_compatibility = self._check_type_compatibility(var1.get("type"), var2.get("type"))
        correlation += type_compatibility * 0.2
        
        # Check for known relationships
        if self._has_known_relationship(var1["name"], var2["name"], context):
            correlation += 0.3
        
        # Domain-specific correlations
        if var1.get("source") == "concept_space" and var2.get("source") == "concept_space":
            if var1.get("space_id") == var2.get("space_id"):
                correlation += 0.1  # Same conceptual space
        
        # Normalize to [-1, 1] range
        correlation = min(1.0, correlation)
        
        # Check for negative correlation indicators
        negative_indicators = ["opposite", "inverse", "contrary", "versus", "but", "however"]
        text_between = self._get_text_between_variables(var1["name"], var2["name"], user_input)
        if any(indicator in text_between.lower() for indicator in negative_indicators):
            correlation *= -1
        
        return correlation
    
    def _calculate_correlation_significance(self, correlation: float, context: Dict[str, Any]) -> float:
        """Calculate statistical significance of correlation"""
        # Simplified significance calculation
        significance = 0.0
        
        # Stronger correlations are more significant
        significance += abs(correlation) * 0.5
        
        # Context support increases significance
        user_input = context.get("user_input", "")
        if "strong" in user_input.lower() or "significant" in user_input.lower():
            significance += 0.2
        
        # Multiple mentions increase significance
        mention_count = context.get("mention_count", 1)
        if mention_count > 1:
            significance += min(0.3, mention_count * 0.1)
        
        # Domain expertise affects significance threshold
        domain = context.get("domain", "general")
        if domain in ["health", "science", "engineering"]:
            # Higher standards for technical domains
            significance *= 0.8
        
        return min(1.0, significance)
    
    async def _check_temporal_criterion(self, var1_name: str, var2_name: str, 
                                      context: Dict[str, Any]) -> float:
        """Check if temporal precedence exists between variables"""
        score = 0.0
        user_input = context.get("user_input", "").lower()
        
        # Temporal keywords
        temporal_patterns = [
            (["before", "prior to", "precedes"], ["after", "following", "follows"], 1.0),
            (["leads to", "causes", "triggers"], ["results from", "caused by", "triggered by"], 0.9),
            (["first", "initially", "begins"], ["then", "subsequently", "ends"], 0.8),
            (["earlier", "previous", "past"], ["later", "next", "future"], 0.7)
        ]
        
        var1_lower = var1_name.lower()
        var2_lower = var2_name.lower()
        
        # Check for explicit temporal relationships
        for before_words, after_words, weight in temporal_patterns:
            # Check var1 -> var2 temporal order
            for before in before_words:
                for after in after_words:
                    if f"{var1_lower} {before}" in user_input and f"{after} {var2_lower}" in user_input:
                        score = max(score, weight)
                    if f"{var1_lower} {before} {var2_lower}" in user_input:
                        score = max(score, weight)
            
            # Check var2 -> var1 temporal order (negative score)
            for before in before_words:
                for after in after_words:
                    if f"{var2_lower} {before}" in user_input and f"{after} {var1_lower}" in user_input:
                        score = min(score, -weight)
                    if f"{var2_lower} {before} {var1_lower}" in user_input:
                        score = min(score, -weight)
        
        # Check positional order as weak evidence
        if score == 0:
            var1_pos = user_input.find(var1_lower)
            var2_pos = user_input.find(var2_lower)
            if var1_pos >= 0 and var2_pos >= 0 and var1_pos < var2_pos:
                score = 0.3  # Weak temporal evidence
        
        # Check for simultaneity (reduces temporal score)
        simultaneity_words = ["simultaneously", "at the same time", "together", "concurrent"]
        if any(word in user_input for word in simultaneity_words):
            score *= 0.5
        
        return max(0, score)  # Return only positive scores
    
    async def _check_mechanism_criterion(self, var1_name: str, var2_name: str, 
                                       context: Dict[str, Any]) -> float:
        """Check if a plausible mechanism exists between variables"""
        score = 0.0
        user_input = context.get("user_input", "").lower()
        
        # Mechanism keywords
        mechanism_patterns = {
            "direct": ["directly", "immediately", "straight"],
            "process": ["through", "via", "by means of", "using"],
            "transformation": ["converts", "transforms", "changes", "becomes"],
            "influence": ["influences", "affects", "impacts", "modifies"],
            "transfer": ["transfers", "transmits", "passes", "communicates"],
            "activation": ["activates", "triggers", "initiates", "starts"],
            "regulation": ["regulates", "controls", "modulates", "adjusts"]
        }
        
        var1_lower = var1_name.lower()
        var2_lower = var2_name.lower()
        
        # Check for explicit mechanisms
        for mechanism_type, keywords in mechanism_patterns.items():
            for keyword in keywords:
                # Check if mechanism keyword appears between variables
                pattern1 = f"{var1_lower} {keyword} {var2_lower}"
                pattern2 = f"{var1_lower} {keyword}"
                pattern3 = f"{keyword} {var2_lower}"
                
                if pattern1 in user_input:
                    score = max(score, 0.9)
                elif pattern2 in user_input and pattern3 in user_input:
                    score = max(score, 0.7)
                elif keyword in user_input:
                    # Mechanism word present but not directly connected
                    score = max(score, 0.3)
        
        # Check for domain-specific mechanisms
        domain = context.get("domain", "general")
        domain_mechanisms = {
            "physics": ["force", "energy", "momentum", "field"],
            "chemistry": ["reaction", "bond", "catalyst", "oxidation"],
            "biology": ["enzyme", "receptor", "pathway", "signal"],
            "psychology": ["perception", "cognition", "emotion", "learning"],
            "economics": ["supply", "demand", "incentive", "market"]
        }
        
        if domain in domain_mechanisms:
            for mechanism in domain_mechanisms[domain]:
                if mechanism in user_input:
                    score = max(score, 0.5)
        
        # Check variable types for compatible mechanisms
        var1_type = context.get("variable_types", {}).get(var1_name, "unknown")
        var2_type = context.get("variable_types", {}).get(var2_name, "unknown")
        
        compatible_types = [
            ("process", "state"),
            ("entity", "property"),
            ("quantity", "quantity"),
            ("rate", "state")
        ]
        
        if (var1_type, var2_type) in compatible_types:
            score = max(score, 0.4)
        
        return score
    
    def _calculate_causal_confidence(self, criteria_scores: Dict[str, float]) -> float:
        """Calculate overall confidence in causal relationship"""
        if not criteria_scores:
            return 0.0
        
        # Weighted average of criteria
        weights = {
            "temporal": 0.3,
            "mechanism": 0.4,
            "correlation": 0.3
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for criterion, score in criteria_scores.items():
            weight = weights.get(criterion, 0.25)
            weighted_sum += score * weight
            total_weight += weight
        
        base_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Adjust confidence based on number of criteria met
        criteria_met = sum(1 for score in criteria_scores.values() if score > 0.5)
        
        if criteria_met >= 3:
            confidence_multiplier = 1.2
        elif criteria_met == 2:
            confidence_multiplier = 1.0
        elif criteria_met == 1:
            confidence_multiplier = 0.7
        else:
            confidence_multiplier = 0.4
        
        final_confidence = base_confidence * confidence_multiplier
        
        # Cap at reasonable bounds
        return max(0.1, min(0.95, final_confidence))
    
    # Helper methods for the above functions
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names"""
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Exact match
        if name1_lower == name2_lower:
            return 1.0
        
        # Substring match
        if name1_lower in name2_lower or name2_lower in name1_lower:
            return 0.7
        
        # Word overlap
        words1 = set(name1_lower.split('_'))
        words2 = set(name2_lower.split('_'))
        
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return overlap / union if union > 0 else 0.0
        
        return 0.0
    
    def _check_type_compatibility(self, type1: str, type2: str) -> float:
        """Check compatibility between variable types"""
        if type1 == type2:
            return 1.0
        
        # Compatible type pairs
        compatibility_matrix = {
            ("quantity", "rate"): 0.8,
            ("state", "process"): 0.7,
            ("entity", "property"): 0.8,
            ("temporal", "state"): 0.6,
            ("spatial", "entity"): 0.7,
            ("boolean", "state"): 0.6
        }
        
        # Check both directions
        return compatibility_matrix.get((type1, type2), 
               compatibility_matrix.get((type2, type1), 0.3))
    
    def _has_known_relationship(self, name1: str, name2: str, context: Dict[str, Any]) -> bool:
        """Check if variables have a known relationship"""
        # Common causal pairs
        known_relationships = [
            ("temperature", "pressure"),
            ("supply", "demand"),
            ("effort", "result"),
            ("input", "output"),
            ("cause", "effect"),
            ("stimulus", "response"),
            ("action", "reaction")
        ]
        
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        for rel1, rel2 in known_relationships:
            if (rel1 in name1_lower and rel2 in name2_lower) or \
               (rel2 in name1_lower and rel1 in name2_lower):
                return True
        
        return False
    
    def _get_text_between_variables(self, var1: str, var2: str, text: str) -> str:
        """Get text between two variables"""
        text_lower = text.lower()
        var1_lower = var1.lower()
        var2_lower = var2.lower()
        
        pos1 = text_lower.find(var1_lower)
        pos2 = text_lower.find(var2_lower)
        
        if pos1 >= 0 and pos2 >= 0:
            start = min(pos1, pos2) + len(var1 if pos1 < pos2 else var2)
            end = max(pos1, pos2)
            return text[start:end]
        
        return ""
