# nyx/core/a2a/enhanced_reasoning_core.py
"""
Enhanced Context-Aware Reasoning Core with all improvements incorporated.
Part 1: Core module with emotion integration, goal alignment, and memory patterns.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import json
from functools import lru_cache, wraps
import time

logger = logging.getLogger(__name__)

# ========================================================================================
# ERROR HANDLING
# ========================================================================================

class ReasoningError(Exception):
    """Base exception for reasoning errors"""
    pass

class CausalDiscoveryError(ReasoningError):
    """Specific error for causal discovery failures"""
    pass

class ConceptualBlendingError(ReasoningError):
    """Error in conceptual blending operations"""
    pass

class ReasoningTimeoutError(ReasoningError):
    """Reasoning operation timed out"""
    pass

# ========================================================================================
# CONFIGURATION MANAGEMENT
# ========================================================================================

@dataclass
class EmotionalInfluenceConfig:
    """Configuration for how emotions influence reasoning"""
    emotion: str
    discovery_threshold_modifier: float = 1.0
    exploration_bonus: float = 0.0
    depth_increase: int = 0
    evidence_requirement_modifier: float = 1.0
    exploration_penalty: float = 0.0
    focus_increase: float = 0.0
    reasoning_style_preference: str = "balanced"
    hypothesis_generation_boost: float = 0.0
    explanation_detail_level: str = "normal"

class ReasoningConfiguration:
    """Centralized configuration with validation"""
    
    def __init__(self):
        self.emotional_influences = {
            "Curiosity": EmotionalInfluenceConfig(
                emotion="Curiosity",
                discovery_threshold_modifier=0.8,
                exploration_bonus=0.3,
                depth_increase=2,
                hypothesis_generation_boost=0.4,
                reasoning_style_preference="exploratory",
                explanation_detail_level="detailed"
            ),
            "Anxiety": EmotionalInfluenceConfig(
                emotion="Anxiety",
                evidence_requirement_modifier=1.5,
                exploration_penalty=0.2,
                focus_increase=0.4,
                reasoning_style_preference="cautious",
                explanation_detail_level="reassuring"
            ),
            "Frustration": EmotionalInfluenceConfig(
                emotion="Frustration",
                discovery_threshold_modifier=0.9,
                focus_increase=0.5,
                reasoning_style_preference="direct",
                explanation_detail_level="concise"
            ),
            "Joy": EmotionalInfluenceConfig(
                emotion="Joy",
                exploration_bonus=0.2,
                hypothesis_generation_boost=0.2,
                reasoning_style_preference="creative",
                explanation_detail_level="enthusiastic"
            ),
            "Confusion": EmotionalInfluenceConfig(
                emotion="Confusion",
                evidence_requirement_modifier=1.3,
                depth_increase=1,
                reasoning_style_preference="systematic",
                explanation_detail_level="step_by_step"
            )
        }
        
        self.goal_alignment_configs = {
            "understanding": {
                "causal_discovery_weight": 0.8,
                "conceptual_exploration_weight": 0.2,
                "intervention_analysis_weight": 0.3,
                "preferred_depth": 3,
                "require_explanations": True
            },
            "problem_solving": {
                "causal_discovery_weight": 0.5,
                "conceptual_exploration_weight": 0.3,
                "intervention_analysis_weight": 0.9,
                "preferred_depth": 2,
                "require_explanations": False
            },
            "creative": {
                "causal_discovery_weight": 0.3,
                "conceptual_exploration_weight": 0.9,
                "intervention_analysis_weight": 0.4,
                "preferred_depth": 4,
                "require_explanations": True
            }
        }
        
        self.memory_pattern_configs = {
            "experiential": {
                "causal_weight": 0.8,
                "conceptual_weight": 0.2,
                "pattern_matching_threshold": 0.6
            },
            "reflective": {
                "causal_weight": 0.3,
                "conceptual_weight": 0.7,
                "pattern_matching_threshold": 0.7
            },
            "procedural": {
                "causal_weight": 0.6,
                "conceptual_weight": 0.4,
                "pattern_matching_threshold": 0.8
            }
        }
    
    def validate(self) -> bool:
        """Validate configuration consistency"""
        for emotion, config in self.emotional_influences.items():
            if config.discovery_threshold_modifier <= 0:
                raise ValueError(f"Invalid discovery threshold for {emotion}")
            if config.evidence_requirement_modifier <= 0:
                raise ValueError(f"Invalid evidence requirement for {emotion}")
        return True
    
    def get_emotional_config(self, emotion: str) -> EmotionalInfluenceConfig:
        """Get configuration for specific emotion with fallback"""
        return self.emotional_influences.get(
            emotion, 
            EmotionalInfluenceConfig(emotion="neutral")
        )

# ========================================================================================
# STATE MANAGEMENT
# ========================================================================================

@dataclass
class ReasoningAttempt:
    """Record of a reasoning attempt"""
    timestamp: datetime
    reasoning_type: str
    input_context: Dict[str, Any]
    approach: Dict[str, Any]
    results: Dict[str, Any]
    success: bool
    duration: float
    models_used: List[str] = field(default_factory=list)
    spaces_used: List[str] = field(default_factory=list)
    cross_module_integration: bool = False

class ReasoningState:
    """Proper state management for reasoning context"""
    
    def __init__(self):
        self.current_models: Set[str] = set()
        self.current_spaces: Set[str] = set()
        self.reasoning_history: List[ReasoningAttempt] = []
        self.performance_metrics: Dict[str, float] = {
            "avg_duration": 0.0,
            "success_rate": 0.0,
            "discovery_rate": 0.0,
            "integration_frequency": 0.0
        }
        self.active_templates: Dict[str, Any] = {}
        self.session_discoveries: List[Dict[str, Any]] = []
        
    def record_reasoning_attempt(self, attempt: ReasoningAttempt):
        """Track reasoning attempts for learning"""
        self.reasoning_history.append(attempt)
        self._update_metrics(attempt)
        
        # Keep only recent history
        if len(self.reasoning_history) > 100:
            self.reasoning_history = self.reasoning_history[-100:]
    
    def _update_metrics(self, attempt: ReasoningAttempt):
        """Update performance metrics based on new attempt"""
        history = self.reasoning_history[-20:]  # Last 20 attempts
        
        if history:
            # Average duration
            durations = [a.duration for a in history]
            self.performance_metrics["avg_duration"] = sum(durations) / len(durations)
            
            # Success rate
            successes = [a.success for a in history]
            self.performance_metrics["success_rate"] = sum(successes) / len(successes)
            
            # Discovery rate
            discoveries = [bool(a.results.get("new_relations_discovered", 0)) for a in history]
            self.performance_metrics["discovery_rate"] = sum(discoveries) / len(discoveries)
            
            # Integration frequency
            integrations = [a.cross_module_integration for a in history]
            self.performance_metrics["integration_frequency"] = sum(integrations) / len(integrations)
    
    def get_relevant_history(self, context: Dict[str, Any], max_items: int = 5) -> List[ReasoningAttempt]:
        """Retrieve relevant past reasoning for current context"""
        relevant = []
        
        # Extract context features
        current_domain = self._extract_domain(context)
        current_type = context.get("reasoning_type", "")
        
        # Find similar attempts
        for attempt in reversed(self.reasoning_history):
            similarity = 0.0
            
            # Domain match
            if self._extract_domain(attempt.input_context) == current_domain:
                similarity += 0.4
            
            # Type match
            if attempt.reasoning_type == current_type:
                similarity += 0.3
            
            # Success bonus
            if attempt.success:
                similarity += 0.2
            
            # Model/space overlap
            if (self.current_models.intersection(set(attempt.models_used)) or
                self.current_spaces.intersection(set(attempt.spaces_used))):
                similarity += 0.1
            
            if similarity >= 0.5:
                relevant.append(attempt)
                
            if len(relevant) >= max_items:
                break
        
        return relevant
    
    def _extract_domain(self, context: Dict[str, Any]) -> str:
        """Extract domain from context"""
        # Simple implementation - could be enhanced
        user_input = context.get("user_input", "").lower()
        for domain in ["health", "economics", "technology", "psychology", "environment"]:
            if domain in user_input:
                return domain
        return "general"

# ========================================================================================
# ENHANCED EMOTION-REASONING INTEGRATION
# ========================================================================================

class EmotionReasoningIntegrator:
    """Enhanced emotion-reasoning integration with rich parameter adjustment"""
    
    def __init__(self, config: ReasoningConfiguration):
        self.config = config
        self.emotion_history: List[Tuple[str, float, datetime]] = []
    
    async def adjust_reasoning_from_emotion(self, 
                                          emotional_data: Dict[str, Any],
                                          reasoning_params: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust reasoning parameters based on emotional state"""
        dominant_emotion = emotional_data.get("dominant_emotion")
        if not dominant_emotion:
            return reasoning_params
        
        emotion_name, strength = dominant_emotion
        emotion_config = self.config.get_emotional_config(emotion_name)
        
        # Record emotion
        self.emotion_history.append((emotion_name, strength, datetime.now()))
        
        # Create adjusted parameters
        adjusted_params = reasoning_params.copy()
        
        # Adjust discovery threshold
        if "discovery_threshold" in adjusted_params:
            adjusted_params["discovery_threshold"] *= emotion_config.discovery_threshold_modifier
        
        # Adjust evidence requirements
        if "min_evidence_strength" in adjusted_params:
            adjusted_params["min_evidence_strength"] *= emotion_config.evidence_requirement_modifier
        
        # Adjust exploration parameters
        if "exploration_depth" in adjusted_params:
            adjusted_params["exploration_depth"] += emotion_config.depth_increase
            
        if "exploration_breadth" in adjusted_params:
            if emotion_config.exploration_bonus > 0:
                adjusted_params["exploration_breadth"] *= (1 + emotion_config.exploration_bonus)
            elif emotion_config.exploration_penalty > 0:
                adjusted_params["exploration_breadth"] *= (1 - emotion_config.exploration_penalty)
        
        # Adjust reasoning style
        adjusted_params["reasoning_style"] = emotion_config.reasoning_style_preference
        
        # Adjust hypothesis generation
        if "hypothesis_generation_rate" in adjusted_params:
            adjusted_params["hypothesis_generation_rate"] *= (1 + emotion_config.hypothesis_generation_boost)
        
        # Adjust explanation style
        adjusted_params["explanation_detail"] = emotion_config.explanation_detail_level
        
        # Add emotion-specific adjustments
        if emotion_name == "Curiosity" and strength > 0.7:
            # High curiosity enables experimental features
            adjusted_params["enable_experimental_methods"] = True
            adjusted_params["novelty_seeking"] = 0.8
            
        elif emotion_name == "Anxiety" and strength > 0.6:
            # High anxiety requires more validation
            adjusted_params["require_validation"] = True
            adjusted_params["confidence_threshold"] = 0.8
            
        elif emotion_name == "Frustration" and strength > 0.7:
            # High frustration triggers alternative approaches
            adjusted_params["try_alternative_methods"] = True
            adjusted_params["simplification_preference"] = 0.7
        
        # Emotional momentum (recent emotional patterns affect reasoning)
        emotional_momentum = self._calculate_emotional_momentum()
        adjusted_params["emotional_momentum"] = emotional_momentum
        
        return adjusted_params
    
    def _calculate_emotional_momentum(self) -> Dict[str, float]:
        """Calculate emotional momentum from recent history"""
        if len(self.emotion_history) < 2:
            return {"stability": 1.0, "direction": "neutral", "intensity": 0.0}
        
        recent_emotions = self.emotion_history[-5:]
        
        # Calculate stability (how consistent emotions are)
        emotion_types = [e[0] for e in recent_emotions]
        unique_emotions = len(set(emotion_types))
        stability = 1.0 - (unique_emotions - 1) / len(emotion_types)
        
        # Calculate direction (positive/negative trend)
        positive_emotions = ["Joy", "Curiosity", "Satisfaction"]
        negative_emotions = ["Anxiety", "Frustration", "Confusion"]
        
        positive_count = sum(1 for e in emotion_types if e in positive_emotions)
        negative_count = sum(1 for e in emotion_types if e in negative_emotions)
        
        if positive_count > negative_count:
            direction = "positive"
        elif negative_count > positive_count:
            direction = "negative"
        else:
            direction = "neutral"
        
        # Calculate intensity trend
        intensities = [e[1] for e in recent_emotions]
        intensity_trend = (intensities[-1] - intensities[0]) / len(intensities)
        
        return {
            "stability": stability,
            "direction": direction,
            "intensity_trend": intensity_trend
        }

# ========================================================================================
# ENHANCED GOAL-DIRECTED REASONING
# ========================================================================================

class GoalDirectedReasoningEngine:
    """Enhanced goal-directed reasoning that actively guides the reasoning process"""
    
    def __init__(self, config: ReasoningConfiguration):
        self.config = config
        self.goal_reasoning_history: Dict[str, List[Dict[str, Any]]] = {}
    
    async def align_reasoning_with_goals(self,
                                       goal_data: Dict[str, Any],
                                       available_models: List[str],
                                       available_spaces: List[str]) -> Dict[str, Any]:
        """Actively guide reasoning based on goals"""
        active_goals = goal_data.get("active_goals", [])
        if not active_goals:
            return self._default_reasoning_plan()
        
        # Analyze goals and create reasoning plan
        reasoning_plan = {
            "primary_approach": None,
            "selected_models": [],
            "selected_spaces": [],
            "reasoning_sequence": [],
            "sub_goals": [],
            "success_criteria": [],
            "resource_allocation": {}
        }
        
        # Process each goal
        for goal in sorted(active_goals, key=lambda g: g.get("priority", 0), reverse=True):
            goal_type = self._categorize_goal(goal)
            goal_config = self.config.goal_alignment_configs.get(goal_type, {})
            
            # Generate sub-goals from causal understanding
            sub_goals = await self._generate_causal_subgoals(goal, available_models)
            reasoning_plan["sub_goals"].extend(sub_goals)
            
            # Select appropriate models and spaces
            selected_models = self._select_models_for_goal(
                goal, available_models, goal_config.get("causal_discovery_weight", 0.5)
            )
            selected_spaces = self._select_spaces_for_goal(
                goal, available_spaces, goal_config.get("conceptual_exploration_weight", 0.5)
            )
            
            reasoning_plan["selected_models"].extend(selected_models)
            reasoning_plan["selected_spaces"].extend(selected_spaces)
            
            # Define reasoning sequence
            sequence_step = {
                "goal_id": goal.get("id"),
                "goal_description": goal.get("description"),
                "approach": goal_type,
                "models": selected_models[:2],  # Top 2 models
                "spaces": selected_spaces[:2],   # Top 2 spaces
                "depth": goal_config.get("preferred_depth", 2),
                "require_explanation": goal_config.get("require_explanations", True)
            }
            
            reasoning_plan["reasoning_sequence"].append(sequence_step)
            
            # Define success criteria
            criteria = self._define_success_criteria(goal, goal_type)
            reasoning_plan["success_criteria"].extend(criteria)
        
        # Determine primary approach
        approach_votes = {}
        for step in reasoning_plan["reasoning_sequence"]:
            approach = step["approach"]
            approach_votes[approach] = approach_votes.get(approach, 0) + 1
        
        reasoning_plan["primary_approach"] = max(approach_votes.items(), key=lambda x: x[1])[0]
        
        # Allocate resources based on goal priorities
        total_priority = sum(g.get("priority", 0.5) for g in active_goals)
        for goal in active_goals:
            goal_priority = goal.get("priority", 0.5)
            reasoning_plan["resource_allocation"][goal.get("id")] = goal_priority / total_priority
        
        # Track goal-reasoning alignment
        self._track_goal_reasoning(active_goals, reasoning_plan)
        
        return reasoning_plan
    
    def _categorize_goal(self, goal: Dict[str, Any]) -> str:
        """Categorize goal to determine reasoning approach"""
        description = goal.get("description", "").lower()
        
        if any(word in description for word in ["understand", "explain", "why", "how"]):
            return "understanding"
        elif any(word in description for word in ["solve", "fix", "improve", "optimize"]):
            return "problem_solving"
        elif any(word in description for word in ["create", "imagine", "design", "invent"]):
            return "creative"
        else:
            return "general"
    
    async def _generate_causal_subgoals(self, 
                                      goal: Dict[str, Any], 
                                      available_models: List[str]) -> List[Dict[str, Any]]:
        """Generate sub-goals based on causal model analysis"""
        sub_goals = []
        
        # For each relevant model, identify intermediate nodes
        goal_keywords = set(goal.get("description", "").lower().split())
        
        for model_id in available_models[:3]:  # Check top 3 models
            # This would interact with actual causal model
            # For now, generating plausible sub-goals
            if "understand" in goal_keywords:
                sub_goals.append({
                    "parent_goal": goal.get("id"),
                    "description": f"Identify key causal factors in {model_id}",
                    "type": "causal_analysis",
                    "model": model_id
                })
            elif "improve" in goal_keywords:
                sub_goals.append({
                    "parent_goal": goal.get("id"),
                    "description": f"Find intervention points in {model_id}",
                    "type": "intervention_analysis",
                    "model": model_id
                })
        
        return sub_goals
    
    def _select_models_for_goal(self, 
                               goal: Dict[str, Any], 
                               available_models: List[str],
                               weight: float) -> List[str]:
        """Select models most relevant to goal"""
        # In production, this would analyze actual model content
        # For now, using heuristics
        goal_keywords = set(goal.get("description", "").lower().split())
        
        scored_models = []
        for model in available_models:
            score = weight
            
            # Boost score for keyword matches
            model_lower = model.lower()
            for keyword in goal_keywords:
                if keyword in model_lower:
                    score += 0.2
            
            scored_models.append((model, score))
        
        # Sort by score and return top models
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return [model for model, _ in scored_models[:3]]
    
    def _select_spaces_for_goal(self,
                               goal: Dict[str, Any],
                               available_spaces: List[str],
                               weight: float) -> List[str]:
        """Select concept spaces most relevant to goal"""
        # Similar to model selection
        goal_keywords = set(goal.get("description", "").lower().split())
        
        scored_spaces = []
        for space in available_spaces:
            score = weight
            
            space_lower = space.lower()
            for keyword in goal_keywords:
                if keyword in space_lower:
                    score += 0.2
            
            scored_spaces.append((space, score))
        
        scored_spaces.sort(key=lambda x: x[1], reverse=True)
        return [space for space, _ in scored_spaces[:3]]
    
    def _define_success_criteria(self, goal: Dict[str, Any], goal_type: str) -> List[Dict[str, Any]]:
        """Define measurable success criteria for goal"""
        criteria = []
        
        if goal_type == "understanding":
            criteria.extend([
                {
                    "criterion": "causal_paths_identified",
                    "threshold": 3,
                    "description": "Identify at least 3 causal paths"
                },
                {
                    "criterion": "explanation_coherence",
                    "threshold": 0.7,
                    "description": "Explanation coherence score > 0.7"
                }
            ])
        elif goal_type == "problem_solving":
            criteria.extend([
                {
                    "criterion": "intervention_points_found",
                    "threshold": 2,
                    "description": "Find at least 2 intervention points"
                },
                {
                    "criterion": "solution_feasibility",
                    "threshold": 0.6,
                    "description": "Solution feasibility > 0.6"
                }
            ])
        elif goal_type == "creative":
            criteria.extend([
                {
                    "criterion": "novel_concepts_generated",
                    "threshold": 5,
                    "description": "Generate at least 5 novel concepts"
                },
                {
                    "criterion": "blend_quality",
                    "threshold": 0.5,
                    "description": "Conceptual blend quality > 0.5"
                }
            ])
        
        return criteria
    
    def _track_goal_reasoning(self, goals: List[Dict[str, Any]], plan: Dict[str, Any]):
        """Track goal-reasoning alignment for learning"""
        for goal in goals:
            goal_id = goal.get("id")
            if goal_id not in self.goal_reasoning_history:
                self.goal_reasoning_history[goal_id] = []
            
            self.goal_reasoning_history[goal_id].append({
                "timestamp": datetime.now(),
                "plan": plan,
                "goal_state": goal.copy()
            })
    
    def _default_reasoning_plan(self) -> Dict[str, Any]:
        """Default reasoning plan when no specific goals"""
        return {
            "primary_approach": "exploratory",
            "selected_models": [],
            "selected_spaces": [],
            "reasoning_sequence": [{
                "approach": "general",
                "depth": 2,
                "require_explanation": True
            }],
            "sub_goals": [],
            "success_criteria": [],
            "resource_allocation": {}
        }
    
    async def suggest_goal_modifications(self, 
                                       goal: Dict[str, Any],
                                       causal_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest goal modifications based on causal impossibilities"""
        suggestions = []
        
        # Check if goal appears causally impossible
        if causal_analysis.get("feasibility_score", 1.0) < 0.3:
            suggestions.append({
                "type": "reformulation",
                "reason": "causal_impossibility",
                "suggestion": f"Consider reformulating goal to focus on controllable factors",
                "alternative_goals": self._generate_alternative_goals(goal, causal_analysis)
            })
        
        # Check if goal is too broad
        if causal_analysis.get("complexity_score", 0) > 0.8:
            suggestions.append({
                "type": "decomposition",
                "reason": "high_complexity",
                "suggestion": "Break down into smaller, manageable sub-goals",
                "sub_goals": self._decompose_complex_goal(goal, causal_analysis)
            })
        
        # Check if prerequisites are missing
        missing_prereqs = causal_analysis.get("missing_prerequisites", [])
        if missing_prereqs:
            suggestions.append({
                "type": "sequencing",
                "reason": "missing_prerequisites",
                "suggestion": "Address prerequisites first",
                "prerequisite_goals": [
                    {"description": f"Establish {prereq}", "priority": 0.8}
                    for prereq in missing_prereqs[:3]
                ]
            })
        
        return suggestions
    
    def _generate_alternative_goals(self, 
                                   original_goal: Dict[str, Any],
                                   causal_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative achievable goals"""
        alternatives = []
        
        # Focus on controllable factors
        controllable = causal_analysis.get("controllable_factors", [])
        if controllable:
            alternatives.append({
                "description": f"Optimize {controllable[0]} to influence outcomes",
                "feasibility": 0.8,
                "similarity_to_original": 0.7
            })
        
        return alternatives
    
    def _decompose_complex_goal(self,
                               goal: Dict[str, Any],
                               causal_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose complex goal into sub-goals"""
        sub_goals = []
        
        # Identify major causal pathways
        pathways = causal_analysis.get("major_pathways", [])
        for i, pathway in enumerate(pathways[:3]):
            sub_goals.append({
                "description": f"Address pathway {i+1}: {pathway.get('description', 'Unknown')}",
                "complexity": 0.4,
                "contributes_to": goal.get("id")
            })
        
        return sub_goals

# ========================================================================================
# ENHANCED MEMORY-INFORMED REASONING
# ========================================================================================

class MemoryInformedReasoningEngine:
    """Enhanced memory integration that actively uses past patterns"""
    
    def __init__(self, config: ReasoningConfiguration):
        self.config = config
        self.pattern_library: Dict[str, List[Dict[str, Any]]] = {
            "causal_patterns": [],
            "conceptual_patterns": [],
            "intervention_patterns": [],
            "failure_patterns": []
        }
        self.pattern_applications: List[Dict[str, Any]] = []
    
    async def inform_reasoning_from_memory(self,
                                         memory_data: Dict[str, Any],
                                         current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Use memory patterns to guide current reasoning"""
        retrieved_memories = memory_data.get("retrieved_memories", [])
        
        # Extract and categorize patterns
        patterns = self._extract_patterns_from_memories(retrieved_memories)
        
        # Match patterns to current situation
        applicable_patterns = self._match_patterns_to_context(patterns, current_context)
        
        # Generate reasoning guidance
        guidance = {
            "recommended_approaches": [],
            "patterns_to_apply": [],
            "warnings": [],
            "shortcuts": [],
            "confidence_adjustments": {}
        }
        
        for pattern in applicable_patterns:
            pattern_type = pattern.get("type")
            
            if pattern_type == "success_pattern":
                # Apply successful reasoning strategies
                guidance["recommended_approaches"].append({
                    "approach": pattern.get("approach"),
                    "expected_success_rate": pattern.get("success_rate", 0.7),
                    "typical_duration": pattern.get("avg_duration", 1.0),
                    "key_factors": pattern.get("key_factors", [])
                })
                
                # Create reasoning shortcut
                shortcut = self._create_reasoning_shortcut(pattern, current_context)
                if shortcut:
                    guidance["shortcuts"].append(shortcut)
                    
            elif pattern_type == "failure_pattern":
                # Warn about potential pitfalls
                guidance["warnings"].append({
                    "pattern": pattern.get("description"),
                    "failure_rate": pattern.get("failure_rate", 0.3),
                    "common_issues": pattern.get("issues", []),
                    "avoidance_strategies": pattern.get("avoidance", [])
                })
                
            elif pattern_type == "causal_pattern":
                # Apply learned causal relationships
                guidance["patterns_to_apply"].append({
                    "pattern": pattern,
                    "application_confidence": self._calculate_pattern_confidence(pattern, current_context),
                    "expected_relations": pattern.get("typical_relations", [])
                })
        
        # Adjust confidence based on memory support
        memory_confidence = self._calculate_memory_confidence(applicable_patterns)
        guidance["confidence_adjustments"] = {
            "memory_support": memory_confidence,
            "pattern_match_quality": self._assess_pattern_match_quality(applicable_patterns),
            "historical_success_rate": self._calculate_historical_success_rate(patterns)
        }
        
        # Update pattern library with new insights
        await self._update_pattern_library(patterns, current_context)
        
        return guidance
    
    def _extract_patterns_from_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract reusable patterns from memories"""
        patterns = []
        
        for memory in memories:
            memory_type = memory.get("memory_type", "")
            content = memory.get("content", {})
            
            if memory_type == "experience":
                # Extract causal patterns
                causal_pattern = self._extract_causal_pattern(content)
                if causal_pattern:
                    patterns.append(causal_pattern)
                    
            elif memory_type == "reflection":
                # Extract conceptual patterns
                conceptual_pattern = self._extract_conceptual_pattern(content)
                if conceptual_pattern:
                    patterns.append(conceptual_pattern)
                    
            elif memory_type == "procedural":
                # Extract intervention patterns
                intervention_pattern = self._extract_intervention_pattern(content)
                if intervention_pattern:
                    patterns.append(intervention_pattern)
        
        return patterns
    
    def _extract_causal_pattern(self, content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract causal pattern from experience memory"""
        if not content.get("causal_elements"):
            return None
        
        return {
            "type": "causal_pattern",
            "domain": content.get("domain", "general"),
            "typical_relations": content.get("causal_elements", {}).get("relations", []),
            "key_factors": content.get("causal_elements", {}).get("key_factors", []),
            "strength": content.get("pattern_strength", 0.5),
            "occurrences": content.get("occurrence_count", 1)
        }
    
    def _extract_conceptual_pattern(self, content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract conceptual pattern from reflection memory"""
        if not content.get("conceptual_insights"):
            return None
        
        return {
            "type": "conceptual_pattern",
            "domain": content.get("domain", "general"),
            "concept_clusters": content.get("conceptual_insights", {}).get("clusters", []),
            "blend_opportunities": content.get("conceptual_insights", {}).get("blends", []),
            "abstraction_level": content.get("abstraction_level", "medium"),
            "creative_potential": content.get("creative_score", 0.5)
        }
    
    def _extract_intervention_pattern(self, content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract intervention pattern from procedural memory"""
        if not content.get("intervention_data"):
            return None
        
        return {
            "type": "intervention_pattern",
            "intervention_type": content.get("intervention_data", {}).get("type"),
            "success_conditions": content.get("intervention_data", {}).get("conditions", []),
            "typical_outcomes": content.get("intervention_data", {}).get("outcomes", []),
            "success_rate": content.get("success_rate", 0.5),
            "side_effects": content.get("side_effects", [])
        }
    
    def _match_patterns_to_context(self,
                                 patterns: List[Dict[str, Any]],
                                 context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match extracted patterns to current context"""
        matched_patterns = []
        
        context_domain = self._extract_context_domain(context)
        context_keywords = set(context.get("user_input", "").lower().split())
        
        for pattern in patterns:
            match_score = 0.0
            
            # Domain matching
            if pattern.get("domain") == context_domain:
                match_score += 0.4
            
            # Keyword matching
            pattern_text = json.dumps(pattern).lower()
            keyword_matches = sum(1 for kw in context_keywords if kw in pattern_text)
            match_score += min(keyword_matches * 0.1, 0.3)
            
            # Type-specific matching
            if pattern["type"] == "causal_pattern" and "why" in context.get("user_input", "").lower():
                match_score += 0.2
            elif pattern["type"] == "conceptual_pattern" and "creative" in context.get("user_input", "").lower():
                match_score += 0.2
            elif pattern["type"] == "intervention_pattern" and "change" in context.get("user_input", "").lower():
                match_score += 0.2
            
            if match_score >= 0.4:
                pattern["match_score"] = match_score
                matched_patterns.append(pattern)
        
        # Sort by match score
        matched_patterns.sort(key=lambda p: p["match_score"], reverse=True)
        
        return matched_patterns[:5]  # Top 5 patterns
    
    def _create_reasoning_shortcut(self,
                                 pattern: Dict[str, Any],
                                 context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a reasoning shortcut based on successful pattern"""
        if pattern.get("success_rate", 0) < 0.7:
            return None  # Only create shortcuts for highly successful patterns
        
        shortcut = {
            "name": f"{pattern['type']}_{pattern.get('domain', 'general')}_shortcut",
            "description": "Fast-track reasoning based on proven pattern",
            "steps": [],
            "expected_duration": pattern.get("avg_duration", 1.0) * 0.5,  # 50% faster
            "confidence": pattern.get("success_rate", 0.7)
        }
        
        # Define shortcut steps based on pattern type
        if pattern["type"] == "causal_pattern":
            shortcut["steps"] = [
                {"action": "apply_known_relations", "data": pattern.get("typical_relations", [])},
                {"action": "focus_on_key_factors", "data": pattern.get("key_factors", [])},
                {"action": "skip_exhaustive_search", "data": {"depth_limit": 2}}
            ]
        elif pattern["type"] == "conceptual_pattern":
            shortcut["steps"] = [
                {"action": "use_concept_clusters", "data": pattern.get("concept_clusters", [])},
                {"action": "apply_blend_template", "data": pattern.get("blend_opportunities", [])},
                {"action": "skip_mapping_search", "data": {"use_cached": True}}
            ]
        
        return shortcut
    
    def _calculate_pattern_confidence(self, pattern: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate confidence in applying a pattern"""
        base_confidence = pattern.get("strength", 0.5)
        
        # Adjust based on pattern occurrences
        occurrences = pattern.get("occurrences", 1)
        if occurrences > 10:
            base_confidence *= 1.2
        elif occurrences < 3:
            base_confidence *= 0.8
        
        # Adjust based on domain match
        if pattern.get("domain") == self._extract_context_domain(context):
            base_confidence *= 1.1
        
        return min(1.0, base_confidence)
    
    def _calculate_memory_confidence(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence from memory support"""
        if not patterns:
            return 0.5  # Neutral confidence
        
        # Weighted average of pattern confidences
        total_confidence = 0.0
        total_weight = 0.0
        
        for pattern in patterns:
            weight = pattern.get("match_score", 0.5) * pattern.get("occurrences", 1)
            confidence = pattern.get("strength", 0.5)
            
            total_confidence += confidence * weight
            total_weight += weight
        
        return total_confidence / total_weight if total_weight > 0 else 0.5
    
    def _assess_pattern_match_quality(self, patterns: List[Dict[str, Any]]) -> float:
        """Assess quality of pattern matches"""
        if not patterns:
            return 0.0
        
        # Average match score of top patterns
        top_patterns = patterns[:3]
        avg_match = sum(p.get("match_score", 0) for p in top_patterns) / len(top_patterns)
        
        return avg_match
    
    def _calculate_historical_success_rate(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate historical success rate from patterns"""
        success_patterns = [p for p in patterns if p.get("type") == "success_pattern"]
        failure_patterns = [p for p in patterns if p.get("type") == "failure_pattern"]
        
        if not success_patterns and not failure_patterns:
            return 0.5  # No historical data
        
        total_attempts = len(success_patterns) + len(failure_patterns)
        success_rate = len(success_patterns) / total_attempts
        
        return success_rate
    
    def _extract_context_domain(self, context: Dict[str, Any]) -> str:
        """Extract domain from context"""
        user_input = context.get("user_input", "").lower()
        
        domains = ["health", "technology", "economics", "psychology", "environment", "education"]
        for domain in domains:
            if domain in user_input:
                return domain
        
        return "general"
    
    async def _update_pattern_library(self, new_patterns: List[Dict[str, Any]], context: Dict[str, Any]):
        """Update pattern library with new patterns"""
        for pattern in new_patterns:
            pattern_type = pattern.get("type")
            
            # Check if pattern already exists
            existing = False
            if pattern_type == "causal_pattern":
                pattern_list = self.pattern_library["causal_patterns"]
            elif pattern_type == "conceptual_pattern":
                pattern_list = self.pattern_library["conceptual_patterns"]
            elif pattern_type == "intervention_pattern":
                pattern_list = self.pattern_library["intervention_patterns"]
            else:
                continue
            
            # Simple deduplication
            for existing_pattern in pattern_list:
                if self._patterns_similar(pattern, existing_pattern):
                    # Update occurrence count
                    existing_pattern["occurrences"] = existing_pattern.get("occurrences", 1) + 1
                    existing = True
                    break
            
            if not existing:
                pattern["first_seen"] = datetime.now()
                pattern["occurrences"] = 1
                pattern_list.append(pattern)
    
    def _patterns_similar(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> bool:
        """Check if two patterns are similar"""
        if pattern1.get("type") != pattern2.get("type"):
            return False
        
        if pattern1.get("domain") != pattern2.get("domain"):
            return False
        
        # Type-specific similarity
        if pattern1["type"] == "causal_pattern":
            # Check if key factors overlap significantly
            factors1 = set(pattern1.get("key_factors", []))
            factors2 = set(pattern2.get("key_factors", []))
            
            if not factors1 or not factors2:
                return False
            
            overlap = len(factors1.intersection(factors2))
            return overlap / min(len(factors1), len(factors2)) > 0.7
        
        return False
    
    def build_reasoning_shortcuts(self) -> List[Dict[str, Any]]:
        """Build reasoning shortcuts from repeated patterns"""
        shortcuts = []
        
        # Analyze pattern library for frequent patterns
        for pattern_type, patterns in self.pattern_library.items():
            frequent_patterns = [p for p in patterns if p.get("occurrences", 1) > 5]
            
            for pattern in frequent_patterns:
                shortcut = {
                    "pattern_type": pattern_type,
                    "domain": pattern.get("domain"),
                    "trigger_conditions": self._extract_trigger_conditions(pattern),
                    "rapid_inference_steps": self._generate_rapid_inference(pattern),
                    "expected_accuracy": self._estimate_shortcut_accuracy(pattern),
                    "usage_count": 0
                }
                
                shortcuts.append(shortcut)
        
        return shortcuts
    
    def _extract_trigger_conditions(self, pattern: Dict[str, Any]) -> List[str]:
        """Extract conditions that trigger this pattern"""
        conditions = []
        
        if pattern["type"] == "causal_pattern":
            conditions.extend([
                f"domain:{pattern.get('domain')}",
                f"has_factors:{','.join(pattern.get('key_factors', [])[:3])}"
            ])
        
        return conditions
    
    def _generate_rapid_inference(self, pattern: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate rapid inference steps from pattern"""
        steps = []
        
        if pattern["type"] == "causal_pattern":
            steps = [
                {"action": "assume_relations", "data": str(pattern.get("typical_relations", []))},
                {"action": "prioritize_factors", "data": str(pattern.get("key_factors", []))}
            ]
        
        return steps
    
    def _estimate_shortcut_accuracy(self, pattern: Dict[str, Any]) -> float:
        """Estimate accuracy of shortcut based on pattern history"""
        occurrences = pattern.get("occurrences", 1)
        base_accuracy = 0.7
        
        # More occurrences = higher confidence
        if occurrences > 20:
            base_accuracy = 0.9
        elif occurrences > 10:
            base_accuracy = 0.8
        
        return base_accuracy

# ========================================================================================
# ENHANCED CREATIVE INTERVENTION GENERATION
# ========================================================================================

class CreativeInterventionGenerator:
    """Sophisticated creative intervention generation system"""
    
    def __init__(self):
        self.intervention_templates = self._load_intervention_templates()
        self.combination_strategies = self._load_combination_strategies()
        self.creativity_metrics = {
            "novelty": 0.0,
            "practicality": 0.0,
            "impact_potential": 0.0,
            "risk_level": 0.0
        }
    
    def _load_intervention_templates(self) -> Dict[str, Any]:
        """Load intervention templates"""
        return {
            "behavioral_nudge": {
                "description": "Subtle environmental changes to influence behavior",
                "applicable_to": ["behavior", "habit", "choice", "decision"],
                "components": ["trigger", "action", "reward"],
                "examples": ["visual cues", "default options", "social proof"]
            },
            "systems_redesign": {
                "description": "Fundamental restructuring of system components",
                "applicable_to": ["system", "process", "workflow", "structure"],
                "components": ["analysis", "redesign", "implementation", "monitoring"],
                "examples": ["process reengineering", "organizational restructuring"]
            },
            "leverage_multiplication": {
                "description": "Amplify impact through strategic leverage points",
                "applicable_to": ["influence", "resource", "network", "relationship"],
                "components": ["leverage_identification", "amplification_strategy", "cascade_planning"],
                "examples": ["viral mechanisms", "network effects", "force multipliers"]
            },
            "paradoxical_intervention": {
                "description": "Counter-intuitive approaches that work through indirect effects",
                "applicable_to": ["conflict", "resistance", "stuckness", "paradox"],
                "components": ["paradox_identification", "reversal_strategy", "reframe"],
                "examples": ["prescribing the symptom", "strategic withdrawal"]
            },
            "emergent_facilitation": {
                "description": "Create conditions for desired properties to emerge",
                "applicable_to": ["creativity", "innovation", "collaboration", "emergence"],
                "components": ["condition_setting", "constraint_design", "emergence_monitoring"],
                "examples": ["innovation labs", "self-organizing teams"]
            }
        }
    
    def _load_combination_strategies(self) -> Dict[str, Any]:
        """Load strategies for combining interventions"""
        return {
            "sequential": {
                "description": "Interventions applied in specific order",
                "timing": "staged",
                "interaction": "building"
            },
            "parallel": {
                "description": "Multiple interventions simultaneously",
                "timing": "concurrent",
                "interaction": "synergistic"
            },
            "adaptive": {
                "description": "Interventions that evolve based on feedback",
                "timing": "dynamic",
                "interaction": "responsive"
            },
            "layered": {
                "description": "Interventions at multiple system levels",
                "timing": "mixed",
                "interaction": "hierarchical"
            }
        }
    
    async def suggest_creative_intervention(self,
                                          model_id: str,
                                          context: Dict[str, Any],
                                          causal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sophisticated creative intervention suggestions"""
        
        # Analyze intervention opportunity
        opportunity = self._analyze_intervention_opportunity(causal_analysis, context)
        
        # Generate base interventions
        base_interventions = await self._generate_base_interventions(
            opportunity, causal_analysis, context
        )
        
        # Apply creativity enhancement
        creative_interventions = await self._enhance_creativity(
            base_interventions, context
        )
        
        # Generate combinations
        intervention_combinations = self._generate_intervention_combinations(
            creative_interventions, context
        )
        
        # Evaluate and rank
        ranked_interventions = self._evaluate_interventions(
            creative_interventions + intervention_combinations, context
        )
        
        # Select best intervention
        best_intervention = ranked_interventions[0] if ranked_interventions else None
        
        if not best_intervention:
            return self._generate_fallback_intervention(causal_analysis, context)
        
        # Add implementation details
        best_intervention["implementation"] = self._design_implementation_strategy(
            best_intervention, context
        )
        
        # Add learning mechanisms
        best_intervention["learning_mechanisms"] = self._design_learning_mechanisms(
            best_intervention
        )
        
        return best_intervention
    
    def _analyze_intervention_opportunity(self,
                                        causal_analysis: Dict[str, Any],
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the intervention opportunity"""
        return {
            "leverage_points": causal_analysis.get("intervention_points", []),
            "constraints": context.get("constraints", []),
            "resources": context.get("available_resources", []),
            "timeline": context.get("timeline", "flexible"),
            "risk_tolerance": context.get("risk_tolerance", "medium"),
            "desired_outcomes": causal_analysis.get("goal_nodes", []),
            "system_dynamics": causal_analysis.get("system_properties", {})
        }
    
    async def _generate_base_interventions(self,
                                         opportunity: Dict[str, Any],
                                         causal_analysis: Dict[str, Any],
                                         context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate base intervention options"""
        interventions = []
        
        for leverage_point in opportunity["leverage_points"][:5]:
            node_name = leverage_point.get("node_name", "").lower()
            
            # Match templates to leverage point
            applicable_templates = []
            for template_name, template in self.intervention_templates.items():
                if any(keyword in node_name for keyword in template["applicable_to"]):
                    applicable_templates.append((template_name, template))
            
            # Generate intervention for each applicable template
            for template_name, template in applicable_templates:
                intervention = {
                    "id": f"{template_name}_{leverage_point.get('node_id')}",
                    "name": f"{template_name.replace('_', ' ').title()} for {leverage_point.get('node_name')}",
                    "template": template_name,
                    "target_node": leverage_point.get("node_name"),
                    "description": template["description"],
                    "components": self._instantiate_components(
                        template["components"], leverage_point, context
                    ),
                    "expected_impact": leverage_point.get("expected_impact", "moderate"),
                    "feasibility": leverage_point.get("feasibility", 0.5),
                    "creativity_score": 0.3  # Base score
                }
                
                interventions.append(intervention)
        
        return interventions
    
    def _instantiate_components(self,
                               components: List[str],
                               leverage_point: Dict[str, Any],
                               context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Instantiate intervention components for specific context"""
        instantiated = []
        
        for component in components:
            instantiated.append({
                "component": component,
                "description": f"{component} for {leverage_point.get('node_name')}",
                "implementation_notes": self._generate_implementation_notes(component, context)
            })
        
        return instantiated
    
    def _generate_implementation_notes(self, component: str, context: Dict[str, Any]) -> str:
        """Generate implementation notes for component"""
        notes_map = {
            "trigger": "Identify natural trigger points in the system",
            "action": "Design specific actions aligned with goals",
            "reward": "Create feedback mechanisms for reinforcement",
            "analysis": "Conduct thorough system analysis",
            "redesign": "Develop new system architecture",
            "monitoring": "Establish metrics and tracking"
        }
        
        return notes_map.get(component, f"Implement {component} based on context")
    
    async def _enhance_creativity(self,
                                interventions: List[Dict[str, Any]],
                                context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhance interventions with creative elements"""
        enhanced = []
        
        for intervention in interventions:
            # Apply creativity techniques
            creative_intervention = intervention.copy()
            
            # Technique 1: Metaphorical transfer
            metaphor = self._generate_metaphor(intervention, context)
            if metaphor:
                creative_intervention["metaphorical_approach"] = metaphor
                creative_intervention["creativity_score"] += 0.2
            
            # Technique 2: Constraint addition/removal
            constraint_play = self._play_with_constraints(intervention, context)
            if constraint_play:
                creative_intervention["constraint_innovation"] = constraint_play
                creative_intervention["creativity_score"] += 0.15
            
            # Technique 3: Reversal
            reversal = self._try_reversal(intervention)
            if reversal:
                creative_intervention["reversal_option"] = reversal
                creative_intervention["creativity_score"] += 0.25
            
            # Technique 4: Cross-domain inspiration
            cross_domain = self._apply_cross_domain(intervention, context)
            if cross_domain:
                creative_intervention["cross_domain_insight"] = cross_domain
                creative_intervention["creativity_score"] += 0.2
            
            enhanced.append(creative_intervention)
        
        return enhanced
    
    def _generate_metaphor(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> Optional[str]:
        """Generate metaphorical approach to intervention"""
        target = intervention.get("target_node", "").lower()
        
        metaphors = {
            "behavior": "Think of behavior as a river - redirect rather than dam",
            "system": "Treat the system as an ecosystem - nurture balance",
            "process": "View process as a dance - find the rhythm",
            "decision": "Consider decisions as crossroads - illuminate paths",
            "relationship": "See relationships as gardens - cultivate growth"
        }
        
        for key, metaphor in metaphors.items():
            if key in target:
                return metaphor
        
        return None
    
    def _play_with_constraints(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> Optional[str]:
        """Play with constraints to enhance creativity"""
        constraints = context.get("constraints", [])
        
        if "limited_resources" in constraints:
            return "Turn resource limitation into creative advantage through frugal innovation"
        elif "time_sensitive" in constraints:
            return "Use time pressure to force rapid prototyping and iteration"
        else:
            return "Add artificial constraints to stimulate creative solutions"
    
    def _try_reversal(self, intervention: Dict[str, Any]) -> Optional[str]:
        """Try reversal technique for creativity"""
        template = intervention.get("template", "")
        
        reversals = {
            "behavioral_nudge": "Instead of nudging toward, create aversion from opposite",
            "systems_redesign": "Preserve what works, redesign around it",
            "leverage_multiplication": "Reduce dependency on single leverage points",
            "paradoxical_intervention": "Apply direct approach where paradox expected"
        }
        
        return reversals.get(template)
    
    def _apply_cross_domain(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> Optional[str]:
        """Apply cross-domain insights"""
        target = intervention.get("target_node", "").lower()
        
        cross_domain_insights = {
            "behavior": "Apply gaming mechanics for engagement",
            "system": "Use biological systems as design inspiration",
            "process": "Borrow from manufacturing lean principles",
            "decision": "Apply AI decision tree logic",
            "relationship": "Use social network dynamics"
        }
        
        for key, insight in cross_domain_insights.items():
            if key in target:
                return insight
        
        return None
    
    def _generate_intervention_combinations(self,
                                          interventions: List[Dict[str, Any]],
                                          context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate creative combinations of interventions"""
        combinations = []
        
        # Only combine if we have multiple interventions
        if len(interventions) < 2:
            return combinations
        
        # Try different combination strategies
        for strategy_name, strategy in self.combination_strategies.items():
            # Select 2-3 interventions that could work together
            compatible_interventions = self._find_compatible_interventions(
                interventions, strategy_name
            )
            
            if len(compatible_interventions) >= 2:
                combination = {
                    "id": f"combined_{strategy_name}_{len(combinations)}",
                    "name": f"{strategy_name.title()} Intervention Combination",
                    "strategy": strategy_name,
                    "components": compatible_interventions[:3],
                    "description": strategy["description"],
                    "timing": strategy["timing"],
                    "interaction": strategy["interaction"],
                    "creativity_score": 0.7,  # Combinations are inherently creative
                    "complexity": len(compatible_interventions) * 0.3
                }
                
                combinations.append(combination)
        
        return combinations
    
    def _find_compatible_interventions(self,
                                     interventions: List[Dict[str, Any]],
                                     strategy: str) -> List[Dict[str, Any]]:
        """Find interventions compatible with combination strategy"""
        if strategy == "sequential":
            # Look for interventions that build on each other
            return sorted(interventions, key=lambda x: x.get("feasibility", 0), reverse=True)[:3]
        elif strategy == "parallel":
            # Look for non-interfering interventions
            diverse = []
            templates_seen = set()
            for intervention in interventions:
                template = intervention.get("template")
                if template not in templates_seen:
                    diverse.append(intervention)
                    templates_seen.add(template)
            return diverse[:3]
        elif strategy == "adaptive":
            # Look for interventions with learning potential
            return [i for i in interventions if i.get("creativity_score", 0) > 0.5][:3]
        else:  # layered
            # Look for interventions at different levels
            return interventions[:3]
    
    def _evaluate_interventions(self,
                              interventions: List[Dict[str, Any]],
                              context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate and rank interventions"""
        evaluated = []
        
        for intervention in interventions:
            # Calculate comprehensive score
            scores = {
                "creativity": intervention.get("creativity_score", 0),
                "feasibility": intervention.get("feasibility", 0.5),
                "impact": self._estimate_impact(intervention),
                "risk": self._assess_risk(intervention, context),
                "resource_efficiency": self._assess_efficiency(intervention, context)
            }
            
            # Weight scores based on context
            weights = self._get_evaluation_weights(context)
            
            total_score = sum(scores[k] * weights.get(k, 1.0) for k in scores)
            
            intervention["evaluation"] = {
                "scores": scores,
                "total_score": total_score,
                "strengths": [k for k, v in scores.items() if v > 0.7],
                "weaknesses": [k for k, v in scores.items() if v < 0.3]
            }
            
            evaluated.append(intervention)
        
        # Sort by total score
        evaluated.sort(key=lambda x: x["evaluation"]["total_score"], reverse=True)
        
        return evaluated
    
    def _estimate_impact(self, intervention: Dict[str, Any]) -> float:
        """Estimate potential impact of intervention"""
        base_impact = 0.5
        
        # Adjust based on intervention type
        if intervention.get("template") == "leverage_multiplication":
            base_impact += 0.3
        elif intervention.get("template") == "systems_redesign":
            base_impact += 0.2
        
        # Adjust for combinations
        if "components" in intervention and len(intervention["components"]) > 1:
            base_impact += 0.2
        
        return min(1.0, base_impact)
    
    def _assess_risk(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Assess risk level of intervention"""
        base_risk = 0.3
        
        # Higher creativity often means higher risk
        base_risk += intervention.get("creativity_score", 0) * 0.3
        
        # Paradoxical interventions are riskier
        if intervention.get("template") == "paradoxical_intervention":
            base_risk += 0.3
        
        # Combinations add complexity risk
        if "components" in intervention:
            base_risk += len(intervention["components"]) * 0.1
        
        return min(1.0, base_risk)
    
    def _assess_efficiency(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Assess resource efficiency"""
        constraints = context.get("constraints", [])
        
        base_efficiency = 0.6
        
        # Simple interventions are more efficient
        if intervention.get("template") == "behavioral_nudge":
            base_efficiency += 0.3
        
        # Complex combinations less efficient
        if "components" in intervention:
            base_efficiency -= len(intervention["components"]) * 0.1
        
        return max(0.1, base_efficiency)
    
    def _get_evaluation_weights(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Get evaluation weights based on context"""
        weights = {
            "creativity": 1.0,
            "feasibility": 1.0,
            "impact": 1.0,
            "risk": -0.5,  # Negative weight for risk
            "resource_efficiency": 0.8
        }
        
        # Adjust based on context
        if "limited_resources" in context.get("constraints", []):
            weights["resource_efficiency"] = 1.5
            weights["feasibility"] = 1.2
        
        if context.get("risk_tolerance") == "high":
            weights["risk"] = -0.2  # Less penalty for risk
            weights["creativity"] = 1.5
        elif context.get("risk_tolerance") == "low":
            weights["risk"] = -1.0  # More penalty for risk
            weights["feasibility"] = 1.5
        
        return weights
    
    def _design_implementation_strategy(self,
                                      intervention: Dict[str, Any],
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Design detailed implementation strategy"""
        strategy = {
            "phases": [],
            "resources_required": [],
            "success_metrics": [],
            "risk_mitigation": [],
            "adaptation_triggers": []
        }
        
        # Define implementation phases
        if "components" in intervention:
            # Multi-component intervention
            for i, component in enumerate(intervention["components"]):
                strategy["phases"].append({
                    "phase": i + 1,
                    "name": f"Implement {component.get('name', 'Component')}",
                    "duration": "2-4 weeks",
                    "activities": component.get("components", []),
                    "dependencies": [f"Phase {j+1}" for j in range(i)]
                })
        else:
            # Single intervention
            strategy["phases"] = self._generate_standard_phases(intervention)
        
        # Define resources
        strategy["resources_required"] = self._identify_required_resources(intervention, context)
        
        # Define success metrics
        strategy["success_metrics"] = self._define_success_metrics(intervention)
        
        # Risk mitigation
        strategy["risk_mitigation"] = self._design_risk_mitigation(intervention)
        
        # Adaptation triggers
        strategy["adaptation_triggers"] = self._define_adaptation_triggers(intervention)
        
        return strategy
    
    def _generate_standard_phases(self, intervention: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate standard implementation phases"""
        return [
            {
                "phase": 1,
                "name": "Preparation",
                "duration": "1 week",
                "activities": ["Stakeholder alignment", "Resource gathering", "Baseline measurement"]
            },
            {
                "phase": 2,
                "name": "Pilot",
                "duration": "2 weeks",
                "activities": ["Small-scale implementation", "Rapid feedback collection", "Adjustment"]
            },
            {
                "phase": 3,
                "name": "Full Implementation",
                "duration": "4-6 weeks",
                "activities": ["Scale up", "Monitor impact", "Continuous optimization"]
            },
            {
                "phase": 4,
                "name": "Evaluation",
                "duration": "1 week",
                "activities": ["Impact assessment", "Lessons learned", "Next steps planning"]
            }
        ]
    
    def _identify_required_resources(self,
                                   intervention: Dict[str, Any],
                                   context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify required resources for implementation"""
        resources = []
        
        # Human resources
        resources.append({
            "type": "human",
            "description": "Project coordinator",
            "quantity": "1 FTE",
            "duration": "Full project"
        })
        
        # Technical resources based on intervention type
        if intervention.get("template") == "systems_redesign":
            resources.append({
                "type": "technical",
                "description": "System analysis tools",
                "quantity": "As needed",
                "duration": "Phase 1-2"
            })
        
        # Financial resources
        resources.append({
            "type": "financial",
            "description": "Implementation budget",
            "quantity": "TBD based on scope",
            "duration": "Full project"
        })
        
        return resources
    
    def _define_success_metrics(self, intervention: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define success metrics for intervention"""
        metrics = []
        
        # Universal metrics
        metrics.extend([
            {
                "metric": "Implementation completion",
                "target": "100%",
                "measurement": "Phase completion tracking"
            },
            {
                "metric": "Stakeholder satisfaction",
                "target": ">80%",
                "measurement": "Survey"
            }
        ])
        
        # Intervention-specific metrics
        if intervention.get("expected_impact") == "very_high":
            metrics.append({
                "metric": "System improvement",
                "target": ">30%",
                "measurement": "Pre/post comparison"
            })
        
        return metrics
    
    def _design_risk_mitigation(self, intervention: Dict[str, Any]) -> List[Dict[str, str]]:
        """Design risk mitigation strategies"""
        mitigations = []
        
        # Based on identified risks
        if intervention.get("evaluation", {}).get("scores", {}).get("risk", 0) > 0.7:
            mitigations.extend([
                {
                    "risk": "High implementation complexity",
                    "mitigation": "Phase implementation with go/no-go decisions"
                },
                {
                    "risk": "Stakeholder resistance",
                    "mitigation": "Extensive communication and involvement plan"
                }
            ])
        
        # Creative interventions need special handling
        if intervention.get("creativity_score", 0) > 0.7:
            mitigations.append({
                "risk": "Unpredictable outcomes",
                "mitigation": "Small-scale pilot with careful monitoring"
            })
        
        return mitigations
    
    def _define_adaptation_triggers(self, intervention: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define triggers for adapting the intervention"""
        triggers = []
        
        triggers.extend([
            {
                "trigger": "Success metric below 50% of target",
                "action": "Pause and reassess approach"
            },
            {
                "trigger": "Unexpected positive outcome",
                "action": "Accelerate and expand implementation"
            },
            {
                "trigger": "Resource constraints emerge",
                "action": "Simplify intervention components"
            }
        ])
        
        return triggers
    
    def _design_learning_mechanisms(self, intervention: Dict[str, Any]) -> Dict[str, Any]:
        """Design mechanisms for learning from intervention"""
        return {
            "data_collection": [
                {
                    "type": "quantitative",
                    "methods": ["Metrics tracking", "Statistical analysis"],
                    "frequency": "Weekly"
                },
                {
                    "type": "qualitative",
                    "methods": ["Interviews", "Observation", "Feedback sessions"],
                    "frequency": "Bi-weekly"
                }
            ],
            "analysis_approach": {
                "regular_reviews": "Weekly team reviews",
                "pattern_identification": "Look for unexpected effects",
                "causal_analysis": "Update causal models with findings"
            },
            "knowledge_capture": {
                "documentation": "Detailed implementation journal",
                "pattern_extraction": "Identify reusable patterns",
                "failure_analysis": "Document what didn't work and why"
            },
            "dissemination": {
                "internal": "Team learning sessions",
                "external": "Case study publication",
                "integration": "Update intervention library"
            }
        }
    
    def _generate_fallback_intervention(self,
                                      causal_analysis: Dict[str, Any],
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback intervention if creative generation fails"""
        return {
            "id": "fallback_standard",
            "name": "Standard Systematic Intervention",
            "description": "Apply systematic improvement approach",
            "template": "systems_redesign",
            "components": [
                {
                    "component": "analysis",
                    "description": "Thorough current state analysis"
                },
                {
                    "component": "design",
                    "description": "Evidence-based redesign"
                },
                {
                    "component": "implementation",
                    "description": "Phased rollout with monitoring"
                }
            ],
            "creativity_score": 0.2,
            "feasibility": 0.8,
            "expected_impact": "moderate"
        }
