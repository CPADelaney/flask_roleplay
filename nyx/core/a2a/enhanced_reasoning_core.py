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
    
    async def _generate_alternative_goals(self, 
                                       original_goal: Dict[str, Any],
                                       causal_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate sophisticated alternative achievable goals"""
        alternatives = []
        
        # Extract key components from original goal
        goal_desc = original_goal.get("description", "").lower()
        goal_keywords = set(goal_desc.split())
        goal_priority = original_goal.get("priority", 0.5)
        
        # Strategy 1: Focus on controllable factors
        controllable = causal_analysis.get("controllable_factors", [])
        if controllable:
            for factor in controllable[:3]:  # Top 3 controllable factors
                # Generate goal focused on this factor
                alternative = {
                    "description": f"Optimize {factor} to improve overall outcomes",
                    "feasibility": 0.8,
                    "similarity_to_original": self._calculate_goal_similarity(
                        f"optimize {factor}", goal_desc
                    ),
                    "type": "controllable_focus",
                    "parent_goal": original_goal.get("id"),
                    "priority": goal_priority * 0.8,
                    "rationale": f"Direct control over {factor} makes this achievable",
                    "expected_impact": causal_analysis.get("factor_impacts", {}).get(factor, 0.5)
                }
                alternatives.append(alternative)
        
        # Strategy 2: Partial goal satisfaction
        if len(goal_keywords) > 3:
            # Break down complex goal into components
            goal_components = self._extract_goal_components(goal_desc)
            
            for component in goal_components[:2]:
                alternative = {
                    "description": f"Achieve {component} as a step toward larger goal",
                    "feasibility": 0.7,
                    "similarity_to_original": 0.6,
                    "type": "partial_satisfaction",
                    "parent_goal": original_goal.get("id"),
                    "priority": goal_priority * 0.7,
                    "rationale": "Focusing on achievable components builds momentum",
                    "expected_impact": 0.4
                }
                alternatives.append(alternative)
        
        # Strategy 3: Indirect approach through causal chains
        indirect_paths = causal_analysis.get("indirect_paths", [])
        for path in indirect_paths[:2]:
            if path.get("feasibility", 0) > 0.5:
                intermediate_target = path.get("intermediate_nodes", [None])[0]
                if intermediate_target:
                    alternative = {
                        "description": f"Target {intermediate_target} to indirectly influence outcomes",
                        "feasibility": path.get("feasibility", 0.6),
                        "similarity_to_original": 0.5,
                        "type": "indirect_approach",
                        "parent_goal": original_goal.get("id"),
                        "priority": goal_priority * 0.6,
                        "rationale": f"Indirect path through {intermediate_target} is more feasible",
                        "causal_path": path,
                        "expected_impact": path.get("total_effect", 0.3)
                    }
                    alternatives.append(alternative)
        
        # Strategy 4: Constraint relaxation
        constraints = causal_analysis.get("binding_constraints", [])
        if constraints:
            relaxed_goal = self._relax_goal_constraints(original_goal, constraints)
            alternative = {
                "description": relaxed_goal["description"],
                "feasibility": 0.75,
                "similarity_to_original": 0.8,
                "type": "constraint_relaxation",
                "parent_goal": original_goal.get("id"),
                "priority": goal_priority * 0.9,
                "rationale": "Relaxing constraints makes goal achievable",
                "relaxed_constraints": relaxed_goal["relaxed"],
                "expected_impact": 0.6
            }
            alternatives.append(alternative)
        
        # Strategy 5: Time-based alternatives
        if causal_analysis.get("time_dynamics"):
            # Short-term version
            short_term = {
                "description": f"Short-term: {self._create_short_term_goal(goal_desc)}",
                "feasibility": 0.85,
                "similarity_to_original": 0.7,
                "type": "temporal_adjustment",
                "parent_goal": original_goal.get("id"),
                "priority": goal_priority,
                "time_horizon": "short",
                "rationale": "Short-term goals are more predictable and controllable",
                "expected_impact": 0.5
            }
            alternatives.append(short_term)
        
        # Rank alternatives by combined score
        for alt in alternatives:
            alt["combined_score"] = (
                alt["feasibility"] * 0.4 +
                alt["similarity_to_original"] * 0.3 +
                alt["expected_impact"] * 0.3
            )
        
        alternatives.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return alternatives[:5]  # Return top 5 alternatives
    
    def _decompose_complex_goal(self,
                               goal: Dict[str, Any],
                               causal_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Sophisticated decomposition of complex goals into sub-goals"""
        sub_goals = []
        
        goal_desc = goal.get("description", "")
        goal_id = goal.get("id", "goal")
        
        # Strategy 1: Linguistic decomposition
        linguistic_components = self._linguistic_goal_decomposition(goal_desc)
        
        # Strategy 2: Causal pathway decomposition
        causal_sub_goals = self._causal_pathway_decomposition(goal, causal_analysis)
        
        # Strategy 3: Functional decomposition
        functional_sub_goals = self._functional_decomposition(goal)
        
        # Strategy 4: Temporal decomposition
        temporal_sub_goals = self._temporal_decomposition(goal, causal_analysis)
        
        # Merge and deduplicate sub-goals
        all_sub_goals = linguistic_components + causal_sub_goals + functional_sub_goals + temporal_sub_goals
        
        # Create proper sub-goal structures
        seen_descriptions = set()
        for i, sub_goal_desc in enumerate(all_sub_goals):
            if sub_goal_desc.lower() not in seen_descriptions:
                seen_descriptions.add(sub_goal_desc.lower())
                
                sub_goal = {
                    "id": f"{goal_id}_sub_{i}",
                    "description": sub_goal_desc,
                    "parent_goal": goal_id,
                    "complexity": self._assess_sub_goal_complexity(sub_goal_desc),
                    "dependencies": self._identify_sub_goal_dependencies(
                        sub_goal_desc, all_sub_goals, causal_analysis
                    ),
                    "priority": self._calculate_sub_goal_priority(
                        sub_goal_desc, goal, causal_analysis
                    ),
                    "estimated_effort": self._estimate_sub_goal_effort(sub_goal_desc),
                    "success_criteria": self._define_sub_goal_criteria(sub_goal_desc),
                    "contributes_to": goal_id,
                    "contribution_weight": 1.0 / max(1, len(all_sub_goals))
                }
                
                sub_goals.append(sub_goal)
        
        # Order sub-goals by dependencies and priority
        ordered_sub_goals = self._order_sub_goals_by_dependencies(sub_goals)
        
        return ordered_sub_goals

    def _linguistic_goal_decomposition(self, goal_desc: str) -> List[str]:
        """Decompose goal using linguistic analysis"""
        sub_goals = []
        
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        
        # Pattern 1: Compound sentences (and, or)
        if " and " in goal_desc:
            parts = goal_desc.split(" and ")
            sub_goals.extend([part.strip() for part in parts if len(part.strip()) > 5])
        
        # Pattern 2: Listed items
        list_pattern = r"(?:including|such as|like|e\.g\.|for example)\s*[:,-]?\s*(.+)"
        list_match = re.search(list_pattern, goal_desc, re.IGNORECASE)
        if list_match:
            items = re.split(r'[,;]|\band\b', list_match.group(1))
            sub_goals.extend([f"Address {item.strip()}" for item in items if item.strip()])
        
        # Pattern 3: Action-object pairs
        action_patterns = [
            r"(improve|increase|enhance|optimize|reduce|minimize|maximize)\s+(\w+(?:\s+\w+)?)",
            r"(develop|create|build|establish)\s+(\w+(?:\s+\w+)?)",
            r"(analyze|evaluate|assess|measure)\s+(\w+(?:\s+\w+)?)"
        ]
        
        for pattern in action_patterns:
            matches = re.finditer(pattern, goal_desc, re.IGNORECASE)
            for match in matches:
                action = match.group(1)
                target = match.group(2)
                if target.lower() not in stop_words:
                    sub_goals.append(f"{action.capitalize()} {target}")
        
        # Pattern 4: Clauses with "by", "through", "via"
        method_pattern = r"(?:by|through|via)\s+(\w+ing\s+.+?)(?:\.|,|;|$)"
        method_matches = re.finditer(method_pattern, goal_desc, re.IGNORECASE)
        for match in method_matches:
            method = match.group(1)
            sub_goals.append(f"Implement: {method}")
        
        return list(set(sub_goals))  # Remove duplicates

    def _causal_pathway_decomposition(self, goal: Dict[str, Any], 
                                    causal_analysis: Dict[str, Any]) -> List[str]:
        """Decompose based on causal pathways"""
        sub_goals = []
        
        # Major pathways to goal
        pathways = causal_analysis.get("major_pathways", [])
        
        for i, pathway in enumerate(pathways[:3]):
            # Key nodes in pathway
            key_nodes = pathway.get("key_nodes", [])
            
            if len(key_nodes) > 2:
                # Create sub-goals for major milestones
                milestone_node = key_nodes[len(key_nodes) // 2]  # Middle node
                sub_goals.append(f"Achieve {milestone_node} as intermediate milestone")
            
            # Bottlenecks in pathway
            bottlenecks = pathway.get("bottlenecks", [])
            for bottleneck in bottlenecks[:2]:
                sub_goals.append(f"Address bottleneck: {bottleneck}")
            
            # Required conditions
            conditions = pathway.get("required_conditions", [])
            for condition in conditions[:2]:
                sub_goals.append(f"Establish condition: {condition}")
        
        return sub_goals

    def _functional_decomposition(self, goal: Dict[str, Any]) -> List[str]:
        """Decompose goal by functions/capabilities needed"""
        sub_goals = []
        goal_desc = goal.get("description", "").lower()
        
        # Common functional categories
        functional_patterns = {
            "data": ["collect", "analyze", "process", "store"],
            "system": ["design", "implement", "test", "deploy"],
            "process": ["define", "optimize", "monitor", "control"],
            "knowledge": ["research", "learn", "document", "share"],
            "resource": ["acquire", "allocate", "manage", "optimize"],
            "quality": ["measure", "improve", "ensure", "validate"]
        }
        
        # Identify relevant functional areas
        for category, functions in functional_patterns.items():
            if category in goal_desc:
                for function in functions:
                    sub_goals.append(f"{function.capitalize()} {category}")
        
        # Generic functional breakdown
        if "improve" in goal_desc or "optimize" in goal_desc:
            sub_goals.extend([
                "Measure current state",
                "Identify improvement opportunities",
                "Implement changes",
                "Monitor results"
            ])
        
        return sub_goals
    
    def _temporal_decomposition(self, goal: Dict[str, Any], 
                              causal_analysis: Dict[str, Any]) -> List[str]:
        """Decompose goal into temporal phases"""
        sub_goals = []
        
        # Time dynamics from causal analysis
        time_dynamics = causal_analysis.get("time_dynamics", {})
        
        # Phase-based decomposition
        phases = [
            ("Immediate", "Quick wins and foundational steps"),
            ("Short-term", "Build momentum and early results"),
            ("Medium-term", "Core implementation and scaling"),
            ("Long-term", "Sustainability and optimization")
        ]
        
        goal_desc = goal.get("description", "")
        
        for phase_name, phase_desc in phases:
            if time_dynamics.get(f"{phase_name.lower()}_feasible", True):
                sub_goals.append(f"{phase_name}: {self._adapt_goal_to_timeframe(goal_desc, phase_name)}")
        
        return sub_goals
    
    # ========================================================================================
    # EXPANDED MEMORY-INFORMED REASONING METHODS
    # ========================================================================================
    
    def _create_reasoning_shortcut(self,
                                 pattern: Dict[str, Any],
                                 current_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create sophisticated reasoning shortcut based on successful pattern"""
        if pattern.get("success_rate", 0) < 0.7:
            return None
        
        pattern_type = pattern.get("type")
        shortcut = {
            "name": f"{pattern_type}_{pattern.get('domain', 'general')}_shortcut_{hash(str(pattern))%1000}",
            "description": f"Fast-track {pattern_type} reasoning based on proven pattern",
            "pattern_id": pattern.get("id", "unknown"),
            "steps": [],
            "expected_duration": pattern.get("avg_duration", 1.0) * 0.5,
            "confidence": pattern.get("success_rate", 0.7),
            "applicable_conditions": self._extract_pattern_conditions(pattern),
            "optimization_level": "high"
        }
        
        # Create sophisticated shortcut steps based on pattern type
        if pattern_type == "causal_pattern":
            shortcut["steps"] = [
                {
                    "action": "apply_known_relations",
                    "description": "Skip discovery, apply proven causal relations",
                    "data": {
                        "relations": pattern.get("typical_relations", []),
                        "confidence_threshold": 0.8,
                        "skip_validation": pattern.get("success_rate", 0) > 0.9
                    }
                },
                {
                    "action": "focus_on_key_factors",
                    "description": "Prioritize known influential factors",
                    "data": {
                        "factors": pattern.get("key_factors", []),
                        "weights": pattern.get("factor_weights", {}),
                        "interaction_effects": pattern.get("interactions", [])
                    }
                },
                {
                    "action": "skip_exhaustive_search",
                    "description": "Use heuristic search with proven paths",
                    "data": {
                        "depth_limit": min(2, pattern.get("typical_depth", 3) - 1),
                        "preferred_paths": pattern.get("successful_paths", []),
                        "pruning_strategy": "aggressive"
                    }
                },
                {
                    "action": "apply_cached_insights",
                    "description": "Reuse insights from similar contexts",
                    "data": {
                        "insight_templates": pattern.get("common_insights", []),
                        "adaptation_rules": pattern.get("context_adaptations", {})
                    }
                }
            ]
            
        elif pattern_type == "conceptual_pattern":
            shortcut["steps"] = [
                {
                    "action": "use_concept_clusters",
                    "description": "Apply pre-identified conceptual groupings",
                    "data": {
                        "clusters": pattern.get("concept_clusters", []),
                        "cluster_relationships": pattern.get("cluster_relations", {}),
                        "semantic_anchors": pattern.get("anchors", [])
                    }
                },
                {
                    "action": "apply_blend_template",
                    "description": "Use successful blending strategies",
                    "data": {
                        "blend_types": pattern.get("successful_blend_types", ["composition"]),
                        "mapping_heuristics": pattern.get("mapping_rules", {}),
                        "quality_thresholds": {"min_mapping_score": 0.7}
                    }
                },
                {
                    "action": "skip_mapping_search",
                    "description": "Use cached conceptual mappings",
                    "data": {
                        "cached_mappings": pattern.get("reliable_mappings", {}),
                        "mapping_confidence": pattern.get("mapping_scores", {}),
                        "fallback_strategy": "similarity_based"
                    }
                },
                {
                    "action": "generate_from_template",
                    "description": "Generate insights using proven templates",
                    "data": {
                        "insight_patterns": pattern.get("insight_templates", []),
                        "customization_params": self._extract_context_params(current_context)
                    }
                }
            ]
            
        elif pattern_type == "intervention_pattern":
            shortcut["steps"] = [
                {
                    "action": "identify_leverage_points",
                    "description": "Focus on proven high-impact nodes",
                    "data": {
                        "leverage_nodes": pattern.get("effective_nodes", []),
                        "impact_scores": pattern.get("impact_history", {}),
                        "selection_criteria": "historical_effectiveness"
                    }
                },
                {
                    "action": "apply_intervention_template",
                    "description": "Use successful intervention strategies",
                    "data": {
                        "templates": pattern.get("intervention_templates", []),
                        "customization_rules": pattern.get("adaptation_rules", {}),
                        "success_predictors": pattern.get("success_factors", [])
                    }
                },
                {
                    "action": "predict_outcomes",
                    "description": "Use historical data for outcome prediction",
                    "data": {
                        "outcome_patterns": pattern.get("typical_outcomes", {}),
                        "confidence_intervals": pattern.get("outcome_variance", {}),
                        "risk_factors": pattern.get("known_risks", [])
                    }
                }
            ]
            
        elif pattern_type == "problem_solving_pattern":
            shortcut["steps"] = [
                {
                    "action": "apply_decomposition_strategy",
                    "description": "Use proven problem breakdown approach",
                    "data": {
                        "decomposition_rules": pattern.get("decomposition_strategy", {}),
                        "granularity_level": pattern.get("optimal_granularity", "medium"),
                        "ordering_heuristic": pattern.get("task_ordering", "dependency_based")
                    }
                },
                {
                    "action": "reuse_solution_components",
                    "description": "Apply solution building blocks",
                    "data": {
                        "solution_components": pattern.get("reusable_components", []),
                        "combination_rules": pattern.get("composition_rules", {}),
                        "adaptation_guidelines": pattern.get("context_adaptations", {})
                    }
                }
            ]
        
        # Add meta-optimization step
        shortcut["steps"].append({
            "action": "optimize_on_the_fly",
            "description": "Continuously optimize based on intermediate results",
            "data": {
                "optimization_triggers": ["low_confidence", "unexpected_result", "high_complexity"],
                "fallback_to_full": pattern.get("success_rate", 0) < 0.85,
                "learning_enabled": True
            }
        })
        
        return shortcut
    
    def _extract_pattern_conditions(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract detailed conditions when pattern is applicable"""
        conditions = []
        
        # Domain condition
        if pattern.get("domain"):
            conditions.append({
                "type": "domain_match",
                "value": pattern["domain"],
                "required": True,
                "weight": 0.3
            })
        
        # Complexity condition
        if pattern.get("complexity_range"):
            conditions.append({
                "type": "complexity_range",
                "min": pattern["complexity_range"].get("min", 0),
                "max": pattern["complexity_range"].get("max", 1),
                "required": False,
                "weight": 0.2
            })
        
        # Context features
        if pattern.get("required_features"):
            for feature in pattern["required_features"]:
                conditions.append({
                    "type": "feature_present",
                    "feature": feature,
                    "required": True,
                    "weight": 0.1
                })
        
        # Success predictors
        if pattern.get("success_predictors"):
            for predictor in pattern["success_predictors"]:
                conditions.append({
                    "type": "predictor_match",
                    "predictor": predictor,
                    "required": False,
                    "weight": 0.15
                })
        
        return conditions
    
    # ========================================================================================
    # EXPANDED TEMPLATE EXECUTION METHODS
    # ========================================================================================
    
    async def _execute_decompose_recursively(self, params: Dict, context: Dict, results: Dict) -> Dict[str, Any]:
        """Execute recursive decomposition of problems"""
        max_depth = params.get("max_depth", 3)
        decomposition_strategy = params.get("strategy", "functional")
        
        # Get the main goal/problem
        main_goal = context.get("goal") or self._extract_goal_from_input(context.get("user_input", ""))
        
        # Initialize decomposition tree
        decomposition_tree = {
            "root": {
                "id": "root",
                "description": main_goal,
                "level": 0,
                "children": [],
                "complexity": 1.0,
                "solved": False
            }
        }
        
        # Recursive decomposition function
        async def decompose_node(node: Dict, current_depth: int):
            if current_depth >= max_depth:
                return
            
            if node["complexity"] < 0.3:  # Simple enough, don't decompose further
                node["solved"] = True
                return
            
            # Apply decomposition strategy
            if decomposition_strategy == "functional":
                sub_problems = self._functional_decomposition_detailed(node["description"])
            elif decomposition_strategy == "temporal":
                sub_problems = self._temporal_decomposition_detailed(node["description"])
            elif decomposition_strategy == "structural":
                sub_problems = self._structural_decomposition(node["description"])
            else:
                sub_problems = self._hybrid_decomposition(node["description"])
            
            # Create child nodes
            for i, sub_problem in enumerate(sub_problems):
                child = {
                    "id": f"{node['id']}_child_{i}",
                    "description": sub_problem["description"],
                    "level": current_depth + 1,
                    "children": [],
                    "complexity": sub_problem.get("complexity", 0.5),
                    "solved": False,
                    "approach": sub_problem.get("approach", "standard"),
                    "dependencies": sub_problem.get("dependencies", [])
                }
                
                node["children"].append(child)
                
                # Recursively decompose child
                await decompose_node(child, current_depth + 1)
        
        # Start decomposition
        await decompose_node(decomposition_tree["root"], 0)
        
        # Extract flat list of all sub-problems
        all_subproblems = []
        
        def extract_subproblems(node: Dict):
            if node["level"] > 0:  # Don't include root
                all_subproblems.append({
                    "id": node["id"],
                    "description": node["description"],
                    "level": node["level"],
                    "complexity": node["complexity"],
                    "parent": node["id"].rsplit("_child_", 1)[0],
                    "approach": node.get("approach", "standard"),
                    "dependencies": node.get("dependencies", [])
                })
            for child in node["children"]:
                extract_subproblems(child)
        
        extract_subproblems(decomposition_tree["root"])
        
        return {
            "action": "decompose_recursively",
            "status": "completed",
            "outputs": {
                "decomposition_tree": decomposition_tree,
                "subproblems": all_subproblems,
                "total_subproblems": len(all_subproblems),
                "max_depth_reached": max(sp["level"] for sp in all_subproblems) if all_subproblems else 0
            },
            "metadata": {
                "strategy": decomposition_strategy,
                "max_depth": max_depth
            }
        }
    
    async def _execute_identify_dependencies(self, params: Dict, context: Dict, results: Dict) -> Dict[str, Any]:
        """Execute dependency identification between sub-problems"""
        subproblems = results.get("outputs", {}).get("subproblems", [])
        
        if not subproblems:
            return {
                "action": "identify_dependencies",
                "status": "skipped",
                "outputs": {"dependencies": []},
                "metadata": {"reason": "no_subproblems"}
            }
        
        dependencies = []
        dependency_graph = defaultdict(list)
        
        # Analyze each pair of subproblems
        for i, sp1 in enumerate(subproblems):
            for j, sp2 in enumerate(subproblems):
                if i >= j:  # Skip self and already analyzed pairs
                    continue
                
                # Check various dependency types
                dep_score = 0.0
                dep_type = None
                dep_strength = 0.0
                
                # 1. Temporal dependency
                temporal_dep = self._check_temporal_dependency(sp1, sp2)
                if temporal_dep["exists"]:
                    dep_score = max(dep_score, temporal_dep["strength"])
                    dep_type = "temporal"
                    dep_strength = temporal_dep["strength"]
                
                # 2. Data dependency
                data_dep = self._check_data_dependency(sp1, sp2)
                if data_dep["exists"]:
                    dep_score = max(dep_score, data_dep["strength"])
                    if data_dep["strength"] > dep_strength:
                        dep_type = "data"
                        dep_strength = data_dep["strength"]
                
                # 3. Resource dependency
                resource_dep = self._check_resource_dependency(sp1, sp2)
                if resource_dep["exists"]:
                    dep_score = max(dep_score, resource_dep["strength"])
                    if resource_dep["strength"] > dep_strength:
                        dep_type = "resource"
                        dep_strength = resource_dep["strength"]
                
                # 4. Logical dependency
                logical_dep = self._check_logical_dependency(sp1, sp2)
                if logical_dep["exists"]:
                    dep_score = max(dep_score, logical_dep["strength"])
                    if logical_dep["strength"] > dep_strength:
                        dep_type = "logical"
                        dep_strength = logical_dep["strength"]
                
                # Record significant dependencies
                if dep_score > 0.3:
                    dependency = {
                        "from": sp1["id"],
                        "to": sp2["id"],
                        "type": dep_type,
                        "strength": dep_strength,
                        "description": f"{sp1['id']} must be completed before {sp2['id']}",
                        "impact": self._calculate_dependency_impact(dep_strength, sp1, sp2)
                    }
                    dependencies.append(dependency)
                    dependency_graph[sp1["id"]].append(sp2["id"])
        
        # Detect circular dependencies
        circular = self._detect_circular_dependencies(dependency_graph)
        
        # Calculate critical path
        critical_path = self._find_critical_path(subproblems, dependencies)
        
        return {
            "action": "identify_dependencies",
            "status": "completed",
            "outputs": {
                "dependencies": dependencies,
                "dependency_count": len(dependencies),
                "dependency_graph": dict(dependency_graph),
                "circular_dependencies": circular,
                "critical_path": critical_path,
                "parallelizable_groups": self._identify_parallel_groups(subproblems, dependency_graph)
            },
            "metadata": {
                "analysis_methods": ["temporal", "data", "resource", "logical"]
            }
        }
    
    async def _execute_identify_source_pattern(self, params: Dict, context: Dict, results: Dict) -> Dict[str, Any]:
        """Execute source pattern identification for analogical reasoning"""
        abstraction_level = params.get("abstraction_level", "medium")
        
        # Extract source domain and pattern from context
        user_input = context.get("user_input", "")
        source_domain = self._identify_source_domain(user_input)
        
        # Identify patterns at specified abstraction level
        patterns = []
        
        # Level 1: Concrete patterns
        concrete_patterns = self._extract_concrete_patterns(user_input, source_domain)
        
        # Level 2: Abstract patterns
        abstract_patterns = self._extract_abstract_patterns(user_input, source_domain)
        
        # Level 3: Meta patterns
        meta_patterns = self._extract_meta_patterns(user_input, source_domain)
        
        # Select patterns based on abstraction level
        if abstraction_level == "low":
            patterns = concrete_patterns
        elif abstraction_level == "high":
            patterns = meta_patterns
        else:  # medium
            patterns = abstract_patterns
        
        # Analyze pattern structure
        analyzed_patterns = []
        for pattern in patterns:
            analyzed = {
                "id": f"pattern_{len(analyzed_patterns)}",
                "description": pattern["description"],
                "structure": self._analyze_pattern_structure(pattern),
                "elements": pattern.get("elements", []),
                "relationships": pattern.get("relationships", []),
                "dynamics": pattern.get("dynamics", {}),
                "abstraction_level": abstraction_level,
                "source_domain": source_domain,
                "transferability": self._assess_pattern_transferability(pattern),
                "key_insights": self._extract_pattern_insights(pattern)
            }
            analyzed_patterns.append(analyzed)
        
        # Identify the primary pattern
        primary_pattern = max(analyzed_patterns, 
                             key=lambda p: p["transferability"]) if analyzed_patterns else None
        
        return {
            "action": "identify_source_pattern",
            "status": "completed",
            "outputs": {
                "source_patterns": analyzed_patterns,
                "primary_pattern": primary_pattern,
                "source_domain": source_domain,
                "pattern_count": len(analyzed_patterns),
                "abstraction_level": abstraction_level
            },
            "metadata": {
                "extraction_methods": ["concrete", "abstract", "meta"]
            }
        }
    
    async def _execute_find_target_mapping(self, params: Dict, context: Dict, results: Dict) -> Dict[str, Any]:
        """Execute target mapping for analogical reasoning"""
        min_similarity = params.get("min_similarity", 0.4)
        
        # Get source pattern from previous step
        source_pattern = results.get("outputs", {}).get("primary_pattern")
        if not source_pattern:
            return {
                "action": "find_target_mapping",
                "status": "failed",
                "outputs": {},
                "error": "No source pattern available"
            }
        
        # Identify target domain
        target_domain = self._identify_target_domain(context, source_pattern["source_domain"])
        
        # Find mappings between source and target
        mappings = []
        
        # Extract target domain elements
        target_elements = self._extract_target_elements(context, target_domain)
        
        # Map source elements to target elements
        for source_elem in source_pattern.get("elements", []):
            best_mapping = None
            best_score = 0.0
            
            for target_elem in target_elements:
                # Calculate mapping score
                mapping_score = self._calculate_mapping_score(
                    source_elem, target_elem, source_pattern, target_domain
                )
                
                if mapping_score > best_score and mapping_score >= min_similarity:
                    best_score = mapping_score
                    best_mapping = {
                        "source_element": source_elem,
                        "target_element": target_elem,
                        "similarity": mapping_score,
                        "mapping_type": self._determine_mapping_type(source_elem, target_elem),
                        "confidence": self._calculate_mapping_confidence(
                            mapping_score, source_elem, target_elem
                        )
                    }
            
            if best_mapping:
                mappings.append(best_mapping)
        
        # Map relationships
        relationship_mappings = self._map_relationships(
            source_pattern.get("relationships", []),
            target_elements,
            mappings
        )
        
        # Validate mapping consistency
        validation_result = self._validate_mapping_consistency(mappings, relationship_mappings)
        
        # Generate mapped pattern
        mapped_pattern = {
            "source_pattern_id": source_pattern["id"],
            "target_domain": target_domain,
            "element_mappings": mappings,
            "relationship_mappings": relationship_mappings,
            "mapping_quality": self._calculate_overall_mapping_quality(mappings, relationship_mappings),
            "validation": validation_result,
            "insights": self._generate_mapping_insights(source_pattern, mappings, target_domain)
        }
        
        return {
            "action": "find_target_mapping",
            "status": "completed",
            "outputs": {
                "mapped_pattern": mapped_pattern,
                "mapping_count": len(mappings),
                "relationship_count": len(relationship_mappings),
                "target_domain": target_domain,
                "mapping_quality": mapped_pattern["mapping_quality"]
            },
            "metadata": {
                "min_similarity_used": min_similarity,
                "validation_passed": validation_result.get("is_valid", False)
            }
        }
    
    async def _execute_custom_action(self, action: str, params: Dict, context: Dict, results: Dict) -> Dict[str, Any]:
        """Execute custom actions defined by templates"""
        
        # Parse custom action type
        action_parts = action.split("_")
        if len(action_parts) < 2:
            return {
                "action": action,
                "status": "failed",
                "error": "Invalid custom action format"
            }
        
        action_category = action_parts[1]
        
        # Route to appropriate custom handler
        if action_category == "analyze":
            return await self._execute_custom_analysis(action, params, context, results)
        elif action_category == "generate":
            return await self._execute_custom_generation(action, params, context, results)
        elif action_category == "evaluate":
            return await self._execute_custom_evaluation(action, params, context, results)
        elif action_category == "transform":
            return await self._execute_custom_transformation(action, params, context, results)
        else:
            # Generic custom action execution
            return await self._execute_generic_custom(action, params, context, results)
    
    async def _execute_custom_analysis(self, action: str, params: Dict, context: Dict, results: Dict) -> Dict[str, Any]:
        """Execute custom analysis actions"""
        analysis_type = action.split("_", 2)[2] if len(action.split("_")) > 2 else "general"
        
        analysis_result = {
            "action": action,
            "status": "completed",
            "outputs": {},
            "metadata": {"analysis_type": analysis_type}
        }
        
        if analysis_type == "sentiment":
            # Sentiment analysis
            text = params.get("text", context.get("user_input", ""))
            sentiment = self._analyze_sentiment_detailed(text)
            analysis_result["outputs"] = {
                "sentiment": sentiment,
                "confidence": sentiment.get("confidence", 0.5)
            }
            
        elif analysis_type == "complexity":
            # Complexity analysis
            target = params.get("target", context.get("user_input", ""))
            complexity = self._analyze_complexity_detailed(target)
            analysis_result["outputs"] = {
                "complexity_score": complexity["score"],
                "complexity_factors": complexity["factors"],
                "complexity_level": complexity["level"]
            }
            
        elif analysis_type == "risk":
            # Risk analysis
            scenario = params.get("scenario", results)
            risks = self._analyze_risks_detailed(scenario, context)
            analysis_result["outputs"] = {
                "risk_factors": risks,
                "overall_risk": self._calculate_overall_risk(risks),
                "mitigation_suggestions": self._suggest_risk_mitigations(risks)
            }
            
        elif analysis_type == "opportunity":
            # Opportunity analysis
            situation = params.get("situation", context)
            opportunities = self._analyze_opportunities(situation, results)
            analysis_result["outputs"] = {
                "opportunities": opportunities,
                "opportunity_score": self._calculate_opportunity_score(opportunities),
                "action_recommendations": self._recommend_opportunity_actions(opportunities)
            }
        
        else:
            # Generic analysis
            analysis_result["outputs"] = {
                "analysis_completed": True,
                "generic_insights": ["Analysis completed for custom type: " + analysis_type]
            }
        
        return analysis_result
    
    # ========================================================================================
    # ENHANCED PATH FINDING AND GRAPH ANALYSIS
    # ========================================================================================
    
    def _find_hierarchical_patterns(self, space) -> List[Dict[str, Any]]:
        """Comprehensive hierarchical pattern detection with multiple strategies"""
        hierarchies = []
        
        # Extended set of hierarchical relations
        hierarchical_relations = [
            # Standard hierarchical relations
            "is_a", "type_of", "kind_of", "instance_of", "example_of",
            "subtype_of", "subclass_of", "specialization_of",
            
            # Part-whole relations
            "part_of", "component_of", "member_of", "subset_of", "belongs_to",
            "contained_in", "consists_of", "comprises",
            
            # Inheritance relations
            "inherits_from", "derives_from", "extends", "implements",
            "based_on", "variant_of",
            
            # Abstraction relations
            "generalizes", "specializes", "abstracts", "refines",
            "instantiates", "realizes",
            
            # Organizational relations
            "reports_to", "supervised_by", "child_of", "branch_of",
            "division_of", "subsidiary_of"
        ]
        
        # Build comprehensive parent-child mappings
        parent_children = defaultdict(list)
        child_parents = defaultdict(list)
        relation_types = defaultdict(lambda: defaultdict(int))
        
        for relation in space.relations:
            rel_type = relation.get("relation_type", "")
            if rel_type in hierarchical_relations:
                parent = relation["source"]
                child = relation["target"]
                parent_children[parent].append(child)
                child_parents[child].append(parent)
                relation_types[parent][rel_type] += 1
        
        # Strategy 1: Find pure hierarchies (single parent)
        pure_roots = self._find_pure_hierarchy_roots(parent_children, child_parents)
        for root in pure_roots:
            hierarchy = self._build_complete_hierarchy(root, parent_children, space, "pure")
            if hierarchy["total_nodes"] > 2:
                hierarchies.append(hierarchy)
        
        # Strategy 2: Find DAG hierarchies (multiple parents allowed)
        dag_roots = self._find_dag_roots(parent_children, child_parents)
        for root in dag_roots:
            hierarchy = self._build_dag_hierarchy(root, parent_children, child_parents, space)
            if hierarchy["total_nodes"] > 2 and hierarchy not in hierarchies:
                hierarchies.append(hierarchy)
        
        # Strategy 3: Find domain-specific hierarchies
        domain_hierarchies = self._find_domain_hierarchies(space, parent_children, relation_types)
        hierarchies.extend(domain_hierarchies)
        
        # Strategy 4: Find implicit hierarchies based on properties
        property_hierarchies = self._find_property_based_hierarchies(space)
        hierarchies.extend(property_hierarchies)
        
        # Analyze and enrich hierarchies
        for hierarchy in hierarchies:
            self._analyze_hierarchy_characteristics(hierarchy, parent_children, child_parents, space)
            self._calculate_hierarchy_metrics(hierarchy, space)
        
        # Rank hierarchies by quality
        hierarchies.sort(key=lambda h: h.get("quality_score", 0), reverse=True)
        
        return hierarchies
    
    def _find_pure_hierarchy_roots(self, parent_children: Dict, child_parents: Dict) -> List[str]:
        """Find roots for pure hierarchies (tree structures)"""
        roots = []
        
        # Find nodes with no parents
        all_nodes = set(parent_children.keys()) | set(child_parents.keys())
        potential_roots = all_nodes - set(child_parents.keys())
        
        # Verify they form tree structures
        for root in potential_roots:
            if self._forms_tree_structure(root, parent_children, set()):
                roots.append(root)
        
        return roots
    
    def _forms_tree_structure(self, node: str, parent_children: Dict, visited: Set[str]) -> bool:
        """Check if node forms a tree structure (no cycles, single parent)"""
        if node in visited:
            return False  # Cycle detected
        
        visited.add(node)
        
        for child in parent_children.get(node, []):
            if not self._forms_tree_structure(child, parent_children, visited):
                return False
        
        return True
    
    def _build_complete_hierarchy(self, root: str, parent_children: Dict, 
                                space: Any, hierarchy_type: str) -> Dict[str, Any]:
        """Build complete hierarchy with rich metadata"""
        hierarchy = {
            "root": root,
            "root_name": space.concepts.get(root, {}).get("name", root),
            "hierarchy_type": hierarchy_type,
            "levels": defaultdict(list),
            "total_nodes": 0,
            "max_depth": 0,
            "tree_structure": {},
            "node_metadata": {},
            "statistics": {}
        }
        
        # BFS to build hierarchy
        queue = [(root, 0, [])]
        visited = set()
        
        while queue:
            node_id, level, ancestors = queue.pop(0)
            
            if node_id in visited:
                continue
                
            visited.add(node_id)
            hierarchy["levels"][level].append(node_id)
            hierarchy["total_nodes"] += 1
            hierarchy["max_depth"] = max(hierarchy["max_depth"], level)
            
            # Store node metadata
            node_data = space.concepts.get(node_id, {})
            hierarchy["node_metadata"][node_id] = {
                "name": node_data.get("name", node_id),
                "level": level,
                "ancestors": ancestors.copy(),
                "properties": node_data.get("properties", {}),
                "child_count": len(parent_children.get(node_id, []))
            }
            
            # Build tree structure
            children = parent_children.get(node_id, [])
            hierarchy["tree_structure"][node_id] = children
            
            # Add children to queue
            for child in children:
                queue.append((child, level + 1, ancestors + [node_id]))
        
        # Calculate statistics
        hierarchy["statistics"] = self._calculate_tree_statistics(hierarchy)
        
        return hierarchy
    
    def _find_conceptual_clusters(self, space) -> List[Dict[str, Any]]:
        """Advanced conceptual clustering with multiple algorithms"""
        clusters = []
        
        # Algorithm 1: Density-based clustering (enhanced)
        density_clusters = self._enhanced_density_clustering(space)
        
        # Algorithm 2: Semantic clustering with embeddings simulation
        semantic_clusters = self._semantic_clustering_advanced(space)
        
        # Algorithm 3: Property-based clustering with feature extraction
        property_clusters = self._property_clustering_advanced(space)
        
        # Algorithm 4: Relational clustering based on connection patterns
        relational_clusters = self._relational_clustering(space)
        
        # Algorithm 5: Hierarchical agglomerative clustering
        hierarchical_clusters = self._hierarchical_clustering(space)
        
        # Merge and reconcile clusters
        merged_clusters = self._advanced_cluster_merging(
            density_clusters, semantic_clusters, property_clusters, 
            relational_clusters, hierarchical_clusters, space
        )
        
        # Post-process clusters
        for cluster in merged_clusters:
            # Calculate quality metrics
            cluster["quality_metrics"] = self._calculate_cluster_quality_metrics(cluster, space)
            
            # Identify cluster characteristics
            cluster["characteristics"] = self._identify_cluster_characteristics(cluster, space)
            
            # Find exemplar concepts
            cluster["exemplars"] = self._find_cluster_exemplars(cluster, space)
            
            # Detect sub-clusters
            cluster["sub_clusters"] = self._detect_sub_clusters(cluster, space)
        
        # Filter high-quality clusters
        quality_threshold = 0.4
        clusters = [c for c in merged_clusters if c["quality_metrics"]["overall_quality"] > quality_threshold]
        
        # Sort by quality and size
        clusters.sort(key=lambda c: (c["quality_metrics"]["overall_quality"], c["size"]), reverse=True)
        
        return clusters
    
    def _enhanced_density_clustering(self, space) -> List[Dict[str, Any]]:
        """Enhanced density-based clustering with adaptive parameters"""
        clusters = []
        visited = set()
        
        # Calculate adaptive density threshold based on graph statistics
        total_concepts = len(space.concepts)
        total_relations = len(space.relations)
        avg_degree = (2 * total_relations) / total_concepts if total_concepts > 0 else 0
        density_threshold = max(0.3, min(0.7, avg_degree / total_concepts))
        
        # Find dense regions
        for start_node in space.concepts:
            if start_node in visited:
                continue
            
            # Try to grow a dense cluster
            cluster = self._grow_adaptive_dense_cluster(
                start_node, space, visited, density_threshold
            )
            
            if len(cluster["members"]) >= 3:
                clusters.append(cluster)
        
        return clusters
    
    def _semantic_clustering_advanced(self, space) -> List[Dict[str, Any]]:
        """Advanced semantic clustering with similarity propagation"""
        clusters = []
        
        # Build semantic similarity matrix
        concepts = list(space.concepts.keys())
        n = len(concepts)
        
        # Initialize similarity matrix
        similarity_matrix = np.zeros((n, n))
        
        # Calculate pairwise similarities
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    concept1 = space.concepts[concepts[i]]
                    concept2 = space.concepts[concepts[j]]
                    
                    # Multi-faceted similarity
                    name_sim = self._calculate_name_similarity_advanced(
                        concept1.get("name", ""), concept2.get("name", "")
                    )
                    prop_sim = self._calculate_property_similarity_advanced(
                        concept1.get("properties", {}), concept2.get("properties", {})
                    )
                    context_sim = self._calculate_contextual_similarity(
                        concepts[i], concepts[j], space
                    )
                    
                    # Weighted combination
                    similarity = 0.4 * name_sim + 0.3 * prop_sim + 0.3 * context_sim
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        # Apply affinity propagation algorithm (simplified)
        clusters_indices = self._affinity_propagation(similarity_matrix)
        
        # Convert to cluster format
        for cluster_idx, member_indices in clusters_indices.items():
            if len(member_indices) >= 3:
                members = [concepts[i] for i in member_indices]
                cluster = {
                    "members": members,
                    "size": len(members),
                    "type": "semantic",
                    "center": concepts[cluster_idx],
                    "avg_similarity": np.mean([
                        similarity_matrix[cluster_idx, i] for i in member_indices
                    ])
                }
                clusters.append(cluster)
        
        return clusters
    
    # ========================================================================================
    # SUPPORTING HELPER METHODS
    # ========================================================================================
    
    def _calculate_goal_similarity(self, goal1: str, goal2: str) -> float:
        """Calculate similarity between two goals"""
        # Tokenize and normalize
        tokens1 = set(goal1.lower().split())
        tokens2 = set(goal2.lower().split())
        
        # Remove stop words
        stop_words = {"the", "a", "an", "to", "for", "of", "in", "on", "at", "by"}
        tokens1 = tokens1 - stop_words
        tokens2 = tokens2 - stop_words
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_goal_components(self, goal_desc: str) -> List[str]:
        """Extract components from complex goal description"""
        components = []
        
        # Split by conjunctions
        parts = re.split(r'\b(?:and|while|by|through)\b', goal_desc)
        
        for part in parts:
            part = part.strip()
            if len(part) > 10:  # Meaningful component
                components.append(part)
        
        return components
    
    def _relax_goal_constraints(self, goal: Dict[str, Any], constraints: List[str]) -> Dict[str, Any]:
        """Relax constraints to make goal more achievable"""
        relaxed_goal = goal.copy()
        relaxed = []
        
        goal_desc = goal.get("description", "")
        
        # Common constraint relaxations
        relaxation_rules = {
            "time": {
                "strict": ["immediately", "within 24 hours", "today"],
                "relaxed": ["soon", "within a week", "this week"]
            },
            "quantity": {
                "strict": ["all", "100%", "complete", "every"],
                "relaxed": ["most", "80%", "substantial", "many"]
            },
            "quality": {
                "strict": ["perfect", "optimal", "best"],
                "relaxed": ["good", "satisfactory", "acceptable"]
            }
        }
        
        # Apply relaxations
        relaxed_desc = goal_desc
        for constraint_type, rules in relaxation_rules.items():
            for strict_term in rules["strict"]:
                if strict_term in relaxed_desc.lower():
                    relaxed_term = rules["relaxed"][0]
                    relaxed_desc = relaxed_desc.replace(strict_term, relaxed_term)
                    relaxed.append(f"{strict_term} -> {relaxed_term}")
        
        relaxed_goal["description"] = relaxed_desc
        relaxed_goal["relaxed"] = relaxed
        
        return relaxed_goal
    
    def _create_short_term_goal(self, original_goal: str) -> str:
        """Create short-term version of goal"""
        # Add short-term modifiers
        short_term_modifiers = [
            "Begin to", "Start", "Initiate", "Lay groundwork for",
            "Take first steps toward", "Establish foundation for"
        ]
        
        modifier = short_term_modifiers[hash(original_goal) % len(short_term_modifiers)]
        
        return f"{modifier} {original_goal.lower()}"
    
    def _functional_decomposition_detailed(self, description: str) -> List[Dict[str, Any]]:
        """Detailed functional decomposition of a problem/goal"""
        sub_problems = []
        
        # Core functional areas
        functions = {
            "input": ["identify inputs", "gather requirements", "validate data"],
            "process": ["design process", "implement logic", "optimize flow"],
            "output": ["define outputs", "format results", "deliver value"],
            "control": ["monitor progress", "handle errors", "ensure quality"],
            "feedback": ["collect feedback", "analyze results", "iterate improvements"]
        }
        
        # Analyze which functions are relevant
        desc_lower = description.lower()
        relevant_functions = []
        
        for func_area, tasks in functions.items():
            if any(keyword in desc_lower for keyword in [func_area, func_area[:-1]]):
                relevant_functions.append(func_area)
        
        # If no specific functions found, use generic decomposition
        if not relevant_functions:
            relevant_functions = ["input", "process", "output"]
        
        # Create sub-problems for each functional area
        for func_area in relevant_functions:
            for task in functions.get(func_area, []):
                sub_problems.append({
                    "description": f"{task} for {description}",
                    "complexity": 0.3,
                    "function": func_area,
                    "approach": "systematic"
                })
        
        return sub_problems
    
    def _temporal_decomposition_detailed(self, description: str) -> List[Dict[str, Any]]:
        """Detailed temporal decomposition"""
        phases = [
            {
                "name": "Preparation",
                "tasks": ["Research", "Planning", "Resource gathering"],
                "complexity": 0.3
            },
            {
                "name": "Initial Implementation", 
                "tasks": ["Prototype", "Test foundations", "Early validation"],
                "complexity": 0.5
            },
            {
                "name": "Core Development",
                "tasks": ["Build main features", "Integration", "Testing"],
                "complexity": 0.7
            },
            {
                "name": "Refinement",
                "tasks": ["Optimization", "Polish", "Documentation"],
                "complexity": 0.4
            },
            {
                "name": "Deployment",
                "tasks": ["Release", "Monitor", "Support"],
                "complexity": 0.5
            }
        ]
        
        sub_problems = []
        
        for phase in phases:
            for task in phase["tasks"]:
                sub_problems.append({
                    "description": f"{phase['name']}: {task} for {description}",
                    "complexity": phase["complexity"],
                    "temporal_phase": phase["name"],
                    "approach": "phased"
                })
        
        return sub_problems
    
    def _structural_decomposition(self, description: str) -> List[Dict[str, Any]]:
        """Structural decomposition based on system components"""
        sub_problems = []
        
        # Common structural patterns
        structures = {
            "layers": ["presentation", "logic", "data"],
            "components": ["interface", "core", "utilities", "integration"],
            "modules": ["input", "processing", "storage", "output", "monitoring"],
            "aspects": ["functional", "performance", "security", "usability"]
        }
        
        # Detect which structural pattern fits
        desc_lower = description.lower()
        
        # Default to component-based
        structure_type = "components"
        
        if any(word in desc_lower for word in ["system", "application", "platform"]):
            structure_type = "layers"
        elif any(word in desc_lower for word in ["module", "modular", "plugin"]):
            structure_type = "modules"
        
        # Create sub-problems for each structural element
        for element in structures[structure_type]:
            sub_problems.append({
                "description": f"Handle {element} aspect of {description}",
                "complexity": 0.5,
                "structure": structure_type,
                "element": element,
                "approach": "architectural"
            })
        
        return sub_problems
    
    def _hybrid_decomposition(self, description: str) -> List[Dict[str, Any]]:
        """Hybrid decomposition combining multiple strategies"""
        sub_problems = []
        
        # Combine different decomposition strategies
        functional = self._functional_decomposition_detailed(description)[:2]
        temporal = self._temporal_decomposition_detailed(description)[:2]
        structural = self._structural_decomposition(description)[:2]
        
        # Add with adjusted complexity
        for sp in functional:
            sp["hybrid_type"] = "functional"
            sub_problems.append(sp)
        
        for sp in temporal:
            sp["hybrid_type"] = "temporal"
            sp["complexity"] *= 0.9  # Slightly reduce complexity
            sub_problems.append(sp)
        
        for sp in structural:
            sp["hybrid_type"] = "structural"
            sp["complexity"] *= 0.8
            sub_problems.append(sp)
        
        return sub_problems
    
    def _check_temporal_dependency(self, sp1: Dict, sp2: Dict) -> Dict[str, Any]:
        """Check if temporal dependency exists between subproblems"""
        result = {"exists": False, "strength": 0.0}
        
        # Check explicit temporal markers
        sp1_desc = sp1.get("description", "").lower()
        sp2_desc = sp2.get("description", "").lower()
        
        # Temporal keywords indicating sp1 before sp2
        before_after_patterns = [
            ("preparation", "implementation"),
            ("planning", "execution"),
            ("design", "build"),
            ("gather", "process"),
            ("initial", "final"),
            ("test", "deploy")
        ]
        
        for before, after in before_after_patterns:
            if before in sp1_desc and after in sp2_desc:
                result["exists"] = True
                result["strength"] = 0.8
                return result
        
        # Check temporal phases
        if sp1.get("temporal_phase") and sp2.get("temporal_phase"):
            phase_order = ["Preparation", "Initial Implementation", "Core Development", 
                          "Refinement", "Deployment"]
            try:
                idx1 = phase_order.index(sp1["temporal_phase"])
                idx2 = phase_order.index(sp2["temporal_phase"])
                if idx1 < idx2:
                    result["exists"] = True
                    result["strength"] = 0.9
            except ValueError:
                pass
        
        return result
    
    def _check_data_dependency(self, sp1: Dict, sp2: Dict) -> Dict[str, Any]:
        """Check if data dependency exists between subproblems"""
        result = {"exists": False, "strength": 0.0}
        
        sp1_desc = sp1.get("description", "").lower()
        sp2_desc = sp2.get("description", "").lower()
        
        # Data flow patterns
        data_patterns = [
            ("output", "input"),
            ("generate", "use"),
            ("create", "process"),
            ("collect", "analyze"),
            ("fetch", "transform"),
            ("produce", "consume")
        ]
        
        for producer, consumer in data_patterns:
            if producer in sp1_desc and consumer in sp2_desc:
                result["exists"] = True
                result["strength"] = 0.7
                return result
        
        return result
    
    def _check_resource_dependency(self, sp1: Dict, sp2: Dict) -> Dict[str, Any]:
        """Check if resource dependency exists between subproblems"""
        result = {"exists": False, "strength": 0.0}
        
        # Check if they compete for same resources
        if sp1.get("resources") and sp2.get("resources"):
            shared_resources = set(sp1["resources"]).intersection(set(sp2["resources"]))
            if shared_resources:
                result["exists"] = True
                result["strength"] = len(shared_resources) / max(
                    len(sp1["resources"]), len(sp2["resources"])
                )
        
        return result
    
    def _check_logical_dependency(self, sp1: Dict, sp2: Dict) -> Dict[str, Any]:
        """Check if logical dependency exists between subproblems"""
        result = {"exists": False, "strength": 0.0}
        
        # Check if sp2 assumes sp1 is complete
        sp1_output = self._extract_output_concept(sp1.get("description", ""))
        sp2_input = self._extract_input_concept(sp2.get("description", ""))
        
        if sp1_output and sp2_input and self._concepts_match_enhanced(sp1_output, sp2_input):
            result["exists"] = True
            result["strength"] = 0.8
        
        return result
    
    def _extract_output_concept(self, description: str) -> Optional[str]:
        """Extract what a task produces"""
        output_patterns = [
            r"produce[s]?\s+(\w+)",
            r"generate[s]?\s+(\w+)",
            r"create[s]?\s+(\w+)",
            r"output[s]?\s+(\w+)",
            r"deliver[s]?\s+(\w+)"
        ]
        
        for pattern in output_patterns:
            match = re.search(pattern, description.lower())
            if match:
                return match.group(1)
        
        return None
    
    def _extract_input_concept(self, description: str) -> Optional[str]:
        """Extract what a task requires"""
        input_patterns = [
            r"use[s]?\s+(\w+)",
            r"require[s]?\s+(\w+)",
            r"need[s]?\s+(\w+)",
            r"process[es]?\s+(\w+)",
            r"take[s]?\s+(\w+)"
        ]
        
        for pattern in input_patterns:
            match = re.search(pattern, description.lower())
            if match:
                return match.group(1)
        
        return None
    
    def _concepts_match_enhanced(self, concept1: str, concept2: str) -> bool:
        """Enhanced concept matching with fuzzy logic"""
        if not concept1 or not concept2:
            return False
        
        c1_lower = concept1.lower()
        c2_lower = concept2.lower()
        
        # Exact match
        if c1_lower == c2_lower:
            return True
        
        # Substring match
        if c1_lower in c2_lower or c2_lower in c1_lower:
            return True
        
        # Stem matching (simplified)
        if len(c1_lower) > 4 and len(c2_lower) > 4:
            if c1_lower[:4] == c2_lower[:4]:
                return True
        
        # Synonym matching (simplified)
        synonyms = {
            "data": ["information", "input", "content"],
            "result": ["output", "outcome", "product"],
            "plan": ["strategy", "approach", "design"],
            "model": ["system", "framework", "structure"]
        }
        
        for key, syns in synonyms.items():
            if c1_lower in [key] + syns and c2_lower in [key] + syns:
                return True
        
        return False
    
    def _calculate_dependency_impact(self, strength: float, sp1: Dict, sp2: Dict) -> str:
        """Calculate impact level of dependency"""
        complexity_factor = (sp1.get("complexity", 0.5) + sp2.get("complexity", 0.5)) / 2
        impact_score = strength * complexity_factor
        
        if impact_score > 0.7:
            return "critical"
        elif impact_score > 0.4:
            return "significant"
        else:
            return "minor"
    
    def _detect_circular_dependencies(self, dep_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Detect circular dependencies using DFS"""
        def dfs(node: str, visited: set, rec_stack: set, path: list) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in dep_graph.get(node, []):
                if neighbor not in visited:
                    cycle = dfs(neighbor, visited, rec_stack, path)
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
            
            path.pop()
            rec_stack.remove(node)
            return None
        
        visited = set()
        cycles = []
        
        for node in dep_graph:
            if node not in visited:
                cycle = dfs(node, visited, set(), [])
                if cycle:
                    cycles.append(cycle)
        
        return cycles
    
    def _find_critical_path(self, subproblems: List[Dict], dependencies: List[Dict]) -> List[str]:
        """Find critical path through dependencies"""
        # Build adjacency list with weights
        graph = defaultdict(list)
        weights = {}
        
        for dep in dependencies:
            from_node = dep["from"]
            to_node = dep["to"]
            weight = dep.get("strength", 0.5)
            graph[from_node].append(to_node)
            weights[(from_node, to_node)] = weight
        
        # Find longest path (critical path)
        # Using topological sort + dynamic programming
        
        # First, get topological order
        in_degree = defaultdict(int)
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] += 1
        
        queue = [sp["id"] for sp in subproblems if in_degree[sp["id"]] == 0]
        topo_order = []
        
        while queue:
            node = queue.pop(0)
            topo_order.append(node)
            
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Calculate longest paths
        dist = defaultdict(float)
        parent = {}
        
        for node in topo_order:
            for neighbor in graph[node]:
                weight = weights.get((node, neighbor), 0.5)
                if dist[node] + weight > dist[neighbor]:
                    dist[neighbor] = dist[node] + weight
                    parent[neighbor] = node
        
        # Find node with maximum distance
        if not dist:
            return []
        
        end_node = max(dist.items(), key=lambda x: x[1])[0]
        
        # Reconstruct path
        path = []
        current = end_node
        while current in parent:
            path.append(current)
            current = parent[current]
        path.append(current)
        path.reverse()
        
        return path
    
    def _identify_parallel_groups(self, subproblems: List[Dict], dep_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Identify groups of subproblems that can be executed in parallel"""
        # Calculate levels using BFS
        levels = defaultdict(list)
        in_degree = defaultdict(int)
        
        # Calculate in-degrees
        all_nodes = {sp["id"] for sp in subproblems}
        for node in dep_graph:
            for neighbor in dep_graph[node]:
                in_degree[neighbor] += 1
        
        # Start with nodes that have no dependencies
        current_level = 0
        queue = [(sp["id"], 0) for sp in subproblems if in_degree[sp["id"]] == 0]
        
        while queue:
            node, level = queue.pop(0)
            levels[level].append(node)
            
            # Process neighbors
            for neighbor in dep_graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append((neighbor, level + 1))
        
        # Convert to list of parallel groups
        parallel_groups = [nodes for level, nodes in sorted(levels.items())]
        
        return parallel_groups

    async def _find_similarity_paths(self, start: str, end: str, graph: Dict[str, Any]) -> List[List[str]]:
        """Find paths through similar concepts using advanced similarity metrics"""
        concepts = graph.get("concepts", {})
        if not concepts or start not in concepts or end not in concepts:
            return []
        
        paths = []
        
        # Strategy 1: Direct similarity bridge (2-hop)
        direct_bridges = await self._find_direct_similarity_bridges(start, end, concepts)
        paths.extend(direct_bridges)
        
        # Strategy 2: Multi-hop similarity paths (up to 4 hops)
        multi_hop_paths = await self._find_multi_hop_similarity_paths(start, end, concepts, max_hops=4)
        paths.extend(multi_hop_paths)
        
        # Strategy 3: Cluster-based paths
        cluster_paths = await self._find_cluster_based_paths(start, end, concepts, graph)
        paths.extend(cluster_paths)
        
        # Strategy 4: Analogy-based paths
        analogy_paths = await self._find_analogy_based_paths(start, end, concepts, graph)
        paths.extend(analogy_paths)
        
        # Remove duplicates and sort by quality
        unique_paths = self._deduplicate_paths(paths)
        scored_paths = [(self._score_similarity_path(path, concepts), path) for path in unique_paths]
        scored_paths.sort(key=lambda x: x[0], reverse=True)
        
        return [path for _, path in scored_paths[:10]]  # Return top 10 paths
    
    async def _find_direct_similarity_bridges(self, start: str, end: str, 
                                            concepts: Dict[str, Any]) -> List[List[str]]:
        """Find 2-hop paths through highly similar intermediate concepts"""
        bridges = []
        similarity_threshold = 0.6
        
        # Calculate similarities for all potential bridges
        bridge_candidates = []
        
        for intermediate_id, intermediate_concept in concepts.items():
            if intermediate_id in [start, end]:
                continue
            
            # Calculate similarity to both start and end
            start_sim = await self._calculate_comprehensive_similarity(
                concepts[start], intermediate_concept
            )
            end_sim = await self._calculate_comprehensive_similarity(
                intermediate_concept, concepts[end]
            )
            
            # Combined score
            combined_score = (start_sim * end_sim) ** 0.5  # Geometric mean
            
            if combined_score > similarity_threshold:
                bridge_candidates.append({
                    "id": intermediate_id,
                    "start_sim": start_sim,
                    "end_sim": end_sim,
                    "combined_score": combined_score
                })
        
        # Sort by combined score
        bridge_candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Create paths from top bridges
        for bridge in bridge_candidates[:5]:
            path = [start, bridge["id"], end]
            bridges.append(path)
        
        return bridges
    
    async def _find_multi_hop_similarity_paths(self, start: str, end: str, 
                                              concepts: Dict[str, Any], 
                                              max_hops: int = 4) -> List[List[str]]:
        """Find multi-hop paths using similarity-guided search"""
        paths = []
        
        # A* search with similarity heuristic
        from heapq import heappush, heappop
        
        # Priority queue: (f_score, g_score, path)
        start_h = await self._similarity_heuristic(start, end, concepts)
        queue = [(start_h, 0, [start])]
        visited = set()
        
        while queue and len(paths) < 5:
            f_score, g_score, path = heappop(queue)
            current = path[-1]
            
            if current == end:
                paths.append(path)
                continue
            
            if len(path) >= max_hops:
                continue
            
            if current in visited:
                continue
            visited.add(current)
            
            # Explore similar neighbors
            neighbors = await self._get_similarity_neighbors(current, concepts, top_k=5)
            
            for neighbor_id, similarity in neighbors:
                if neighbor_id not in path:  # Avoid cycles
                    new_g = g_score + (1 - similarity)  # Cost is inverse of similarity
                    new_h = await self._similarity_heuristic(neighbor_id, end, concepts)
                    new_f = new_g + new_h
                    new_path = path + [neighbor_id]
                    
                    heappush(queue, (new_f, new_g, new_path))
        
        return paths
    
    async def _find_cluster_based_paths(self, start: str, end: str, 
                                       concepts: Dict[str, Any], 
                                       graph: Dict[str, Any]) -> List[List[str]]:
        """Find paths through concept clusters"""
        paths = []
        
        # First, identify clusters
        clusters = await self._identify_concept_clusters(concepts)
        
        # Find which clusters contain start and end
        start_cluster = None
        end_cluster = None
        
        for cluster_id, cluster in clusters.items():
            if start in cluster["members"]:
                start_cluster = cluster_id
            if end in cluster["members"]:
                end_cluster = cluster_id
        
        if not start_cluster or not end_cluster:
            return paths
        
        if start_cluster == end_cluster:
            # Same cluster - find intra-cluster path
            cluster = clusters[start_cluster]
            intra_path = await self._find_intra_cluster_path(start, end, cluster, concepts)
            if intra_path:
                paths.append(intra_path)
        else:
            # Different clusters - find inter-cluster paths
            bridge_nodes = await self._find_cluster_bridges(start_cluster, end_cluster, clusters)
            
            for bridge in bridge_nodes[:3]:
                # Path: start -> start_cluster_exit -> bridge -> end_cluster_entry -> end
                path = await self._construct_cluster_bridge_path(
                    start, end, bridge, start_cluster, end_cluster, clusters, concepts
                )
                if path:
                    paths.append(path)
        
        return paths
    
    async def _find_analogy_based_paths(self, start: str, end: str, 
                                       concepts: Dict[str, Any], 
                                       graph: Dict[str, Any]) -> List[List[str]]:
        """Find paths using analogical reasoning"""
        paths = []
        
        # Find analogical relationships
        analogies = await self._find_concept_analogies(start, end, concepts)
        
        for analogy in analogies[:3]:
            # Build path through analogy
            if analogy["type"] == "proportional":
                # A:B :: C:D type analogy
                path = [start, analogy["intermediate1"], analogy["intermediate2"], end]
            elif analogy["type"] == "structural":
                # Structural mapping
                path = [start] + analogy["mapping_nodes"] + [end]
            elif analogy["type"] == "functional":
                # Functional analogy
                path = [start, analogy["function_node"], end]
            else:
                continue
            
            # Validate path connectivity
            if self._validate_analogy_path(path, concepts):
                paths.append(path)
        
        return paths
    
    async def _calculate_comprehensive_similarity(self, concept1: Dict[str, Any], 
                                                concept2: Dict[str, Any]) -> float:
        """Calculate comprehensive similarity between concepts"""
        # Multiple similarity dimensions
        similarities = []
        weights = []
        
        # 1. Name similarity (enhanced)
        name_sim = self._calculate_advanced_name_similarity(
            concept1.get("name", ""), concept2.get("name", "")
        )
        similarities.append(name_sim)
        weights.append(0.25)
        
        # 2. Property similarity (structural)
        prop_sim = self._calculate_structural_property_similarity(
            concept1.get("properties", {}), concept2.get("properties", {})
        )
        similarities.append(prop_sim)
        weights.append(0.20)
        
        # 3. Semantic field similarity
        semantic_sim = self._calculate_semantic_field_similarity(concept1, concept2)
        similarities.append(semantic_sim)
        weights.append(0.20)
        
        # 4. Relational similarity
        rel_sim = self._calculate_relational_similarity(concept1, concept2)
        similarities.append(rel_sim)
        weights.append(0.15)
        
        # 5. Functional similarity
        func_sim = self._calculate_functional_similarity(concept1, concept2)
        similarities.append(func_sim)
        weights.append(0.10)
        
        # 6. Contextual similarity
        context_sim = self._calculate_contextual_similarity_advanced(concept1, concept2)
        similarities.append(context_sim)
        weights.append(0.10)
        
        # Weighted combination
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(similarities, weights))
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_advanced_name_similarity(self, name1: str, name2: str) -> float:
        """Advanced name similarity using multiple techniques"""
        if not name1 or not name2:
            return 0.0
        
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Exact match
        if name1_lower == name2_lower:
            return 1.0
        
        similarities = []
        
        # 1. Token overlap (Jaccard)
        tokens1 = set(name1_lower.split())
        tokens2 = set(name2_lower.split())
        if tokens1 and tokens2:
            jaccard = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
            similarities.append(jaccard)
        
        # 2. Character n-gram similarity
        def get_ngrams(s, n=3):
            return set(s[i:i+n] for i in range(len(s)-n+1))
        
        ngrams1 = get_ngrams(name1_lower, 3)
        ngrams2 = get_ngrams(name2_lower, 3)
        if ngrams1 and ngrams2:
            ngram_sim = len(ngrams1.intersection(ngrams2)) / len(ngrams1.union(ngrams2))
            similarities.append(ngram_sim)
        
        # 3. Edit distance (normalized)
        edit_dist = self._calculate_edit_distance(name1_lower, name2_lower)
        max_len = max(len(name1_lower), len(name2_lower))
        if max_len > 0:
            edit_sim = 1.0 - (edit_dist / max_len)
            similarities.append(edit_sim)
        
        # 4. Prefix/suffix matching
        common_prefix_len = len(self._longest_common_prefix(name1_lower, name2_lower))
        common_suffix_len = len(self._longest_common_suffix(name1_lower, name2_lower))
        prefix_suffix_sim = (common_prefix_len + common_suffix_len) / (len(name1_lower) + len(name2_lower))
        similarities.append(prefix_suffix_sim)
        
        # 5. Semantic similarity (using word embeddings simulation)
        semantic_sim = self._simulate_word_embedding_similarity(name1_lower, name2_lower)
        similarities.append(semantic_sim)
        
        # Combine similarities
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_structural_property_similarity(self, props1: Dict[str, Any], 
                                                props2: Dict[str, Any]) -> float:
        """Calculate structural similarity between property sets"""
        if not props1 and not props2:
            return 1.0  # Both empty
        if not props1 or not props2:
            return 0.0  # One empty
        
        # 1. Key overlap
        keys1 = set(props1.keys())
        keys2 = set(props2.keys())
        key_jaccard = len(keys1.intersection(keys2)) / len(keys1.union(keys2))
        
        # 2. Value similarity for common keys
        common_keys = keys1.intersection(keys2)
        value_similarities = []
        
        for key in common_keys:
            val1 = props1[key]
            val2 = props2[key]
            
            if type(val1) == type(val2):
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Numeric similarity
                    if max(abs(val1), abs(val2)) > 0:
                        sim = 1 - abs(val1 - val2) / max(abs(val1), abs(val2))
                    else:
                        sim = 1.0
                elif isinstance(val1, str) and isinstance(val2, str):
                    # String similarity
                    sim = self._calculate_string_similarity(val1, val2)
                elif isinstance(val1, list) and isinstance(val2, list):
                    # List similarity
                    sim = self._calculate_list_similarity(val1, val2)
                elif isinstance(val1, dict) and isinstance(val2, dict):
                    # Recursive dict similarity
                    sim = self._calculate_structural_property_similarity(val1, val2)
                else:
                    sim = 1.0 if val1 == val2 else 0.0
            else:
                sim = 0.0
            
            value_similarities.append(sim)
        
        # 3. Structure depth similarity
        depth1 = self._calculate_property_depth(props1)
        depth2 = self._calculate_property_depth(props2)
        depth_sim = 1 - abs(depth1 - depth2) / max(depth1, depth2, 1)
        
        # Combine
        weights = [0.4, 0.4, 0.2]  # key overlap, value sim, depth sim
        scores = [key_jaccard, 
                  np.mean(value_similarities) if value_similarities else 0, 
                  depth_sim]
        
        return sum(s * w for s, w in zip(scores, weights))
    
    # ========================================================================================
    # EXPANDED PROPERTY MANAGEMENT METHODS
    # ========================================================================================
    
    def _compare_compatibility(self, props1: Dict[str, Any], props2: Dict[str, Any]) -> Dict[str, Any]:
        """Sophisticated property compatibility analysis"""
        comparison = {
            "compatible": True,
            "compatibility_score": 1.0,
            "conflicts": [],
            "complementary": [],
            "synergies": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Deep compatibility analysis
        common_keys = set(props1.keys()).intersection(set(props2.keys()))
        unique_to_1 = set(props1.keys()) - set(props2.keys())
        unique_to_2 = set(props2.keys()) - set(props1.keys())
        
        # 1. Analyze conflicts in common properties
        conflict_severity_total = 0
        for key in common_keys:
            val1 = props1[key]
            val2 = props2[key]
            
            conflict = self._analyze_property_conflict(key, val1, val2)
            if conflict["has_conflict"]:
                comparison["conflicts"].append(conflict)
                comparison["compatible"] = False
                conflict_severity_total += conflict["severity"]
        
        # 2. Analyze complementary properties
        complementary_analysis = self._analyze_complementary_properties(
            unique_to_1, unique_to_2, props1, props2
        )
        comparison["complementary"] = complementary_analysis["complementary_pairs"]
        
        # 3. Identify synergies
        synergies = self._identify_property_synergies(props1, props2, common_keys)
        comparison["synergies"] = synergies
        
        # 4. Type system compatibility
        type_compatibility = self._analyze_type_compatibility(props1, props2)
        if not type_compatibility["compatible"]:
            comparison["warnings"].extend(type_compatibility["warnings"])
            comparison["compatible"] = False
        
        # 5. Domain compatibility
        domain_compat = self._analyze_domain_compatibility(props1, props2)
        if domain_compat["score"] < 0.3:
            comparison["warnings"].append({
                "type": "domain_mismatch",
                "message": "Properties appear to be from incompatible domains",
                "domains": domain_compat["domains"]
            })
        
        # 6. Calculate overall compatibility score
        scores = {
            "conflict_score": max(0, 1 - conflict_severity_total / max(len(common_keys), 1)),
            "complementary_score": len(comparison["complementary"]) / max(len(unique_to_1) + len(unique_to_2), 1),
            "synergy_score": len(synergies) / max(len(common_keys), 1),
            "type_score": type_compatibility["score"],
            "domain_score": domain_compat["score"]
        }
        
        weights = {
            "conflict_score": 0.35,
            "complementary_score": 0.20,
            "synergy_score": 0.15,
            "type_score": 0.20,
            "domain_score": 0.10
        }
        
        comparison["compatibility_score"] = sum(
            scores[k] * weights[k] for k in scores
        )
        
        # 7. Generate recommendations
        comparison["recommendations"] = self._generate_compatibility_recommendations(
            comparison, props1, props2
        )
        
        return comparison
    
    def _analyze_property_conflict(self, key: str, val1: Any, val2: Any) -> Dict[str, Any]:
        """Detailed analysis of property conflicts"""
        conflict = {
            "has_conflict": False,
            "property": key,
            "value1": val1,
            "value2": val2,
            "conflict_type": None,
            "severity": 0.0,
            "resolution_suggestions": []
        }
        
        # Type mismatch
        if type(val1) != type(val2):
            conflict["has_conflict"] = True
            conflict["conflict_type"] = "type_mismatch"
            conflict["severity"] = 0.8
            
            # Suggest resolutions
            if isinstance(val1, str) and isinstance(val2, (int, float)):
                conflict["resolution_suggestions"].append(
                    f"Convert string '{val1}' to numeric type"
                )
            elif isinstance(val1, (int, float)) and isinstance(val2, str):
                conflict["resolution_suggestions"].append(
                    f"Convert numeric {val1} to string type"
                )
        
        # Value conflicts for same type
        elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if abs(val1 - val2) > max(abs(val1), abs(val2)) * 0.5:
                conflict["has_conflict"] = True
                conflict["conflict_type"] = "numeric_discrepancy"
                conflict["severity"] = abs(val1 - val2) / max(abs(val1), abs(val2), 1)
                conflict["resolution_suggestions"].extend([
                    f"Use average: {(val1 + val2) / 2}",
                    f"Use maximum: {max(val1, val2)}",
                    f"Use minimum: {min(val1, val2)}"
                ])
        
        elif isinstance(val1, str) and isinstance(val2, str):
            if val1.lower() != val2.lower():
                # Check for semantic conflicts
                if self._are_semantically_opposite(val1, val2):
                    conflict["has_conflict"] = True
                    conflict["conflict_type"] = "semantic_opposition"
                    conflict["severity"] = 0.9
                    conflict["resolution_suggestions"].append(
                        "Values are semantically opposite - requires domain expert resolution"
                    )
                else:
                    conflict["has_conflict"] = True
                    conflict["conflict_type"] = "value_mismatch"
                    conflict["severity"] = 0.5
                    conflict["resolution_suggestions"].extend([
                        f"Concatenate: '{val1} / {val2}'",
                        "Choose based on priority or timestamp",
                        "Create multi-valued property"
                    ])
        
        elif isinstance(val1, list) and isinstance(val2, list):
            # List conflicts
            set1 = set(val1) if all(isinstance(x, (str, int, float)) for x in val1) else None
            set2 = set(val2) if all(isinstance(x, (str, int, float)) for x in val2) else None
            
            if set1 and set2 and set1 != set2:
                conflict["has_conflict"] = True
                conflict["conflict_type"] = "list_mismatch"
                conflict["severity"] = len(set1.symmetric_difference(set2)) / len(set1.union(set2))
                conflict["resolution_suggestions"].extend([
                    f"Union: {list(set1.union(set2))}",
                    f"Intersection: {list(set1.intersection(set2))}",
                    "Preserve both as alternatives"
                ])
        
        # Special handling for critical properties
        critical_properties = ["id", "type", "version", "schema", "format"]
        if key in critical_properties and conflict["has_conflict"]:
            conflict["severity"] = min(conflict["severity"] * 1.5, 1.0)
        
        return conflict
    
    def _analyze_complementary_properties(self, unique1: Set[str], unique2: Set[str],
                                         props1: Dict[str, Any], props2: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how unique properties complement each other"""
        analysis = {
            "complementary_pairs": [],
            "coverage_increase": 0.0,
            "domain_expansion": []
        }
        
        # Find semantic complements
        for prop1 in unique1:
            for prop2 in unique2:
                if self._are_complementary_properties(prop1, prop2):
                    analysis["complementary_pairs"].append({
                        "prop1": prop1,
                        "prop2": prop2,
                        "relationship": self._get_complement_relationship(prop1, prop2),
                        "value1": props1[prop1],
                        "value2": props2[prop2]
                    })
        
        # Calculate coverage increase
        total_props = len(set(props1.keys()).union(set(props2.keys())))
        original_coverage = len(props1) / total_props if total_props > 0 else 0
        combined_coverage = 1.0  # All properties
        analysis["coverage_increase"] = combined_coverage - original_coverage
        
        # Identify domain expansion
        domains1 = self._extract_property_domains(props1)
        domains2 = self._extract_property_domains(props2)
        
        for domain in domains2:
            if domain not in domains1:
                analysis["domain_expansion"].append({
                    "domain": domain,
                    "properties": [p for p in unique2 if self._property_in_domain(p, domain)]
                })
        
        return analysis
    
    def _identify_property_synergies(self, props1: Dict[str, Any], props2: Dict[str, Any],
                                   common_keys: Set[str]) -> List[Dict[str, Any]]:
        """Identify synergistic property combinations"""
        synergies = []
        
        # Synergy patterns
        synergy_patterns = [
            {
                "pattern": ["input_format", "output_format"],
                "benefit": "Complete data transformation pipeline"
            },
            {
                "pattern": ["min_value", "max_value"],
                "benefit": "Complete range specification"
            },
            {
                "pattern": ["requirement", "implementation"],
                "benefit": "Requirement-implementation traceability"
            },
            {
                "pattern": ["problem", "solution"],
                "benefit": "Problem-solution mapping"
            },
            {
                "pattern": ["cause", "effect"],
                "benefit": "Causal relationship clarity"
            }
        ]
        
        # Check for pattern matches
        all_props = set(props1.keys()).union(set(props2.keys()))
        
        for pattern_def in synergy_patterns:
            pattern = pattern_def["pattern"]
            if all(any(p in prop for prop in all_props) for p in pattern):
                # Found synergy
                matching_props = {
                    p: [prop for prop in all_props if p in prop][0] 
                    for p in pattern
                }
                
                synergies.append({
                    "type": "pattern_match",
                    "pattern": pattern,
                    "properties": matching_props,
                    "benefit": pattern_def["benefit"],
                    "strength": 0.8
                })
        
        # Value-based synergies
        for key in common_keys:
            val1 = props1[key]
            val2 = props2[key]
            
            # Numeric range synergy
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if val1 != val2:
                    synergies.append({
                        "type": "range_definition",
                        "property": key,
                        "values": [val1, val2],
                        "benefit": f"Defines range [{min(val1, val2)}, {max(val1, val2)}]",
                        "strength": 0.6
                    })
            
            # List union synergy
            elif isinstance(val1, list) and isinstance(val2, list):
                union_benefit = len(set(val1).union(set(val2))) > max(len(val1), len(val2))
                if union_benefit:
                    synergies.append({
                        "type": "list_enrichment",
                        "property": key,
                        "benefit": "Combined lists provide richer options",
                        "strength": 0.7
                    })
        
        return synergies
    
    # ========================================================================================
    # EXPANDED EXPLANATION GENERATION METHODS
    # ========================================================================================
    
    def _generate_causal_example(self, start: str, end: str) -> str:
        """Generate rich, context-aware examples for causal relationships"""
        # Comprehensive example database organized by domains
        example_database = {
            # Scientific/Physical
            ("temperature", "pressure"): {
                "example": "In a closed container, increasing temperature causes gas molecules to move faster, colliding more frequently with container walls, thus increasing pressure (Gay-Lussac's Law)",
                "domain": "physics",
                "confidence": 0.95
            },
            ("pressure", "volume"): {
                "example": "When you squeeze a balloon (increase pressure), it gets smaller (decreases volume) - demonstrating Boyle's Law where pressure and volume are inversely related",
                "domain": "physics",
                "confidence": 0.95
            },
            ("mass", "gravity"): {
                "example": "The more massive an object (like Earth vs. Moon), the stronger its gravitational pull - which is why astronauts can jump higher on the Moon",
                "domain": "physics",
                "confidence": 0.98
            },
            
            # Biological/Health
            ("exercise", "endorphins"): {
                "example": "During exercise, your body releases endorphins - natural 'feel-good' chemicals that act as painkillers and mood elevators, creating the 'runner's high'",
                "domain": "health",
                "confidence": 0.90
            },
            ("sleep deprivation", "cognitive performance"): {
                "example": "After just one night of poor sleep, reaction times slow by 50%, decision-making suffers, and memory consolidation is impaired - similar to being legally drunk",
                "domain": "neuroscience",
                "confidence": 0.92
            },
            ("stress", "inflammation"): {
                "example": "Chronic stress triggers cortisol release, which disrupts immune function and promotes inflammatory cytokines, contributing to conditions from arthritis to heart disease",
                "domain": "health",
                "confidence": 0.88
            },
            
            # Psychological/Behavioral  
            ("practice", "automaticity"): {
                "example": "When learning to drive, you initially think about every action, but after ~50 hours of practice, neural pathways strengthen through myelination, making driving automatic",
                "domain": "psychology",
                "confidence": 0.91
            },
            ("social isolation", "depression"): {
                "example": "Humans are wired for connection - isolation reduces oxytocin and serotonin while increasing stress hormones, creating a biochemical foundation for depression",
                "domain": "psychology",
                "confidence": 0.87
            },
            ("positive reinforcement", "behavior"): {
                "example": "When a child receives praise for sharing toys, dopamine reinforces the neural pathways, making them 3x more likely to share again - the basis of operant conditioning",
                "domain": "psychology",
                "confidence": 0.89
            },
            
            # Economic/Business
            ("interest rates", "investment"): {
                "example": "When central banks lower interest rates, borrowing becomes cheaper, making business expansion more attractive - like how 2020's near-zero rates fueled tech startup growth",
                "domain": "economics",
                "confidence": 0.85
            },
            ("scarcity", "value"): {
                "example": "Limited edition sneakers that cost $50 to make sell for $5000 - artificial scarcity triggers psychological value perception, regardless of intrinsic worth",
                "domain": "economics",
                "confidence": 0.90
            },
            ("automation", "employment"): {
                "example": "While ATMs were supposed to eliminate bank tellers, they actually increased teller jobs by 13% by making branches cheaper to operate, enabling more locations",
                "domain": "economics",
                "confidence": 0.82
            },
            
            # Environmental/Climate
            ("deforestation", "rainfall"): {
                "example": "Amazon rainforest trees release 20 billion tons of water daily through transpiration, creating rain clouds - cutting trees directly reduces regional rainfall by up to 25%",
                "domain": "environment",
                "confidence": 0.91
            },
            ("ocean temperature", "hurricanes"): {
                "example": "Every 1°C increase in ocean surface temperature increases hurricane wind speeds by 15-20mph, as warm water provides the energy that fuels these storms",
                "domain": "climate",
                "confidence": 0.93
            },
            
            # Technology/Computing
            ("data volume", "model accuracy"): {
                "example": "GPT-2 trained on 40GB of text achieved 85% accuracy, while GPT-3 with 570GB reached 95% - demonstrating the scaling laws of machine learning",
                "domain": "technology",
                "confidence": 0.88
            },
            ("network effect", "platform value"): {
                "example": "WhatsApp with 100 users has 4,950 possible connections, but with 1000 users has 499,500 - value grows exponentially with n(n-1)/2 connections",
                "domain": "technology",
                "confidence": 0.94
            },
            
            # Social/Cultural
            ("education", "social mobility"): {
                "example": "College graduates earn 84% more over their lifetime than high school graduates, and their children are 5x more likely to escape poverty",
                "domain": "sociology",
                "confidence": 0.86
            },
            ("diversity", "innovation"): {
                "example": "Companies with diverse teams are 35% more likely to outperform homogeneous ones - different perspectives create cognitive friction that sparks creative solutions",
                "domain": "business",
                "confidence": 0.83
            }
        }
        
        # First, try exact and reverse match
        start_lower = start.lower()
        end_lower = end.lower()
        
        # Direct lookup
        if (start_lower, end_lower) in example_database:
            return example_database[(start_lower, end_lower)]["example"]
        
        # Reverse lookup
        if (end_lower, start_lower) in example_database:
            example_data = example_database[(end_lower, start_lower)]
            return f"(Reverse relationship) {example_data['example']}"
        
        # Try partial matches with word stems
        start_stem = self._get_word_stem(start_lower)
        end_stem = self._get_word_stem(end_lower)
        
        for (key_start, key_end), data in example_database.items():
            key_start_stem = self._get_word_stem(key_start)
            key_end_stem = self._get_word_stem(key_end)
            
            if (start_stem in key_start_stem or key_start_stem in start_stem) and \
               (end_stem in key_end_stem or key_end_stem in end_stem):
                return f"{data['example']} (Similar relationship: {key_start} → {key_end})"
        
        # Try semantic similarity matching
        best_match = None
        best_score = 0.0
        
        for (key_start, key_end), data in example_database.items():
            score = self._calculate_concept_pair_similarity(
                (start_lower, end_lower), (key_start, key_end)
            )
            
            if score > best_score and score > 0.6:
                best_score = score
                best_match = (key_start, key_end, data)
        
        if best_match:
            key_start, key_end, data = best_match
            return f"{data['example']} (Analogous to: {key_start} → {key_end})"
        
        # Domain-based example generation
        return self._generate_domain_specific_example(start, end)
    
    def _generate_domain_specific_example(self, start: str, end: str) -> str:
        """Generate domain-specific example when no direct match exists"""
        # Identify domains for both concepts
        start_domain = self._identify_concept_domain(start)
        end_domain = self._identify_concept_domain(end)
        
        # Domain-specific templates
        domain_templates = {
            "physical": "Changes in {start} create measurable effects on {end} through physical mechanisms, similar to how force affects acceleration",
            "biological": "{start} influences {end} through complex biological pathways involving multiple feedback loops and regulatory mechanisms",
            "psychological": "{start} shapes {end} through cognitive and emotional processing, affecting neural pathways and behavioral patterns",
            "economic": "Market dynamics show that {start} drives {end} through supply-demand interactions and price signaling mechanisms",
            "technological": "In technological systems, {start} determines {end} through algorithmic processes and data transformations",
            "social": "Social research indicates {start} impacts {end} through interpersonal dynamics and cultural transmission",
            "environmental": "Environmental studies demonstrate how {start} affects {end} through ecosystem interactions and feedback cycles"
        }
        
        # Generate example based on domains
        if start_domain == end_domain and start_domain in domain_templates:
            base_template = domain_templates[start_domain]
        else:
            # Cross-domain relationship
            base_template = "Cross-domain analysis reveals that {start} influences {end} through indirect mechanisms and emergent properties"
        
        # Enhance with specific details
        example = base_template.format(start=start, end=end)
        
        # Add quantitative element if applicable
        if any(term in start.lower() + end.lower() for term in ["increase", "decrease", "more", "less"]):
            example += ", with studies showing significant correlation (p < 0.05) between changes in these variables"
        
        # Add mechanism hint
        mechanism = self._infer_causal_mechanism(start, end, start_domain, end_domain)
        if mechanism:
            example += f". The likely mechanism involves {mechanism}"
        
        return example
    
    def _get_word_stem(self, word: str) -> str:
        """Simple stemming for word matching"""
        # Remove common suffixes
        suffixes = ["ing", "ed", "er", "est", "ly", "tion", "sion", "ment", "ness", "ity"]
        
        word_lower = word.lower()
        for suffix in suffixes:
            if word_lower.endswith(suffix) and len(word_lower) - len(suffix) >= 3:
                return word_lower[:-len(suffix)]
        
        return word_lower
    
    def _calculate_concept_pair_similarity(self, pair1: Tuple[str, str], 
                                         pair2: Tuple[str, str]) -> float:
        """Calculate similarity between two concept pairs"""
        # Compare both directions
        forward_sim = (self._calculate_string_similarity(pair1[0], pair2[0]) + 
                       self._calculate_string_similarity(pair1[1], pair2[1])) / 2
        
        reverse_sim = (self._calculate_string_similarity(pair1[0], pair2[1]) + 
                       self._calculate_string_similarity(pair1[1], pair2[0])) / 2
        
        return max(forward_sim, reverse_sim)
    
    def _identify_concept_domain(self, concept: str) -> str:
        """Identify the domain of a concept"""
        concept_lower = concept.lower()
        
        # Domain keywords mapping
        domain_keywords = {
            "physical": ["temperature", "pressure", "force", "energy", "mass", "velocity", "momentum"],
            "biological": ["cell", "gene", "protein", "organism", "tissue", "enzyme", "metabolism"],
            "psychological": ["mind", "emotion", "thought", "behavior", "cognition", "perception", "memory"],
            "economic": ["money", "market", "price", "supply", "demand", "trade", "investment"],
            "technological": ["data", "algorithm", "system", "network", "software", "computing"],
            "social": ["society", "culture", "group", "relationship", "community", "interaction"],
            "environmental": ["climate", "ecosystem", "pollution", "resource", "habitat", "species"]
        }
        
        # Check each domain
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in concept_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # Return highest scoring domain
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return "general"
    
    def _infer_causal_mechanism(self, start: str, end: str, 
                               start_domain: str, end_domain: str) -> str:
        """Infer likely causal mechanism based on concepts and domains"""
        # Mechanism templates by domain combination
        mechanisms = {
            ("physical", "physical"): "energy transfer and conservation laws",
            ("biological", "biological"): "biochemical signaling and regulatory feedback",
            ("psychological", "psychological"): "cognitive processing and neural adaptation",
            ("economic", "economic"): "market forces and rational agent behavior",
            ("technological", "technological"): "information processing and system dynamics",
            
            # Cross-domain mechanisms
            ("physical", "biological"): "physical forces affecting biological structures",
            ("biological", "psychological"): "neurobiological processes affecting mental states",
            ("psychological", "social"): "individual behaviors aggregating to social patterns",
            ("economic", "social"): "economic incentives shaping social behaviors",
            ("environmental", "biological"): "environmental pressures driving biological responses"
        }
        
        # Look up mechanism
        mechanism = mechanisms.get((start_domain, end_domain))
        if not mechanism:
            mechanism = mechanisms.get((end_domain, start_domain))
        
        if not mechanism:
            mechanism = "complex multi-factor interactions"
        
        return mechanism
    
    # ========================================================================================
    # EXPANDED UNCERTAINTY METHODS
    # ========================================================================================
    
    def _analyze_risks_detailed(self, scenario: Dict[str, Any], 
                               context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Comprehensive risk analysis"""
        risks = []
        
        # Risk categories to analyze
        risk_categories = [
            {
                "category": "technical",
                "factors": ["complexity", "novelty", "dependencies", "scalability"],
                "weight": 0.25
            },
            {
                "category": "resource", 
                "factors": ["budget", "time", "personnel", "infrastructure"],
                "weight": 0.20
            },
            {
                "category": "market",
                "factors": ["competition", "demand", "timing", "regulations"],
                "weight": 0.15
            },
            {
                "category": "operational",
                "factors": ["processes", "quality", "efficiency", "continuity"],
                "weight": 0.20
            },
            {
                "category": "strategic",
                "factors": ["alignment", "opportunity_cost", "reputation", "flexibility"],
                "weight": 0.20
            }
        ]
        
        # Analyze each category
        for cat_info in risk_categories:
            category = cat_info["category"]
            
            # Identify risks in this category
            category_risks = self._identify_category_risks(
                scenario, context, category, cat_info["factors"]
            )
            
            for risk in category_risks:
                # Calculate risk score
                probability = risk.get("probability", 0.5)
                impact = risk.get("impact", 0.5)
                risk_score = probability * impact
                
                # Determine risk level
                if risk_score > 0.7:
                    risk_level = "critical"
                elif risk_score > 0.4:
                    risk_level = "high"
                elif risk_score > 0.2:
                    risk_level = "medium"
                else:
                    risk_level = "low"
                
                risk_entry = {
                    "id": f"risk_{len(risks)}",
                    "category": category,
                    "description": risk["description"],
                    "probability": probability,
                    "impact": impact,
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "factors": risk.get("factors", []),
                    "triggers": risk.get("triggers", []),
                    "early_warnings": self._identify_early_warnings(risk),
                    "cascading_risks": self._identify_cascading_risks(risk, scenario)
                }
                
                risks.append(risk_entry)
        
        # Sort by risk score
        risks.sort(key=lambda r: r["risk_score"], reverse=True)
        
        return risks
    
    def _identify_category_risks(self, scenario: Dict[str, Any], context: Dict[str, Any],
                               category: str, factors: List[str]) -> List[Dict[str, Any]]:
        """Identify risks within a specific category"""
        category_risks = []
        
        # Category-specific risk patterns
        risk_patterns = {
            "technical": {
                "complexity": {
                    "threshold": 0.7,
                    "risks": [
                        {
                            "description": "System complexity exceeds team capability",
                            "probability": 0.6,
                            "impact": 0.8
                        },
                        {
                            "description": "Integration challenges between components",
                            "probability": 0.7,
                            "impact": 0.6
                        }
                    ]
                },
                "novelty": {
                    "threshold": 0.8,
                    "risks": [
                        {
                            "description": "Unproven technology adoption risks",
                            "probability": 0.7,
                            "impact": 0.7
                        }
                    ]
                }
            },
            "resource": {
                "budget": {
                    "threshold": 0.6,
                    "risks": [
                        {
                            "description": "Budget overrun due to scope creep",
                            "probability": 0.6,
                            "impact": 0.7
                        }
                    ]
                },
                "time": {
                    "threshold": 0.7,
                    "risks": [
                        {
                            "description": "Timeline compression leading to quality issues",
                            "probability": 0.7,
                            "impact": 0.6
                        }
                    ]
                }
            }
        }
        
        # Analyze each factor
        if category in risk_patterns:
            for factor in factors:
                if factor in risk_patterns[category]:
                    factor_score = self._calculate_factor_score(scenario, context, category, factor)
                    
                    if factor_score > risk_patterns[category][factor]["threshold"]:
                        for risk_template in risk_patterns[category][factor]["risks"]:
                            risk = risk_template.copy()
                            risk["factors"] = [factor]
                            risk["factor_score"] = factor_score
                            category_risks.append(risk)
        
        return category_risks
    
    def _calculate_factor_score(self, scenario: Dict[str, Any], context: Dict[str, Any],
                              category: str, factor: str) -> float:
        """Calculate score for a specific risk factor"""
        # Simplified scoring logic - would be more complex in production
        base_score = 0.5
        
        # Adjust based on scenario properties
        if "complexity" in scenario and factor == "complexity":
            base_score = scenario["complexity"]
        elif "novelty" in scenario and factor == "novelty":
            base_score = scenario["novelty"]
        elif "constraints" in context:
            if "limited_budget" in context["constraints"] and factor == "budget":
                base_score = 0.8
            elif "tight_timeline" in context["constraints"] and factor == "time":
                base_score = 0.9
        
        return base_score
    
    def _identify_early_warnings(self, risk: Dict[str, Any]) -> List[str]:
        """Identify early warning signs for a risk"""
        warnings = []
        
        # General warning patterns
        if risk["probability"] > 0.6:
            warnings.append("Historical data shows high occurrence rate")
        
        if "complexity" in risk.get("factors", []):
            warnings.extend([
                "Team expressing concerns about technical challenges",
                "Increasing number of design revisions",
                "Dependencies between components growing"
            ])
        
        if "budget" in risk.get("factors", []):
            warnings.extend([
                "Actual costs exceeding estimates by >10%",
                "Frequent budget revision requests",
                "Unplanned resource requirements emerging"
            ])
        
        if "time" in risk.get("factors", []):
            warnings.extend([
                "Milestone slippage becoming common",
                "Increasing overtime requirements",
                "Quality shortcuts being discussed"
            ])
        
        return warnings[:3]  # Top 3 warnings
    
    def _identify_cascading_risks(self, risk: Dict[str, Any], 
                                scenario: Dict[str, Any]) -> List[str]:
        """Identify risks that could cascade from this risk"""
        cascading = []
        
        # Risk cascade patterns
        cascade_patterns = {
            "technical": ["operational", "time", "budget"],
            "budget": ["resource", "quality", "scope"],
            "time": ["quality", "team_morale", "stakeholder_confidence"],
            "quality": ["reputation", "customer_satisfaction", "rework"]
        }
        
        risk_category = risk.get("category", "")
        if risk_category in cascade_patterns:
            for cascade_type in cascade_patterns[risk_category]:
                cascading.append(f"Could trigger {cascade_type} risks")
        
        return cascading
    
    def _calculate_overall_risk(self, risks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate risk metrics"""
        if not risks:
            return {"level": "low", "score": 0.0, "distribution": {}}
        
        # Aggregate metrics
        total_score = sum(r["risk_score"] for r in risks)
        max_score = max(r["risk_score"] for r in risks)
        critical_count = sum(1 for r in risks if r["risk_level"] == "critical")
        high_count = sum(1 for r in risks if r["risk_level"] == "high")
        
        # Overall risk level
        if critical_count > 0 or max_score > 0.8:
            level = "critical"
        elif high_count > 2 or max_score > 0.6:
            level = "high"
        elif total_score / len(risks) > 0.4:
            level = "medium"
        else:
            level = "low"
        
        # Risk distribution
        distribution = {}
        for risk in risks:
            category = risk["category"]
            if category not in distribution:
                distribution[category] = {"count": 0, "total_score": 0}
            distribution[category]["count"] += 1
            distribution[category]["total_score"] += risk["risk_score"]
        
        return {
            "level": level,
            "score": total_score / len(risks),
            "max_individual_score": max_score,
            "critical_risks": critical_count,
            "high_risks": high_count,
            "total_risks": len(risks),
            "distribution": distribution
        }
    
    def _suggest_risk_mitigations(self, risks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest risk mitigation strategies"""
        mitigations = []
        
        # Focus on top risks
        top_risks = risks[:5]
        
        for risk in top_risks:
            mitigation_strategies = []
            
            # Category-specific strategies
            if risk["category"] == "technical":
                mitigation_strategies.extend([
                    {
                        "strategy": "Incremental development",
                        "description": "Break complex features into smaller, testable increments",
                        "cost": "low",
                        "effectiveness": 0.7
                    },
                    {
                        "strategy": "Technical spikes",
                        "description": "Conduct proof-of-concept for high-risk components",
                        "cost": "medium",
                        "effectiveness": 0.8
                    }
                ])
            
            elif risk["category"] == "resource":
                mitigation_strategies.extend([
                    {
                        "strategy": "Resource buffering",
                        "description": "Add 20% contingency to resource estimates",
                        "cost": "medium",
                        "effectiveness": 0.6
                    },
                    {
                        "strategy": "Phased resource allocation",
                        "description": "Allocate resources based on milestone achievement",
                        "cost": "low",
                        "effectiveness": 0.7
                    }
                ])
            
            elif risk["category"] == "operational":
                mitigation_strategies.extend([
                    {
                        "strategy": "Process automation",
                        "description": "Automate repetitive operational tasks",
                        "cost": "high",
                        "effectiveness": 0.8
                    },
                    {
                        "strategy": "Redundancy planning",
                        "description": "Build backup systems for critical operations",
                        "cost": "high",
                        "effectiveness": 0.9
                    }
                ])
            
            # Select best strategies based on risk score
            selected_strategies = sorted(
                mitigation_strategies,
                key=lambda s: s["effectiveness"] / (1 + {"low": 0, "medium": 0.5, "high": 1}[s["cost"]]),
                reverse=True
            )[:2]
            
            mitigations.append({
                "risk_id": risk["id"],
                "risk_description": risk["description"],
                "strategies": selected_strategies,
                "combined_effectiveness": 1 - np.prod([1 - s["effectiveness"] for s in selected_strategies])
            })
        
        return mitigations

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
