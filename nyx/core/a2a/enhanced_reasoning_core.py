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
import re
import numpy as np
import math
import nltk
import spacy
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

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
        """Enhanced domain extraction using multiple signals"""
        user_input = context.get("user_input", "").lower()
        
        # Expanded domain definitions with more keywords and patterns
        domain_definitions = {
            "health": {
                "keywords": ["health", "medical", "disease", "treatment", "patient", "doctor", 
                            "hospital", "symptom", "diagnosis", "medicine", "therapy", "wellness",
                            "nutrition", "exercise", "mental health", "pandemic", "vaccine"],
                "patterns": [r"(?:treat|cure|heal|diagnose)\s+\w+", r"\w+\s+(?:syndrome|disorder|disease)"],
                "entities": ["DISEASE", "MEDICATION", "SYMPTOM", "MEDICAL_PROCEDURE"]
            },
            "technology": {
                "keywords": ["technology", "software", "hardware", "computer", "algorithm", "data",
                            "programming", "code", "system", "network", "AI", "machine learning",
                            "database", "cloud", "cybersecurity", "API", "framework"],
                "patterns": [r"(?:develop|build|code|program)\s+\w+", r"\w+\s+(?:algorithm|system|software)"],
                "entities": ["TECHNOLOGY", "PROGRAMMING_LANGUAGE", "SOFTWARE", "HARDWARE"]
            },
            "economics": {
                "keywords": ["economics", "economy", "market", "finance", "money", "trade", "investment",
                            "stock", "bond", "GDP", "inflation", "recession", "supply", "demand",
                            "price", "cost", "revenue", "profit", "budget"],
                "patterns": [r"(?:buy|sell|trade|invest)\s+\w+", r"\w+\s+(?:market|economy|sector)"],
                "entities": ["MONEY", "CURRENCY", "FINANCIAL_INSTRUMENT", "ECONOMIC_INDICATOR"]
            },
            "psychology": {
                "keywords": ["psychology", "mind", "behavior", "emotion", "cognition", "personality",
                            "mental", "consciousness", "memory", "learning", "motivation", "perception",
                            "anxiety", "depression", "therapy", "counseling"],
                "patterns": [r"(?:feel|think|believe|perceive)\s+\w+", r"\w+\s+(?:behavior|emotion|trait)"],
                "entities": ["EMOTION", "COGNITIVE_PROCESS", "MENTAL_STATE", "PERSONALITY_TRAIT"]
            },
            "environment": {
                "keywords": ["environment", "climate", "ecology", "nature", "pollution", "conservation",
                            "sustainability", "renewable", "ecosystem", "biodiversity", "carbon",
                            "greenhouse", "global warming", "recycling", "habitat"],
                "patterns": [r"(?:protect|conserve|pollute)\s+\w+", r"\w+\s+(?:ecosystem|habitat|species)"],
                "entities": ["NATURAL_RESOURCE", "POLLUTANT", "ECOSYSTEM", "ENVIRONMENTAL_ISSUE"]
            },
            "education": {
                "keywords": ["education", "learning", "teaching", "school", "university", "student",
                            "teacher", "curriculum", "course", "degree", "knowledge", "skill",
                            "training", "academic", "research", "study"],
                "patterns": [r"(?:learn|teach|study)\s+\w+", r"\w+\s+(?:course|class|subject)"],
                "entities": ["EDUCATIONAL_INSTITUTION", "SUBJECT", "DEGREE", "SKILL"]
            }
        }
        
        # Score each domain
        domain_scores = {}
        
        for domain, definition in domain_definitions.items():
            score = 0.0
            
            # Keyword matching with weights
            keywords = definition["keywords"]
            for i, keyword in enumerate(keywords):
                if keyword in user_input:
                    # Earlier keywords in list are more definitive
                    weight = 1.0 - (i / len(keywords)) * 0.5
                    score += weight
            
            # Pattern matching
            patterns = definition["patterns"]
            for pattern in patterns:
                matches = re.findall(pattern, user_input)
                score += len(matches) * 0.5
            
            # Named entity recognition if spacy is available
            if nlp and context.get("enable_nlp", True):
                try:
                    doc = nlp(user_input)
                    for ent in doc.ents:
                        if ent.label_ in definition.get("entities", []):
                            score += 1.0
                except:
                    pass
            
            # Context clues from conversation history
            history = context.get("conversation_history", [])
            for past_message in history[-5:]:  # Look at last 5 messages
                past_text = past_message.get("text", "").lower()
                keyword_matches = sum(1 for kw in keywords[:5] if kw in past_text)
                score += keyword_matches * 0.1
            
            domain_scores[domain] = score
        
        # Also check for cross-domain indicators
        multi_domain_indicators = {
            ("health", "technology"): ["health tech", "medical AI", "digital health", "telemedicine"],
            ("economics", "technology"): ["fintech", "cryptocurrency", "algorithmic trading", "digital economy"],
            ("environment", "technology"): ["cleantech", "renewable energy tech", "smart grid", "environmental monitoring"],
            ("education", "technology"): ["edtech", "online learning", "educational software", "e-learning"]
        }
        
        for (domain1, domain2), indicators in multi_domain_indicators.items():
            if any(indicator in user_input for indicator in indicators):
                domain_scores[domain1] = domain_scores.get(domain1, 0) + 0.3
                domain_scores[domain2] = domain_scores.get(domain2, 0) + 0.3
        
        # Return highest scoring domain
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            if best_domain[1] > 0.5:  # Minimum confidence threshold
                return best_domain[0]
        
        # Fallback: try to infer from intent
        if any(word in user_input for word in ["diagnose", "treat", "cure", "symptom"]):
            return "health"
        elif any(word in user_input for word in ["code", "algorithm", "software", "data"]):
            return "technology"
        elif any(word in user_input for word in ["market", "price", "invest", "economic"]):
            return "economics"
        
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
        """Generate sophisticated causal sub-goals based on causal model analysis"""
        sub_goals = []
        goal_desc = goal.get("description", "").lower()
        goal_id = goal.get("id", "goal")
        
        # Parse goal to understand structure
        goal_components = self._parse_goal_structure(goal_desc)
        
        # For each available model, analyze causal pathways
        for model_id in available_models[:3]:
            # Get model details (in production, this would fetch actual model)
            model_info = await self._get_causal_model_info(model_id)
            
            if not model_info:
                continue
            
            # Identify relevant nodes in the causal model
            relevant_nodes = self._find_relevant_causal_nodes(goal_components, model_info)
            
            # Find causal paths to goal
            if goal_components.get("target_outcome"):
                paths = self._find_causal_paths_to_outcome(
                    goal_components["target_outcome"], 
                    relevant_nodes, 
                    model_info
                )
                
                # Generate sub-goals for each path
                for path in paths[:2]:  # Top 2 paths
                    path_subgoals = self._generate_path_subgoals(path, goal, model_info)
                    sub_goals.extend(path_subgoals)
            
            # Identify key intervention points
            intervention_points = self._identify_intervention_points(relevant_nodes, model_info)
            
            for point in intervention_points[:3]:
                sub_goal = {
                    "parent_goal": goal_id,
                    "description": f"Establish control over {point['node_name']} to influence {goal_components.get('target_outcome', 'outcome')}",
                    "type": "intervention_preparation",
                    "model": model_id,
                    "causal_importance": point.get("importance", 0.5),
                    "prerequisites": point.get("prerequisites", []),
                    "expected_effect": point.get("expected_effect", "moderate")
                }
                sub_goals.append(sub_goal)
            
            # Generate measurement sub-goals
            measurement_goals = self._generate_measurement_subgoals(goal_components, model_info)
            sub_goals.extend(measurement_goals)
        
        # Deduplicate and prioritize sub-goals
        sub_goals = self._deduplicate_and_prioritize_subgoals(sub_goals)
        
        return sub_goals

    def _parse_goal_structure(self, goal_desc: str) -> Dict[str, Any]:
        """Parse goal description to extract key components"""
        components = {
            "action": None,
            "target": None,
            "target_outcome": None,
            "constraints": [],
            "success_metrics": []
        }
        
        # Action extraction patterns
        action_patterns = {
            "improve": ["improve", "enhance", "optimize", "increase", "boost"],
            "reduce": ["reduce", "decrease", "minimize", "lower", "diminish"],
            "maintain": ["maintain", "sustain", "preserve", "keep", "stabilize"],
            "achieve": ["achieve", "reach", "attain", "accomplish", "realize"],
            "understand": ["understand", "analyze", "investigate", "explore", "examine"]
        }
        
        for action_type, keywords in action_patterns.items():
            if any(kw in goal_desc for kw in keywords):
                components["action"] = action_type
                # Extract what follows the action verb
                for kw in keywords:
                    if kw in goal_desc:
                        idx = goal_desc.index(kw) + len(kw)
                        remaining = goal_desc[idx:].strip()
                        # Extract target (next 1-3 words)
                        words = remaining.split()
                        if words:
                            components["target"] = " ".join(words[:3])
                        break
                break
        
        # Extract outcome indicators
        outcome_patterns = [
            r"(?:to|in order to|so that)\s+(.+?)(?:\.|,|;|$)",
            r"(?:resulting in|leading to|causing)\s+(.+?)(?:\.|,|;|$)",
            r"(?:outcome|result|goal):\s*(.+?)(?:\.|,|;|$)"
        ]
        
        for pattern in outcome_patterns:
            match = re.search(pattern, goal_desc)
            if match:
                components["target_outcome"] = match.group(1).strip()
                break
        
        # Extract constraints
        constraint_keywords = ["without", "while", "maintaining", "preserving", "avoiding"]
        for keyword in constraint_keywords:
            if keyword in goal_desc:
                idx = goal_desc.index(keyword)
                constraint = goal_desc[idx:idx+50].split(".")[0]
                components["constraints"].append(constraint)
        
        # Extract success metrics if mentioned
        metric_patterns = [
            r"(?:by|increase by|improve by)\s+(\d+%?)",
            r"(?:reach|achieve|attain)\s+(\d+\s*\w+)",
            r"(?:target|goal|threshold):\s*(.+?)(?:\.|,|;|$)"
        ]
        
        for pattern in metric_patterns:
            matches = re.findall(pattern, goal_desc)
            components["success_metrics"].extend(matches)
        
        return components

async def _get_causal_model_info(self, model_id: str) -> Dict[str, Any]:
    """Get detailed information about a causal model"""
    # In production, this would fetch from a model repository
    # For now, return structured example data
    
    model_templates = {
        "health_outcomes": {
            "nodes": {
                "lifestyle_factors": {"type": "input", "modifiable": True},
                "dietary_habits": {"type": "input", "modifiable": True},
                "exercise_level": {"type": "input", "modifiable": True},
                "genetic_factors": {"type": "input", "modifiable": False},
                "stress_level": {"type": "intermediate", "modifiable": True},
                "metabolic_health": {"type": "intermediate", "modifiable": False},
                "cardiovascular_health": {"type": "outcome", "modifiable": False},
                "overall_wellness": {"type": "outcome", "modifiable": False}
            },
            "edges": [
                {"from": "lifestyle_factors", "to": "stress_level", "strength": 0.7},
                {"from": "dietary_habits", "to": "metabolic_health", "strength": 0.8},
                {"from": "exercise_level", "to": "cardiovascular_health", "strength": 0.9},
                {"from": "stress_level", "to": "cardiovascular_health", "strength": -0.6},
                {"from": "metabolic_health", "to": "overall_wellness", "strength": 0.7},
                {"from": "cardiovascular_health", "to": "overall_wellness", "strength": 0.8}
            ]
        },
        "business_performance": {
            "nodes": {
                "market_conditions": {"type": "input", "modifiable": False},
                "marketing_spend": {"type": "input", "modifiable": True},
                "product_quality": {"type": "input", "modifiable": True},
                "customer_satisfaction": {"type": "intermediate", "modifiable": False},
                "brand_reputation": {"type": "intermediate", "modifiable": False},
                "sales_volume": {"type": "outcome", "modifiable": False},
                "revenue": {"type": "outcome", "modifiable": False},
                "profitability": {"type": "outcome", "modifiable": False}
            },
            "edges": [
                {"from": "product_quality", "to": "customer_satisfaction", "strength": 0.9},
                {"from": "customer_satisfaction", "to": "brand_reputation", "strength": 0.8},
                {"from": "marketing_spend", "to": "sales_volume", "strength": 0.6},
                {"from": "brand_reputation", "to": "sales_volume", "strength": 0.7},
                {"from": "sales_volume", "to": "revenue", "strength": 0.95},
                {"from": "revenue", "to": "profitability", "strength": 0.7}
            ]
        }
    }
    
    # Select appropriate template based on model_id
    if "health" in model_id.lower():
        return model_templates.get("health_outcomes", {})
    elif "business" in model_id.lower() or "performance" in model_id.lower():
        return model_templates.get("business_performance", {})
    else:
        # Generate a generic model structure
        return self._generate_generic_causal_model(model_id)

def _generate_generic_causal_model(self, model_id: str) -> Dict[str, Any]:
    """Generate a generic causal model structure"""
    return {
        "nodes": {
            f"{model_id}_input_1": {"type": "input", "modifiable": True},
            f"{model_id}_input_2": {"type": "input", "modifiable": True},
            f"{model_id}_intermediate_1": {"type": "intermediate", "modifiable": False},
            f"{model_id}_outcome_1": {"type": "outcome", "modifiable": False}
        },
        "edges": [
            {"from": f"{model_id}_input_1", "to": f"{model_id}_intermediate_1", "strength": 0.7},
            {"from": f"{model_id}_input_2", "to": f"{model_id}_intermediate_1", "strength": 0.6},
            {"from": f"{model_id}_intermediate_1", "to": f"{model_id}_outcome_1", "strength": 0.8}
        ]
    }
    
    def _select_models_for_goal(self, 
                                       goal: Dict[str, Any], 
                                       available_models: List[str],
                                       weight: float) -> List[str]:
        """Enhanced model selection based on deep analysis"""
        goal_desc = goal.get("description", "").lower()
        goal_keywords = set(goal_desc.split())
        
        # Remove common words
        stop_words = {"the", "a", "an", "to", "of", "in", "for", "and", "or", "but", "with"}
        goal_keywords = goal_keywords - stop_words
        
        # Analyze goal type and requirements
        goal_analysis = self._analyze_goal_requirements(goal)
        
        scored_models = []
        
        for model in available_models:
            score = weight  # Base score from weight
            
            # Model name analysis
            model_lower = model.lower()
            model_tokens = set(model_lower.split("_"))
            
            # Direct keyword matching
            keyword_overlap = len(goal_keywords.intersection(model_tokens))
            score += keyword_overlap * 0.2
            
            # Semantic similarity using word stems
            stem_matches = 0
            for goal_word in goal_keywords:
                goal_stem = self._get_word_stem(goal_word)
                for model_word in model_tokens:
                    model_stem = self._get_word_stem(model_word)
                    if goal_stem == model_stem and len(goal_stem) > 3:
                        stem_matches += 1
            score += stem_matches * 0.15
            
            # Goal type compatibility
            if goal_analysis["type"] == "understanding" and any(term in model_lower for term in ["analysis", "diagnostic", "explanatory"]):
                score += 0.3
            elif goal_analysis["type"] == "optimization" and any(term in model_lower for term in ["optimization", "improvement", "performance"]):
                score += 0.3
            elif goal_analysis["type"] == "prediction" and any(term in model_lower for term in ["prediction", "forecast", "projection"]):
                score += 0.3
            
            # Domain compatibility
            model_domain = self._infer_model_domain(model)
            if model_domain == goal_analysis.get("domain"):
                score += 0.25
            
            # Complexity matching
            if goal_analysis.get("complexity") == "high" and any(term in model_lower for term in ["complex", "advanced", "multi"]):
                score += 0.2
            elif goal_analysis.get("complexity") == "low" and any(term in model_lower for term in ["simple", "basic", "elementary"]):
                score += 0.2
            
            # Temporal alignment
            if goal_analysis.get("temporal") and any(term in model_lower for term in ["temporal", "time", "dynamic", "evolution"]):
                score += 0.2
            
            scored_models.append((model, score))
        
        # Sort by score and return top models
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        # Dynamic selection based on score distribution
        if scored_models:
            top_score = scored_models[0][1]
            # Include models within 20% of top score
            threshold = top_score * 0.8
            selected = [model for model, score in scored_models if score >= threshold]
            # But limit to top 5
            return selected[:5]
        
        return []:
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

    def _analyze_goal_requirements(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze goal to understand its requirements"""
        goal_desc = goal.get("description", "").lower()
        
        requirements = {
            "type": "general",
            "domain": None,
            "complexity": "medium",
            "temporal": False,
            "requires_prediction": False,
            "requires_optimization": False,
            "requires_understanding": False
        }
        
        # Determine goal type
        if any(word in goal_desc for word in ["understand", "why", "how", "explain", "analyze"]):
            requirements["type"] = "understanding"
            requirements["requires_understanding"] = True
        elif any(word in goal_desc for word in ["optimize", "improve", "maximize", "minimize", "enhance"]):
            requirements["type"] = "optimization"
            requirements["requires_optimization"] = True
        elif any(word in goal_desc for word in ["predict", "forecast", "project", "estimate", "anticipate"]):
            requirements["type"] = "prediction"
            requirements["requires_prediction"] = True
        
        # Assess complexity
        complexity_indicators = {
            "high": ["complex", "sophisticated", "advanced", "multi-factor", "comprehensive"],
            "low": ["simple", "basic", "straightforward", "elementary", "fundamental"]
        }
        
        for level, indicators in complexity_indicators.items():
            if any(ind in goal_desc for ind in indicators):
                requirements["complexity"] = level
                break
        
        # Check temporal requirements
        temporal_indicators = ["over time", "temporal", "dynamic", "evolve", "trend", "historical"]
        requirements["temporal"] = any(ind in goal_desc for ind in temporal_indicators)
        
        # Extract domain
        requirements["domain"] = self._extract_domain({"user_input": goal_desc})
        
        return requirements

    def _infer_model_domain(self, model_name: str) -> str:
        """Infer domain from model name"""
        model_lower = model_name.lower()
        
        domain_indicators = {
            "health": ["health", "medical", "clinical", "patient", "disease", "treatment"],
            "business": ["business", "market", "sales", "revenue", "customer", "performance"],
            "technology": ["tech", "system", "software", "data", "algorithm", "network"],
            "environment": ["environment", "climate", "ecology", "pollution", "sustainability"],
            "social": ["social", "community", "population", "demographic", "behavioral"]
        }
        
        for domain, indicators in domain_indicators.items():
            if any(ind in model_lower for ind in indicators):
                return domain
        
        return "general"
    
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
        
        if sp1_output and sp2_input and self._concepts_match(sp1_output, sp2_input):
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
    
    def _concepts_match(self, concept1: str, concept2: str) -> bool:
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
        """Enhanced pattern similarity checking with multiple criteria"""
        if pattern1.get("type") != pattern2.get("type"):
            return False
        
        if pattern1.get("domain") != pattern2.get("domain"):
            return False
        
        pattern_type = pattern1["type"]
        
        # Type-specific similarity checking
        if pattern_type == "causal_pattern":
            return self._causal_patterns_similar(pattern1, pattern2)
        elif pattern_type == "conceptual_pattern":
            return self._conceptual_patterns_similar(pattern1, pattern2)
        elif pattern_type == "intervention_pattern":
            return self._intervention_patterns_similar(pattern1, pattern2)
        elif pattern_type == "success_pattern":
            return self._success_patterns_similar(pattern1, pattern2)
        else:
            # Generic pattern similarity
            return self._generic_patterns_similar(pattern1, pattern2)

    def _causal_patterns_similar(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> bool:
        """Check similarity between causal patterns"""
        # Extract key components
        factors1 = set(pattern1.get("key_factors", []))
        factors2 = set(pattern2.get("key_factors", []))
        
        relations1 = pattern1.get("typical_relations", [])
        relations2 = pattern2.get("typical_relations", [])
        
        # Factor overlap check
        if factors1 and factors2:
            factor_overlap = len(factors1.intersection(factors2)) / min(len(factors1), len(factors2))
            if factor_overlap < 0.6:
                return False
        
        # Relation similarity check
        if relations1 and relations2:
            relation_similarity = self._calculate_relation_set_similarity(relations1, relations2)
            if relation_similarity < 0.5:
                return False
        
        # Strength similarity
        strength_diff = abs(pattern1.get("strength", 0.5) - pattern2.get("strength", 0.5))
        if strength_diff > 0.3:
            return False
        
        return True
    
    def _conceptual_patterns_similar(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> bool:
        """Check similarity between conceptual patterns"""
        # Compare abstraction levels
        if pattern1.get("abstraction_level") != pattern2.get("abstraction_level"):
            return False
        
        # Compare concept clusters
        clusters1 = pattern1.get("concept_clusters", [])
        clusters2 = pattern2.get("concept_clusters", [])
        
        if clusters1 and clusters2:
            cluster_similarity = self._calculate_cluster_similarity(clusters1, clusters2)
            if cluster_similarity < 0.6:
                return False
        
        # Compare creative potential
        creative_diff = abs(pattern1.get("creative_potential", 0.5) - pattern2.get("creative_potential", 0.5))
        if creative_diff > 0.4:
            return False
        
        return True
    
    def _calculate_relation_set_similarity(self, relations1: List[Any], relations2: List[Any]) -> float:
        """Calculate similarity between two sets of relations"""
        if not relations1 or not relations2:
            return 0.0
        
        # Convert relations to comparable format
        rel_set1 = set()
        rel_set2 = set()
        
        for rel in relations1:
            if isinstance(rel, dict):
                rel_key = f"{rel.get('source', '')}_{rel.get('relation_type', '')}_{rel.get('target', '')}"
            else:
                rel_key = str(rel)
            rel_set1.add(rel_key)
        
        for rel in relations2:
            if isinstance(rel, dict):
                rel_key = f"{rel.get('source', '')}_{rel.get('relation_type', '')}_{rel.get('target', '')}"
            else:
                rel_key = str(rel)
            rel_set2.add(rel_key)
        
        # Calculate Jaccard similarity
        intersection = len(rel_set1.intersection(rel_set2))
        union = len(rel_set1.union(rel_set2))
        
        return intersection / union if union > 0 else 0.0
    
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

    # ========================================================================================
    # GOAL DECOMPOSITION HELPER METHODS
    # ========================================================================================
    
    def _adapt_goal_to_timeframe(self, goal_desc: str, phase_name: str) -> str:
        """Adapt goal description to specific timeframe"""
        timeframe_adaptations = {
            "Immediate": {
                "prefixes": ["Quickly", "Immediately", "Right away"],
                "focus": "actionable first steps",
                "scope": "narrow"
            },
            "Short-term": {
                "prefixes": ["Soon", "In the near term", "Shortly"],
                "focus": "foundational progress",
                "scope": "focused"
            },
            "Medium-term": {
                "prefixes": ["Progressively", "Over time", "Steadily"],
                "focus": "substantial advancement",
                "scope": "expanded"
            },
            "Long-term": {
                "prefixes": ["Eventually", "Ultimately", "In the long run"],
                "focus": "complete realization",
                "scope": "comprehensive"
            }
        }
        
        adaptation = timeframe_adaptations.get(phase_name, timeframe_adaptations["Short-term"])
        prefix = adaptation["prefixes"][hash(goal_desc) % len(adaptation["prefixes"])]
        
        # Simplify goal for immediate timeframe
        if phase_name == "Immediate":
            # Extract core action from goal
            action_words = ["improve", "create", "develop", "analyze", "optimize", "build"]
            for word in action_words:
                if word in goal_desc.lower():
                    return f"{prefix} begin to {word} initial aspects"
        
        return f"{prefix} {goal_desc.lower()} with {adaptation['focus']}"
    
    def _assess_sub_goal_complexity(self, sub_goal_desc: str) -> float:
        """Assess complexity of a sub-goal"""
        complexity_score = 0.3  # Base complexity
        
        # Factors that increase complexity
        complexity_indicators = {
            "multiple": 0.1,
            "integrate": 0.15,
            "coordinate": 0.15,
            "optimize": 0.1,
            "analyze": 0.1,
            "complex": 0.2,
            "system": 0.1,
            "comprehensive": 0.15
        }
        
        desc_lower = sub_goal_desc.lower()
        for indicator, weight in complexity_indicators.items():
            if indicator in desc_lower:
                complexity_score += weight
        
        # Length also indicates complexity
        word_count = len(sub_goal_desc.split())
        if word_count > 15:
            complexity_score += 0.1
        elif word_count > 25:
            complexity_score += 0.2
        
        # Number of conjunctions suggests multiple parts
        conjunctions = ["and", "or", "while", "but"]
        conjunction_count = sum(1 for conj in conjunctions if f" {conj} " in desc_lower)
        complexity_score += conjunction_count * 0.05
        
        return min(1.0, complexity_score)
    
    def _identify_sub_goal_dependencies(self, sub_goal_desc: str, 
                                      all_sub_goals: List[str], 
                                      causal_analysis: Dict[str, Any]) -> List[str]:
        """Identify dependencies for a sub-goal"""
        dependencies = []
        desc_lower = sub_goal_desc.lower()
        
        # Sequential dependencies based on process order
        sequential_patterns = [
            ("implement", ["design", "plan", "analyze"]),
            ("deploy", ["build", "test", "implement"]),
            ("optimize", ["measure", "analyze", "implement"]),
            ("evaluate", ["implement", "monitor", "collect"]),
            ("scale", ["pilot", "validate", "optimize"])
        ]
        
        for key_term, prereqs in sequential_patterns:
            if key_term in desc_lower:
                for prereq in prereqs:
                    for other_goal in all_sub_goals:
                        if prereq in other_goal.lower() and other_goal != sub_goal_desc:
                            dependencies.append(f"Requires: {other_goal}")
        
        # Data dependencies
        if "analyze" in desc_lower or "process" in desc_lower:
            for other_goal in all_sub_goals:
                if "collect" in other_goal.lower() or "gather" in other_goal.lower():
                    dependencies.append(f"Data from: {other_goal}")
        
        # Resource dependencies from causal analysis
        if causal_analysis.get("shared_resources"):
            for resource in causal_analysis["shared_resources"]:
                if resource.lower() in desc_lower:
                    dependencies.append(f"Shares resource: {resource}")
        
        return list(set(dependencies))[:3]  # Return top 3 unique dependencies
    
    def _calculate_sub_goal_priority(self, sub_goal_desc: str, 
                                   parent_goal: Dict[str, Any], 
                                   causal_analysis: Dict[str, Any]) -> float:
        """Calculate priority for a sub-goal"""
        base_priority = parent_goal.get("priority", 0.5) * 0.8  # Inherit from parent
        
        # Priority modifiers
        priority_boosts = {
            "critical": 0.3,
            "essential": 0.25,
            "foundation": 0.2,
            "prerequisite": 0.2,
            "immediate": 0.15,
            "urgent": 0.15,
            "blocker": 0.25
        }
        
        desc_lower = sub_goal_desc.lower()
        for term, boost in priority_boosts.items():
            if term in desc_lower:
                base_priority += boost
        
        # Check if it's on critical path
        if causal_analysis.get("critical_path_nodes"):
            for node in causal_analysis["critical_path_nodes"]:
                if node.lower() in desc_lower:
                    base_priority += 0.2
        
        # Early phase sub-goals get priority boost
        if any(phase in desc_lower for phase in ["preparation", "initial", "first"]):
            base_priority += 0.1
        
        return min(1.0, base_priority)
    
    def _estimate_sub_goal_effort(self, sub_goal_desc: str) -> str:
        """Estimate effort required for sub-goal"""
        desc_lower = sub_goal_desc.lower()
        
        # Effort indicators
        high_effort_indicators = [
            "comprehensive", "complete", "entire", "all",
            "system-wide", "organization-wide", "redesign", "overhaul"
        ]
        
        medium_effort_indicators = [
            "develop", "implement", "create", "build",
            "analyze", "design", "integrate", "optimize"
        ]
        
        low_effort_indicators = [
            "identify", "document", "review", "assess",
            "gather", "collect", "update", "minor"
        ]
        
        # Check indicators
        high_count = sum(1 for ind in high_effort_indicators if ind in desc_lower)
        medium_count = sum(1 for ind in medium_effort_indicators if ind in desc_lower)
        low_count = sum(1 for ind in low_effort_indicators if ind in desc_lower)
        
        # Determine effort level
        if high_count > 0 or (medium_count > 2):
            return "high"
        elif medium_count > 0 or (low_count > 3):
            return "medium"
        else:
            return "low"
    
    def _define_sub_goal_criteria(self, sub_goal_desc: str) -> List[Dict[str, Any]]:
        """Define success criteria for sub-goal"""
        criteria = []
        desc_lower = sub_goal_desc.lower()
        
        # Universal criteria
        criteria.append({
            "criterion": "completion",
            "measure": "Task completed as specified",
            "target": "100%",
            "verification": "Deliverable review"
        })
        
        # Type-specific criteria
        if "analyze" in desc_lower:
            criteria.append({
                "criterion": "analysis_quality",
                "measure": "Insights generated",
                "target": "≥3 actionable insights",
                "verification": "Peer review"
            })
        
        if "implement" in desc_lower or "build" in desc_lower:
            criteria.append({
                "criterion": "functionality",
                "measure": "Feature working as designed",
                "target": "All tests passing",
                "verification": "Testing suite"
            })
        
        if "optimize" in desc_lower:
            criteria.append({
                "criterion": "improvement",
                "measure": "Performance gain",
                "target": "≥20% improvement",
                "verification": "Before/after metrics"
            })
        
        if "document" in desc_lower:
            criteria.append({
                "criterion": "documentation_quality",
                "measure": "Completeness and clarity",
                "target": "All sections complete",
                "verification": "Documentation review"
            })
        
        return criteria
    
    def _order_sub_goals_by_dependencies(self, sub_goals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Order sub-goals considering dependencies"""
        # Build dependency graph
        dependency_graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        for sub_goal in sub_goals:
            sub_goal_id = sub_goal["id"]
            for dep in sub_goal.get("dependencies", []):
                # Extract dependency ID from description
                for other in sub_goals:
                    if other["description"] in dep:
                        dependency_graph[other["id"]].append(sub_goal_id)
                        in_degree[sub_goal_id] += 1
        
        # Topological sort
        ordered = []
        queue = [sg for sg in sub_goals if in_degree[sg["id"]] == 0]
        
        while queue:
            # Sort queue by priority for tie-breaking
            queue.sort(key=lambda x: x.get("priority", 0), reverse=True)
            current = queue.pop(0)
            ordered.append(current)
            
            # Update in-degrees
            for neighbor in dependency_graph[current["id"]]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    neighbor_obj = next(sg for sg in sub_goals if sg["id"] == neighbor)
                    queue.append(neighbor_obj)
        
        # Add any remaining (circular dependencies)
        remaining = [sg for sg in sub_goals if sg not in ordered]
        ordered.extend(sorted(remaining, key=lambda x: x.get("priority", 0), reverse=True))
        
        return ordered
    
    def _extract_context_params(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant parameters from context for customization"""
        params = {
            "domain": self._extract_context_domain(context),
            "complexity_level": "medium",
            "user_expertise": "intermediate",
            "time_pressure": "normal",
            "resource_constraints": []
        }
        
        # Analyze user input for complexity indicators
        user_input = context.get("user_input", "").lower()
        
        if any(term in user_input for term in ["simple", "basic", "easy"]):
            params["complexity_level"] = "low"
        elif any(term in user_input for term in ["complex", "advanced", "sophisticated"]):
            params["complexity_level"] = "high"
        
        # Check for expertise indicators
        if any(term in user_input for term in ["beginner", "new to", "learning"]):
            params["user_expertise"] = "beginner"
        elif any(term in user_input for term in ["expert", "experienced", "advanced"]):
            params["user_expertise"] = "expert"
        
        # Check for time pressure
        if any(term in user_input for term in ["urgent", "asap", "immediately", "quickly"]):
            params["time_pressure"] = "high"
        elif any(term in user_input for term in ["whenever", "no rush", "eventually"]):
            params["time_pressure"] = "low"
        
        # Extract constraints
        if "constraints" in context:
            params["resource_constraints"] = context["constraints"]
        
        return params
    
    # ========================================================================================
    # STRING AND SIMILARITY CALCULATION METHODS
    # ========================================================================================
    
    def _calculate_edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings"""
        if len(s1) < len(s2):
            return self._calculate_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # j+1 instead of j since previous_row and current_row are one character longer
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _longest_common_prefix(self, s1: str, s2: str) -> str:
        """Find longest common prefix of two strings"""
        min_len = min(len(s1), len(s2))
        for i in range(min_len):
            if s1[i] != s2[i]:
                return s1[:i]
        return s1[:min_len]
    
    def _longest_common_suffix(self, s1: str, s2: str) -> str:
        """Find longest common suffix of two strings"""
        s1_rev = s1[::-1]
        s2_rev = s2[::-1]
        prefix = self._longest_common_prefix(s1_rev, s2_rev)
        return prefix[::-1]
    
    def _simulate_word_embedding_similarity(self, word1: str, word2: str) -> float:
        """Enhanced word embedding similarity with semantic relationships"""
        if not word1 or not word2:
            return 0.0
        
        word1_lower = word1.lower().strip()
        word2_lower = word2.lower().strip()
        
        if word1_lower == word2_lower:
            return 1.0
        
        # Expanded semantic relationships
        semantic_relationships = {
            # Synonyms and near-synonyms
            "synonyms": [
                {"increase", "boost", "raise", "enhance", "improve", "augment", "amplify"},
                {"decrease", "reduce", "lower", "diminish", "minimize", "lessen", "shrink"},
                {"create", "make", "build", "construct", "develop", "generate", "produce"},
                {"analyze", "examine", "study", "investigate", "assess", "evaluate", "inspect"},
                {"problem", "issue", "challenge", "difficulty", "obstacle", "impediment", "hurdle"},
                {"solution", "answer", "resolution", "fix", "remedy", "approach", "method"},
                {"good", "positive", "beneficial", "favorable", "advantageous", "excellent", "great"},
                {"bad", "negative", "harmful", "detrimental", "adverse", "poor", "terrible"},
                {"fast", "quick", "rapid", "swift", "speedy", "prompt", "immediate"},
                {"slow", "gradual", "leisurely", "delayed", "prolonged", "sluggish"},
                {"big", "large", "huge", "massive", "enormous", "substantial", "significant"},
                {"small", "little", "tiny", "minor", "slight", "minimal", "negligible"}
            ],
            # Antonyms
            "antonyms": [
                ("increase", "decrease"), ("up", "down"), ("left", "right"), ("in", "out"),
                ("true", "false"), ("yes", "no"), ("positive", "negative"), ("start", "stop"),
                ("hot", "cold"), ("fast", "slow"), ("high", "low"), ("new", "old"),
                ("open", "closed"), ("begin", "end"), ("success", "failure"), ("win", "lose")
            ],
            # Hierarchical relationships
            "hypernyms": {
                "car": "vehicle", "dog": "animal", "rose": "flower", "python": "programming_language",
                "table": "furniture", "apple": "fruit", "doctor": "professional", "novel": "book"
            },
            # Part-whole relationships
            "meronyms": {
                "wheel": "car", "page": "book", "branch": "tree", "room": "house",
                "ingredient": "recipe", "chapter": "book", "employee": "company"
            },
            # Functional relationships
            "functional": {
                "hammer": "nail", "key": "lock", "pen": "paper", "teacher": "student",
                "doctor": "patient", "buyer": "seller", "question": "answer"
            }
        }
        
        # Check synonyms
        for syn_group in semantic_relationships["synonyms"]:
            if word1_lower in syn_group and word2_lower in syn_group:
                # Calculate similarity based on semantic distance within group
                return 0.85 + 0.1 * (1.0 / (1 + abs(list(syn_group).index(word1_lower) - list(syn_group).index(word2_lower))))
        
        # Check antonyms
        for ant1, ant2 in semantic_relationships["antonyms"]:
            if (word1_lower == ant1 and word2_lower == ant2) or (word1_lower == ant2 and word2_lower == ant1):
                return 0.2  # Antonyms have low similarity but not zero
        
        # Check hierarchical relationships
        if word1_lower in semantic_relationships["hypernyms"]:
            if semantic_relationships["hypernyms"][word1_lower] == word2_lower:
                return 0.7  # Specific to general
        
        # Check part-whole relationships
        if word1_lower in semantic_relationships["meronyms"]:
            if semantic_relationships["meronyms"][word1_lower] == word2_lower:
                return 0.6  # Part to whole
        
        # Check functional relationships
        if word1_lower in semantic_relationships["functional"]:
            if semantic_relationships["functional"][word1_lower] == word2_lower:
                return 0.65  # Functionally related
        
        # Morphological similarity
        common_prefix = self._longest_common_prefix(word1_lower, word2_lower)
        if len(common_prefix) >= 4:  # Significant prefix match
            return 0.5 + 0.1 * (len(common_prefix) / max(len(word1_lower), len(word2_lower)))
        
        # Use character n-gram similarity as fallback
        return self._calculate_ngram_similarity(word1_lower, word2_lower)
    
    def _calculate_ngram_similarity(self, word1: str, word2: str, n: int = 3) -> float:
        """Calculate n-gram based similarity"""
        if len(word1) < n or len(word2) < n:
            # Fall back to smaller n-grams
            n = min(len(word1), len(word2), 2)
        
        if n < 2:
            # Too short for meaningful n-grams
            return 0.1 if word1[0] == word2[0] else 0.0
        
        ngrams1 = set(word1[i:i+n] for i in range(len(word1)-n+1))
        ngrams2 = set(word2[i:i+n] for i in range(len(word2)-n+1))
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0

    
    def _calculate_string_similarity(self, s1: str, s2: str) -> float:
        """Calculate overall string similarity"""
        if not s1 or not s2:
            return 0.0
        
        s1_lower = s1.lower()
        s2_lower = s2.lower()
        
        if s1_lower == s2_lower:
            return 1.0
        
        # Combine multiple similarity measures
        similarities = []
        
        # Token-based similarity
        tokens1 = set(s1_lower.split())
        tokens2 = set(s2_lower.split())
        if tokens1 and tokens2:
            token_sim = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
            similarities.append(token_sim)
        
        # Edit distance similarity
        edit_dist = self._calculate_edit_distance(s1_lower, s2_lower)
        max_len = max(len(s1_lower), len(s2_lower))
        edit_sim = 1.0 - (edit_dist / max_len) if max_len > 0 else 0
        similarities.append(edit_sim)
        
        # Character n-gram similarity
        def get_char_ngrams(s, n=2):
            return set(s[i:i+n] for i in range(len(s)-n+1))
        
        if len(s1_lower) > 1 and len(s2_lower) > 1:
            ngrams1 = get_char_ngrams(s1_lower)
            ngrams2 = get_char_ngrams(s2_lower)
            if ngrams1 and ngrams2:
                ngram_sim = len(ngrams1.intersection(ngrams2)) / len(ngrams1.union(ngrams2))
                similarities.append(ngram_sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_list_similarity(self, list1: List[Any], list2: List[Any]) -> float:
        """Calculate similarity between two lists"""
        if not list1 and not list2:
            return 1.0
        if not list1 or not list2:
            return 0.0
        
        # Convert to sets if possible
        try:
            set1 = set(list1)
            set2 = set(list2)
            
            # Jaccard similarity
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return intersection / union if union > 0 else 0.0
        except TypeError:
            # Elements not hashable, use position-based similarity
            matches = 0
            max_len = max(len(list1), len(list2))
            
            for i in range(min(len(list1), len(list2))):
                if list1[i] == list2[i]:
                    matches += 1
            
            return matches / max_len if max_len > 0 else 0.0
    
    def _calculate_property_depth(self, props: Dict[str, Any], depth: int = 0) -> int:
        """Calculate maximum depth of nested properties"""
        if not isinstance(props, dict):
            return depth
        
        max_depth = depth
        for value in props.values():
            if isinstance(value, dict):
                max_depth = max(max_depth, self._calculate_property_depth(value, depth + 1))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        max_depth = max(max_depth, self._calculate_property_depth(item, depth + 1))
        
        return max_depth
    
    # ========================================================================================
    # ADVANCED SIMILARITY CALCULATION METHODS
    # ========================================================================================
    
    def _calculate_semantic_field_similarity(self, concept1: Dict[str, Any], 
                                           concept2: Dict[str, Any]) -> float:
        """Calculate similarity based on semantic fields"""
        # Extract semantic indicators
        name1 = concept1.get("name", "").lower()
        name2 = concept2.get("name", "").lower()
        
        # Define semantic fields
        semantic_fields = {
            "temporal": ["time", "duration", "period", "moment", "schedule", "timeline"],
            "spatial": ["location", "position", "area", "region", "space", "place"],
            "quantitative": ["amount", "number", "quantity", "measure", "count", "value"],
            "qualitative": ["quality", "characteristic", "property", "attribute", "feature"],
            "process": ["method", "procedure", "process", "technique", "approach", "strategy"],
            "state": ["condition", "status", "state", "situation", "circumstance"],
            "action": ["action", "activity", "operation", "task", "function", "behavior"],
            "entity": ["object", "entity", "thing", "item", "element", "component"]
        }
        
        # Find semantic fields for each concept
        fields1 = set()
        fields2 = set()
        
        for field, indicators in semantic_fields.items():
            if any(ind in name1 for ind in indicators):
                fields1.add(field)
            if any(ind in name2 for ind in indicators):
                fields2.add(field)
        
        # Add fields from properties
        props1 = concept1.get("properties", {})
        props2 = concept2.get("properties", {})
        
        for field, indicators in semantic_fields.items():
            if any(ind in str(props1).lower() for ind in indicators):
                fields1.add(field)
            if any(ind in str(props2).lower() for ind in indicators):
                fields2.add(field)
        
        # Calculate field overlap
        if not fields1 and not fields2:
            return 0.5  # No semantic field information
        
        if not fields1 or not fields2:
            return 0.0
        
        intersection = len(fields1.intersection(fields2))
        union = len(fields1.union(fields2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_relational_similarity(self, concept1: Dict[str, Any], 
                                       concept2: Dict[str, Any]) -> float:
        """Calculate similarity based on relational patterns"""
        # In a full implementation, this would analyze the relations each concept participates in
        # For now, we'll use property-based heuristics
        
        props1 = concept1.get("properties", {})
        props2 = concept2.get("properties", {})
        
        # Look for relational properties
        relational_props = ["relates_to", "connected_to", "depends_on", "influences", 
                           "caused_by", "leads_to", "part_of", "contains"]
        
        relations1 = {k: v for k, v in props1.items() if any(rp in k for rp in relational_props)}
        relations2 = {k: v for k, v in props2.items() if any(rp in k for rp in relational_props)}
        
        if not relations1 and not relations2:
            return 0.5  # No relational information
        
        # Compare relational patterns
        pattern_similarity = 0.0
        pattern_count = 0
        
        for rel_type in relational_props:
            rels1 = [v for k, v in relations1.items() if rel_type in k]
            rels2 = [v for k, v in relations2.items() if rel_type in k]
            
            if rels1 or rels2:
                if rels1 and rels2:
                    # Both have this relation type
                    pattern_similarity += 0.8
                else:
                    # Only one has this relation type
                    pattern_similarity += 0.2
                pattern_count += 1
        
        return pattern_similarity / pattern_count if pattern_count > 0 else 0.0
    
    def _calculate_functional_similarity(self, concept1: Dict[str, Any], 
                                       concept2: Dict[str, Any]) -> float:
        """Calculate similarity based on functional roles"""
        # Extract functional indicators
        name1 = concept1.get("name", "").lower()
        name2 = concept2.get("name", "").lower()
        props1 = concept1.get("properties", {})
        props2 = concept2.get("properties", {})
        
        # Define functional categories
        functional_categories = {
            "input": ["input", "source", "data", "information", "parameter"],
            "output": ["output", "result", "product", "outcome", "return"],
            "process": ["process", "transform", "compute", "calculate", "analyze"],
            "storage": ["store", "save", "cache", "memory", "database"],
            "control": ["control", "manage", "coordinate", "regulate", "govern"],
            "interface": ["interface", "api", "endpoint", "connection", "bridge"],
            "validation": ["validate", "verify", "check", "ensure", "confirm"]
        }
        
        # Identify functions for each concept
        functions1 = set()
        functions2 = set()
        
        for func, indicators in functional_categories.items():
            if any(ind in name1 or ind in str(props1) for ind in indicators):
                functions1.add(func)
            if any(ind in name2 or ind in str(props2) for ind in indicators):
                functions2.add(func)
        
        # Calculate functional overlap
        if not functions1 and not functions2:
            return 0.5
        
        if not functions1 or not functions2:
            return 0.0
        
        intersection = len(functions1.intersection(functions2))
        union = len(functions1.union(functions2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_contextual_similarity_advanced(self, concept1: Dict[str, Any], 
                                                concept2: Dict[str, Any]) -> float:
        """Calculate advanced contextual similarity"""
        # This would ideally use the actual graph context
        # For now, use property overlap as a proxy
        
        props1 = set(concept1.get("properties", {}).keys())
        props2 = set(concept2.get("properties", {}).keys())
        
        if not props1 and not props2:
            return 0.5
        
        # Property key overlap
        key_overlap = len(props1.intersection(props2)) / len(props1.union(props2)) if props1.union(props2) else 0
        
        # Value similarity for common properties
        common_props = props1.intersection(props2)
        value_similarities = []
        
        for prop in common_props:
            val1 = concept1["properties"][prop]
            val2 = concept2["properties"][prop]
            
            if type(val1) == type(val2):
                if isinstance(val1, (str, int, float)):
                    if val1 == val2:
                        value_similarities.append(1.0)
                    else:
                        value_similarities.append(0.0)
        
        value_sim = np.mean(value_similarities) if value_similarities else 0.0
        
        # Combine
        return 0.6 * key_overlap + 0.4 * value_sim
    
    # ========================================================================================
    # PROPERTY ANALYSIS METHODS
    # ========================================================================================
    
    def _are_semantically_opposite(self, val1: str, val2: str) -> bool:
        """Check if two values are semantically opposite"""
        opposites = [
            ("increase", "decrease"), ("up", "down"), ("left", "right"),
            ("true", "false"), ("yes", "no"), ("positive", "negative"),
            ("hot", "cold"), ("fast", "slow"), ("high", "low"),
            ("open", "closed"), ("start", "stop"), ("begin", "end"),
            ("enable", "disable"), ("allow", "deny"), ("accept", "reject")
        ]
        
        val1_lower = val1.lower()
        val2_lower = val2.lower()
        
        for opp1, opp2 in opposites:
            if (opp1 in val1_lower and opp2 in val2_lower) or \
               (opp2 in val1_lower and opp1 in val2_lower):
                return True
        
        return False
    
    def _are_complementary_properties(self, prop1: str, prop2: str) -> bool:
        """Check if two properties are complementary"""
        complementary_pairs = [
            ("min", "max"), ("start", "end"), ("from", "to"),
            ("source", "target"), ("input", "output"), ("question", "answer"),
            ("problem", "solution"), ("cause", "effect"), ("before", "after"),
            ("width", "height"), ("latitude", "longitude"), ("x", "y")
        ]
        
        prop1_lower = prop1.lower()
        prop2_lower = prop2.lower()
        
        for comp1, comp2 in complementary_pairs:
            if (comp1 in prop1_lower and comp2 in prop2_lower) or \
               (comp2 in prop1_lower and comp1 in prop2_lower):
                return True
        
        return False
    
    def _get_complement_relationship(self, prop1: str, prop2: str) -> str:
        """Get the type of complementary relationship"""
        prop1_lower = prop1.lower()
        prop2_lower = prop2.lower()
        
        if ("min" in prop1_lower and "max" in prop2_lower) or \
           ("max" in prop1_lower and "min" in prop2_lower):
            return "range_bounds"
        
        if ("start" in prop1_lower and "end" in prop2_lower) or \
           ("from" in prop1_lower and "to" in prop2_lower):
            return "interval_bounds"
        
        if ("input" in prop1_lower and "output" in prop2_lower) or \
           ("source" in prop1_lower and "target" in prop2_lower):
            return "flow_endpoints"
        
        if ("cause" in prop1_lower and "effect" in prop2_lower):
            return "causal_pair"
        
        return "complementary"
    
    def _extract_property_domains(self, props: Dict[str, Any]) -> Set[str]:
        """Extract domains from properties"""
        domains = set()
        
        domain_indicators = {
            "technical": ["system", "code", "api", "database", "algorithm"],
            "business": ["revenue", "cost", "profit", "customer", "market"],
            "temporal": ["time", "date", "duration", "schedule", "deadline"],
            "spatial": ["location", "position", "coordinate", "area", "region"],
            "quantitative": ["amount", "count", "number", "quantity", "measure"],
            "qualitative": ["quality", "rating", "category", "type", "class"]
        }
        
        props_str = str(props).lower()
        
        for domain, indicators in domain_indicators.items():
            if any(ind in props_str for ind in indicators):
                domains.add(domain)
        
        return domains
    
    def _property_in_domain(self, prop: str, domain: str) -> bool:
        """Check if a property belongs to a domain"""
        domain_keywords = {
            "technical": ["system", "code", "api", "database", "algorithm"],
            "business": ["revenue", "cost", "profit", "customer", "market"],
            "temporal": ["time", "date", "duration", "schedule", "deadline"],
            "spatial": ["location", "position", "coordinate", "area", "region"],
            "quantitative": ["amount", "count", "number", "quantity", "measure"],
            "qualitative": ["quality", "rating", "category", "type", "class"]
        }
        
        keywords = domain_keywords.get(domain, [])
        prop_lower = prop.lower()
        
        return any(keyword in prop_lower for keyword in keywords)
    
    def _analyze_type_compatibility(self, props1: Dict[str, Any], 
                                   props2: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze type system compatibility between properties"""
        result = {
            "compatible": True,
            "score": 1.0,
            "warnings": [],
            "type_conflicts": []
        }
        
        # Extract type information
        types1 = self._extract_types(props1)
        types2 = self._extract_types(props2)
        
        # Check for type system conflicts
        if types1.get("schema") and types2.get("schema"):
            if types1["schema"] != types2["schema"]:
                result["compatible"] = False
                result["warnings"].append({
                    "type": "schema_mismatch",
                    "message": f"Different schemas: {types1['schema']} vs {types2['schema']}"
                })
                result["score"] *= 0.5
        
        # Check for data type conflicts
        common_fields = set(types1.get("fields", {}).keys()).intersection(
            set(types2.get("fields", {}).keys())
        )
        
        for field in common_fields:
            type1 = types1["fields"][field]
            type2 = types2["fields"][field]
            
            if type1 != type2:
                result["type_conflicts"].append({
                    "field": field,
                    "type1": type1,
                    "type2": type2
                })
                result["score"] *= 0.8
        
        return result
    
    def _extract_types(self, props: Dict[str, Any]) -> Dict[str, Any]:
        """Extract type information from properties"""
        types = {
            "schema": props.get("schema") or props.get("$schema"),
            "fields": {}
        }
        
        for key, value in props.items():
            if key in ["type", "dataType", "data_type"]:
                types["primary_type"] = value
            elif isinstance(value, dict) and "type" in value:
                types["fields"][key] = value["type"]
            else:
                # Infer type from value
                types["fields"][key] = type(value).__name__
        
        return types
    
    def _analyze_domain_compatibility(self, props1: Dict[str, Any], 
                                    props2: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze domain compatibility between properties"""
        domains1 = self._extract_property_domains(props1)
        domains2 = self._extract_property_domains(props2)
        
        if not domains1 and not domains2:
            return {"score": 0.5, "domains": []}
        
        if not domains1 or not domains2:
            return {"score": 0.2, "domains": list(domains1.union(domains2))}
        
        intersection = domains1.intersection(domains2)
        union = domains1.union(domains2)
        
        score = len(intersection) / len(union) if union else 0
        
        return {
            "score": score,
            "domains": list(union),
            "common_domains": list(intersection),
            "unique_to_1": list(domains1 - domains2),
            "unique_to_2": list(domains2 - domains1)
        }
    
    def _generate_compatibility_recommendations(self, comparison: Dict[str, Any],
                                              props1: Dict[str, Any],
                                              props2: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations for improving compatibility"""
        recommendations = []
        
        # Handle conflicts
        if comparison["conflicts"]:
            recommendations.append({
                "type": "conflict_resolution",
                "priority": "high",
                "action": "Resolve property conflicts",
                "details": f"Address {len(comparison['conflicts'])} conflicts, starting with highest severity"
            })
        
        # Leverage complementary properties
        if comparison["complementary"]:
            recommendations.append({
                "type": "integration_opportunity",
                "priority": "medium",
                "action": "Integrate complementary properties",
                "details": f"Combine {len(comparison['complementary'])} complementary property pairs"
            })
        
        # Build on synergies
        if comparison["synergies"]:
            recommendations.append({
                "type": "synergy_exploitation",
                "priority": "medium",
                "action": "Maximize synergistic effects",
                "details": f"Leverage {len(comparison['synergies'])} identified synergies"
            })
        
        # Address warnings
        if comparison["warnings"]:
            recommendations.append({
                "type": "risk_mitigation",
                "priority": "high",
                "action": "Address compatibility warnings",
                "details": f"Resolve {len(comparison['warnings'])} compatibility issues"
            })
        
        return recommendations
    
    # ========================================================================================
    # CLUSTERING METHODS
    # ========================================================================================
    
    def _grow_adaptive_dense_cluster(self, start_node: str, space: Any, 
                                   visited: Set[str], density_threshold: float) -> Dict[str, Any]:
        """Grow a dense cluster from a starting node"""
        cluster = {
            "members": [start_node],
            "size": 1,
            "type": "density",
            "density": 1.0,
            "center": start_node
        }
        
        visited.add(start_node)
        frontier = [start_node]
        
        while frontier:
            current = frontier.pop(0)
            
            # Get neighbors
            neighbors = self._get_node_neighbors(current, space)
            
            # Calculate local density
            local_density = len(neighbors) / len(space.concepts) if space.concepts else 0
            
            if local_density >= density_threshold:
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        cluster["members"].append(neighbor)
                        cluster["size"] += 1
                        frontier.append(neighbor)
        
        # Calculate final cluster density
        internal_edges = 0
        for member in cluster["members"]:
            member_neighbors = self._get_node_neighbors(member, space)
            internal_edges += len([n for n in member_neighbors if n in cluster["members"]])
        
        max_edges = cluster["size"] * (cluster["size"] - 1)
        cluster["density"] = internal_edges / max_edges if max_edges > 0 else 0
        
        return cluster
    
    def _get_node_neighbors(self, node: str, space: Any) -> List[str]:
        """Get neighbors of a node in the concept space"""
        neighbors = []
        
        for relation in space.relations:
            if relation.get("source") == node:
                neighbors.append(relation.get("target"))
            elif relation.get("target") == node and not relation.get("directed", True):
                neighbors.append(relation.get("source"))
        
        return list(set(neighbors))
    
    def _calculate_name_similarity_advanced(self, name1: str, name2: str) -> float:
        """Advanced name similarity calculation"""
        # Already implemented in parent, but adding more sophisticated features
        if not name1 or not name2:
            return 0.0
        
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        if name1_lower == name2_lower:
            return 1.0
        
        # Multiple similarity measures
        similarities = []
        
        # Word-level similarity
        words1 = set(name1_lower.split())
        words2 = set(name2_lower.split())
        
        if words1 and words2:
            # Exact word overlap
            word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
            similarities.append(word_overlap)
            
            # Stem overlap
            stems1 = {self._get_word_stem(w) for w in words1}
            stems2 = {self._get_word_stem(w) for w in words2}
            stem_overlap = len(stems1.intersection(stems2)) / len(stems1.union(stems2))
            similarities.append(stem_overlap * 0.8)
        
        # Character-level similarity
        char_sim = self._calculate_string_similarity(name1_lower, name2_lower)
        similarities.append(char_sim)
        
        # Semantic similarity
        semantic_sim = self._simulate_word_embedding_similarity(name1_lower, name2_lower)
        similarities.append(semantic_sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_property_similarity_advanced(self, props1: Dict[str, Any], 
                                              props2: Dict[str, Any]) -> float:
        """Advanced property similarity calculation"""
        if not props1 and not props2:
            return 1.0
        if not props1 or not props2:
            return 0.0
        
        return self._calculate_structural_property_similarity(props1, props2)
    
    def _calculate_contextual_similarity(self, node1: str, node2: str, space: Any) -> float:
        """Calculate contextual similarity based on graph neighborhood"""
        # Get neighbors
        neighbors1 = set(self._get_node_neighbors(node1, space))
        neighbors2 = set(self._get_node_neighbors(node2, space))
        
        if not neighbors1 and not neighbors2:
            return 0.5
        
        if not neighbors1 or not neighbors2:
            return 0.0
        
        # Neighbor overlap
        overlap = len(neighbors1.intersection(neighbors2))
        union = len(neighbors1.union(neighbors2))
        
        return overlap / union if union > 0 else 0.0
    
    def _affinity_propagation(self, similarity_matrix: np.ndarray) -> Dict[int, List[int]]:
        """Simplified affinity propagation clustering"""
        n = similarity_matrix.shape[0]
        
        # Initialize
        availability = np.zeros((n, n))
        responsibility = np.zeros((n, n))
        
        # Parameters
        damping = 0.5
        max_iter = 50
        
        for _ in range(max_iter):
            # Update responsibilities
            for i in range(n):
                for k in range(n):
                    max_val = -np.inf
                    for kp in range(n):
                        if kp != k:
                            val = availability[i, kp] + similarity_matrix[i, kp]
                            if val > max_val:
                                max_val = val
                    
                    responsibility[i, k] = (1 - damping) * (similarity_matrix[i, k] - max_val) + \
                                           damping * responsibility[i, k]
            
            # Update availabilities
            for i in range(n):
                for k in range(n):
                    if i == k:
                        sum_val = 0
                        for ip in range(n):
                            if ip != k:
                                sum_val += max(0, responsibility[ip, k])
                        availability[i, k] = (1 - damping) * sum_val + damping * availability[i, k]
                    else:
                        sum_val = 0
                        for ip in range(n):
                            if ip != i and ip != k:
                                sum_val += max(0, responsibility[ip, k])
                        availability[i, k] = (1 - damping) * min(0, responsibility[k, k] + sum_val) + \
                                            damping * availability[i, k]
        
        # Find exemplars
        exemplars = []
        for i in range(n):
            if responsibility[i, i] + availability[i, i] > 0:
                exemplars.append(i)
        
        # Assign points to clusters
        clusters = defaultdict(list)
        for i in range(n):
            if i in exemplars:
                clusters[i].append(i)
            else:
                # Find best exemplar
                best_exemplar = None
                best_score = -np.inf
                
                for ex in exemplars:
                    score = similarity_matrix[i, ex]
                    if score > best_score:
                        best_score = score
                        best_exemplar = ex
                
                if best_exemplar is not None:
                    clusters[best_exemplar].append(i)
        
        return dict(clusters)
    
    # ========================================================================================
    # ADDITIONAL CLUSTERING AND HIERARCHY METHODS
    # ========================================================================================
    
    def _find_dag_roots(self, parent_children: Dict, child_parents: Dict) -> List[str]:
        """Find roots for DAG hierarchies"""
        all_nodes = set(parent_children.keys()) | set(child_parents.keys())
        
        # Nodes with no parents or only self-references
        roots = []
        for node in all_nodes:
            parents = child_parents.get(node, [])
            if not parents or (len(parents) == 1 and parents[0] == node):
                # Verify it has children
                if parent_children.get(node):
                    roots.append(node)
        
        return roots
    
    def _build_dag_hierarchy(self, root: str, parent_children: Dict, 
                            child_parents: Dict, space: Any) -> Dict[str, Any]:
        """Build DAG hierarchy allowing multiple parents"""
        hierarchy = {
            "root": root,
            "hierarchy_type": "dag",
            "levels": defaultdict(list),
            "total_nodes": 0,
            "max_depth": 0,
            "dag_structure": defaultdict(dict),
            "node_metadata": {}
        }
        
        # Modified BFS for DAG
        visited = set()
        queue = [(root, 0)]
        node_levels = {}
        
        while queue:
            node_id, level = queue.pop(0)
            
            if node_id in node_levels:
                # Update to maximum level
                node_levels[node_id] = max(node_levels[node_id], level)
                continue
            
            node_levels[node_id] = level
            visited.add(node_id)
            
            hierarchy["levels"][level].append(node_id)
            hierarchy["total_nodes"] = len(visited)
            hierarchy["max_depth"] = max(hierarchy["max_depth"], level)
            
            # Store node metadata
            node_data = space.concepts.get(node_id, {})
            hierarchy["node_metadata"][node_id] = {
                "name": node_data.get("name", node_id),
                "level": level,
                "parents": child_parents.get(node_id, []),
                "children": parent_children.get(node_id, [])
            }
            
            # Add children
            for child in parent_children.get(node_id, []):
                queue.append((child, level + 1))
        
        return hierarchy
    
    def _find_domain_hierarchies(self, space: Any, parent_children: Dict, 
                               relation_types: Dict) -> List[Dict[str, Any]]:
        """Find domain-specific hierarchies"""
        hierarchies = []
        
        # Group by relation type
        for node, rel_counts in relation_types.items():
            dominant_relation = max(rel_counts.items(), key=lambda x: x[1])[0] if rel_counts else None
            
            if dominant_relation and parent_children.get(node):
                hierarchy = self._build_complete_hierarchy(node, parent_children, space, f"domain_{dominant_relation}")
                
                if hierarchy["total_nodes"] > 3:
                    hierarchy["domain_relation"] = dominant_relation
                    hierarchies.append(hierarchy)
        
        return hierarchies
    
    def _find_property_based_hierarchies(self, space: Any) -> List[Dict[str, Any]]:
        """Find hierarchies based on property inheritance"""
        hierarchies = []
        
        # Build property inheritance map
        property_inheritance = defaultdict(list)
        
        for concept_id, concept in space.concepts.items():
            props = set(concept.get("properties", {}).keys())
            
            # Find potential parents (concepts with subset of properties)
            for other_id, other_concept in space.concepts.items():
                if other_id != concept_id:
                    other_props = set(other_concept.get("properties", {}).keys())
                    
                    if other_props and props and other_props.issubset(props):
                        property_inheritance[other_id].append(concept_id)
        
        # Find roots and build hierarchies
        roots = [node for node in property_inheritance if node not in 
                 set(child for children in property_inheritance.values() for child in children)]
        
        for root in roots:
            hierarchy = self._build_complete_hierarchy(root, property_inheritance, space, "property_based")
            
            if hierarchy["total_nodes"] > 2:
                hierarchies.append(hierarchy)
        
        return hierarchies
    
    def _analyze_hierarchy_characteristics(self, hierarchy: Dict[str, Any], 
                                         parent_children: Dict, child_parents: Dict, 
                                         space: Any):
        """Analyze characteristics of a hierarchy"""
        chars = hierarchy.setdefault("characteristics", {})
        
        # Balance analysis
        level_sizes = [len(nodes) for nodes in hierarchy["levels"].values()]
        chars["balance"] = np.std(level_sizes) / np.mean(level_sizes) if level_sizes else 0
        
        # Branching factor
        branching_factors = []
        for node in hierarchy["tree_structure"]:
            children = len(hierarchy["tree_structure"][node])
            if children > 0:
                branching_factors.append(children)
        
        chars["avg_branching_factor"] = np.mean(branching_factors) if branching_factors else 0
        chars["max_branching_factor"] = max(branching_factors) if branching_factors else 0
        
        # Depth distribution
        chars["depth_distribution"] = {
            level: count for level, count in enumerate(level_sizes)
        }
        
        # Semantic coherence
        chars["semantic_coherence"] = self._calculate_hierarchy_coherence(hierarchy, space)
    
    def _calculate_hierarchy_coherence(self, hierarchy: Dict[str, Any], space: Any) -> float:
        """Calculate semantic coherence of hierarchy"""
        coherence_scores = []
        
        # Check parent-child semantic similarity
        for parent, children in hierarchy.get("tree_structure", {}).items():
            parent_concept = space.concepts.get(parent, {})
            
            for child in children:
                child_concept = space.concepts.get(child, {})
                
                similarity = self._calculate_comprehensive_similarity(parent_concept, child_concept)
                coherence_scores.append(similarity)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_hierarchy_metrics(self, hierarchy: Dict[str, Any], space: Any):
        """Calculate quality metrics for hierarchy"""
        metrics = hierarchy.setdefault("quality_metrics", {})
        
        # Completeness (% of concepts included)
        total_concepts = len(space.concepts)
        metrics["completeness"] = hierarchy["total_nodes"] / total_concepts if total_concepts > 0 else 0
        
        # Depth appropriateness
        ideal_depth = math.log2(hierarchy["total_nodes"]) if hierarchy["total_nodes"] > 1 else 1
        metrics["depth_appropriateness"] = 1 - abs(hierarchy["max_depth"] - ideal_depth) / ideal_depth
        
        # Coherence
        metrics["coherence"] = hierarchy["characteristics"].get("semantic_coherence", 0)
        
        # Overall quality
        metrics["quality_score"] = np.mean([
            metrics["completeness"],
            metrics["depth_appropriateness"],
            metrics["coherence"]
        ])
    
    def _calculate_tree_statistics(self, hierarchy: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics for tree structure"""
        stats = {
            "total_nodes": hierarchy["total_nodes"],
            "max_depth": hierarchy["max_depth"],
            "levels": len(hierarchy["levels"]),
            "leaf_nodes": 0,
            "internal_nodes": 0,
            "avg_depth": 0
        }
        
        # Count leaf and internal nodes
        depths = []
        for node_id, node_meta in hierarchy["node_metadata"].items():
            if not hierarchy["tree_structure"].get(node_id):
                stats["leaf_nodes"] += 1
            else:
                stats["internal_nodes"] += 1
            
            depths.append(node_meta["level"])
        
        stats["avg_depth"] = np.mean(depths) if depths else 0
        
        return stats
    
    # ========================================================================================
    # ADDITIONAL CLUSTERING ALGORITHMS
    # ========================================================================================
    
    def _property_clustering_advanced(self, space: Any) -> List[Dict[str, Any]]:
        """Advanced property-based clustering"""
        clusters = []
        concepts = list(space.concepts.items())
        
        # Build property vectors
        all_props = set()
        for _, concept in concepts:
            all_props.update(concept.get("properties", {}).keys())
        
        prop_list = list(all_props)
        
        # Create binary property matrix
        property_matrix = []
        for _, concept in concepts:
            props = concept.get("properties", {})
            vector = [1 if prop in props else 0 for prop in prop_list]
            property_matrix.append(vector)
        
        if not property_matrix:
            return clusters
        
        # Cluster using property patterns
        property_matrix = np.array(property_matrix)
        
        # Find concepts with similar property patterns
        similarity_threshold = 0.7
        clustered = set()
        
        for i in range(len(concepts)):
            if i in clustered:
                continue
            
            cluster = {
                "members": [concepts[i][0]],
                "size": 1,
                "type": "property_based",
                "common_properties": []
            }
            clustered.add(i)
            
            for j in range(i + 1, len(concepts)):
                if j in clustered:
                    continue
                
                # Calculate property similarity
                if len(prop_list) > 0:
                    similarity = np.dot(property_matrix[i], property_matrix[j]) / len(prop_list)
                    
                    if similarity >= similarity_threshold:
                        cluster["members"].append(concepts[j][0])
                        cluster["size"] += 1
                        clustered.add(j)
            
            if cluster["size"] >= 3:
                # Find common properties
                common_props = []
                for prop_idx, prop in enumerate(prop_list):
                    if all(property_matrix[concepts.index((cid, space.concepts[cid]))][prop_idx] 
                          for cid in cluster["members"] if cid in space.concepts):
                        common_props.append(prop)
                
                cluster["common_properties"] = common_props
                clusters.append(cluster)
        
        return clusters
    
    def _relational_clustering(self, space: Any) -> List[Dict[str, Any]]:
        """Cluster based on relational patterns"""
        clusters = []
        
        # Build relation patterns for each concept
        relation_patterns = defaultdict(lambda: defaultdict(int))
        
        for relation in space.relations:
            source = relation.get("source")
            target = relation.get("target")
            rel_type = relation.get("relation_type", "unknown")
            
            relation_patterns[source][f"out_{rel_type}"] += 1
            relation_patterns[target][f"in_{rel_type}"] += 1
        
        # Convert to vectors
        all_patterns = set()
        for patterns in relation_patterns.values():
            all_patterns.update(patterns.keys())
        
        pattern_list = list(all_patterns)
        concepts = list(relation_patterns.keys())
        
        # Create pattern matrix
        pattern_matrix = []
        for concept in concepts:
            patterns = relation_patterns[concept]
            vector = [patterns.get(pat, 0) for pat in pattern_list]
            pattern_matrix.append(vector)
        
        if not pattern_matrix:
            return clusters
        
        # Normalize vectors
        pattern_matrix = np.array(pattern_matrix)
        norms = np.linalg.norm(pattern_matrix, axis=1)
        pattern_matrix = pattern_matrix / (norms[:, np.newaxis] + 1e-10)
        
        # Cluster using cosine similarity
        similarity_threshold = 0.8
        clustered = set()
        
        for i in range(len(concepts)):
            if i in clustered:
                continue
            
            cluster = {
                "members": [concepts[i]],
                "size": 1,
                "type": "relational",
                "pattern": pattern_matrix[i].tolist()
            }
            clustered.add(i)
            
            for j in range(i + 1, len(concepts)):
                if j in clustered:
                    continue
                
                # Cosine similarity
                similarity = np.dot(pattern_matrix[i], pattern_matrix[j])
                
                if similarity >= similarity_threshold:
                    cluster["members"].append(concepts[j])
                    cluster["size"] += 1
                    clustered.add(j)
            
            if cluster["size"] >= 3:
                clusters.append(cluster)
        
        return clusters
    
    def _hierarchical_clustering(self, space: Any) -> List[Dict[str, Any]]:
        """Hierarchical agglomerative clustering"""
        clusters = []
        concepts = list(space.concepts.keys())
        
        if len(concepts) < 3:
            return clusters
        
        # Build distance matrix
        n = len(concepts)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self._calculate_comprehensive_similarity(
                    space.concepts[concepts[i]], 
                    space.concepts[concepts[j]]
                )
                distance = 1 - similarity
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        # Simple agglomerative clustering
        active_clusters = [{i} for i in range(n)]
        cluster_distances = distance_matrix.copy()
        
        while len(active_clusters) > 1:
            # Find closest clusters
            min_dist = np.inf
            merge_i, merge_j = -1, -1
            
            for i in range(len(active_clusters)):
                for j in range(i + 1, len(active_clusters)):
                    # Average linkage
                    dist = np.mean([
                        cluster_distances[ci, cj]
                        for ci in active_clusters[i]
                        for cj in active_clusters[j]
                    ])
                    
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            if min_dist > 0.7:  # Stop if clusters too far apart
                break
            
            # Merge clusters
            new_cluster = active_clusters[merge_i].union(active_clusters[merge_j])
            active_clusters = [c for i, c in enumerate(active_clusters) 
                              if i not in [merge_i, merge_j]]
            active_clusters.append(new_cluster)
            
            # Record cluster if size appropriate
            if 3 <= len(new_cluster) <= len(concepts) * 0.5:
                clusters.append({
                    "members": [concepts[i] for i in new_cluster],
                    "size": len(new_cluster),
                    "type": "hierarchical",
                    "merge_distance": min_dist
                })
        
        return clusters
    
    def _advanced_cluster_merging(self, *cluster_lists, space: Any) -> List[Dict[str, Any]]:
        """Merge clusters from different algorithms intelligently"""
        all_clusters = []
        for cl in cluster_lists:
            all_clusters.extend(cl)
        
        if not all_clusters:
            return []
        
        # Build cluster similarity matrix
        n = len(all_clusters)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                members1 = set(all_clusters[i]["members"])
                members2 = set(all_clusters[j]["members"])
                
                # Jaccard similarity of members
                if members1 or members2:
                    similarity = len(members1.intersection(members2)) / len(members1.union(members2))
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        # Merge similar clusters
        merged = []
        used = set()
        
        for i in range(n):
            if i in used:
                continue
            
            # Find all similar clusters
            similar = [i]
            for j in range(n):
                if j != i and j not in used and similarity_matrix[i, j] > 0.5:
                    similar.append(j)
                    used.add(j)
            
            # Merge
            merged_cluster = {
                "members": list(set().union(*[set(all_clusters[idx]["members"]) for idx in similar])),
                "size": 0,
                "type": "merged",
                "source_types": list(set(all_clusters[idx]["type"] for idx in similar)),
                "merge_count": len(similar)
            }
            
            merged_cluster["size"] = len(merged_cluster["members"])
            
            if merged_cluster["size"] >= 3:
                merged.append(merged_cluster)
        
        return merged
    
    def _calculate_cluster_quality_metrics(self, cluster: Dict[str, Any], space: Any) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for a cluster"""
        metrics = {
            "cohesion": 0.0,
            "separation": 0.0,
            "stability": 0.0,
            "coverage": 0.0,
            "overall_quality": 0.0
        }
        
        members = cluster["members"]
        if len(members) < 2:
            return metrics
        
        # Cohesion: average pairwise similarity within cluster
        cohesion_scores = []
        for i, member1 in enumerate(members):
            for member2 in members[i+1:]:
                if member1 in space.concepts and member2 in space.concepts:
                    sim = self._calculate_comprehensive_similarity(
                        space.concepts[member1], space.concepts[member2]
                    )
                    cohesion_scores.append(sim)
        
        metrics["cohesion"] = np.mean(cohesion_scores) if cohesion_scores else 0
        
        # Separation: average distance to non-cluster members
        non_members = [c for c in space.concepts if c not in members]
        if non_members:
            separation_scores = []
            for member in members:
                for non_member in non_members[:10]:  # Sample for efficiency
                    if member in space.concepts and non_member in space.concepts:
                        sim = self._calculate_comprehensive_similarity(
                            space.concepts[member], space.concepts[non_member]
                        )
                        separation_scores.append(1 - sim)
            
            metrics["separation"] = np.mean(separation_scores) if separation_scores else 0
        
        # Stability: based on cluster type and size
        if cluster["type"] in ["semantic", "property_based"]:
            metrics["stability"] = 0.8
        elif cluster["type"] == "density":
            metrics["stability"] = 0.7
        else:
            metrics["stability"] = 0.6
        
        # Coverage
        metrics["coverage"] = len(members) / len(space.concepts) if space.concepts else 0
        
        # Overall quality
        metrics["overall_quality"] = np.mean([
            metrics["cohesion"],
            metrics["separation"],
            metrics["stability"],
            min(metrics["coverage"] * 5, 1.0)  # Normalize coverage
        ])
        
        return metrics
    
    def _identify_cluster_characteristics(self, cluster: Dict[str, Any], space: Any) -> Dict[str, Any]:
        """Identify key characteristics of a cluster"""
        chars = {
            "dominant_properties": [],
            "common_patterns": [],
            "cluster_role": "unknown",
            "key_concepts": []
        }
        
        members = cluster["members"]
        
        # Find dominant properties
        property_counts = defaultdict(int)
        for member in members:
            if member in space.concepts:
                for prop in space.concepts[member].get("properties", {}):
                    property_counts[prop] += 1
        
        # Properties in >70% of members
        threshold = len(members) * 0.7
        chars["dominant_properties"] = [
            prop for prop, count in property_counts.items() 
            if count >= threshold
        ]
        
        # Identify cluster role
        if cluster["type"] == "hierarchical":
            chars["cluster_role"] = "taxonomic_group"
        elif cluster["type"] == "semantic":
            chars["cluster_role"] = "semantic_category"
        elif cluster["type"] == "relational":
            chars["cluster_role"] = "functional_group"
        
        # Key concepts (most connected within cluster)
        connection_counts = defaultdict(int)
        for relation in space.relations:
            if relation["source"] in members and relation["target"] in members:
                connection_counts[relation["source"]] += 1
                connection_counts[relation["target"]] += 1
        
        # Top 3 most connected
        chars["key_concepts"] = [
            k for k, v in sorted(connection_counts.items(), 
                               key=lambda x: x[1], reverse=True)[:3]
        ]
        
        return chars
    
    def _find_cluster_exemplars(self, cluster: Dict[str, Any], space: Any) -> List[str]:
        """Find exemplar concepts that best represent the cluster"""
        members = cluster["members"]
        if len(members) <= 3:
            return members
        
        exemplars = []
        
        # Method 1: Centrality-based exemplar
        centrality_scores = {}
        for member in members:
            if member in space.concepts:
                # Calculate average similarity to other members
                similarities = []
                for other in members:
                    if other != member and other in space.concepts:
                        sim = self._calculate_comprehensive_similarity(
                            space.concepts[member], space.concepts[other]
                        )
                        similarities.append(sim)
                
                centrality_scores[member] = np.mean(similarities) if similarities else 0
        
        # Top 3 by centrality
        central_exemplars = sorted(centrality_scores.items(), 
                                  key=lambda x: x[1], reverse=True)[:3]
        exemplars.extend([ex[0] for ex in central_exemplars])
        
        # Method 2: Coverage-based exemplar
        # Find minimal set that covers all major properties
        if cluster.get("dominant_properties"):
            covered_props = set()
            coverage_exemplars = []
            
            for member in members:
                if member in space.concepts:
                    member_props = set(space.concepts[member].get("properties", {}).keys())
                    new_props = member_props - covered_props
                    
                    if new_props:
                        coverage_exemplars.append(member)
                        covered_props.update(member_props)
                        
                        if len(coverage_exemplars) >= 2:
                            break
            
            exemplars.extend(coverage_exemplars)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_exemplars = []
        for ex in exemplars:
            if ex not in seen:
                seen.add(ex)
                unique_exemplars.append(ex)
        
        return unique_exemplars[:3]
    
    def _detect_sub_clusters(self, cluster: Dict[str, Any], space: Any) -> List[Dict[str, Any]]:
        """Detect sub-clusters within a larger cluster"""
        members = cluster["members"]
        if len(members) < 6:  # Too small to have meaningful sub-clusters
            return []
        
        # Build similarity matrix for cluster members
        member_indices = {member: i for i, member in enumerate(members)}
        n = len(members)
        similarity_matrix = np.zeros((n, n))
        
        for i, member1 in enumerate(members):
            for j, member2 in enumerate(members[i+1:], i+1):
                if member1 in space.concepts and member2 in space.concepts:
                    sim = self._calculate_comprehensive_similarity(
                        space.concepts[member1], space.concepts[member2]
                    )
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        # Apply clustering to find sub-groups
        sub_clusters = []
        threshold = 0.8  # Higher threshold for sub-clusters
        clustered = set()
        
        for i in range(n):
            if i in clustered:
                continue
            
            sub_cluster = {
                "members": [members[i]],
                "parent_cluster": cluster.get("id", "unknown")
            }
            clustered.add(i)
            
            for j in range(n):
                if j not in clustered and similarity_matrix[i, j] >= threshold:
                    sub_cluster["members"].append(members[j])
                    clustered.add(j)
            
            if len(sub_cluster["members"]) >= 3:
                sub_cluster["size"] = len(sub_cluster["members"])
                sub_clusters.append(sub_cluster)
        
        return sub_clusters
    
    # ========================================================================================
    # PATH FINDING HELPER METHODS
    # ========================================================================================
    
    async def _similarity_heuristic(self, current: str, goal: str, concepts: Dict[str, Any]) -> float:
        """Heuristic function for A* search based on similarity"""
        if current == goal:
            return 0.0
        
        if current not in concepts or goal not in concepts:
            return 1.0
        
        similarity = await self._calculate_comprehensive_similarity(
            concepts[current], concepts[goal]
        )
        
        # Convert similarity to distance
        return 1.0 - similarity
    
    async def _get_similarity_neighbors(self, node: str, concepts: Dict[str, Any], 
                                      top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k most similar neighbors"""
        if node not in concepts:
            return []
        
        neighbors = []
        node_concept = concepts[node]
        
        for other_id, other_concept in concepts.items():
            if other_id != node:
                similarity = await self._calculate_comprehensive_similarity(
                    node_concept, other_concept
                )
                neighbors.append((other_id, similarity))
        
        # Sort by similarity and return top-k
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:top_k]
    
    async def _identify_concept_clusters(self, concepts: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Identify concept clusters for path finding"""
        # Simplified clustering for path finding
        clusters = {}
        cluster_id = 0
        
        # Use property-based clustering
        clustered = set()
        
        for concept_id, concept in concepts.items():
            if concept_id in clustered:
                continue
            
            # Start new cluster
            cluster = {
                "id": f"cluster_{cluster_id}",
                "members": [concept_id],
                "center": concept_id,
                "properties": set(concept.get("properties", {}).keys())
            }
            clustered.add(concept_id)
            
            # Find similar concepts
            for other_id, other_concept in concepts.items():
                if other_id in clustered:
                    continue
                
                other_props = set(other_concept.get("properties", {}).keys())
                if cluster["properties"] and other_props:
                    overlap = len(cluster["properties"].intersection(other_props))
                    union = len(cluster["properties"].union(other_props))
                    
                    if union > 0 and overlap / union > 0.6:
                        cluster["members"].append(other_id)
                        clustered.add(other_id)
            
            if len(cluster["members"]) >= 2:
                clusters[cluster["id"]] = cluster
                cluster_id += 1
        
        return clusters
    
    async def _find_intra_cluster_path(self, start: str, end: str, 
                                     cluster: Dict[str, Any], 
                                     concepts: Dict[str, Any]) -> Optional[List[str]]:
        """Find path within a cluster"""
        members = cluster["members"]
        
        if start not in members or end not in members:
            return None
        
        # For small clusters, try direct path
        if len(members) <= 5:
            return [start, end]
        
        # Find intermediate node with high similarity to both
        best_intermediate = None
        best_score = 0.0
        
        for member in members:
            if member not in [start, end]:
                score1 = await self._calculate_comprehensive_similarity(
                    concepts[start], concepts[member]
                )
                score2 = await self._calculate_comprehensive_similarity(
                    concepts[member], concepts[end]
                )
                
                combined_score = (score1 * score2) ** 0.5
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_intermediate = member
        
        if best_intermediate and best_score > 0.5:
            return [start, best_intermediate, end]
        
        return [start, end]
    
    async def _find_cluster_bridges(self, cluster1_id: str, cluster2_id: str, 
                                  clusters: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find bridge nodes between clusters"""
        bridges = []
        
        cluster1 = clusters.get(cluster1_id, {})
        cluster2 = clusters.get(cluster2_id, {})
        
        if not cluster1 or not cluster2:
            return bridges
        
        # Find nodes with properties from both clusters
        props1 = cluster1.get("properties", set())
        props2 = cluster2.get("properties", set())
        
        # Look for concepts that bridge the property gap
        common_props = props1.intersection(props2)
        unique_props1 = props1 - props2
        unique_props2 = props2 - props1
        
        # Create synthetic bridge description
        bridges.append({
            "type": "property_bridge",
            "common_properties": list(common_props),
            "bridges_from": list(unique_props1)[:3],
            "bridges_to": list(unique_props2)[:3]
        })
        
        return bridges
    
    async def _construct_cluster_bridge_path(self, start: str, end: str, 
                                           bridge: Dict[str, Any],
                                           start_cluster: str, end_cluster: str,
                                           clusters: Dict[str, Dict[str, Any]],
                                           concepts: Dict[str, Any]) -> Optional[List[str]]:
        """Construct path using cluster bridge"""
        # Find exit point from start cluster
        start_cluster_members = clusters[start_cluster]["members"]
        exit_point = start
        
        # Find entry point to end cluster
        end_cluster_members = clusters[end_cluster]["members"]
        entry_point = end
        
        # Look for better exit/entry points based on bridge properties
        if bridge["type"] == "property_bridge":
            # Find member of start cluster with most bridge properties
            best_exit_score = 0
            for member in start_cluster_members:
                if member in concepts:
                    member_props = set(concepts[member].get("properties", {}).keys())
                    score = len(member_props.intersection(set(bridge["common_properties"])))
                    
                    if score > best_exit_score:
                        best_exit_score = score
                        exit_point = member
            
            # Find member of end cluster with most bridge properties
            best_entry_score = 0
            for member in end_cluster_members:
                if member in concepts:
                    member_props = set(concepts[member].get("properties", {}).keys())
                    score = len(member_props.intersection(set(bridge["common_properties"])))
                    
                    if score > best_entry_score:
                        best_entry_score = score
                        entry_point = member
        
        # Construct path
        path = [start]
        if start != exit_point:
            path.append(exit_point)
        if exit_point != entry_point:
            path.append(entry_point)
        if entry_point != end:
            path.append(end)
        
        return path
    
    async def _find_concept_analogies(self, start: str, end: str, 
                                    concepts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find analogical relationships between concepts"""
        analogies = []
        
        if start not in concepts or end not in concepts:
            return analogies
        
        start_concept = concepts[start]
        end_concept = concepts[end]
        
        # Type 1: Proportional analogy (A:B :: C:D)
        # Find concepts with similar property relationships
        start_props = set(start_concept.get("properties", {}).keys())
        end_props = set(end_concept.get("properties", {}).keys())
        
        prop_diff = start_props.symmetric_difference(end_props)
        
        if prop_diff:
            # Look for other concept pairs with similar differences
            for concept1_id, concept1 in concepts.items():
                if concept1_id in [start, end]:
                    continue
                
                props1 = set(concept1.get("properties", {}).keys())
                
                for concept2_id, concept2 in concepts.items():
                    if concept2_id in [start, end, concept1_id]:
                        continue
                    
                    props2 = set(concept2.get("properties", {}).keys())
                    diff2 = props1.symmetric_difference(props2)
                    
                    if len(diff2) > 0 and len(prop_diff.intersection(diff2)) / len(prop_diff) > 0.6:
                        analogies.append({
                            "type": "proportional",
                            "intermediate1": concept1_id,
                            "intermediate2": concept2_id,
                            "similarity": len(prop_diff.intersection(diff2)) / len(prop_diff)
                        })
        
        # Type 2: Structural analogy
        # Find concepts that share structural patterns
        if start_props and end_props:
            pattern_similarity = len(start_props.intersection(end_props)) / len(start_props.union(end_props))
            
            if pattern_similarity > 0.3 and pattern_similarity < 0.7:
                # Find intermediate concepts that bridge the gap
                mapping_nodes = []
                
                for concept_id, concept in concepts.items():
                    if concept_id in [start, end]:
                        continue
                    
                    props = set(concept.get("properties", {}).keys())
                    
                    # Check if it shares properties with both
                    start_overlap = len(props.intersection(start_props))
                    end_overlap = len(props.intersection(end_props))
                    
                    if start_overlap > 0 and end_overlap > 0:
                        mapping_nodes.append(concept_id)
                
                if mapping_nodes:
                    analogies.append({
                        "type": "structural",
                        "mapping_nodes": mapping_nodes[:3],
                        "pattern_similarity": pattern_similarity
                    })
        
        # Type 3: Functional analogy
        # Based on property patterns suggesting similar functions
        functional_patterns = {
            "input_output": {"input", "output", "process"},
            "container": {"contains", "capacity", "items"},
            "controller": {"controls", "regulates", "monitors"}
        }
        
        for pattern_name, pattern_props in functional_patterns.items():
            start_match = len(start_props.intersection(pattern_props)) / len(pattern_props)
            end_match = len(end_props.intersection(pattern_props)) / len(pattern_props)
            
            if start_match > 0.5 and end_match > 0.5:
                # Find a concept that exemplifies this function
                for concept_id, concept in concepts.items():
                    if concept_id in [start, end]:
                        continue
                    
                    props = set(concept.get("properties", {}).keys())
                    match = len(props.intersection(pattern_props)) / len(pattern_props)
                    
                    if match > 0.8:
                        analogies.append({
                            "type": "functional",
                            "function_node": concept_id,
                            "function_type": pattern_name
                        })
                        break
        
        # Sort by quality
        analogies.sort(key=lambda x: x.get("similarity", 0.5), reverse=True)
        
        return analogies[:5]
    
    def _validate_analogy_path(self, path: List[str], concepts: Dict[str, Any]) -> bool:
        """Validate that an analogy-based path is reasonable"""
        if len(path) < 2:
            return False
        
        # Check that all nodes exist
        for node in path:
            if node not in concepts:
                return False
        
        # Check that consecutive nodes have some similarity
        for i in range(len(path) - 1):
            concept1 = concepts[path[i]]
            concept2 = concepts[path[i + 1]]
            
            # They should share at least some properties
            props1 = set(concept1.get("properties", {}).keys())
            props2 = set(concept2.get("properties", {}).keys())
            
            if props1 and props2 and not props1.intersection(props2):
                return False
        
        return True
    
    def _deduplicate_paths(self, paths: List[List[str]]) -> List[List[str]]:
        """Remove duplicate paths"""
        unique_paths = []
        seen = set()
        
        for path in paths:
            path_tuple = tuple(path)
            if path_tuple not in seen:
                seen.add(path_tuple)
                unique_paths.append(path)
        
        return unique_paths
    
    def _score_similarity_path(self, path: List[str], concepts: Dict[str, Any]) -> float:
        """Score a similarity path based on quality"""
        if len(path) < 2:
            return 0.0
        
        scores = []
        
        # Score based on consecutive similarities
        for i in range(len(path) - 1):
            if path[i] in concepts and path[i + 1] in concepts:
                sim = self._calculate_comprehensive_similarity(
                    concepts[path[i]], concepts[path[i + 1]]
                )
                scores.append(sim)
        
        if not scores:
            return 0.0
        
        # Penalize longer paths
        length_penalty = 1.0 / (1 + len(path) - 2)
        
        # Combine average similarity with length penalty
        return np.mean(scores) * length_penalty
    
    # ========================================================================================
    # PATTERN EXTRACTION AND ANALYSIS METHODS
    # ========================================================================================
    
    def _extract_goal_from_input(self, user_input: str) -> str:
        """Enhanced goal extraction with linguistic analysis"""
        # Remove question indicators
        question_words = ["how", "what", "why", "when", "where", "who", "which", 
                         "can", "could", "would", "should", "is", "are", "do", "does"]
        
        # Parse sentence structure
        sentences = self._split_into_sentences(user_input)
        
        goal_candidates = []
        
        for sentence in sentences:
            # Look for goal patterns
            goal_patterns = [
                # Infinitive patterns
                r"(?:want|need|aim|plan|intend|hope|wish|try|seek|strive)\s+to\s+(.+?)(?:\.|,|;|$)",
                # Purpose patterns
                r"(?:in order to|so that|to)\s+(.+?)(?:\.|,|;|$)",
                # Goal statement patterns
                r"(?:goal|objective|target|purpose|mission)\s+(?:is|:|-)?\s*(.+?)(?:\.|,|;|$)",
                # Imperative patterns (direct commands)
                r"^(?:please\s+)?([A-Z][a-z]+(?:\s+\w+)*?)(?:\.|!|$)",
                # Desire patterns
                r"(?:I'd like to|I want to|We need to|Let's)\s+(.+?)(?:\.|,|;|$)"
            ]
            
            for pattern in goal_patterns:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                if matches:
                    for match in matches:
                        # Clean up the match
                        cleaned = match.strip()
                        # Remove trailing punctuation
                        cleaned = re.sub(r'[.!?,;]+$', '', cleaned)
                        # Remove leading articles
                        cleaned = re.sub(r'^(the|a|an)\s+', '', cleaned, flags=re.IGNORECASE)
                        
                        if len(cleaned) > 5:  # Minimum meaningful length
                            goal_candidates.append({
                                "text": cleaned,
                                "confidence": 0.8 if pattern.startswith("(?:goal|objective)") else 0.6,
                                "source_pattern": pattern[:20] + "..."
                            })
            
            # If no explicit goal patterns, look for action verbs
            if not goal_candidates:
                action_verbs = ["improve", "increase", "decrease", "optimize", "create", "build",
                               "analyze", "develop", "implement", "achieve", "enhance", "reduce"]
                
                words = sentence.lower().split()
                for i, word in enumerate(words):
                    if word in action_verbs and i < len(words) - 1:
                        # Extract from action verb to end
                        goal_text = " ".join(words[i:])
                        goal_candidates.append({
                            "text": goal_text,
                            "confidence": 0.5,
                            "source_pattern": "action_verb"
                        })
                        break
        
        # Select best goal candidate
        if goal_candidates:
            # Sort by confidence and length (prefer more specific goals)
            goal_candidates.sort(key=lambda x: (x["confidence"], len(x["text"])), reverse=True)
            return goal_candidates[0]["text"]
        
        # Fallback: clean up input and return
        words = user_input.lower().split()
        filtered_words = [w for w in words if w not in question_words]
        return " ".join(filtered_words).strip()

    
    def _identify_source_domain(self, user_input: str) -> str:
        """Identify source domain from user input"""
        input_lower = user_input.lower()
        
        # Domain keywords
        domains = {
            "biology": ["organism", "cell", "evolution", "species", "ecosystem"],
            "physics": ["force", "energy", "motion", "wave", "particle"],
            "chemistry": ["reaction", "molecule", "compound", "element", "bond"],
            "psychology": ["mind", "behavior", "emotion", "cognition", "personality"],
            "economics": ["market", "supply", "demand", "price", "trade"],
            "technology": ["system", "algorithm", "data", "network", "software"],
            "sociology": ["society", "culture", "group", "community", "social"]
        }
        
        domain_scores = {}
        
        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in input_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return "general"
    
    def _extract_concrete_patterns(self, user_input: str, source_domain: str) -> List[Dict[str, Any]]:
        """Enhanced concrete pattern extraction using NLP techniques"""
        patterns = []
        
        # Preprocess text
        sentences = self._split_into_sentences(user_input)
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue
            
            # Use spaCy for entity and dependency parsing if available
            if nlp:
                try:
                    doc = nlp(sentence)
                    
                    # Extract entities
                    entities = [(ent.text, ent.label_) for ent in doc.ents]
                    
                    # Extract subject-verb-object patterns
                    svo_patterns = self._extract_svo_patterns(doc)
                    
                    # Extract noun phrases
                    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
                    
                    # Extract relationships from dependency parse
                    relationships = self._extract_dependency_relationships(doc)
                    
                    if entities or svo_patterns:
                        patterns.append({
                            "description": sentence.strip(),
                            "elements": entities,
                            "noun_phrases": noun_phrases,
                            "relationships": relationships,
                            "svo_patterns": svo_patterns,
                            "level": "concrete",
                            "domain": source_domain,
                            "confidence": 0.8 if entities else 0.6
                        })
                except:
                    # Fallback to rule-based extraction
                    patterns.append(self._extract_pattern_rule_based(sentence, source_domain))
            else:
                # Pure rule-based extraction
                patterns.append(self._extract_pattern_rule_based(sentence, source_domain))
        
        return [p for p in patterns if p is not None]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved handling"""
        # Handle common abbreviations
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr)\.\s*', r'\1<DOT> ', text)
        text = re.sub(r'\b(Inc|Ltd|Corp|Co)\.\s*', r'\1<DOT> ', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore dots
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        return sentences
    
    def _extract_svo_patterns(self, doc) -> List[Dict[str, str]]:
        """Extract subject-verb-object patterns from spaCy doc"""
        patterns = []
        
        for token in doc:
            if token.pos_ == "VERB":
                subject = None
                obj = None
                
                # Find subject
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = child.text
                    elif child.dep_ in ["dobj", "pobj"]:
                        obj = child.text
                
                if subject:
                    patterns.append({
                        "subject": subject,
                        "verb": token.text,
                        "object": obj or "?"
                    })
        
        return patterns
    
    def _extract_dependency_relationships(self, doc) -> List[Dict[str, Any]]:
        """Extract relationships from dependency parse"""
        relationships = []
        
        for token in doc:
            if token.dep_ != "ROOT" and token.head.text != token.text:
                relationships.append({
                    "from": token.head.text,
                    "to": token.text,
                    "relation": token.dep_,
                    "from_pos": token.head.pos_,
                    "to_pos": token.pos_
                })
        
        return relationships
    
    def _extract_pattern_rule_based(self, sentence: str, source_domain: str) -> Optional[Dict[str, Any]]:
        """Rule-based pattern extraction as fallback"""
        # Simple POS tagging patterns
        noun_pattern = r'\b[A-Z][a-z]+\b'
        verb_pattern = r'\b(?:is|are|was|were|has|have|had|will|would|can|could|should|must|may|might|' \
                       r'do|does|did|make|makes|made|take|takes|took|get|gets|got|' \
                       r'give|gives|gave|find|finds|found|think|thinks|thought|' \
                       r'tell|tells|told|become|becomes|became|leave|leaves|left|' \
                       r'feel|feels|felt|bring|brings|brought|begin|begins|began|' \
                       r'keep|keeps|kept|hold|holds|held|write|writes|wrote|' \
                       r'provide|provides|provided|sit|sits|sat|stand|stands|stood|' \
                       r'lose|loses|lost|pay|pays|paid|meet|meets|met|' \
                       r'ing|ed|es|s)\b'
        
        nouns = re.findall(noun_pattern, sentence)
        verbs = re.findall(verb_pattern, sentence, re.IGNORECASE)
        
        if nouns or verbs:
            return {
                "description": sentence.strip(),
                "elements": nouns,
                "relationships": verbs,
                "level": "concrete",
                "domain": source_domain,
                "confidence": 0.5
            }
        
        return None
    
    def _extract_abstract_patterns(self, user_input: str, source_domain: str) -> List[Dict[str, Any]]:
        """Extract abstract patterns from input"""
        patterns = []
        
        # Abstract pattern indicators
        abstract_indicators = {
            "causation": ["causes", "leads to", "results in", "produces", "affects"],
            "correlation": ["relates to", "associated with", "linked to", "connected to"],
            "transformation": ["becomes", "transforms", "changes into", "evolves"],
            "hierarchy": ["consists of", "contains", "includes", "part of", "type of"],
            "flow": ["flows", "moves", "transfers", "passes", "propagates"]
        }
        
        input_lower = user_input.lower()
        
        for pattern_type, indicators in abstract_indicators.items():
            for indicator in indicators:
                if indicator in input_lower:
                    # Extract context around indicator
                    index = input_lower.find(indicator)
                    start = max(0, index - 50)
                    end = min(len(input_lower), index + 50)
                    context = user_input[start:end]
                    
                    patterns.append({
                        "description": context,
                        "pattern_type": pattern_type,
                        "indicator": indicator,
                        "level": "abstract",
                        "domain": source_domain,
                        "elements": self._extract_pattern_elements(context, pattern_type)
                    })
        
        return patterns
    
    def _extract_meta_patterns(self, user_input: str, source_domain: str) -> List[Dict[str, Any]]:
        """Extract meta-level patterns from input"""
        patterns = []
        
        # Meta-pattern types
        meta_patterns = {
            "feedback_loop": ["feedback", "reinforces", "amplifies", "dampens", "stabilizes"],
            "emergence": ["emerges", "arises from", "self-organizes", "spontaneous"],
            "optimization": ["optimizes", "maximizes", "minimizes", "balances", "efficient"],
            "adaptation": ["adapts", "evolves", "learns", "adjusts", "responds"],
            "network_effect": ["network", "connections", "interactions", "spreads", "propagates"]
        }
        
        input_lower = user_input.lower()
        
        for pattern_name, keywords in meta_patterns.items():
            if any(keyword in input_lower for keyword in keywords):
                patterns.append({
                    "description": f"Meta-pattern: {pattern_name}",
                    "pattern_type": pattern_name,
                    "level": "meta",
                    "domain": source_domain,
                    "characteristics": self._get_meta_pattern_characteristics(pattern_name),
                    "elements": []
                })
        
        return patterns
    
    def _extract_pattern_elements(self, context: str, pattern_type: str) -> List[str]:
        """Extract elements involved in a pattern"""
        elements = []
        
        # Pattern-specific extraction rules
        if pattern_type == "causation":
            # Look for cause and effect
            parts = context.split("causes")
            if len(parts) == 2:
                elements.extend(["cause: " + parts[0].strip(), "effect: " + parts[1].strip()])
        
        elif pattern_type == "hierarchy":
            # Look for parent and children
            if "contains" in context:
                parts = context.split("contains")
                if len(parts) == 2:
                    elements.extend(["parent: " + parts[0].strip(), "children: " + parts[1].strip()])
        
        # Default: extract nouns
        if not elements:
            words = context.split()
            for word in words:
                if word[0].isupper() and len(word) > 2:
                    elements.append(word)
        
        return elements[:5]  # Limit to 5 elements
    
    def _get_meta_pattern_characteristics(self, pattern_name: str) -> Dict[str, Any]:
        """Get characteristics of meta-patterns"""
        characteristics = {
            "feedback_loop": {
                "dynamics": "circular_causality",
                "stability": "variable",
                "time_scale": "continuous"
            },
            "emergence": {
                "dynamics": "bottom_up",
                "predictability": "low",
                "complexity": "high"
            },
            "optimization": {
                "dynamics": "goal_seeking",
                "constraints": "present",
                "trade_offs": "likely"
            },
            "adaptation": {
                "dynamics": "responsive",
                "learning": "present",
                "flexibility": "high"
            },
            "network_effect": {
                "dynamics": "multiplicative",
                "scaling": "non_linear",
                "connectivity": "critical"
            }
        }
        
        return characteristics.get(pattern_name, {})
    
    def _analyze_pattern_structure(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the structure of a pattern"""
        structure = {
            "complexity": "simple",
            "components": len(pattern.get("elements", [])),
            "relationships": len(pattern.get("relationships", [])),
            "abstraction_level": pattern.get("level", "unknown"),
            "pattern_type": pattern.get("pattern_type", "generic")
        }
        
        # Assess complexity
        total_components = structure["components"] + structure["relationships"]
        if total_components > 5:
            structure["complexity"] = "complex"
        elif total_components > 2:
            structure["complexity"] = "moderate"
        
        # Identify structure type
        if pattern.get("pattern_type") == "hierarchy":
            structure["structure_type"] = "tree"
        elif pattern.get("pattern_type") == "flow":
            structure["structure_type"] = "directed_graph"
        elif pattern.get("pattern_type") == "network_effect":
            structure["structure_type"] = "network"
        else:
            structure["structure_type"] = "generic"
        
        return structure
    
    def _assess_pattern_transferability(self, pattern: Dict[str, Any]) -> float:
        """Assess how transferable a pattern is to other domains"""
        transferability = 0.5  # Base score
        
        # More abstract patterns are more transferable
        if pattern.get("level") == "meta":
            transferability += 0.3
        elif pattern.get("level") == "abstract":
            transferability += 0.2
        
        # Generic patterns are more transferable
        pattern_type = pattern.get("pattern_type", "")
        universal_patterns = ["causation", "hierarchy", "flow", "feedback_loop", "optimization"]
        
        if pattern_type in universal_patterns:
            transferability += 0.2
        
        # Patterns with fewer domain-specific elements are more transferable
        elements = pattern.get("elements", [])
        domain = pattern.get("domain", "")
        
        domain_specific_count = sum(1 for elem in elements if domain in str(elem).lower())
        if domain_specific_count == 0:
            transferability += 0.1
        elif domain_specific_count > len(elements) / 2:
            transferability -= 0.2
        
        return min(1.0, max(0.0, transferability))
    
    def _extract_pattern_insights(self, pattern: Dict[str, Any]) -> List[str]:
        """Extract key insights from a pattern"""
        insights = []
        
        pattern_type = pattern.get("pattern_type", "")
        level = pattern.get("level", "")
        
        # Level-specific insights
        if level == "meta":
            insights.append(f"This represents a {pattern_type} meta-pattern that appears across many systems")
        elif level == "abstract":
            insights.append(f"Abstract {pattern_type} relationship that can be instantiated in various ways")
        
        # Pattern-specific insights
        if pattern_type == "causation":
            insights.append("Causal relationship suggests intervention opportunities")
        elif pattern_type == "feedback_loop":
            insights.append("Feedback dynamics can lead to amplification or stabilization")
        elif pattern_type == "hierarchy":
            insights.append("Hierarchical structure enables decomposition and aggregation")
        elif pattern_type == "emergence":
            insights.append("Emergent properties arise from component interactions")
        
        # Transferability insight
        transferability = self._assess_pattern_transferability(pattern)
        if transferability > 0.7:
            insights.append("Highly transferable pattern applicable across domains")
        elif transferability < 0.3:
            insights.append("Domain-specific pattern with limited transferability")
        
        return insights
    
    # ========================================================================================
    # TARGET MAPPING METHODS
    # ========================================================================================
    
    def _identify_target_domain(self, context: Dict[str, Any], source_domain: str) -> str:
        """Identify target domain for analogical reasoning"""
        user_input = context.get("user_input", "").lower()
        
        # Look for explicit domain transitions
        transition_phrases = ["like in", "similar to", "as in", "compared to", "analogous to"]
        
        for phrase in transition_phrases:
            if phrase in user_input:
                index = user_input.find(phrase)
                after_phrase = user_input[index + len(phrase):].strip()
                
                # Extract domain from what follows
                words = after_phrase.split()[:3]  # Look at first few words
                potential_domain = " ".join(words)
                
                # Check against known domains
                domains = ["biology", "physics", "chemistry", "psychology", "economics", 
                          "technology", "sociology", "business", "nature", "engineering"]
                
                for domain in domains:
                    if domain in potential_domain:
                        return domain
        
        # If no explicit target, choose complementary domain
        domain_complements = {
            "biology": "technology",
            "technology": "biology",
            "physics": "economics",
            "economics": "physics",
            "psychology": "sociology",
            "sociology": "psychology"
        }
        
        return domain_complements.get(source_domain, "general")
    
    def _extract_target_elements(self, context: Dict[str, Any], target_domain: str) -> List[Dict[str, Any]]:
        """Extract elements from target domain"""
        elements = []
        
        # Domain-specific element templates
        domain_elements = {
            "technology": ["system", "component", "interface", "data", "process", "user", "network"],
            "biology": ["organism", "cell", "gene", "protein", "ecosystem", "species", "function"],
            "economics": ["market", "agent", "resource", "price", "supply", "demand", "utility"],
            "physics": ["particle", "force", "energy", "field", "wave", "mass", "motion"],
            "psychology": ["mind", "behavior", "emotion", "memory", "perception", "learning", "motivation"],
            "business": ["company", "product", "customer", "revenue", "strategy", "competition", "innovation"]
        }
        
        # Get relevant elements for target domain
        relevant_elements = domain_elements.get(target_domain, ["entity", "property", "relation", "process"])
        
        # Create element structures
        for i, elem_type in enumerate(relevant_elements[:5]):
            elements.append({
                "id": f"{target_domain}_{elem_type}_{i}",
                "type": elem_type,
                "domain": target_domain,
                "properties": self._get_element_properties(elem_type, target_domain)
            })
        
        return elements
    
    def _get_element_properties(self, elem_type: str, domain: str) -> Dict[str, Any]:
        """Get typical properties for an element type in a domain"""
        # Simplified property assignment
        properties = {
            "domain": domain,
            "type": elem_type
        }
        
        # Add domain-specific properties
        if domain == "technology" and elem_type == "system":
            properties.update({
                "complexity": "high",
                "components": "multiple",
                "interfaces": "defined"
            })
        elif domain == "biology" and elem_type == "organism":
            properties.update({
                "living": True,
                "reproduces": True,
                "evolves": True
            })
        
        return properties
    
    def _calculate_mapping_score(self, source_elem: Any, target_elem: Dict[str, Any],
                               source_pattern: Dict[str, Any], target_domain: str) -> float:
        """Calculate mapping score between source and target elements"""
        score = 0.0
        
        # Type compatibility
        if isinstance(source_elem, dict):
            source_type = source_elem.get("type", "")
            target_type = target_elem.get("type", "")
            
            # Check for functional similarity
            type_mappings = {
                ("cause", "input"): 0.8,
                ("effect", "output"): 0.8,
                ("parent", "system"): 0.7,
                ("children", "component"): 0.7,
                ("flow", "process"): 0.8
            }
            
            for (s_type, t_type), mapping_score in type_mappings.items():
                if s_type in source_type and t_type in target_type:
                    score += mapping_score
        
        # Structural similarity
        if source_pattern.get("pattern_type") == "hierarchy" and "system" in target_elem.get("type", ""):
            score += 0.3
        elif source_pattern.get("pattern_type") == "flow" and "process" in target_elem.get("type", ""):
            score += 0.3
        
        # Abstract property matching
        if source_pattern.get("level") in ["abstract", "meta"]:
            score += 0.2  # Abstract patterns map more easily
        
        return min(1.0, score)
    
    def _determine_mapping_type(self, source_elem: Any, target_elem: Dict[str, Any]) -> str:
        """Determine the type of mapping between elements"""
        if isinstance(source_elem, str):
            # Simple string mapping
            if "cause" in source_elem and "input" in target_elem.get("type", ""):
                return "causal_to_functional"
            elif "effect" in source_elem and "output" in target_elem.get("type", ""):
                return "result_mapping"
        
        # Default mapping types
        source_type = source_elem.get("type", "") if isinstance(source_elem, dict) else str(source_elem)
        target_type = target_elem.get("type", "")
        
        if source_type == target_type:
            return "direct_correspondence"
        elif "parent" in source_type and "system" in target_type:
            return "hierarchical_mapping"
        else:
            return "analogical_mapping"
    
    def _calculate_mapping_confidence(self, mapping_score: float, 
                                    source_elem: Any, target_elem: Dict[str, Any]) -> float:
        """Calculate confidence in a mapping"""
        confidence = mapping_score  # Start with mapping score
        
        # Boost confidence for clear mappings
        if mapping_score > 0.8:
            confidence *= 1.1
        
        # Reduce confidence for ambiguous mappings
        if isinstance(source_elem, str) and len(source_elem) < 10:
            confidence *= 0.9  # Short descriptions are less reliable
        
        # Domain expertise factor (simplified)
        confidence *= 0.85  # Assume moderate domain expertise
        
        return min(1.0, confidence)
    
    def _map_relationships(self, source_relationships: List[Any], 
                         target_elements: List[Dict[str, Any]], 
                         element_mappings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map relationships from source to target"""
        relationship_mappings = []
        
        # Create element mapping lookup
        mapping_lookup = {}
        for mapping in element_mappings:
            source = mapping.get("source_element")
            target = mapping.get("target_element")
            if isinstance(source, dict):
                source_key = source.get("id", str(source))
            else:
                source_key = str(source)
            mapping_lookup[source_key] = target
        
        # Map each relationship
        for rel in source_relationships[:5]:  # Limit for efficiency
            if isinstance(rel, str):
                # Simple relationship
                rel_mapping = {
                    "source_relationship": rel,
                    "mapped_relationship": self._map_relationship_type(rel),
                    "confidence": 0.7
                }
                relationship_mappings.append(rel_mapping)
            elif isinstance(rel, dict):
                # Complex relationship
                source_node = rel.get("source")
                target_node = rel.get("target")
                rel_type = rel.get("type", "relates_to")
                
                # Try to find mapped elements
                mapped_source = mapping_lookup.get(source_node)
                mapped_target = mapping_lookup.get(target_node)
                
                if mapped_source and mapped_target:
                    rel_mapping = {
                        "source_relationship": rel,
                        "mapped_relationship": {
                            "source": mapped_source,
                            "target": mapped_target,
                            "type": self._map_relationship_type(rel_type)
                        },
                        "confidence": 0.8
                    }
                    relationship_mappings.append(rel_mapping)
        
        return relationship_mappings
    
    def _map_relationship_type(self, rel_type: str) -> str:
        """Map relationship type to target domain"""
        # Relationship type mappings
        mappings = {
            "causes": "triggers",
            "leads_to": "results_in",
            "contains": "includes",
            "part_of": "component_of",
            "flows_to": "transfers_to",
            "depends_on": "requires",
            "transforms": "processes"
        }
        
        return mappings.get(rel_type, rel_type)
    
    def _validate_mapping_consistency(self, element_mappings: List[Dict[str, Any]], 
                                    relationship_mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate consistency of mappings"""
        validation = {
            "is_valid": True,
            "consistency_score": 1.0,
            "issues": []
        }
        
        # Check element mapping coverage
        if len(element_mappings) < 2:
            validation["issues"].append("Insufficient element mappings")
            validation["consistency_score"] *= 0.7
        
        # Check relationship preservation
        if len(relationship_mappings) == 0 and len(element_mappings) > 2:
            validation["issues"].append("No relationships mapped despite multiple elements")
            validation["consistency_score"] *= 0.8
        
        # Check mapping confidence distribution
        confidences = [m.get("confidence", 0.5) for m in element_mappings]
        if confidences:
            avg_confidence = np.mean(confidences)
            if avg_confidence < 0.5:
                validation["issues"].append("Low average mapping confidence")
                validation["consistency_score"] *= 0.6
                validation["is_valid"] = False
        
        # Check for orphaned elements
        mapped_elements = set()
        for rel in relationship_mappings:
            if isinstance(rel.get("mapped_relationship"), dict):
                mapped_elements.add(rel["mapped_relationship"].get("source"))
                mapped_elements.add(rel["mapped_relationship"].get("target"))
        
        orphaned_count = len(element_mappings) - len(mapped_elements)
        if orphaned_count > len(element_mappings) / 2:
            validation["issues"].append("Many mapped elements not used in relationships")
            validation["consistency_score"] *= 0.85
        
        return validation
    
    def _calculate_overall_mapping_quality(self, element_mappings: List[Dict[str, Any]], 
                                         relationship_mappings: List[Dict[str, Any]]) -> float:
        """Calculate overall quality of mapping"""
        if not element_mappings:
            return 0.0
        
        # Element mapping quality
        element_scores = [m.get("similarity", 0) * m.get("confidence", 1) 
                         for m in element_mappings]
        element_quality = np.mean(element_scores) if element_scores else 0
        
        # Relationship mapping quality
        rel_scores = [m.get("confidence", 0.5) for m in relationship_mappings]
        rel_quality = np.mean(rel_scores) if rel_scores else 0
        
        # Coverage quality
        coverage = len(element_mappings) / 10  # Normalize to expected ~10 elements
        coverage_quality = min(1.0, coverage)
        
        # Combine with weights
        quality = (0.4 * element_quality + 
                   0.3 * rel_quality + 
                   0.3 * coverage_quality)
        
        return quality
    
    def _generate_mapping_insights(self, source_pattern: Dict[str, Any], 
                                 mappings: List[Dict[str, Any]], 
                                 target_domain: str) -> List[str]:
        """Generate insights from the mapping"""
        insights = []
        
        # Pattern transfer insight
        pattern_type = source_pattern.get("pattern_type", "generic")
        insights.append(f"The {pattern_type} pattern from {source_pattern.get('domain', 'source')} "
                       f"can be applied to {target_domain} through analogical mapping")
        
        # Mapping strength insight
        if len(mappings) > 5:
            insights.append("Strong mapping with multiple correspondence points identified")
        elif len(mappings) > 2:
            insights.append("Moderate mapping with key elements successfully transferred")
        else:
            insights.append("Weak mapping - consider alternative source patterns")
        
        # Application suggestions
        if pattern_type == "causation":
            insights.append(f"Causal relationships in {target_domain} can be analyzed using this mapping")
        elif pattern_type == "hierarchy":
            insights.append(f"Hierarchical structure suggests decomposition opportunities in {target_domain}")
        elif pattern_type == "flow":
            insights.append(f"Flow patterns indicate process optimization potential in {target_domain}")
        
        return insights
    
    # ========================================================================================
    # CUSTOM ACTION EXECUTION METHODS
    # ========================================================================================
    
    async def _execute_custom_generation(self, action: str, params: Dict, 
                                       context: Dict, results: Dict) -> Dict[str, Any]:
        """Execute custom generation actions"""
        generation_type = action.split("_", 2)[2] if len(action.split("_")) > 2 else "general"
        
        result = {
            "action": action,
            "status": "completed",
            "outputs": {},
            "metadata": {"generation_type": generation_type}
        }
        
        if generation_type == "hypothesis":
            # Generate hypotheses
            hypotheses = self._generate_hypotheses(params, context, results)
            result["outputs"] = {
                "hypotheses": hypotheses,
                "count": len(hypotheses)
            }
        
        elif generation_type == "solution":
            # Generate solutions
            solutions = self._generate_solutions(params, context, results)
            result["outputs"] = {
                "solutions": solutions,
                "count": len(solutions)
            }
        
        elif generation_type == "question":
            # Generate questions
            questions = self._generate_questions(params, context, results)
            result["outputs"] = {
                "questions": questions,
                "count": len(questions)
            }
        
        return result
    
    async def _execute_custom_evaluation(self, action: str, params: Dict, 
                                       context: Dict, results: Dict) -> Dict[str, Any]:
        """Execute custom evaluation actions"""
        evaluation_type = action.split("_", 2)[2] if len(action.split("_")) > 2 else "general"
        
        result = {
            "action": action,
            "status": "completed",
            "outputs": {},
            "metadata": {"evaluation_type": evaluation_type}
        }
        
        target = params.get("target", results)
        
        if evaluation_type == "feasibility":
            feasibility = self._evaluate_feasibility(target, context)
            result["outputs"] = feasibility
        
        elif evaluation_type == "quality":
            quality = self._evaluate_quality(target, context)
            result["outputs"] = quality
        
        elif evaluation_type == "impact":
            impact = self._evaluate_impact(target, context)
            result["outputs"] = impact
        
        return result
    
    async def _execute_custom_transformation(self, action: str, params: Dict, 
                                           context: Dict, results: Dict) -> Dict[str, Any]:
        """Execute custom transformation actions"""
        transformation_type = action.split("_", 2)[2] if len(action.split("_")) > 2 else "general"
        
        result = {
            "action": action,
            "status": "completed",
            "outputs": {},
            "metadata": {"transformation_type": transformation_type}
        }
        
        input_data = params.get("input", results)
        
        if transformation_type == "abstract":
            # Transform to abstract representation
            abstracted = self._transform_to_abstract(input_data)
            result["outputs"] = {"abstracted": abstracted}
        
        elif transformation_type == "concrete":
            # Transform to concrete representation
            concrete = self._transform_to_concrete(input_data)
            result["outputs"] = {"concrete": concrete}
        
        elif transformation_type == "structured":
            # Transform to structured format
            structured = self._transform_to_structured(input_data)
            result["outputs"] = {"structured": structured}
        
        return result
    
    async def _execute_generic_custom(self, action: str, params: Dict, 
                                    context: Dict, results: Dict) -> Dict[str, Any]:
        """Execute generic custom actions"""
        return {
            "action": action,
            "status": "completed",
            "outputs": {
                "message": f"Executed custom action: {action}",
                "params_received": list(params.keys()),
                "context_available": bool(context),
                "previous_results": bool(results)
            },
            "metadata": {"action_type": "generic_custom"}
        }
    
    # ========================================================================================
    # ANALYSIS HELPER METHODS
    # ========================================================================================
    
    def _analyze_sentiment_detailed(self, text: str) -> Dict[str, Any]:
        """Detailed sentiment analysis"""
        sentiment = {
            "polarity": "neutral",
            "score": 0.0,
            "confidence": 0.5,
            "emotions": {},
            "aspects": []
        }
        
        text_lower = text.lower()
        
        # Sentiment indicators
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "positive", 
                         "success", "achieve", "improve", "benefit", "opportunity"]
        negative_words = ["bad", "poor", "terrible", "awful", "negative", "fail", 
                         "problem", "issue", "difficult", "challenge", "risk"]
        
        # Count sentiment words
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate polarity
        if positive_count > negative_count:
            sentiment["polarity"] = "positive"
            sentiment["score"] = min(1.0, positive_count / 10)
        elif negative_count > positive_count:
            sentiment["polarity"] = "negative"
            sentiment["score"] = -min(1.0, negative_count / 10)
        
        # Emotion detection
        emotion_indicators = {
            "joy": ["happy", "joy", "excited", "delighted"],
            "anger": ["angry", "frustrated", "annoyed", "furious"],
            "fear": ["afraid", "scared", "worried", "anxious"],
            "sadness": ["sad", "depressed", "disappointed", "unhappy"],
            "surprise": ["surprised", "amazed", "astonished", "unexpected"]
        }
        
        for emotion, indicators in emotion_indicators.items():
            count = sum(1 for ind in indicators if ind in text_lower)
            if count > 0:
                sentiment["emotions"][emotion] = min(1.0, count / 3)
        
        # Confidence based on clarity of sentiment
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words > 5:
            sentiment["confidence"] = 0.9
        elif total_sentiment_words > 2:
            sentiment["confidence"] = 0.7
        else:
            sentiment["confidence"] = 0.4
        
        return sentiment
    
    def _analyze_complexity_detailed(self, target: str) -> Dict[str, Any]:
        """Detailed complexity analysis"""
        complexity = {
            "score": 0.3,  # Base complexity
            "factors": [],
            "level": "low"
        }
        
        # Length factor
        word_count = len(target.split())
        if word_count > 50:
            complexity["score"] += 0.2
            complexity["factors"].append("high_word_count")
        elif word_count > 20:
            complexity["score"] += 0.1
            complexity["factors"].append("moderate_word_count")
        
        # Structural complexity
        if any(conj in target.lower() for conj in ["and", "or", "but", "while", "although"]):
            complexity["score"] += 0.1
            complexity["factors"].append("multiple_clauses")
        
        # Technical terms
        technical_indicators = ["algorithm", "system", "process", "mechanism", "framework",
                              "architecture", "protocol", "interface", "implementation"]
        tech_count = sum(1 for term in technical_indicators if term in target.lower())
        if tech_count > 2:
            complexity["score"] += 0.2
            complexity["factors"].append("technical_content")
        
        # Nested structures
        if target.count("(") > 1 or target.count("[") > 1:
            complexity["score"] += 0.1
            complexity["factors"].append("nested_structures")
        
        # Determine level
        if complexity["score"] > 0.7:
            complexity["level"] = "high"
        elif complexity["score"] > 0.4:
            complexity["level"] = "medium"
        else:
            complexity["level"] = "low"
        
        return complexity
    
    def _analyze_opportunities(self, situation: Dict[str, Any], 
                                      results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhanced opportunity analysis with sophisticated pattern detection"""
        opportunities = []
        
        # Analyze situation comprehensively
        situation_analysis = self._analyze_situation_deeply(situation, results)
        
        # Detect opportunity patterns
        opportunity_patterns = [
            {
                "pattern": "unmet_need",
                "indicators": ["gap", "missing", "lack", "need", "require", "demand"],
                "opportunity_type": "solution_development",
                "template": "Develop solution to address {need}"
            },
            {
                "pattern": "inefficiency",
                "indicators": ["slow", "expensive", "complex", "difficult", "inefficient", "waste"],
                "opportunity_type": "optimization",
                "template": "Optimize {process} to improve efficiency"
            },
            {
                "pattern": "emerging_trend",
                "indicators": ["growing", "increasing", "trending", "emerging", "new", "novel"],
                "opportunity_type": "early_adoption",
                "template": "Capitalize on emerging trend in {area}"
            },
            {
                "pattern": "underutilized_resource",
                "indicators": ["unused", "available", "excess", "spare", "underutilized"],
                "opportunity_type": "resource_optimization",
                "template": "Better utilize {resource}"
            },
            {
                "pattern": "synergy_potential",
                "indicators": ["combine", "integrate", "merge", "synergy", "complement"],
                "opportunity_type": "integration",
                "template": "Create synergy by combining {elements}"
            },
            {
                "pattern": "automation_potential",
                "indicators": ["manual", "repetitive", "routine", "standardized", "predictable"],
                "opportunity_type": "automation",
                "template": "Automate {process} to save time and resources"
            }
        ]
        
        # Check each pattern
        situation_text = json.dumps(situation).lower()
        
        for pattern_def in opportunity_patterns:
            # Calculate pattern match score
            match_score = 0.0
            matched_indicators = []
            
            for indicator in pattern_def["indicators"]:
                if indicator in situation_text:
                    match_score += 1.0 / len(pattern_def["indicators"])
                    matched_indicators.append(indicator)
            
            if match_score > 0.3 or len(opportunities) < 3:  # Ensure minimum opportunities
                # Extract specific details
                opportunity_details = self._extract_opportunity_details(
                    situation, pattern_def, matched_indicators, situation_analysis
                )
                
                opportunity = {
                    "type": pattern_def["opportunity_type"],
                    "pattern": pattern_def["pattern"],
                    "description": opportunity_details["description"],
                    "specific_action": opportunity_details["action"],
                    "confidence": min(1.0, match_score + 0.3),
                    "potential_impact": opportunity_details["impact"],
                    "implementation_difficulty": opportunity_details["difficulty"],
                    "time_to_value": opportunity_details["time_to_value"],
                    "resources_required": opportunity_details["resources"],
                    "success_factors": opportunity_details["success_factors"],
                    "risks": opportunity_details["risks"]
                }
                
                opportunities.append(opportunity)
        
        # Analyze cross-cutting opportunities
        if len(opportunities) > 1:
            meta_opportunities = self._identify_meta_opportunities(opportunities, situation_analysis)
            opportunities.extend(meta_opportunities)
        
        # Rank opportunities
        opportunities = self._rank_opportunities(opportunities)
        
        return opportunities[:5]  # Top 5 opportunities
    
    def _analyze_situation_deeply(self, situation: Dict[str, Any], 
                                results: Dict[str, Any]) -> Dict[str, Any]:
        """Deep analysis of situation to support opportunity identification"""
        analysis = {
            "key_entities": [],
            "relationships": [],
            "constraints": [],
            "resources": [],
            "trends": [],
            "pain_points": []
        }
        
        # Extract from situation
        if isinstance(situation, dict):
            # Look for common keys
            for key in ["entities", "actors", "components", "elements"]:
                if key in situation:
                    analysis["key_entities"].extend(situation[key])
            
            for key in ["constraints", "limitations", "barriers"]:
                if key in situation:
                    analysis["constraints"].extend(situation[key])
            
            for key in ["resources", "assets", "capabilities"]:
                if key in situation:
                    analysis["resources"].extend(situation[key])
        
        # Extract from results
        if isinstance(results, dict):
            # Look for patterns in results
            if "causal_relations" in results:
                analysis["relationships"].extend(results["causal_relations"])
            
            if "trends" in results or "patterns" in results:
                analysis["trends"].extend(results.get("trends", []))
                analysis["trends"].extend(results.get("patterns", []))
        
        # Identify pain points through keyword analysis
        situation_text = str(situation).lower()
        pain_indicators = ["problem", "issue", "challenge", "difficult", "struggle", 
                          "pain", "friction", "bottleneck", "obstacle"]
        
        for indicator in pain_indicators:
            if indicator in situation_text:
                analysis["pain_points"].append(indicator)
        
        return analysis
    
    def _extract_opportunity_details(self, situation: Dict[str, Any], pattern_def: Dict[str, Any],
                                   matched_indicators: List[str], 
                                   situation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract specific details for an opportunity"""
        details = {
            "description": "",
            "action": "",
            "impact": "moderate",
            "difficulty": "medium",
            "time_to_value": "medium-term",
            "resources": [],
            "success_factors": [],
            "risks": []
        }
        
        # Generate specific description
        template = pattern_def["template"]
        
        # Try to fill in template with actual values
        if "{need}" in template and situation_analysis["pain_points"]:
            details["description"] = template.format(need=situation_analysis["pain_points"][0])
        elif "{process}" in template:
            # Look for process-related terms
            processes = self._extract_processes(situation)
            if processes:
                details["description"] = template.format(process=processes[0])
        elif "{area}" in template and situation_analysis["key_entities"]:
            details["description"] = template.format(area=situation_analysis["key_entities"][0])
        elif "{resource}" in template and situation_analysis["resources"]:
            details["description"] = template.format(resource=situation_analysis["resources"][0])
        elif "{elements}" in template and len(situation_analysis["key_entities"]) >= 2:
            elements = " and ".join(situation_analysis["key_entities"][:2])
            details["description"] = template.format(elements=elements)
        else:
            details["description"] = f"Opportunity: {pattern_def['pattern'].replace('_', ' ')}"
        
        # Generate specific action
        opportunity_type = pattern_def["opportunity_type"]
        if opportunity_type == "solution_development":
            details["action"] = "Design and implement targeted solution"
            details["impact"] = "high"
            details["difficulty"] = "high"
            details["resources"] = ["development team", "domain expertise", "implementation budget"]
        elif opportunity_type == "optimization":
            details["action"] = "Analyze and streamline current process"
            details["impact"] = "moderate"
            details["difficulty"] = "medium"
            details["resources"] = ["process analyst", "stakeholder time"]
        elif opportunity_type == "automation":
            details["action"] = "Implement automation technology"
            details["impact"] = "high"
            details["difficulty"] = "medium"
            details["time_to_value"] = "short-term"
            details["resources"] = ["automation tools", "technical expertise"]
        
        # Success factors based on type
        if opportunity_type in ["solution_development", "integration"]:
            details["success_factors"] = ["stakeholder buy-in", "clear requirements", "iterative development"]
        elif opportunity_type == "optimization":
            details["success_factors"] = ["baseline metrics", "process documentation", "continuous monitoring"]
        
        # Common risks
        details["risks"] = ["resistance to change", "resource constraints", "implementation complexity"]
        
        return details
    
    def _extract_processes(self, situation: Any) -> List[str]:
        """Extract process-related terms from situation"""
        processes = []
        
        situation_text = str(situation).lower()
        
        # Common process indicators
        process_patterns = [
            r"(\w+ing)\s+process",
            r"process\s+of\s+(\w+)",
            r"(\w+)\s+workflow",
            r"(\w+)\s+procedure",
            r"(\w+)\s+method"
        ]
        
        for pattern in process_patterns:
            matches = re.findall(pattern, situation_text)
            processes.extend(matches)
        
        # Also look for verb-based processes
        verb_processes = ["planning", "executing", "monitoring", "analyzing", 
                         "reporting", "coordinating", "managing"]
        for process in verb_processes:
            if process in situation_text:
                processes.append(process)
        
        return list(set(processes))[:3]  # Unique, top 3
    
    def _identify_meta_opportunities(self, opportunities: List[Dict[str, Any]], 
                                   situation_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify higher-level opportunities from combinations"""
        meta_opportunities = []
        
        # Look for complementary opportunities
        for i, opp1 in enumerate(opportunities):
            for j, opp2 in enumerate(opportunities[i+1:], i+1):
                if self._opportunities_complementary(opp1, opp2):
                    meta_opp = {
                        "type": "combined_opportunity",
                        "pattern": "synergistic_combination",
                        "description": f"Combine {opp1['type']} with {opp2['type']} for multiplied impact",
                        "specific_action": f"Implement integrated approach addressing both opportunities",
                        "confidence": (opp1["confidence"] + opp2["confidence"]) / 2 * 0.9,
                        "potential_impact": "very_high",
                        "implementation_difficulty": "high",
                        "time_to_value": "long-term",
                        "resources_required": list(set(opp1["resources_required"] + opp2["resources_required"])),
                        "success_factors": ["coordination", "integrated planning", "sustained commitment"],
                        "risks": ["complexity", "resource strain", "coordination challenges"],
                        "component_opportunities": [opp1["type"], opp2["type"]]
                    }
                    meta_opportunities.append(meta_opp)
        
        return meta_opportunities
    
    def _opportunities_complementary(self, opp1: Dict[str, Any], opp2: Dict[str, Any]) -> bool:
        """Check if two opportunities are complementary"""
        # Complementary patterns
        complementary_pairs = [
            ("optimization", "automation"),
            ("solution_development", "integration"),
            ("resource_optimization", "optimization"),
            ("early_adoption", "solution_development")
        ]
        
        type1, type2 = opp1["type"], opp2["type"]
        
        return (type1, type2) in complementary_pairs or (type2, type1) in complementary_pairs
    
    def _rank_opportunities(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank opportunities by multiple criteria"""
        for opp in opportunities:
            # Calculate composite score
            impact_scores = {"low": 0.2, "moderate": 0.5, "high": 0.8, "very_high": 1.0}
            difficulty_scores = {"low": 1.0, "medium": 0.6, "high": 0.3}  # Inverse for difficulty
            time_scores = {"short-term": 1.0, "medium-term": 0.7, "long-term": 0.4}
            
            impact = impact_scores.get(opp["potential_impact"], 0.5)
            difficulty = difficulty_scores.get(opp["implementation_difficulty"], 0.5)
            time = time_scores.get(opp["time_to_value"], 0.5)
            confidence = opp["confidence"]
            
            # Weighted score
            opp["composite_score"] = (
                impact * 0.35 +
                difficulty * 0.25 +
                time * 0.20 +
                confidence * 0.20
            )
        
        # Sort by composite score
        opportunities.sort(key=lambda x: x["composite_score"], reverse=True)
        
        return opportunities
    
    # ========================================================================================
    # ENHANCED SUBGOAL DEDUPLICATION AND PRIORITIZATION
    # ========================================================================================
    
    def _deduplicate_and_prioritize_subgoals(self, sub_goals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate and prioritize sub-goals with sophisticated logic"""
        if not sub_goals:
            return []
        
        # Group similar sub-goals
        subgoal_groups = defaultdict(list)
        
        for subgoal in sub_goals:
            # Create a signature for grouping
            signature = self._create_subgoal_signature(subgoal)
            subgoal_groups[signature].append(subgoal)
        
        # Merge and prioritize within groups
        deduplicated = []
        
        for signature, group in subgoal_groups.items():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Merge similar sub-goals
                merged = self._merge_similar_subgoals(group)
                deduplicated.append(merged)
        
        # Calculate priority scores
        for subgoal in deduplicated:
            subgoal["priority_score"] = self._calculate_subgoal_priority_score(subgoal)
        
        # Sort by priority
        deduplicated.sort(key=lambda x: x["priority_score"], reverse=True)
        
        # Add sequence numbers
        for i, subgoal in enumerate(deduplicated):
            subgoal["sequence_number"] = i + 1
            subgoal["total_subgoals"] = len(deduplicated)
        
        return deduplicated
    
    def _create_subgoal_signature(self, subgoal: Dict[str, Any]) -> str:
        """Create a signature for subgoal grouping"""
        # Extract key components
        components = []
        
        # Type
        components.append(subgoal.get("type", "unknown"))
        
        # Target node or entity
        if "node" in subgoal:
            components.append(subgoal["node"])
        elif "target" in subgoal:
            components.append(str(subgoal["target"]))
        
        # Extract key words from description
        description = subgoal.get("description", "").lower()
        key_verbs = ["establish", "measure", "control", "adjust", "monitor", "achieve"]
        
        for verb in key_verbs:
            if verb in description:
                components.append(verb)
                break
        
        return "_".join(components)
    
    def _merge_similar_subgoals(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge a group of similar sub-goals"""
        # Start with the most detailed sub-goal
        merged = max(group, key=lambda x: len(x.get("description", "")))
        
        # Combine unique elements from all sub-goals
        all_prerequisites = []
        all_models = []
        importance_scores = []
        
        for subgoal in group:
            # Collect prerequisites
            if "prerequisites" in subgoal:
                all_prerequisites.extend(subgoal["prerequisites"])
            
            # Collect models
            if "model" in subgoal:
                all_models.append(subgoal["model"])
            
            # Collect importance scores
            if "causal_importance" in subgoal:
                importance_scores.append(subgoal["causal_importance"])
            elif "importance" in subgoal:
                importance_scores.append(subgoal["importance"])
        
        # Update merged sub-goal
        if all_prerequisites:
            merged["prerequisites"] = list(set(all_prerequisites))
        
        if all_models:
            merged["models"] = list(set(all_models))
            merged["model_count"] = len(merged["models"])
        
        if importance_scores:
            merged["average_importance"] = np.mean(importance_scores)
            merged["max_importance"] = max(importance_scores)
        
        # Indicate this is a merged sub-goal
        merged["is_merged"] = True
        merged["merge_count"] = len(group)
        
        return merged
    
    def _calculate_subgoal_priority_score(self, subgoal: Dict[str, Any]) -> float:
        """Calculate comprehensive priority score for a sub-goal"""
        score = 0.5  # Base score
        
        # Type-based priority
        type_priorities = {
            "intervention_preparation": 0.8,
            "measurement": 0.7,
            "path_milestone": 0.6,
            "causal_analysis": 0.5
        }
        score += type_priorities.get(subgoal.get("type", ""), 0) * 0.2
        
        # Importance-based priority
        importance = subgoal.get("average_importance", subgoal.get("importance", 0.5))
        score += importance * 0.3
        
        # Feasibility-based priority
        feasibility = subgoal.get("feasibility", 0.5)
        score += feasibility * 0.2
        
        # Position in causal path
        if "sequence_position" in subgoal:
            path_length = subgoal.get("path_length", 1)
            position = subgoal["sequence_position"]
            # Earlier positions get higher priority
            position_score = 1.0 - (position / path_length)
            score += position_score * 0.15
        
        # Modifiability bonus
        if subgoal.get("modifiable", False):
            score += 0.1
        
        # Model support (more models = higher confidence)
        if subgoal.get("model_count", 0) > 1:
            score += 0.05
        
        return min(1.0, score)
    
    def _calculate_opportunity_score(self, opportunities: List[Dict[str, Any]]) -> float:
        """Calculate overall opportunity score"""
        if not opportunities:
            return 0.0
        
        scores = []
        for opp in opportunities:
            base_score = opp.get("confidence", 0.5)
            
            # Adjust for impact
            impact = opp.get("potential_impact", "moderate")
            if impact == "high":
                base_score *= 1.3
            elif impact == "low":
                base_score *= 0.7
            
            scores.append(base_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _recommend_opportunity_actions(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Recommend actions based on opportunities"""
        recommendations = []
        
        # Sort by confidence
        sorted_opps = sorted(opportunities, 
                            key=lambda x: x.get("confidence", 0.5), 
                            reverse=True)
        
        for i, opp in enumerate(sorted_opps[:3]):  # Top 3
            rec = {
                "priority": f"P{i+1}",
                "opportunity": opp["type"],
                "action": opp["recommended_action"],
                "reasoning": f"High confidence ({opp.get('confidence', 0.5):.2f}) opportunity",
                "timeline": "short-term" if i == 0 else "medium-term"
            }
            recommendations.append(rec)
        
        return recommendations
    
    # ========================================================================================
    # GENERATION HELPER METHODS
    # ========================================================================================
    
    def _generate_hypotheses(self, params: Dict, context: Dict, results: Dict) -> List[Dict[str, Any]]:
        """Generate hypotheses based on context and results"""
        hypotheses = []
        
        # Extract relevant information
        domain = context.get("domain", "general")
        topic = params.get("topic", context.get("user_input", ""))
        
        # Hypothesis templates
        templates = [
            "If {condition}, then {outcome} due to {mechanism}",
            "{factor} influences {target} through {pathway}",
            "The relationship between {var1} and {var2} is mediated by {mediator}",
            "{cause} leads to {effect} under {conditions}"
        ]
        
        # Generate domain-specific hypotheses
        if domain == "science":
            hypotheses.extend([
                {
                    "hypothesis": "Increased temperature accelerates reaction rate through enhanced molecular kinetics",
                    "testable": True,
                    "variables": ["temperature", "reaction_rate"],
                    "mechanism": "molecular_kinetics"
                },
                {
                    "hypothesis": "System complexity emerges from simple rule interactions",
                    "testable": True,
                    "variables": ["rule_complexity", "system_behavior"],
                    "mechanism": "emergence"
                }
            ])
        
        # Generate context-specific hypotheses
        if "causal" in str(results).lower():
            hypotheses.append({
                "hypothesis": "The observed causal relationship is strengthened by feedback loops",
                "testable": True,
                "variables": ["causal_strength", "feedback_presence"],
                "mechanism": "reinforcement"
            })
        
        # Ensure minimum hypotheses
        while len(hypotheses) < 3:
            hypotheses.append({
                "hypothesis": f"Factor X influences outcome Y through mechanism Z (hypothesis {len(hypotheses)+1})",
                "testable": True,
                "variables": ["factor_x", "outcome_y"],
                "mechanism": "unknown"
            })
        
        return hypotheses
    
    def _generate_solutions(self, params: Dict, context: Dict, results: Dict) -> List[Dict[str, Any]]:
        """Generate solution options"""
        solutions = []
        
        problem = params.get("problem", context.get("user_input", ""))
        constraints = context.get("constraints", [])
        
        # Solution generation strategies
        strategies = [
            {
                "approach": "direct",
                "description": "Address the problem directly",
                "pros": ["Simple", "Fast"],
                "cons": ["May miss root cause"]
            },
            {
                "approach": "systematic",
                "description": "Break down and solve systematically",
                "pros": ["Thorough", "Scalable"],
                "cons": ["Time-consuming"]
            },
            {
                "approach": "innovative",
                "description": "Apply creative problem-solving",
                "pros": ["Novel solutions", "Breakthrough potential"],
                "cons": ["Higher risk"]
            }
        ]
        
        for strategy in strategies:
            solution = {
                "approach": strategy["approach"],
                "description": strategy["description"],
                "implementation_steps": self._generate_implementation_steps(strategy["approach"]),
                "pros": strategy["pros"],
                "cons": strategy["cons"],
                "feasibility": self._assess_solution_feasibility(strategy, constraints),
                "estimated_effort": self._estimate_solution_effort(strategy["approach"])
            }
            solutions.append(solution)
        
        return solutions
    
    def _generate_questions(self, params: Dict, context: Dict, results: Dict) -> List[Dict[str, Any]]:
        """Generate insightful questions"""
        questions = []
        
        topic = params.get("topic", context.get("user_input", ""))
        question_type = params.get("type", "exploratory")
        
        # Question templates by type
        if question_type == "exploratory":
            question_templates = [
                "What factors influence {topic}?",
                "How does {topic} change over time?",
                "What are the boundaries of {topic}?",
                "What assumptions underlie {topic}?"
            ]
        elif question_type == "analytical":
            question_templates = [
                "What patterns exist in {topic}?",
                "How do components of {topic} interact?",
                "What causes variation in {topic}?",
                "What predicts {topic} outcomes?"
            ]
        elif question_type == "evaluative":
            question_templates = [
                "How effective is {topic}?",
                "What are the strengths and weaknesses of {topic}?",
                "How does {topic} compare to alternatives?",
                "What improvements could be made to {topic}?"
            ]
        else:
            question_templates = [
                "What is the nature of {topic}?",
                "Why is {topic} important?",
                "How can {topic} be measured?",
                "What are the implications of {topic}?"
            ]
        
        # Generate questions
        for i, template in enumerate(question_templates[:4]):
            questions.append({
                "question": template.format(topic=topic),
                "type": question_type,
                "priority": "high" if i < 2 else "medium",
                "follow_ups": self._generate_follow_up_questions(template, topic)
            })
        
        return questions
    
    def _generate_implementation_steps(self, approach: str) -> List[str]:
        """Generate implementation steps for a solution approach"""
        steps_map = {
            "direct": [
                "Identify the immediate issue",
                "Apply targeted intervention",
                "Monitor results",
                "Adjust as needed"
            ],
            "systematic": [
                "Analyze problem structure",
                "Decompose into sub-problems",
                "Solve sub-problems sequentially",
                "Integrate solutions",
                "Validate complete solution"
            ],
            "innovative": [
                "Challenge assumptions",
                "Generate creative alternatives",
                "Prototype solutions",
                "Test and iterate",
                "Scale successful approach"
            ]
        }
        
        return steps_map.get(approach, ["Define", "Plan", "Execute", "Review"])
    
    def _assess_solution_feasibility(self, strategy: Dict[str, Any], 
                                   constraints: List[str]) -> float:
        """Assess feasibility of a solution strategy"""
        feasibility = 0.8  # Base feasibility
        
        # Check constraints
        if "limited_resources" in constraints:
            if strategy["approach"] == "systematic":
                feasibility -= 0.2  # Resource intensive
        
        if "time_critical" in constraints:
            if strategy["approach"] == "innovative":
                feasibility -= 0.3  # Takes time to develop
        
        if "low_risk" in constraints:
            if strategy["approach"] == "innovative":
                feasibility -= 0.2  # Higher risk
        
        return max(0.1, feasibility)
    
    def _estimate_solution_effort(self, approach: str) -> str:
        """Estimate effort required for solution approach"""
        effort_map = {
            "direct": "low",
            "systematic": "high",
            "innovative": "medium-high"
        }
        
        return effort_map.get(approach, "medium")
    
    def _generate_follow_up_questions(self, template: str, topic: str) -> List[str]:
        """Generate follow-up questions"""
        follow_ups = []
        
        if "factors" in template:
            follow_ups.extend([
                f"Which factors are most influential for {topic}?",
                f"How do these factors interact?"
            ])
        elif "patterns" in template:
            follow_ups.extend([
                f"Are these patterns consistent across contexts?",
                f"What drives these patterns?"
            ])
        elif "effective" in template:
            follow_ups.extend([
                f"How is effectiveness measured?",
                f"What are the success criteria?"
            ])
        
        return follow_ups[:2]
    
    # ========================================================================================
    # EVALUATION HELPER METHODS  
    # ========================================================================================
    
    def _evaluate_feasibility(self, target: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate feasibility of a target"""
        feasibility = {
            "overall_score": 0.5,
            "factors": {},
            "risks": [],
            "enablers": [],
            "recommendation": ""
        }
        
        # Evaluate different feasibility factors
        factors = {
            "technical": self._assess_technical_feasibility(target),
            "resource": self._assess_resource_feasibility(target, context),
            "temporal": self._assess_temporal_feasibility(target, context),
            "organizational": self._assess_organizational_feasibility(target)
        }
        
        feasibility["factors"] = factors
        
        # Calculate overall score
        weights = {"technical": 0.3, "resource": 0.3, "temporal": 0.2, "organizational": 0.2}
        feasibility["overall_score"] = sum(factors[k] * weights[k] for k in factors)
        
        # Identify risks and enablers
        if factors["technical"] < 0.5:
            feasibility["risks"].append("Technical complexity may pose challenges")
        if factors["resource"] < 0.5:
            feasibility["risks"].append("Resource constraints could limit implementation")
        
        if factors["technical"] > 0.7:
            feasibility["enablers"].append("Strong technical foundation available")
        if factors["organizational"] > 0.7:
            feasibility["enablers"].append("Good organizational alignment")
        
        # Make recommendation
        if feasibility["overall_score"] > 0.7:
            feasibility["recommendation"] = "Highly feasible - proceed with confidence"
        elif feasibility["overall_score"] > 0.5:
            feasibility["recommendation"] = "Moderately feasible - proceed with risk mitigation"
        else:
            feasibility["recommendation"] = "Low feasibility - consider alternatives"
        
        return feasibility
    
    def _evaluate_quality(self, target: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate quality of a target"""
        quality = {
            "overall_score": 0.5,
            "dimensions": {},
            "strengths": [],
            "improvements": []
        }
        
        # Quality dimensions
        dimensions = {
            "completeness": self._assess_completeness(target),
            "correctness": self._assess_correctness(target),
            "clarity": self._assess_clarity(target),
            "consistency": self._assess_consistency(target),
            "relevance": self._assess_relevance(target, context)
        }
        
        quality["dimensions"] = dimensions
        
        # Calculate overall score
        quality["overall_score"] = np.mean(list(dimensions.values()))
        
        # Identify strengths and improvements
        for dim, score in dimensions.items():
            if score > 0.7:
                quality["strengths"].append(f"High {dim}")
            elif score < 0.5:
                quality["improvements"].append(f"Improve {dim}")
        
        return quality
    
    def _evaluate_impact(self, target: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate potential impact"""
        impact = {
            "magnitude": "medium",
            "scope": "moderate",
            "duration": "medium-term",
            "categories": {},
            "overall_assessment": ""
        }
        
        # Assess different impact categories
        categories = {
            "direct": 0.6,  # Direct impact score
            "indirect": 0.4,  # Indirect/ripple effects
            "strategic": 0.5,  # Strategic value
            "operational": 0.7   # Operational improvement
        }
        
        impact["categories"] = categories
        
        # Determine magnitude
        avg_impact = np.mean(list(categories.values()))
        if avg_impact > 0.7:
            impact["magnitude"] = "high"
        elif avg_impact < 0.4:
            impact["magnitude"] = "low"
        
        # Overall assessment
        if impact["magnitude"] == "high":
            impact["overall_assessment"] = "Significant positive impact expected"
        elif impact["magnitude"] == "medium":
            impact["overall_assessment"] = "Moderate positive impact likely"
        else:
            impact["overall_assessment"] = "Limited impact anticipated"
        
        return impact
    
    # ========================================================================================
    # FEASIBILITY ASSESSMENT HELPER METHODS
    # ========================================================================================
    
    def _assess_technical_feasibility(self, target: Any) -> float:
        """Enhanced technical feasibility assessment"""
        feasibility_score = 0.5  # Base score
        
        if isinstance(target, dict):
            # Assess based on multiple factors
            factors = {
                "complexity": 0.0,
                "technology_readiness": 0.0,
                "skill_availability": 0.0,
                "integration_difficulty": 0.0,
                "scalability": 0.0
            }
            
            # Complexity assessment
            complexity = target.get("complexity", "medium")
            complexity_scores = {"low": 0.9, "medium": 0.6, "high": 0.3, "very_high": 0.1}
            factors["complexity"] = complexity_scores.get(complexity, 0.5)
            
            # Technology readiness
            tech_maturity = target.get("technology_maturity", "established")
            maturity_scores = {
                "experimental": 0.2,
                "emerging": 0.4,
                "developing": 0.6,
                "established": 0.8,
                "mature": 0.9
            }
            factors["technology_readiness"] = maturity_scores.get(tech_maturity, 0.5)
            
            # Required skills assessment
            required_skills = target.get("required_skills", [])
            if not required_skills:
                factors["skill_availability"] = 0.8
            else:
                # Assess skill rarity
                rare_skills = ["quantum_computing", "advanced_ml", "blockchain", "neuroscience"]
                rare_count = sum(1 for skill in required_skills if skill in rare_skills)
                factors["skill_availability"] = max(0.2, 1.0 - (rare_count * 0.2))
            
            # Integration difficulty
            dependencies = target.get("dependencies", [])
            factors["integration_difficulty"] = max(0.2, 1.0 - (len(dependencies) * 0.1))
            
            # Scalability assessment
            scalability_indicators = target.get("scalability_indicators", {})
            if scalability_indicators.get("linear_scaling", False):
                factors["scalability"] = 0.8
            elif scalability_indicators.get("sub_linear_scaling", False):
                factors["scalability"] = 0.6
            else:
                factors["scalability"] = 0.4
            
            # Calculate weighted score
            weights = {
                "complexity": 0.25,
                "technology_readiness": 0.25,
                "skill_availability": 0.20,
                "integration_difficulty": 0.15,
                "scalability": 0.15
            }
            
            feasibility_score = sum(factors[k] * weights[k] for k in factors)
        
        return feasibility_score
    
    def _assess_resource_feasibility(self, target: Any, context: Dict[str, Any]) -> float:
        """Enhanced resource feasibility assessment"""
        resource_score = 0.5  # Base score
        
        constraints = context.get("constraints", [])
        resources = context.get("available_resources", {})
        
        if isinstance(target, dict):
            required_resources = target.get("required_resources", {})
            
            # Assess different resource types
            resource_types = {
                "budget": {"weight": 0.3, "score": 0.5},
                "time": {"weight": 0.25, "score": 0.5},
                "personnel": {"weight": 0.25, "score": 0.5},
                "infrastructure": {"weight": 0.2, "score": 0.5}
            }
            
            # Budget assessment
            if "budget" in required_resources and "budget" in resources:
                required_budget = required_resources["budget"]
                available_budget = resources["budget"]
                if available_budget >= required_budget:
                    resource_types["budget"]["score"] = 0.9
                else:
                    resource_types["budget"]["score"] = available_budget / required_budget
            
            # Time assessment
            if "time" in required_resources and "timeline" in context:
                required_time = required_resources["time"]  # in days
                available_time = context.get("timeline", {}).get("days", 90)
                if available_time >= required_time:
                    resource_types["time"]["score"] = 0.8
                else:
                    resource_types["time"]["score"] = available_time / required_time
            
            # Personnel assessment
            if "personnel" in required_resources:
                required_count = required_resources["personnel"].get("count", 1)
                required_skills = set(required_resources["personnel"].get("skills", []))
                
                available_personnel = resources.get("personnel", {})
                available_count = available_personnel.get("count", 0)
                available_skills = set(available_personnel.get("skills", []))
                
                count_score = min(1.0, available_count / required_count) if required_count > 0 else 1.0
                skill_score = len(available_skills.intersection(required_skills)) / len(required_skills) if required_skills else 1.0
                
                resource_types["personnel"]["score"] = (count_score + skill_score) / 2
            
            # Infrastructure assessment
            if "infrastructure" in required_resources:
                required_infra = set(required_resources["infrastructure"])
                available_infra = set(resources.get("infrastructure", []))
                
                resource_types["infrastructure"]["score"] = len(available_infra.intersection(required_infra)) / len(required_infra) if required_infra else 1.0
            
            # Apply constraint penalties
            if "limited_resources" in constraints:
                for resource_type in resource_types:
                    resource_types[resource_type]["score"] *= 0.7
            
            # Calculate weighted score
            resource_score = sum(rt["score"] * rt["weight"] for rt in resource_types.values())
        
        return resource_score

    
    def _assess_temporal_feasibility(self, target: Any, context: Dict[str, Any]) -> float:
        """Assess temporal feasibility"""
        constraints = context.get("constraints", [])
        
        if "time_critical" in constraints or "urgent" in constraints:
            return 0.4
        elif "flexible_timeline" in constraints:
            return 0.8
        
        return 0.6  # Default medium feasibility
    
    def _assess_organizational_feasibility(self, target: Any) -> float:
        """Assess organizational feasibility"""
        # Simplified assessment
        if isinstance(target, dict):
            if target.get("requires_change_management"):
                return 0.4
            elif target.get("aligns_with_culture"):
                return 0.8
        
        return 0.6  # Default medium feasibility
    
    # ========================================================================================
    # QUALITY ASSESSMENT HELPER METHODS
    # ========================================================================================
    
    def _assess_completeness(self, target: Any) -> float:
        """Assess completeness"""
        if isinstance(target, dict):
            required_fields = ["description", "approach", "implementation"]
            present_fields = sum(1 for field in required_fields if field in target)
            return present_fields / len(required_fields)
        elif isinstance(target, list):
            return min(1.0, len(target) / 5)  # Expect at least 5 items
        
        return 0.5
    
    def _assess_correctness(self, target: Any) -> float:
        """Assess correctness"""
        # Simplified assessment - in production would validate against rules
        return 0.7  # Assume mostly correct
    
    def _assess_clarity(self, target: Any) -> float:
        """Assess clarity"""
        if isinstance(target, str):
            # Check for clarity indicators
            if len(target) > 200:  # Too long
                return 0.4
            elif len(target) < 20:  # Too short
                return 0.3
            else:
                return 0.8
        
        return 0.6
    
    def _assess_consistency(self, target: Any) -> float:
        """Assess consistency"""
        # Check for internal consistency
        if isinstance(target, dict):
            # Check if related fields align
            return 0.7  # Simplified
        elif isinstance(target, list):
            # Check if all items follow same format
            if all(isinstance(item, type(target[0])) for item in target):
                return 0.9
        
        return 0.6
    
    def _assess_relevance(self, target: Any, context: Dict[str, Any]) -> float:
        """Assess relevance to context"""
        # Check alignment with context
        if context.get("domain"):
            # Simplified relevance check
            return 0.7
        
        return 0.5
    
    # ========================================================================================
    # TRANSFORMATION HELPER METHODS
    # ========================================================================================
    
    def _transform_to_abstract(self, input_data: Any) -> Dict[str, Any]:
        """Transform data to abstract representation"""
        abstract = {
            "type": type(input_data).__name__,
            "structure": "",
            "patterns": [],
            "properties": {}
        }
        
        if isinstance(input_data, dict):
            abstract["structure"] = "hierarchical"
            abstract["properties"] = {
                "depth": self._calculate_dict_depth(input_data),
                "breadth": len(input_data),
                "key_types": list(set(type(k).__name__ for k in input_data.keys()))
            }
        elif isinstance(input_data, list):
            abstract["structure"] = "sequential"
            abstract["properties"] = {
                "length": len(input_data),
                "homogeneous": len(set(type(item).__name__ for item in input_data)) == 1
            }
        elif isinstance(input_data, str):
            abstract["structure"] = "textual"
            abstract["properties"] = {
                "length": len(input_data),
                "words": len(input_data.split())
            }
        
        return abstract
    
    def _transform_to_concrete(self, input_data: Any) -> Dict[str, Any]:
        """Transform data to concrete representation"""
        concrete = {
            "examples": [],
            "instances": [],
            "specific_values": {}
        }
        
        if isinstance(input_data, dict):
            # Extract concrete examples
            for key, value in list(input_data.items())[:3]:  # First 3 items
                concrete["examples"].append({
                    "key": key,
                    "value": str(value)[:100],  # Truncate long values
                    "type": type(value).__name__
                })
        elif isinstance(input_data, list):
            concrete["instances"] = input_data[:5]  # First 5 items
        
        return concrete
    
    def _transform_to_structured(self, input_data: Any) -> Dict[str, Any]:
        """Transform data to structured format"""
        structured = {
            "schema": {},
            "data": {},
            "metadata": {}
        }
        
        if isinstance(input_data, dict):
            # Extract schema
            structured["schema"] = {
                key: type(value).__name__ for key, value in input_data.items()
            }
            structured["data"] = input_data
        elif isinstance(input_data, list):
            # Convert to structured format
            structured["schema"] = {"items": "array"}
            structured["data"] = {"items": input_data}
        else:
            # Wrap in structure
            structured["schema"] = {"value": type(input_data).__name__}
            structured["data"] = {"value": input_data}
        
        structured["metadata"] = {
            "transformed_at": datetime.now().isoformat(),
            "original_type": type(input_data).__name__
        }
        
        return structured
    
    def _calculate_dict_depth(self, d: Dict, current_depth: int = 0) -> int:
        """Calculate maximum depth of nested dictionary"""
        if not isinstance(d, dict):
            return current_depth
        
        if not d:
            return current_depth
        
        return max(self._calculate_dict_depth(v, current_depth + 1) 
                   for v in d.values() if isinstance(v, dict))
    
    def _find_relevant_causal_nodes(self, goal_components: Dict[str, Any], 
                                  model_info: Dict[str, Any]) -> List[str]:
        """Find nodes in causal model relevant to goal"""
        relevant_nodes = []
        
        if not model_info.get("nodes"):
            return relevant_nodes
        
        target = goal_components.get("target", "").lower()
        outcome = goal_components.get("target_outcome", "").lower()
        action = goal_components.get("action", "")
        
        for node_id, node_info in model_info["nodes"].items():
            node_name = node_id.lower()
            relevance_score = 0.0
            
            # Check if node matches target
            if target and (target in node_name or node_name in target):
                relevance_score += 0.8
            
            # Check if node matches outcome
            if outcome and (outcome in node_name or node_name in outcome):
                relevance_score += 0.7
            
            # Check node type relevance
            if action == "improve" and node_info.get("type") == "outcome":
                relevance_score += 0.3
            elif action == "understand" and node_info.get("type") == "intermediate":
                relevance_score += 0.3
            elif action == "control" and node_info.get("modifiable", False):
                relevance_score += 0.4
            
            # Check semantic similarity
            if target:
                similarity = self._simulate_word_embedding_similarity(
                    node_name.replace("_", " "),
                    target
                )
                relevance_score += similarity * 0.5
            
            if relevance_score > 0.5:
                relevant_nodes.append((node_id, relevance_score))
        
        # Sort by relevance and return node IDs
        relevant_nodes.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in relevant_nodes]

    def _find_causal_paths_to_outcome(self, outcome: str, relevant_nodes: List[str], 
                                     model_info: Dict[str, Any]) -> List[List[str]]:
        """Find causal paths leading to a specific outcome"""
        if not model_info.get("edges"):
            return []
        
        # Build directed graph
        graph = nx.DiGraph()
        
        # Add edges from model
        for edge in model_info["edges"]:
            source = edge.get("from")
            target = edge.get("to")
            strength = edge.get("strength", 0.5)
            
            if source and target:
                graph.add_edge(source, target, weight=strength)
        
        # Find outcome nodes
        outcome_nodes = []
        for node in graph.nodes():
            if outcome.lower() in node.lower() or node.lower() in outcome.lower():
                outcome_nodes.append(node)
        
        if not outcome_nodes:
            # Look for nodes of type 'outcome'
            for node_id, node_info in model_info.get("nodes", {}).items():
                if node_info.get("type") == "outcome":
                    outcome_nodes.append(node_id)
        
        paths = []
        
        # Find paths from relevant nodes to outcome nodes
        for start_node in relevant_nodes:
            for end_node in outcome_nodes:
                if start_node in graph and end_node in graph:
                    try:
                        # Find all simple paths
                        all_paths = list(nx.all_simple_paths(graph, start_node, end_node, cutoff=5))
                        
                        # Score and filter paths
                        scored_paths = []
                        for path in all_paths:
                            score = self._score_causal_path(path, graph, model_info)
                            scored_paths.append((score, path))
                        
                        # Sort by score and add top paths
                        scored_paths.sort(key=lambda x: x[0], reverse=True)
                        paths.extend([path for _, path in scored_paths[:2]])  # Top 2 paths per pair
                        
                    except nx.NetworkXNoPath:
                        continue
        
        # Deduplicate paths
        unique_paths = []
        seen = set()
        for path in paths:
            path_tuple = tuple(path)
            if path_tuple not in seen:
                seen.add(path_tuple)
                unique_paths.append(path)
        
        return unique_paths[:5]  # Return top 5 unique paths
    
    def _score_causal_path(self, path: List[str], graph: nx.DiGraph, 
                          model_info: Dict[str, Any]) -> float:
        """Score a causal path based on multiple criteria"""
        if len(path) < 2:
            return 0.0
        
        score = 1.0
        
        # Path length penalty (shorter is better)
        score *= (1.0 / (1 + len(path) - 2))
        
        # Edge strength product
        edge_strengths = []
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i+1])
            if edge_data:
                edge_strengths.append(abs(edge_data.get("weight", 0.5)))
        
        if edge_strengths:
            # Use geometric mean to avoid zero products
            score *= np.exp(np.mean(np.log(edge_strengths + 1e-10)))
        
        # Node importance
        node_importance = 0.0
        for node in path[1:-1]:  # Intermediate nodes
            node_info = model_info.get("nodes", {}).get(node, {})
            if node_info.get("type") == "intermediate":
                node_importance += 0.2
            if node_info.get("modifiable", False):
                node_importance += 0.3
        
        score *= (1 + node_importance)
        
        return score
    
    def _generate_path_subgoals(self, path: List[str], goal: Dict[str, Any], 
                              model_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate sub-goals for each node in a causal path"""
        subgoals = []
        
        for i, node in enumerate(path):
            node_info = model_info.get("nodes", {}).get(node, {})
            
            # Skip if not modifiable and not a measurement point
            if not node_info.get("modifiable", False) and node_info.get("type") != "intermediate":
                continue
            
            subgoal = {
                "parent_goal": goal.get("id"),
                "description": self._generate_node_subgoal_description(node, node_info, i, len(path)),
                "type": "path_milestone",
                "node": node,
                "sequence_position": i,
                "path_length": len(path),
                "node_type": node_info.get("type", "unknown"),
                "modifiable": node_info.get("modifiable", False),
                "priority": self._calculate_path_node_priority(i, len(path), node_info)
            }
            
            # Add dependencies
            if i > 0:
                subgoal["depends_on"] = [f"Control over {path[i-1]}"]
            
            subgoals.append(subgoal)
        
        return subgoals
    
    def _generate_node_subgoal_description(self, node: str, node_info: Dict[str, Any], 
                                          position: int, path_length: int) -> str:
        """Generate description for a node-based subgoal"""
        node_name = node.replace("_", " ").title()
        node_type = node_info.get("type", "factor")
        
        if position == 0:
            if node_info.get("modifiable", False):
                return f"Establish initial control over {node_name}"
            else:
                return f"Measure baseline level of {node_name}"
        elif position == path_length - 1:
            return f"Achieve target level of {node_name}"
        else:
            if node_info.get("modifiable", False):
                return f"Adjust {node_name} to influence downstream effects"
            else:
                return f"Monitor changes in {node_name} as intermediate indicator"
    
    # ========================================================================================
    # ENHANCED INTERVENTION POINT IDENTIFICATION
    # ========================================================================================
    
    def _identify_intervention_points(self, relevant_nodes: List[str], 
                                    model_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify key intervention points in the causal model"""
        intervention_points = []
        
        # Build graph for analysis
        graph = nx.DiGraph()
        for edge in model_info.get("edges", []):
            graph.add_edge(edge["from"], edge["to"], weight=edge.get("strength", 0.5))
        
        for node in relevant_nodes:
            if node not in graph:
                continue
            
            node_info = model_info.get("nodes", {}).get(node, {})
            
            # Skip non-modifiable nodes
            if not node_info.get("modifiable", False):
                continue
            
            # Calculate intervention value
            intervention_value = self._calculate_intervention_value(node, graph, model_info)
            
            # Identify downstream effects
            downstream_effects = self._analyze_downstream_effects(node, graph, model_info)
            
            # Calculate feasibility
            feasibility = self._calculate_intervention_feasibility(node, node_info, model_info)
            
            intervention_point = {
                "node_id": node,
                "node_name": node.replace("_", " ").title(),
                "importance": intervention_value,
                "feasibility": feasibility,
                "downstream_effects": downstream_effects,
                "prerequisites": self._identify_prerequisites(node, graph, model_info),
                "expected_effort": self._estimate_intervention_effort(node_info),
                "risk_level": self._assess_intervention_risk(node, downstream_effects)
            }
            
            intervention_points.append(intervention_point)
        
        # Sort by combined score (importance * feasibility)
        intervention_points.sort(
            key=lambda x: x["importance"] * x["feasibility"], 
            reverse=True
        )
        
        return intervention_points
    
    def _calculate_intervention_value(self, node: str, graph: nx.DiGraph, 
                                    model_info: Dict[str, Any]) -> float:
        """Calculate the value of intervening at a specific node"""
        value = 0.0
        
        # Centrality measures
        try:
            # Betweenness centrality (how often node appears on shortest paths)
            betweenness = nx.betweenness_centrality(graph).get(node, 0)
            value += betweenness * 0.3
            
            # Out-degree centrality (how many nodes it influences)
            out_degree = graph.out_degree(node)
            max_out_degree = max(graph.out_degree(n) for n in graph.nodes()) if graph.nodes() else 1
            value += (out_degree / max_out_degree) * 0.3 if max_out_degree > 0 else 0
            
            # PageRank (importance based on incoming links)
            pagerank = nx.pagerank(graph).get(node, 0)
            value += pagerank * 0.2
        except:
            # Fallback to simple metrics
            value = 0.5
        
        # Boost value for nodes that affect outcomes
        outcome_nodes = [n for n, info in model_info.get("nodes", {}).items() 
                         if info.get("type") == "outcome"]
        
        for outcome in outcome_nodes:
            if nx.has_path(graph, node, outcome):
                path_length = nx.shortest_path_length(graph, node, outcome)
                value += 0.2 / (1 + path_length)
        
        return min(1.0, value)
    
    def _analyze_downstream_effects(self, node: str, graph: nx.DiGraph, 
                                   model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the downstream effects of intervening at a node"""
        effects = {
            "direct_effects": [],
            "indirect_effects": [],
            "total_affected_nodes": 0,
            "outcome_impacts": []
        }
        
        # Direct effects (immediate successors)
        for successor in graph.successors(node):
            edge_data = graph.get_edge_data(node, successor)
            effects["direct_effects"].append({
                "node": successor,
                "strength": edge_data.get("weight", 0.5) if edge_data else 0.5
            })
        
        # Indirect effects (2-3 hops)
        visited = {node}
        current_level = {node}
        
        for level in range(1, 4):  # Up to 3 hops
            next_level = set()
            for current_node in current_level:
                for successor in graph.successors(current_node):
                    if successor not in visited:
                        visited.add(successor)
                        next_level.add(successor)
                        
                        if level > 1:  # Indirect effect
                            # Calculate propagated strength
                            try:
                                paths = list(nx.all_simple_paths(graph, node, successor, cutoff=level))
                                if paths:
                                    # Average strength across paths
                                    path_strengths = []
                                    for path in paths:
                                        strength = 1.0
                                        for i in range(len(path) - 1):
                                            edge_data = graph.get_edge_data(path[i], path[i+1])
                                            strength *= abs(edge_data.get("weight", 0.5)) if edge_data else 0.5
                                        path_strengths.append(strength)
                                    
                                    avg_strength = np.mean(path_strengths)
                                    effects["indirect_effects"].append({
                                        "node": successor,
                                        "strength": avg_strength,
                                        "distance": level
                                    })
                            except:
                                pass
            
            current_level = next_level
        
        effects["total_affected_nodes"] = len(visited) - 1
        
        # Outcome impacts
        outcome_nodes = [n for n, info in model_info.get("nodes", {}).items() 
                         if info.get("type") == "outcome"]
        
        for outcome in outcome_nodes:
            if outcome in visited:
                impact_level = "high" if outcome in [e["node"] for e in effects["direct_effects"]] else "moderate"
                effects["outcome_impacts"].append({
                    "outcome": outcome,
                    "impact_level": impact_level
                })
        
        return effects
    
    # ========================================================================================
    # ENHANCED MEASUREMENT AND MONITORING
    # ========================================================================================
    
    def _generate_measurement_subgoals(self, goal_components: Dict[str, Any], 
                                     model_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate measurement and monitoring sub-goals"""
        measurement_goals = []
        
        # Identify what needs to be measured
        measurement_targets = []
        
        # Target outcome measurement
        if goal_components.get("target_outcome"):
            measurement_targets.append({
                "type": "outcome",
                "name": goal_components["target_outcome"],
                "frequency": "continuous",
                "priority": "high"
            })
        
        # Success metric measurements
        for metric in goal_components.get("success_metrics", []):
            measurement_targets.append({
                "type": "metric",
                "name": metric,
                "frequency": "periodic",
                "priority": "high"
            })
        
        # Key intermediate variables
        for node_id, node_info in model_info.get("nodes", {}).items():
            if node_info.get("type") == "intermediate":
                # Check if it's on a relevant path
                relevance = self._assess_node_measurement_relevance(node_id, goal_components, model_info)
                if relevance > 0.5:
                    measurement_targets.append({
                        "type": "intermediate",
                        "name": node_id.replace("_", " "),
                        "frequency": "periodic",
                        "priority": "medium" if relevance > 0.7 else "low"
                    })
        
        # Generate sub-goals for each measurement target
        for target in measurement_targets:
            subgoal = {
                "parent_goal": goal_components.get("goal_id", "main"),
                "description": f"Establish measurement system for {target['name']}",
                "type": "measurement",
                "measurement_details": {
                    "target": target["name"],
                    "target_type": target["type"],
                    "frequency": target["frequency"],
                    "method": self._suggest_measurement_method(target),
                    "baseline_required": True,
                    "tracking_duration": "throughout_intervention"
                },
                "priority": target["priority"]
            }
            
            measurement_goals.append(subgoal)
        
        # Add a data integration sub-goal
        if len(measurement_targets) > 2:
            measurement_goals.append({
                "parent_goal": goal_components.get("goal_id", "main"),
                "description": "Integrate measurement data for comprehensive analysis",
                "type": "measurement_integration",
                "priority": "medium"
            })
        
        return measurement_goals
    
    def _assess_node_measurement_relevance(self, node_id: str, goal_components: Dict[str, Any], 
                                         model_info: Dict[str, Any]) -> float:
        """Assess how relevant a node is for measurement"""
        relevance = 0.0
        
        # Check if node name matches goal components
        node_name_lower = node_id.lower()
        
        if goal_components.get("target"):
            if goal_components["target"].lower() in node_name_lower:
                relevance += 0.5
        
        if goal_components.get("target_outcome"):
            if any(word in node_name_lower for word in goal_components["target_outcome"].lower().split()):
                relevance += 0.3
        
        # Check if node is on critical paths
        # This is simplified - in production would do actual path analysis
        node_info = model_info.get("nodes", {}).get(node_id, {})
        if node_info.get("type") == "intermediate":
            relevance += 0.2
        
        return min(1.0, relevance)
    
    def _suggest_measurement_method(self, target: Dict[str, Any]) -> str:
        """Suggest appropriate measurement method for a target"""
        target_type = target.get("type", "generic")
        target_name = target.get("name", "").lower()
        
        # Type-based suggestions
        if target_type == "outcome":
            if "satisfaction" in target_name:
                return "Survey with validated satisfaction scale"
            elif "performance" in target_name:
                return "Key performance indicators (KPIs) tracking"
            elif "health" in target_name:
                return "Clinical measurements and health assessments"
            else:
                return "Outcome-specific quantitative metrics"
        
        elif target_type == "metric":
            # Parse metric type
            if "%" in str(target_name) or "percent" in target_name:
                return "Percentage calculation from relevant data"
            elif any(word in target_name for word in ["count", "number", "quantity"]):
                return "Direct counting or enumeration"
            elif any(word in target_name for word in ["time", "duration", "speed"]):
                return "Time-based measurement"
            else:
                return "Direct metric measurement"
        
        elif target_type == "intermediate":
            return "Periodic sampling or continuous monitoring"
        
        else:
            return "Appropriate measurement method based on variable type"
    
    # ========================================================================================
    # ENHANCED CALCULATION METHODS
    # ========================================================================================
    
    def _calculate_intervention_feasibility(self, node: str, node_info: Dict[str, Any], 
                                          model_info: Dict[str, Any]) -> float:
        """Calculate comprehensive feasibility of intervening at a node"""
        feasibility = 0.5  # Base feasibility
        
        # Factor 1: Modifiability
        if node_info.get("modifiable", False):
            feasibility += 0.3
        else:
            feasibility -= 0.3
        
        # Factor 2: Accessibility (simplified - would check actual constraints)
        if node_info.get("type") == "input":
            feasibility += 0.2  # Inputs are usually more accessible
        elif node_info.get("type") == "intermediate":
            feasibility += 0.0  # Neutral
        else:
            feasibility -= 0.1  # Outcomes are harder to directly modify
        
        # Factor 3: Number of incoming influences (fewer is easier)
        edges = model_info.get("edges", [])
        incoming_edges = [e for e in edges if e.get("to") == node]
        if len(incoming_edges) == 0:
            feasibility += 0.2  # No competing influences
        elif len(incoming_edges) <= 2:
            feasibility += 0.1
        else:
            feasibility -= 0.1 * min(3, len(incoming_edges) - 2)
        
        # Factor 4: Historical success (simplified)
        if "control" in node.lower() or "input" in node.lower():
            feasibility += 0.1
        
        # Factor 5: Resource requirements (estimated)
        if any(term in node.lower() for term in ["complex", "system", "infrastructure"]):
            feasibility -= 0.2
        elif any(term in node.lower() for term in ["simple", "basic", "behavior"]):
            feasibility += 0.1
        
        return max(0.1, min(1.0, feasibility))
    
    def _identify_prerequisites(self, node: str, graph: nx.DiGraph, 
                              model_info: Dict[str, Any]) -> List[str]:
        """Identify prerequisites for intervening at a node"""
        prerequisites = []
        
        # Check for required upstream controls
        for predecessor in graph.predecessors(node):
            pred_info = model_info.get("nodes", {}).get(predecessor, {})
            if pred_info.get("modifiable", False):
                edge_data = graph.get_edge_data(predecessor, node)
                if edge_data and abs(edge_data.get("weight", 0)) > 0.5:
                    prerequisites.append(f"Control over {predecessor.replace('_', ' ')}")
        
        # Check for measurement requirements
        if model_info.get("nodes", {}).get(node, {}).get("type") == "intermediate":
            prerequisites.append(f"Ability to measure {node.replace('_', ' ')}")
        
        # Check for resource requirements
        node_lower = node.lower()
        if "technology" in node_lower or "system" in node_lower:
            prerequisites.append("Technical infrastructure")
        elif "behavior" in node_lower or "habit" in node_lower:
            prerequisites.append("Behavior change capability")
        elif "policy" in node_lower or "regulation" in node_lower:
            prerequisites.append("Policy influence or authority")
        
        return prerequisites[:3]  # Limit to top 3 prerequisites
    
    def _estimate_intervention_effort(self, node_info: Dict[str, Any]) -> str:
        """Estimate effort required for intervention"""
        effort_score = 0.5  # Base effort
        
        # Adjust based on node properties
        if not node_info.get("modifiable", True):
            effort_score += 0.3
        
        if node_info.get("type") == "outcome":
            effort_score += 0.2
        elif node_info.get("type") == "input":
            effort_score -= 0.1
        
        # Complexity indicators (would be more sophisticated in production)
        if node_info.get("complexity", "medium") == "high":
            effort_score += 0.2
        elif node_info.get("complexity", "medium") == "low":
            effort_score -= 0.2
        
        # Convert to categorical
        if effort_score < 0.3:
            return "low"
        elif effort_score < 0.7:
            return "medium"
        else:
            return "high"
    
    def _assess_intervention_risk(self, node: str, downstream_effects: Dict[str, Any]) -> str:
        """Assess risk level of intervention"""
        risk_score = 0.3  # Base risk
        
        # Factor 1: Number of affected nodes
        affected_count = downstream_effects.get("total_affected_nodes", 0)
        if affected_count > 10:
            risk_score += 0.3
        elif affected_count > 5:
            risk_score += 0.2
        elif affected_count > 2:
            risk_score += 0.1
        
        # Factor 2: Outcome impacts
        outcome_impacts = downstream_effects.get("outcome_impacts", [])
        if len(outcome_impacts) > 2:
            risk_score += 0.2
        elif len(outcome_impacts) > 0:
            risk_score += 0.1
        
        # Factor 3: Strength of effects
        strong_effects = [e for e in downstream_effects.get("direct_effects", []) 
                          if abs(e.get("strength", 0)) > 0.7]
        if len(strong_effects) > 2:
            risk_score += 0.2
        
        # Factor 4: Node criticality (simplified)
        if any(term in node.lower() for term in ["critical", "essential", "core", "fundamental"]):
            risk_score += 0.2
        
        # Convert to categorical
        if risk_score < 0.4:
            return "low"
        elif risk_score < 0.7:
            return "medium"
        else:
            return "high"
