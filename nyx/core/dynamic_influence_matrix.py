# nyx/core/dynamic_influence_matrix.py

import logging
import copy
import json
import datetime
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union

logger = logging.getLogger(__name__)

class DynamicInfluenceMatrix:
    """
    Dynamic influence matrix that manages integration weights between Nyx components.
    
    This class implements a learning matrix that controls how different components
    influence each other, with context-dependent adjustments and optimization over time.
    
    Key features:
    - Maintains a weight matrix between all core components
    - Adjusts weights based on context (emotional state, complexity, etc.)
    - Learns from feedback and interaction outcomes
    - Tracks historical changes to enable pattern analysis
    - Supports task-specific optimization profiles
    """
    
    def __init__(self, components=None):
        # Default component list if not provided
        self.components = components or [
            "memory", "emotion", "reasoning", "reflection", 
            "experience", "adaptation", "feedback", "meta",
            "knowledge"
        ]
        
        # Create component index mapping for faster lookups
        self.component_indices = {comp: i for i, comp in enumerate(self.components)}
        
        # Initialize the matrix with default weights
        self.matrix = self._initialize_matrix()
        
        # History of matrix states for analysis and learning
        self.history = []
        self.history_limit = 100  # Limit history size
        
        # Context-specific adjustment rules
        self.context_adjustments = {}
        
        # Learning parameters
        self.learning_rate = 0.05
        self.momentum = 0.1
        self.previous_update = {}
        
        # Stabilization parameters
        self.min_weight = 0.1  # Minimum influence weight
        self.max_weight = 0.9  # Maximum influence weight
        
        # Tracking data
        self.last_update = datetime.datetime.now()
        self.update_count = 0
        self.learning_events = 0
        
        # Save initial state to history
        self._save_history()
    
    def _initialize_matrix(self) -> Dict[str, Dict[str, float]]:
        """Initialize the matrix with empirically-derived default weights"""
        # Create a dictionary of dictionaries for each component pair
        matrix = {}
        for source in self.components:
            matrix[source] = {}
            for target in self.components:
                if source == target:
                    # Self-influence is set to 0.6 by default
                    matrix[source][target] = 0.6
                else:
                    # Default moderate influence (0.3)
                    matrix[source][target] = 0.3
        
        # Set specific known influence values from the existing system
        if "memory" in matrix and "emotion" in matrix:
            matrix["memory"]["emotion"] = 0.3  # Original memory_to_emotion_influence
        if "emotion" in matrix and "memory" in matrix:
            matrix["emotion"]["memory"] = 0.4  # Original emotion_to_memory_influence
            
        # Set other empirically useful influence weights
        if "reasoning" in matrix:
            if "memory" in matrix["reasoning"]:
                matrix["reasoning"]["memory"] = 0.5  # Reasoning strongly influences memory formation
            if "reflection" in matrix["reasoning"]:
                matrix["reasoning"]["reflection"] = 0.6  # Reasoning strongly influences reflection
                
        if "emotion" in matrix and "reasoning" in matrix["emotion"]:
            matrix["emotion"]["reasoning"] = 0.35  # Emotions moderately influence reasoning
            
        if "experience" in matrix and "memory" in matrix["experience"]:
            matrix["experience"]["memory"] = 0.55  # Experiences strongly influence memory
            
        if "adaptation" in matrix:
            for target in matrix["adaptation"]:
                matrix["adaptation"][target] = 0.4  # Adaptation has broad influence
                
        if "feedback" in matrix:
            for target in matrix["feedback"]:
                if target != "feedback":
                    matrix["feedback"][target] = 0.45  # Feedback influences all components
            
        return matrix
    
    def get_influence(self, source: str, target: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Get the influence weight from source to target component,
        adjusted for the current context if provided.
        
        Args:
            source: Source component name
            target: Target component name
            context: Optional context information for adjustments
            
        Returns:
            Influence weight (0.0-1.0)
        """
        # Handle missing components
        if source not in self.matrix or target not in self.matrix[source]:
            return 0.0
            
        # Start with the base weight
        weight = self.matrix[source][target]
        
        # Apply context-specific adjustments if context provided
        if context and isinstance(context, dict):
            # Apply emotional state adjustments
            if "emotional_state" in context:
                weight = self._apply_emotional_adjustments(source, target, weight, context["emotional_state"])
            
            # Apply complexity adjustments
            if "complexity" in context and isinstance(context["complexity"], (int, float)):
                weight = self._apply_complexity_adjustments(source, target, weight, context["complexity"])
            
            # Apply volatility adjustments
            if "volatility" in context and isinstance(context["volatility"], (int, float)):
                weight = self._apply_volatility_adjustments(source, target, weight, context["volatility"])
            
            # Apply interaction type adjustments
            if "interaction_type" in context:
                weight = self._apply_interaction_adjustments(source, target, weight, context["interaction_type"])
                
            # Apply user profile adjustments if available
            if "user_profile" in context:
                weight = self._apply_user_profile_adjustments(source, target, weight, context["user_profile"])
        
        # Ensure weight is within acceptable bounds
        return max(self.min_weight, min(self.max_weight, weight))
    
    def _apply_emotional_adjustments(self, source: str, target: str, weight: float, 
                                     emotional_state: Dict[str, Any]) -> float:
        """Apply adjustments based on emotional state"""
        # Extract emotional state data
        dominant_emotion = None
        intensity = 0.5
        valence = 0.0
        
        # Handle different emotional state formats
        if isinstance(emotional_state, dict):
            # Get dominant emotion
            if "dominant_emotion" in emotional_state:
                dominant_emotion = emotional_state["dominant_emotion"]
            elif "primary_emotion" in emotional_state:
                dominant_emotion = emotional_state["primary_emotion"]
                
            # Get intensity
            if "dominant_value" in emotional_state:
                intensity = emotional_state["dominant_value"]
            elif "primary_intensity" in emotional_state:
                intensity = emotional_state["primary_intensity"]
            elif "arousal" in emotional_state:
                intensity = emotional_state["arousal"]
                
            # Get valence
            if "valence" in emotional_state:
                valence = emotional_state["valence"]
        
        # Adjustments based on emotional intensity
        if intensity > 0.7:
            # High emotional intensity
            if source == "emotion":
                # Emotions have stronger influence when intense
                weight *= 1.2
            elif target == "emotion" and source == "reasoning":
                # Reasoning has less effect on strong emotions
                weight *= 0.8
        
        # Adjustments based on specific emotions
        if dominant_emotion:
            if dominant_emotion in ["Joy", "Trust", "Love"]:
                # Positive emotions
                if source == "emotion" and target == "memory":
                    # Positive emotions enhance memory influence
                    weight *= 1.1
                elif source == "emotion" and target == "reasoning":
                    # Positive emotions slightly enhance reasoning
                    weight *= 1.05
                    
            elif dominant_emotion in ["Fear", "Anger", "Frustration"]:
                # Negative high-arousal emotions
                if source == "emotion":
                    # These emotions have stronger influence overall
                    weight *= 1.15
                if target == "reasoning" and source != "reasoning":
                    # External influences on reasoning are reduced (tunnel vision)
                    weight *= 0.9
                    
            elif dominant_emotion in ["Sadness"]:
                # Negative low-arousal emotions
                if source == "memory" and target == "emotion":
                    # Memories have stronger emotional impact when sad
                    weight *= 1.2
                    
            elif dominant_emotion in ["Surprise"]:
                # Surprise enhances learning
                if target == "memory" or target == "adaptation":
                    weight *= 1.15
                    
        # Adjustments based on emotional valence
        if abs(valence) > 0.6:
            # Strong valence (either positive or negative)
            if source == "emotion" and target == "memory":
                # Strong emotions enhance memory formation
                weight *= 1.1
                
        return weight
    
    def _apply_complexity_adjustments(self, source: str, target: str, weight: float, 
                                     complexity: float) -> float:
        """Apply adjustments based on context complexity"""
        if complexity > 0.7:
            # High complexity situations
            if source == "reasoning":
                # Reasoning has more influence in complex situations
                weight *= 1.15
            elif source == "memory" and target == "reasoning":
                # Memory has more influence on reasoning in complex situations
                weight *= 1.1
            elif source == "emotion" and target == "reasoning":
                # Emotions have less influence on reasoning in complex situations
                weight *= 0.9
                
        elif complexity < 0.3:
            # Low complexity situations
            if source == "reasoning" and target != "reasoning":
                # Reasoning has less outgoing influence in simple contexts
                weight *= 0.9
            elif source == "emotion":
                # Emotions have slightly more influence in simple contexts
                weight *= 1.05
                
        return weight
    
    def _apply_volatility_adjustments(self, source: str, target: str, weight: float, 
                                     volatility: float) -> float:
        """Apply adjustments based on context volatility"""
        if volatility > 0.6:
            # High volatility situations
            if source == "adaptation":
                # Adaptation component has more influence in volatile situations
                weight *= 1.2
            elif source == "memory" and target != "memory":
                # Memory has less reliable influence in volatile contexts
                weight *= 0.9
            elif source == "reasoning" and target == "adaptation":
                # Reasoning has more influence on adaptation in volatile contexts
                weight *= 1.15
                
        elif volatility < 0.3:
            # Low volatility (stable) situations
            if source == "memory":
                # Memory has more reliable influence in stable contexts
                weight *= 1.1
            elif source == "adaptation":
                # Adaptation has less urgent role in stable contexts
                weight *= 0.9
                
        return weight
    
    def _apply_interaction_adjustments(self, source: str, target: str, weight: float, 
                                      interaction_type: str) -> float:
        """Apply adjustments based on interaction type"""
        # Emotional interactions
        if interaction_type in ["emotional", "personal", "empathetic"]:
            if source == "emotion":
                weight *= 1.2
            elif source == "reasoning" and target == "emotion":
                weight *= 0.85
            elif source == "experience" and target == "emotion":
                weight *= 1.15
                
        # Analytical interactions
        elif interaction_type in ["analytical", "logical", "problem_solving"]:
            if source == "reasoning":
                weight *= 1.25
            elif source == "emotion" and target != "emotion":
                weight *= 0.8
            elif source == "memory" and target == "reasoning":
                weight *= 1.15
                
        # Creative interactions
        elif interaction_type in ["creative", "generative", "imaginative"]:
            if source == "reflection":
                weight *= 1.2
            elif source == "memory" and target == "reflection":
                weight *= 1.15
            elif source == "emotion" and target == "reflection":
                weight *= 1.1
                
        # Narrative interactions
        elif interaction_type in ["storytelling", "narrative", "recall"]:
            if source == "memory":
                weight *= 1.2
            elif source == "experience":
                weight *= 1.25
            elif source == "emotion" and target == "memory":
                weight *= 1.1
                
        # Social interactions
        elif interaction_type in ["social", "conversational", "casual"]:
            if source == "emotion":
                weight *= 1.1
            if source == "experience":
                weight *= 1.1
                
        return weight
    
    def _apply_user_profile_adjustments(self, source: str, target: str, weight: float, 
                                       user_profile: Dict[str, Any]) -> float:
        """Apply adjustments based on user profile preferences"""
        # This would be customized based on your user profiling system
        # Example implementation:
        
        if "preferences" in user_profile:
            preferences = user_profile["preferences"]
            
            # Adjust for emotional preference
            if "emotional_sensitivity" in preferences:
                sensitivity = preferences["emotional_sensitivity"]
                if source == "emotion":
                    weight *= (0.8 + (sensitivity * 0.4))  # Scale between 0.8-1.2 based on sensitivity
                    
            # Adjust for analytical preference
            if "analytical_focus" in preferences:
                focus = preferences["analytical_focus"]
                if source == "reasoning":
                    weight *= (0.8 + (focus * 0.4))
                    
            # Adjust for memory preference
            if "memory_focus" in preferences:
                memory_focus = preferences["memory_focus"]
                if source == "memory" or target == "memory":
                    weight *= (0.9 + (memory_focus * 0.2))
                    
        return weight
    
    def update_matrix(self, source: str, target: str, adjustment: float, decay: float = 0.97) -> None:
        """
        Update the influence weight with the given adjustment.
        
        Args:
            source: Source component name
            target: Target component name
            adjustment: Value to adjust the weight by
            decay: Decay factor for the adjustment
        """
        # Handle missing components
        if source not in self.matrix or target not in self.matrix[source]:
            return
            
        # Save current state before updating
        self._save_history()
        
        # Apply adjustment with learning rate
        current_weight = self.matrix[source][target]
        update_value = adjustment * self.learning_rate
        
        # Apply momentum if there was a previous update
        if (source, target) in self.previous_update:
            update_value += self.previous_update[(source, target)] * self.momentum
        
        # Calculate new weight
        new_weight = current_weight + update_value
        
        # Ensure weight is within bounds
        self.matrix[source][target] = max(self.min_weight, min(self.max_weight, new_weight))
        
        # Save this update for momentum calculation in the future
        self.previous_update[(source, target)] = update_value
        
        # Apply subtle decay to other weights (normalization)
        self._apply_normalized_decay(source, target, decay)
        
        # Update tracking stats
        self.last_update = datetime.datetime.now()
        self.update_count += 1
    
    def _apply_normalized_decay(self, updated_source: str, updated_target: str, decay: float) -> None:
        """Apply normalized decay to maintain balance in the system"""
        # Apply subtle decay to all outgoing connections from the source
        for target in self.matrix[updated_source]:
            if target != updated_target:
                self.matrix[updated_source][target] *= decay
                # Ensure minimum weight
                self.matrix[updated_source][target] = max(self.min_weight, self.matrix[updated_source][target])
                
        # Apply even subtler decay to all incoming connections to the target
        for source in self.components:
            if source in self.matrix and updated_target in self.matrix[source] and source != updated_source:
                self.matrix[source][updated_target] *= (decay + 0.01)
                # Ensure minimum weight
                self.matrix[source][updated_target] = max(self.min_weight, self.matrix[source][updated_target])
    
    def learn_from_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """
        Update matrix weights based on feedback about component performance.
        
        Args:
            feedback_data: Dictionary with component performance data
        """
        # Extract component performance scores
        component_scores = feedback_data.get("component_scores", {})
        
        if not component_scores:
            return
            
        # Calculate adjustments based on relative performance
        total_score = sum(component_scores.values())
        
        if total_score <= 0:
            return
            
        # Save history before batch updates
        self._save_history()
        
        # Components that performed well get more influence
        for component, score in component_scores.items():
            if component not in self.components:
                continue
                
            # Calculate normalized performance score (0-1 scale)
            normalized_score = score / total_score
            
            # Adjust this component's influence on others
            for target in self.components:
                if component != target and component in self.matrix and target in self.matrix[component]:
                    # Adjustment: positive for high performers, negative for low performers
                    # Baseline: 0.5 is neutral performance
                    adjustment = (normalized_score - 0.5) * 0.15
                    
                    # Apply the adjustment (no decay in batch updates)
                    current_weight = self.matrix[component][target]
                    new_weight = current_weight + (adjustment * self.learning_rate)
                    self.matrix[component][target] = max(self.min_weight, min(self.max_weight, new_weight))
                    
                    # Track update for momentum
                    self.previous_update[(component, target)] = adjustment * self.learning_rate
        
        # Apply normalization after batch updates
        self._normalize_matrix()
            
        # Track learning events
        self.learning_events += 1
        self.last_update = datetime.datetime.now()
        self.update_count += 1
    
    def _normalize_matrix(self) -> None:
        """Normalize the matrix to prevent weights from growing too large collectively"""
        # Calculate average outgoing influence for each component
        for source in self.components:
            if source not in self.matrix:
                continue
                
            outgoing_weights = [self.matrix[source][target] for target in self.matrix[source]]
            if not outgoing_weights:
                continue
                
            avg_weight = sum(outgoing_weights) / len(outgoing_weights)
            
            # If average weight is too high, scale down
            if avg_weight > 0.6:
                scale_factor = 0.6 / avg_weight
                for target in self.matrix[source]:
                    self.matrix[source][target] *= scale_factor
                    # Ensure minimum weight
                    self.matrix[source][target] = max(self.min_weight, self.matrix[source][target])
    
    def optimize_for_task(self, task_type: str, intensity: float = 0.2) -> None:
        """
        Optimize the matrix for a specific task type.
        
        Args:
            task_type: Type of task to optimize for
            intensity: Intensity of the optimization (0.0-1.0)
        """
        # Predefined patterns for different task types
        patterns = {
            "emotional": {
                # Emotion-focused task patterns
                "emotion": {
                    "memory": 0.7, "reasoning": 0.4, "reflection": 0.6, 
                    "experience": 0.6, "adaptation": 0.5
                },
                "memory": {
                    "emotion": 0.7, "reasoning": 0.5, "reflection": 0.6
                },
                "experience": {
                    "emotion": 0.7, "reasoning": 0.4, "memory": 0.6
                }
            },
            "analytical": {
                # Analysis-focused task patterns
                "reasoning": {
                    "memory": 0.7, "emotion": 0.3, "reflection": 0.6, 
                    "experience": 0.5, "adaptation": 0.6
                },
                "memory": {
                    "reasoning": 0.7, "emotion": 0.4, "reflection": 0.5
                },
                "knowledge": {
                    "reasoning": 0.7, "memory": 0.6, "reflection": 0.6
                }
            },
            "creative": {
                # Creative task patterns
                "reflection": {
                    "memory": 0.6, "emotion": 0.7, "reasoning": 0.6, 
                    "experience": 0.6, "adaptation": 0.5
                },
                "emotion": {
                    "reflection": 0.7, "reasoning": 0.5, "memory": 0.6
                },
                "memory": {
                    "reflection": 0.7, "emotion": 0.6, "experience": 0.6
                }
            },
            "memory_intensive": {
                # Memory-focused task patterns
                "memory": {
                    "emotion": 0.6, "reasoning": 0.7, "reflection": 0.6, 
                    "experience": 0.7, "adaptation": 0.5
                },
                "experience": {
                    "memory": 0.7, "reflection": 0.6, "emotion": 0.6
                },
                "reasoning": {
                    "memory": 0.7, "reflection": 0.6, "adaptation": 0.5
                }
            },
            "narrative": {
                # Storytelling task patterns
                "experience": {
                    "memory": 0.7, "emotion": 0.7, "reasoning": 0.5, 
                    "reflection": 0.6, "adaptation": 0.4
                },
                "memory": {
                    "experience": 0.7, "emotion": 0.6, "reflection": 0.6
                },
                "emotion": {
                    "experience": 0.7, "memory": 0.6, "reflection": 0.6
                }
            },
            "social": {
                # Social interaction patterns
                "emotion": {
                    "memory": 0.6, "reasoning": 0.5, "reflection": 0.6,
                    "experience": 0.7, "adaptation": 0.6
                },
                "experience": {
                    "emotion": 0.7, "memory": 0.6, "reflection": 0.5
                },
                "memory": {
                    "emotion": 0.6, "experience": 0.7, "reasoning": 0.5
                }
            },
            "learning": {
                # Learning-focused patterns
                "adaptation": {
                    "memory": 0.7, "reasoning": 0.7, "reflection": 0.7,
                    "experience": 0.6, "emotion": 0.5
                },
                "memory": {
                    "adaptation": 0.7, "reflection": 0.6, "reasoning": 0.6
                },
                "reflection": {
                    "adaptation": 0.7, "memory": 0.6, "reasoning": 0.6
                }
            }
        }
        
        # Apply the pattern if it exists
        if task_type in patterns:
            # Save history before optimizing
            self._save_history()
            
            # Scale intensity to prevent extreme changes
            applied_intensity = min(0.8, max(0.1, intensity))
            
            for source, targets in patterns[task_type].items():
                if source not in self.matrix:
                    continue
                    
                for target, ideal_weight in targets.items():
                    if target not in self.matrix[source]:
                        continue
                        
                    current_weight = self.matrix[source][target]
                    # Calculate weighted adjustment toward ideal weight
                    adjustment = (ideal_weight - current_weight) * applied_intensity
                    
                    # Apply adjustment directly without decay
                    new_weight = current_weight + adjustment
                    self.matrix[source][target] = max(self.min_weight, min(self.max_weight, new_weight))
            
            # Apply normalization after task optimization
            self._normalize_matrix()
            
            # Update tracking stats
            self.last_update = datetime.datetime.now()
            self.update_count += 1
            
            return True
            
        return False
    
    def adapt_to_user(self, user_profile: Dict[str, Any], intensity: float = 0.3) -> None:
        """
        Adapt the influence matrix to better match a specific user's needs.
        
        Args:
            user_profile: User profile data
            intensity: Intensity of the adaptation
        """
        if not isinstance(user_profile, dict):
            return False
            
        # Save history before adapting
        self._save_history()
        
        # Scale intensity to prevent extreme changes
        applied_intensity = min(0.7, max(0.1, intensity))
        
        # Apply adaptations based on profile attributes
        if "emotional_sensitivity" in user_profile:
            sensitivity = user_profile["emotional_sensitivity"]
            if isinstance(sensitivity, (int, float)) and 0 <= sensitivity <= 1:
                # Adjust emotion component influence
                for target in self.components:
                    if "emotion" in self.matrix and target in self.matrix["emotion"] and target != "emotion":
                        ideal_weight = 0.4 + (sensitivity * 0.4)  # Range from 0.4 to 0.8
                        current_weight = self.matrix["emotion"][target]
                        adjustment = (ideal_weight - current_weight) * applied_intensity
                        self.matrix["emotion"][target] = max(self.min_weight, min(self.max_weight, current_weight + adjustment))
        
        if "analytical_preference" in user_profile:
            preference = user_profile["analytical_preference"]
            if isinstance(preference, (int, float)) and 0 <= preference <= 1:
                # Adjust reasoning component influence
                for target in self.components:
                    if "reasoning" in self.matrix and target in self.matrix["reasoning"] and target != "reasoning":
                        ideal_weight = 0.4 + (preference * 0.4)  # Range from 0.4 to 0.8
                        current_weight = self.matrix["reasoning"][target]
                        adjustment = (ideal_weight - current_weight) * applied_intensity
                        self.matrix["reasoning"][target] = max(self.min_weight, min(self.max_weight, current_weight + adjustment))
        
        if "experience_focus" in user_profile:
            focus = user_profile["experience_focus"]
            if isinstance(focus, (int, float)) and 0 <= focus <= 1:
                # Adjust experience/memory components influence
                for source in ["experience", "memory"]:
                    for target in self.components:
                        if source in self.matrix and target in self.matrix[source] and target != source:
                            ideal_weight = 0.4 + (focus * 0.4)  # Range from 0.4 to 0.8
                            current_weight = self.matrix[source][target]
                            adjustment = (ideal_weight - current_weight) * applied_intensity
                            self.matrix[source][target] = max(self.min_weight, min(self.max_weight, current_weight + adjustment))
        
        # Apply normalization after user adaptation
        self._normalize_matrix()
        
        # Update tracking stats
        self.last_update = datetime.datetime.now()
        self.update_count += 1
        
        return True
    
    def detect_patterns(self) -> Dict[str, Any]:
        """
        Analyze history to detect patterns in influence changes.
        
        Returns:
            Dictionary with detected patterns
        """
        if len(self.history) < 5:
            return {"patterns_detected": False, "reason": "Insufficient history"}
        
        patterns = {
            "increasing": [],
            "decreasing": [],
            "oscillating": [],
            "stable": []
        }
        
        # Sample history at regular intervals if it's very large
        history_samples = self.history
        if len(self.history) > 20:
            sample_indices = [int(i * len(self.history) / 20) for i in range(20)]
            history_samples = [self.history[i] for i in sample_indices]
        
        # Analyze each component pair
        for source in self.components:
            for target in self.components:
                if source == target:
                    continue
                
                # Extract history for this pair
                values = []
                for h in history_samples:
                    if source in h and target in h[source]:
                        values.append(h[source][target])
                
                if not values or len(values) < 5:
                    continue
                    
                # Calculate trend
                diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
                avg_diff = sum(diffs) / len(diffs)
                
                # Check for oscillation
                sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
                oscillation_rate = sign_changes / (len(diffs) - 1) if len(diffs) > 1 else 0
                
                # Calculate standard deviation to measure stability
                std_dev = np.std(values) if np else sum((v - sum(values)/len(values))**2 for v in values)**0.5 / len(values)
                
                # Categorize pattern
                if std_dev < 0.02:
                    patterns["stable"].append((source, target, values[-1]))
                elif oscillation_rate > 0.4:
                    patterns["oscillating"].append((source, target, values[-1]))
                elif avg_diff > 0.01:
                    patterns["increasing"].append((source, target, values[-1]))
                elif avg_diff < -0.01:
                    patterns["decreasing"].append((source, target, values[-1]))
        
        # Generate insights
        insights = []
        
        if len(patterns["increasing"]) > 3:
            top_increasing = sorted(patterns["increasing"], key=lambda x: x[2], reverse=True)[:3]
            sources = set(s for s, _, _ in top_increasing)
            if len(sources) == 1:
                insights.append(f"The {next(iter(sources))} component is gaining influence across the system")
            else:
                insights.append(f"Multiple components are gaining influence: {', '.join(sources)}")
                
        if len(patterns["decreasing"]) > 3:
            top_decreasing = sorted(patterns["decreasing"], key=lambda x: x[2])[:3]
            sources = set(s for s, _, _ in top_decreasing)
            if len(sources) == 1:
                insights.append(f"The {next(iter(sources))} component is losing influence across the system")
            else:
                insights.append(f"Multiple components are losing influence: {', '.join(sources)}")
                
        if len(patterns["oscillating"]) > 3:
            sources = set(s for s, _, _ in patterns["oscillating"][:3])
            insights.append(f"Unstable influence patterns detected in: {', '.join(sources)}")
            
        if len(patterns["stable"]) > len(self.components) * 2:
            insights.append("The matrix is showing strong stability in most component relationships")
        
        return {
            "patterns_detected": True,
            "patterns": {
                "increasing": [(s, t) for s, t, _ in patterns["increasing"]],
                "decreasing": [(s, t) for s, t, _ in patterns["decreasing"]],
                "oscillating": [(s, t) for s, t, _ in patterns["oscillating"]],
                "stable": [(s, t) for s, t, _ in patterns["stable"]],
            },
            "insights": insights,
            "history_length": len(self.history)
        }
    
    def reset_to_defaults(self) -> None:
        """Reset the matrix to default values"""
        # Save history before reset
        self._save_history()
        
        # Reset matrix
        self.matrix = self._initialize_matrix()
        self.previous_update = {}
        
        # Update tracking stats
        self.last_update = datetime.datetime.now()
        self.update_count += 1
    
    def _save_history(self) -> None:
        """Save current matrix state to history"""
        # Don't save if no change since last save
        if self.history and self._matrices_equal(self.matrix, self.history[-1]):
            return
            
        self.history.append(copy.deepcopy(self.matrix))
        
        # Limit history size
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]
    
    def _matrices_equal(self, matrix1: Dict, matrix2: Dict) -> bool:
        """Check if two matrices are effectively equal"""
        for source in self.components:
            if source not in matrix1 or source not in matrix2:
                return False
                
            for target in self.components:
                if target not in matrix1[source] or target not in matrix2[source]:
                    return False
                    
                # Compare with small tolerance for floating point differences
                if abs(matrix1[source][target] - matrix2[source][target]) > 0.0001:
                    return False
        
        return True
    
    def get_influence_stats(self) -> Dict[str, Any]:
        """Get statistics about the influence matrix"""
        # Calculate overall stats
        all_weights = []
        for source in self.matrix:
            for target in self.matrix[source]:
                all_weights.append(self.matrix[source][target])
        
        if not all_weights:
            return {"error": "No weights found in matrix"}
        
        stats = {
            "total_influence": sum(all_weights),
            "max_influence": max(all_weights),
            "min_influence": min(all_weights),
            "avg_influence": sum(all_weights) / len(all_weights),
            "std_dev": np.std(all_weights) if np else (sum((w - sum(all_weights)/len(all_weights))**2 for w in all_weights) / len(all_weights))**0.5,
            "update_count": self.update_count,
            "learning_events": self.learning_events,
            "last_update": self.last_update.isoformat(),
            "component_influence": {}
        }
        
        # Calculate influence metrics for each component
        for component in self.components:
            if component not in self.matrix:
                continue
                
            # Outgoing influence (how much this component influences others)
            outgoing = sum(self.matrix[component].get(target, 0) for target in self.components if target != component)
            outgoing_count = sum(1 for target in self.components if target != component and target in self.matrix[component])
            
            # Incoming influence (how much this component is influenced by others)
            incoming = sum(self.matrix[source].get(component, 0) for source in self.components if source != component and source in self.matrix)
            incoming_count = sum(1 for source in self.components if source != component and source in self.matrix and component in self.matrix[source])
            
            # Calculate averages
            avg_outgoing = outgoing / outgoing_count if outgoing_count else 0
            avg_incoming = incoming / incoming_count if incoming_count else 0
            
            # Calculate influence ratio and balance
            influence_ratio = outgoing / incoming if incoming > 0 else float('inf')
            influence_balance = outgoing - incoming
            
            stats["component_influence"][component] = {
                "outgoing": outgoing,
                "incoming": incoming,
                "avg_outgoing": avg_outgoing,
                "avg_incoming": avg_incoming,
                "influence_ratio": influence_ratio,
                "influence_balance": influence_balance
            }
        
        # Find dominant and dominated components
        influence_balances = [(comp, data["influence_balance"]) for comp, data in stats["component_influence"].items()]
        dominant = sorted(influence_balances, key=lambda x: x[1], reverse=True)[:2]
        dominated = sorted(influence_balances, key=lambda x: x[1])[:2]
        
        stats["dominant_components"] = [{"component": comp, "balance": bal} for comp, bal in dominant]
        stats["dominated_components"] = [{"component": comp, "balance": bal} for comp, bal in dominated]
        
        return stats
    
    def save_to_file(self, filename: str) -> bool:
        """Save matrix state to a file"""
        try:
            data = {
                "matrix": self.matrix,
                "components": self.components,
                "history": self.history[-10:],  # Save only recent history
                "learning_rate": self.learning_rate,
                "momentum": self.momentum,
                "update_count": self.update_count,
                "learning_events": self.learning_events,
                "last_update": self.last_update.isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving influence matrix: {str(e)}")
            return False
    
    def load_from_file(self, filename: str) -> bool:
        """Load matrix state from a file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.matrix = data.get("matrix", self._initialize_matrix())
            self.components = data.get("components", self.components)
            self.history = data.get("history", [])
            self.learning_rate = data.get("learning_rate", 0.05)
            self.momentum = data.get("momentum", 0.1)
            self.update_count = data.get("update_count", 0)
            self.learning_events = data.get("learning_events", 0)
            
            try:
                self.last_update = datetime.datetime.fromisoformat(data.get("last_update", datetime.datetime.now().isoformat()))
            except:
                self.last_update = datetime.datetime.now()
            
            return True
        except Exception as e:
            logger.error(f"Error loading influence matrix: {str(e)}")
            return False
    
    def get_matrix_visualization_data(self) -> Dict[str, Any]:
        """Get data for visualizing the influence matrix"""
        nodes = [{"id": comp, "group": i + 1} for i, comp in enumerate(self.components)]
        links = []
        
        for source in self.matrix:
            for target in self.matrix[source]:
                weight = self.matrix[source][target]
                if weight > 0.15:  # Filter out weak connections for clearer visualization
                    links.append({
                        "source": source,
                        "target": target,
                        "value": weight * 10  # Scale for visualization
                    })
        
        return {
            "nodes": nodes,
            "links": links
        }
