# nyx/core/integration_utils.py

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class ComponentIntegration:
    """
    Utility class to facilitate integration between Nyx components
    using the dynamic influence matrix. This class provides helper methods
    for components to interact with the influence matrix system.
    """
    
    @staticmethod
    async def get_influence_weight(influence_matrix, source: str, target: str, context: Dict[str, Any]) -> float:
        """
        Get the influence weight between two components considering context
        
        Args:
            influence_matrix: The component influence matrix
            source: Source component name
            target: Target component name
            context: Current context data
            
        Returns:
            Influence weight (0.0-1.0)
        """
        if not influence_matrix:
            # Default weights if no matrix is available
            default_weights = {
                ("memory", "emotion"): 0.3,
                ("emotion", "memory"): 0.4,
                ("reasoning", "reflection"): 0.7,
                ("experience", "emotion"): 0.6,
                ("emotion", "reasoning"): 0.4,
                ("knowledge", "memory"): 0.7,
                ("memory", "reflection"): 0.6,
                ("adaptation", "meta"): 0.7,
                ("feedback", "adaptation"): 0.8
            }
            
            key = (source, target)
            return default_weights.get(key, 0.5)  # Default to 0.5 for undefined pairs
        
        try:
            # Use contextual influence if context is provided
            if context:
                return influence_matrix.calculate_contextual_influence(source, target, context)
            # Otherwise use base influence
            return influence_matrix.get_influence(source, target)
        except Exception as e:
            logger.warning(f"Error getting influence weight {source}->{target}: {e}")
            return 0.5  # Default fallback
    
    @staticmethod
    async def track_component_activity(activity_tracker: Dict[str, float], 
                                    component: str, 
                                    activity_level: float) -> Dict[str, float]:
        """
        Track activity level for a component
        
        Args:
            activity_tracker: Dictionary tracking component activities
            component: Component name
            activity_level: Activity level (0.0-1.0)
            
        Returns:
            Updated activity tracker
        """
        if not activity_tracker:
            activity_tracker = {}
        
        # Ensure activity level is in valid range
        activity_level = max(0.0, min(1.0, activity_level))
        
        # Update activity tracker
        activity_tracker[component] = activity_level
        
        return activity_tracker
    
    @staticmethod
    async def calculate_multi_component_influence(influence_matrix, 
                                               source_components: List[str], 
                                               target: str, 
                                               context: Dict[str, Any]) -> float:
        """
        Calculate combined influence from multiple components on a target
        
        Args:
            influence_matrix: The component influence matrix
            source_components: List of source component names
            target: Target component name
            context: Current context data
            
        Returns:
            Combined influence weight (0.0-1.0)
        """
        if not influence_matrix or not source_components:
            return 0.5  # Default
        
        # Get individual influence weights
        weights = []
        for source in source_components:
            weight = await ComponentIntegration.get_influence_weight(
                influence_matrix, source, target, context
            )
            weights.append(weight)
        
        # Calculate weighted average
        if not weights:
            return 0.5
            
        return sum(weights) / len(weights)
    
    @staticmethod
    async def prepare_context_with_emotional_influence(original_context: Dict[str, Any], 
                                                   influence_matrix,
                                                   emotion_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance context with emotional information based on influence weights
        
        Args:
            original_context: Original context dictionary
            influence_matrix: The component influence matrix
            emotion_state: Current emotional state
            
        Returns:
            Enhanced context with emotional influence
        """
        enhanced_context = original_context.copy() if original_context else {}
        
        # Add emotional state if not already present
        if "emotional_state" not in enhanced_context and emotion_state:
            enhanced_context["emotional_state"] = emotion_state
        
        # Calculate emotional influence weight for target components
        target_components = ["memory", "reasoning", "reflection", "experience"]
        
        for target in target_components:
            influence_key = f"emotion_to_{target}_influence"
            
            # Get dynamic influence weight
            influence = await ComponentIntegration.get_influence_weight(
                influence_matrix, "emotion", target, enhanced_context
            )
            
            # Add to context
            enhanced_context[influence_key] = influence
        
        return enhanced_context
    
    @staticmethod
    async def collect_influence_data(influence_matrix,
                                  component_activity: Dict[str, float],
                                  interaction_metrics: Dict[str, float],
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect influence-related data for learning and analysis
        
        Args:
            influence_matrix: The component influence matrix
            component_activity: Component activity levels
            interaction_metrics: Interaction outcome metrics
            context: Current context
            
        Returns:
            Collected influence data
        """
        if not influence_matrix:
            return {}
        
        # Extract key influence pairs from active components
        active_components = [c for c, level in component_activity.items() if level > 0.3]
        influence_data = []
        
        for source in active_components:
            for target in active_components:
                if source != target:
                    influence = await ComponentIntegration.get_influence_weight(
                        influence_matrix, source, target, context
                    )
                    
                    influence_data.append({
                        "source": source,
                        "target": target,
                        "weight": influence,
                        "source_activity": component_activity.get(source, 0.0),
                        "target_activity": component_activity.get(target, 0.0)
                    })
        
        # Calculate success metrics
        avg_success = sum(interaction_metrics.values()) / len(interaction_metrics) if interaction_metrics else 0.5
        
        return {
            "timestamp": datetime.now().isoformat(),
            "influence_pairs": influence_data,
            "component_activity": component_activity.copy(),
            "metrics": interaction_metrics.copy(),
            "avg_success": avg_success,
            "context_type": context.get("interaction_type", "general") if context else "general"
        }
    
    @staticmethod
    async def calculate_optimal_weights(tracked_data: List[Dict[str, Any]], 
                                     influence_matrix) -> Dict[str, Any]:
        """
        Calculate optimal influence weights based on tracked interaction data
        
        Args:
            tracked_data: List of tracked interaction data points
            influence_matrix: The component influence matrix
            
        Returns:
            Optimization recommendations
        """
        if not tracked_data or not influence_matrix:
            return {"status": "insufficient_data", "recommendations": {}}
        
        # Group data by success level
        successful = []
        neutral = []
        unsuccessful = []
        
        for data in tracked_data:
            avg_success = data.get("avg_success", 0.5)
            
            if avg_success > 0.7:
                successful.append(data)
            elif avg_success < 0.4:
                unsuccessful.append(data)
            else:
                neutral.append(data)
        
        # If not enough data in any category, return
        if len(successful) < 3:
            return {"status": "insufficient_successful_data", "recommendations": {}}
        
        # Calculate average influence weights for successful interactions
        successful_weights = {}
        for data in successful:
            for pair in data.get("influence_pairs", []):
                source = pair.get("source")
                target = pair.get("target")
                weight = pair.get("weight")
                
                if not all([source, target, weight is not None]):
                    continue
                    
                key = f"{source}_{target}"
                if key not in successful_weights:
                    successful_weights[key] = []
                    
                successful_weights[key].append(weight)
        
        # Average out successful weights
        recommendations = {}
        for key, weights in successful_weights.items():
            if len(weights) >= 3:  # Only use keys with enough data
                source, target = key.split("_")
                current_weight = influence_matrix.get_influence(source, target)
                optimal_weight = sum(weights) / len(weights)
                
                # Only recommend changes if significant difference
                if abs(current_weight - optimal_weight) > 0.1:
                    recommendations[key] = {
                        "source": source,
                        "target": target,
                        "current_weight": current_weight,
                        "recommended_weight": optimal_weight,
                        "difference": optimal_weight - current_weight,
                        "data_points": len(weights)
                    }
        
        return {
            "status": "success",
            "successful_interactions": len(successful),
            "neutral_interactions": len(neutral),
            "unsuccessful_interactions": len(unsuccessful),
            "recommendations": recommendations
        }

class NyxEmotionalIntegration:
    """
    Specialized integration utilities for the emotional core and other components
    """
    
    @staticmethod
    async def apply_memory_to_emotion_effects(memory_data: List[Dict[str, Any]],
                                           emotional_core,
                                           influence_matrix,
                                           context: Dict[str, Any]) -> Dict[str, float]:
        """
        Apply dynamically weighted effects from memories to emotions
        
        Args:
            memory_data: List of memory data
            emotional_core: The emotional core component
            influence_matrix: The component influence matrix
            context: Current context
            
        Returns:
            Emotional changes applied
        """
        if not memory_data or not emotional_core:
            return {}
        
        # Get the influence weight
        memory_emotion_influence = await ComponentIntegration.get_influence_weight(
            influence_matrix, "memory", "emotion", context
        )
        
        # Calculate emotional impact for each memory
        impacts = {}
        
        for memory in memory_data:
            # Extract emotional context
            emotional_context = memory.get("metadata", {}).get("emotional_context", {})
            if not emotional_context:
                continue
                
            # Get primary emotion and intensity
            primary_emotion = emotional_context.get("primary_emotion")
            primary_intensity = emotional_context.get("primary_intensity", 0.5)
            
            if not primary_emotion:
                continue
                
            # Calculate impact factors
            relevance = memory.get("relevance", 0.5)
            recency = 1.0  # Default recency factor
            
            # Calculate timestamp-based recency if available
            timestamp_str = memory.get("metadata", {}).get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    days_old = (datetime.now() - timestamp).days
                    recency = max(0.5, 1.0 - (days_old / 30))  # Decay over 30 days, minimum 0.5
                except (ValueError, TypeError):
                    pass
            
            # Calculate impact value with dynamic influence weight
            impact = primary_intensity * relevance * recency * memory_emotion_influence * 0.1
            
            # Aggregate impacts by emotion
            if primary_emotion not in impacts:
                impacts[primary_emotion] = 0.0
            impacts[primary_emotion] += impact
        
        # Apply emotional impacts to emotional core
        for emotion, impact in impacts.items():
            if impact > 0.01:  # Only apply non-trivial impacts
                emotional_core.update_emotion(emotion, impact)
        
        return impacts

class NyxMemoryIntegration:
    """
    Specialized integration utilities for the memory system and other components
    """
    
    @staticmethod
    async def enhance_memory_retrieval(query: str,
                                    memory_core,
                                    influence_matrix,
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance memory retrieval using dynamic influence weights
        
        Args:
            query: Memory retrieval query
            memory_core: The memory core component
            influence_matrix: The component influence matrix
            context: Current context
            
        Returns:
            Enhanced retrieval parameters
        """
        if not memory_core:
            return {}
        
        # Default retrieval parameters
        retrieval_params = {
            "query": query,
            "memory_types": ["observation", "reflection", "abstraction", "experience"],
            "limit": 5,
            "min_significance": 3
        }
        
        # Enhance based on component influences if matrix is available
        if influence_matrix:
            # Get emotion influence on memory retrieval
            emotion_memory_influence = await ComponentIntegration.get_influence_weight(
                influence_matrix, "emotion", "memory", context
            )
            
            # Get reasoning influence on memory retrieval
            reasoning_memory_influence = await ComponentIntegration.get_influence_weight(
                influence_matrix, "reasoning", "memory", context
            )
            
            # Get knowledge influence on memory retrieval
            knowledge_memory_influence = await ComponentIntegration.get_influence_weight(
                influence_matrix, "knowledge", "memory", context
            )
            
            # Adjust retrieval parameters based on influences
            
            # Emotional influence adjusts significance threshold
            if emotion_memory_influence > 0.6:
                # High emotional influence - prioritize emotionally significant memories
                retrieval_params["min_significance"] = 4  # Higher threshold for significance
                
                # Add emotional state for context matching if available
                if "emotional_state" in context:
                    retrieval_params["emotional_state"] = context["emotional_state"]
            elif emotion_memory_influence < 0.3:
                # Low emotional influence - be more inclusive with significance
                retrieval_params["min_significance"] = 2
            
            # Reasoning influence adjusts memory types prioritized
            if reasoning_memory_influence > 0.6:
                # High reasoning influence - prioritize abstractions and reflections
                retrieval_params["memory_types"] = ["abstraction", "reflection", "observation", "experience"]
            
            # Knowledge influence adjusts retrieval limit
            if knowledge_memory_influence > 0.6:
                # High knowledge influence - retrieve more memories for better knowledge context
                retrieval_params["limit"] = 8
            elif knowledge_memory_influence < 0.3:
                # Low knowledge influence - retrieve fewer memories to reduce noise
                retrieval_params["limit"] = 3
        
        return retrieval_params

class NyxReasoningIntegration:
    """
    Specialized integration utilities for the reasoning system and other components
    """
    
    @staticmethod
    async def determine_reasoning_approach(query: str,
                                        influence_matrix,
                                        emotional_state: Dict[str, Any],
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the optimal reasoning approach based on dynamic influences
        
        Args:
            query: The reasoning query
            influence_matrix: The component influence matrix
            emotional_state: Current emotional state
            context: Current context
            
        Returns:
            Reasoning approach configuration
        """
        # Default reasoning configuration
        reasoning_config = {
            "approach": "balanced",
            "depth": 5,
            "creativity": 0.5,
            "emotion_consideration": 0.5
        }
        
        if not influence_matrix:
            return reasoning_config
        
        # Get relevant influence weights
        emotion_reasoning_influence = await ComponentIntegration.get_influence_weight(
            influence_matrix, "emotion", "reasoning", context
        )
        
        memory_reasoning_influence = await ComponentIntegration.get_influence_weight(
            influence_matrix, "memory", "reasoning", context
        )
        
        knowledge_reasoning_influence = await ComponentIntegration.get_influence_weight(
            influence_matrix, "knowledge", "reasoning", context
        )
        
        # Adjust reasoning configuration based on influences
        
        # Emotion influence impacts creativity and emotion consideration
        reasoning_config["emotion_consideration"] = emotion_reasoning_influence
        
        if emotion_reasoning_influence > 0.7:
            # High emotional influence
            reasoning_config["approach"] = "intuitive"
            reasoning_config["creativity"] = 0.7
        elif emotion_reasoning_influence < 0.3:
            # Low emotional influence
            reasoning_config["approach"] = "analytical"
            reasoning_config["creativity"] = 0.3
        
        # Memory influence impacts depth
        if memory_reasoning_influence > 0.6:
            # Deep reasoning with memory connections
            reasoning_config["depth"] = 7
        elif memory_reasoning_influence < 0.3:
            # Shallow reasoning with fewer memory considerations
            reasoning_config["depth"] = 3
        
        # Knowledge influence impacts approach
        if knowledge_reasoning_influence > 0.7:
            # Knowledge-heavy approach
            if reasoning_config["approach"] == "intuitive":
                # Balance with emotional influence
                reasoning_config["approach"] = "insightful"
            else:
                reasoning_config["approach"] = "scholarly"
        
        return reasoning_config
