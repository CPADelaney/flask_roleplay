# nyx/core/a2a/context_aware_prediction_engine.py

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import math

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwarePredictionEngine(ContextAwareModule):
    """
    Advanced PredictionEngine with full context distribution capabilities
    """
    
    def __init__(self, original_prediction_engine):
        super().__init__("prediction_engine")
        self.original_engine = original_prediction_engine
        self.context_subscriptions = [
            "emotional_state_update", "pattern_detected", "goal_progress",
            "memory_consolidation", "relationship_milestone", "behavioral_trend",
            "prediction_feedback", "context_stability_change", "system_state_update",
            "cross_module_correlation", "temporal_pattern", "causal_chain_detected"
        ]
        
        # Enhanced prediction tracking
        self.context_enhanced_predictions = {}
        self.cross_module_predictions = []
        self.prediction_accuracy_by_context = {}
        self.context_stability_history = []
        
        # Advanced prediction parameters
        self.context_weight = 0.3  # How much context influences predictions
        self.cross_module_bonus = 0.2  # Bonus for cross-module coherent predictions
        self.stability_threshold = 0.7  # Minimum stability for high-confidence predictions
    
    async def on_context_received(self, context: SharedContext):
        """Process incoming context for prediction generation"""
        logger.debug(f"PredictionEngine received context for user: {context.user_id}")
        
        # Analyze context stability for prediction confidence
        stability_analysis = await self._analyze_context_stability(context)
        self.context_stability_history.append({
            "timestamp": datetime.now().isoformat(),
            "stability": stability_analysis["overall_stability"],
            "factors": stability_analysis["stability_factors"]
        })
        
        # Identify prediction opportunities from context
        prediction_opportunities = await self._identify_prediction_opportunities(context)
        
        # Check for cross-module prediction possibilities
        cross_module_potential = await self._analyze_cross_module_potential(context)
        
        # Generate initial predictions if opportunities exist
        if prediction_opportunities:
            for opportunity in prediction_opportunities[:2]:  # Limit initial predictions
                asyncio.create_task(self._generate_contextual_prediction(opportunity, context))
        
        # Send prediction context
        await self.send_context_update(
            update_type="prediction_context_available",
            data={
                "context_stability": stability_analysis["overall_stability"],
                "prediction_opportunities": len(prediction_opportunities),
                "cross_module_potential": cross_module_potential,
                "active_predictions": len(self.original_engine.predictions),
                "confidence_modifier": stability_analysis.get("confidence_modifier", 0)
            },
            priority=ContextPriority.MEDIUM
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules for prediction enhancement"""
        
        if update.update_type == "pattern_detected":
            # Use detected patterns to improve predictions
            pattern_data = update.data
            pattern_type = pattern_data.get("pattern_type")
            pattern_strength = pattern_data.get("strength", 0.5)
            
            if pattern_strength > 0.6:
                await self._incorporate_pattern_into_predictions(pattern_type, pattern_data)
        
        elif update.update_type == "emotional_state_update":
            # Update emotional predictions
            emotional_data = update.data
            await self._update_emotional_predictions(emotional_data)
        
        elif update.update_type == "goal_progress":
            # Predict future goal outcomes
            goal_data = update.data
            execution_results = goal_data.get("execution_results", [])
            
            if execution_results:
                await self._predict_goal_trajectory(execution_results, goal_data)
        
        elif update.update_type == "behavioral_trend":
            # Incorporate behavioral trends into predictions
            trend_data = update.data
            trend_type = trend_data.get("trend_type")
            trend_direction = trend_data.get("direction")
            
            await self._adjust_predictions_for_trend(trend_type, trend_direction, trend_data)
        
        elif update.update_type == "prediction_feedback":
            # Learn from prediction accuracy feedback
            feedback_data = update.data
            prediction_id = feedback_data.get("prediction_id")
            accuracy = feedback_data.get("accuracy", 0.5)
            
            if prediction_id:
                await self._process_prediction_feedback(prediction_id, accuracy, feedback_data)
        
        elif update.update_type == "context_stability_change":
            # Adjust prediction confidence based on stability
            stability_data = update.data
            new_stability = stability_data.get("stability_score", 0.5)
            
            await self._adjust_prediction_confidence(new_stability)
        
        elif update.update_type == "temporal_pattern":
            # Use temporal patterns for time-based predictions
            temporal_data = update.data
            pattern = temporal_data.get("pattern")
            
            if pattern:
                await self._generate_temporal_predictions(pattern, temporal_data)
        
        elif update.update_type == "causal_chain_detected":
            # Use causal chains for outcome predictions
            causal_data = update.data
            chain = causal_data.get("causal_chain", [])
            
            if chain:
                await self._predict_causal_outcomes(chain, causal_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with context-aware prediction generation"""
        # Extract prediction query from input
        prediction_query = await self._extract_prediction_query(context.user_input)
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Build comprehensive prediction context
        prediction_context = await self._build_prediction_context(context, messages)
        
        # Generate predictions with context enhancement
        predictions = []
        
        if prediction_query["requires_prediction"]:
            # Use original engine with enhanced context
            history = await self._extract_relevant_history(context, messages)
            
            prediction_result = await self.original_engine.generate_prediction(
                context=prediction_context,
                history=history,
                query_type=prediction_query.get("query_type")
            )
            
            if prediction_result:
                predictions.append(prediction_result)
                
                # Track as context-enhanced
                self.context_enhanced_predictions[prediction_result.prediction_id] = {
                    "context_summary": self._summarize_context(context),
                    "cross_module_factors": len(messages),
                    "stability_score": prediction_context.get("stability_score", 0.5),
                    "timestamp": datetime.now().isoformat()
                }
        
        # Check for implicit prediction needs
        implicit_predictions = await self._check_implicit_prediction_needs(context, messages)
        for implicit_pred in implicit_predictions:
            pred_result = await self._generate_implicit_prediction(implicit_pred, context)
            if pred_result:
                predictions.append(pred_result)
        
        # Send prediction summary
        if predictions:
            await self.send_context_update(
                update_type="predictions_generated",
                data={
                    "prediction_count": len(predictions),
                    "prediction_ids": [p.prediction_id for p in predictions],
                    "average_confidence": sum(p.confidence for p in predictions) / len(predictions),
                    "context_enhanced": True
                },
                priority=ContextPriority.MEDIUM
            )
        
        return {
            "prediction_processing": True,
            "predictions_generated": len(predictions),
            "prediction_query": prediction_query,
            "context_stability": prediction_context.get("stability_score", 0.5)
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze prediction patterns and accuracy"""
        # Get prediction performance metrics
        performance = await self.original_engine.get_performance_metrics()
        
        # Analyze context-specific accuracy
        context_accuracy = await self._analyze_context_specific_accuracy()
        
        # Identify prediction patterns
        prediction_patterns = await self._analyze_prediction_patterns()
        
        # Evaluate cross-module prediction success
        cross_module_analysis = await self._analyze_cross_module_predictions()
        
        # Generate prediction insights
        prediction_insights = await self._generate_prediction_insights(
            performance, context_accuracy, prediction_patterns
        )
        
        # Identify areas for improvement
        improvement_areas = await self._identify_improvement_areas(context)
        
        return {
            "overall_performance": performance,
            "context_specific_accuracy": context_accuracy,
            "prediction_patterns": prediction_patterns,
            "cross_module_analysis": cross_module_analysis,
            "prediction_insights": prediction_insights,
            "improvement_areas": improvement_areas,
            "context_enhancement_ratio": self._calculate_context_enhancement_ratio()
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize prediction insights for decision making"""
        # Get active predictions relevant to current context
        relevant_predictions = await self._get_contextually_relevant_predictions(context)
        
        # Generate prediction synthesis
        prediction_synthesis = {
            "featured_predictions": [],
            "confidence_summary": "",
            "recommended_actions": [],
            "uncertainty_areas": [],
            "meta_predictions": []
        }
        
        # Process featured predictions
        for pred_id, pred_data in relevant_predictions[:3]:  # Top 3 predictions
            prediction = self.original_engine.predictions.get(pred_id)
            if prediction:
                featured = {
                    "prediction_id": pred_id,
                    "summary": self._summarize_prediction(prediction),
                    "confidence": prediction["result"]["confidence"],
                    "horizon": prediction["result"]["prediction_horizon"],
                    "context_factors": pred_data.get("context_factors", [])
                }
                prediction_synthesis["featured_predictions"].append(featured)
        
        # Generate confidence summary
        if prediction_synthesis["featured_predictions"]:
            avg_confidence = sum(p["confidence"] for p in prediction_synthesis["featured_predictions"]) / len(prediction_synthesis["featured_predictions"])
            prediction_synthesis["confidence_summary"] = self._generate_confidence_summary(avg_confidence, context)
        
        # Add recommended actions based on predictions
        messages = await self.get_cross_module_messages()
        recommendations = await self._generate_action_recommendations(relevant_predictions, context, messages)
        prediction_synthesis["recommended_actions"] = recommendations
        
        # Identify uncertainty areas
        uncertainty = await self._identify_uncertainty_areas(context, relevant_predictions)
        prediction_synthesis["uncertainty_areas"] = uncertainty
        
        # Generate meta-predictions
        meta_predictions = await self._generate_meta_predictions(context)
        prediction_synthesis["meta_predictions"] = meta_predictions
        
        # Send synthesis update
        await self.send_context_update(
            update_type="prediction_synthesis_complete",
            data={
                "synthesis": prediction_synthesis,
                "predictions_synthesized": len(prediction_synthesis["featured_predictions"]),
                "has_recommendations": len(recommendations) > 0,
                "uncertainty_level": len(uncertainty) / 5.0  # Normalize to 0-1
            },
            priority=ContextPriority.MEDIUM
        )
        
        return prediction_synthesis
    
    # Advanced prediction methods
    
    async def _analyze_context_stability(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze stability of context for prediction confidence"""
        stability_factors = {
            "emotional_stability": 0.5,
            "topic_consistency": 0.5,
            "interaction_pattern_stability": 0.5,
            "temporal_consistency": 0.5,
            "cross_module_coherence": 0.5
        }
        
        # Emotional stability
        if context.emotional_state:
            # Check if emotions are stable or volatile
            if hasattr(self, "emotional_history"):
                recent_emotions = self.emotional_history[-5:]
                if len(recent_emotions) >= 2:
                    # Calculate emotional volatility
                    emotion_changes = 0
                    for i in range(1, len(recent_emotions)):
                        if recent_emotions[i] != recent_emotions[i-1]:
                            emotion_changes += 1
                    stability_factors["emotional_stability"] = 1.0 - (emotion_changes / (len(recent_emotions) - 1))
        
        # Topic consistency
        if context.session_context and "topic_history" in context.session_context:
            topic_history = context.session_context["topic_history"]
            if len(topic_history) >= 2:
                # Check topic changes
                topic_changes = sum(1 for i in range(1, len(topic_history)) if topic_history[i] != topic_history[i-1])
                stability_factors["topic_consistency"] = 1.0 - (topic_changes / (len(topic_history) - 1))
        
        # Temporal consistency
        if context.temporal_context:
            # Stable time of day, consistent interaction patterns
            stability_factors["temporal_consistency"] = 0.7  # Default good stability
        
        # Cross-module coherence
        coherence_score = await self._calculate_cross_module_coherence(context)
        stability_factors["cross_module_coherence"] = coherence_score
        
        # Calculate overall stability
        overall_stability = sum(stability_factors.values()) / len(stability_factors)
        
        # Calculate confidence modifier
        confidence_modifier = 0.0
        if overall_stability > self.stability_threshold:
            confidence_modifier = 0.2
        elif overall_stability < 0.3:
            confidence_modifier = -0.2
        
        return {
            "overall_stability": overall_stability,
            "stability_factors": stability_factors,
            "confidence_modifier": confidence_modifier,
            "stability_assessment": self._assess_stability_level(overall_stability)
        }
    
    async def _identify_prediction_opportunities(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify opportunities for contextual predictions"""
        opportunities = []
        
        # Emotional trajectory predictions
        if context.emotional_state:
            current_emotion = context.emotional_state.get("primary_emotion", {}).get("name")
            if current_emotion:
                opportunities.append({
                    "type": "emotional_trajectory",
                    "query_type": "emotional",
                    "context_factors": ["current_emotion", "emotional_history"],
                    "confidence_boost": 0.1
                })
        
        # Goal outcome predictions
        if context.goal_context and context.goal_context.get("active_goals"):
            opportunities.append({
                "type": "goal_outcomes",
                "query_type": "goal",
                "context_factors": ["active_goals", "past_performance"],
                "confidence_boost": 0.15
            })
        
        # Interaction pattern predictions
        if context.session_context and context.session_context.get("interaction_count", 0) > 5:
            opportunities.append({
                "type": "interaction_patterns",
                "query_type": "behavioral",
                "context_factors": ["interaction_history", "communication_style"],
                "confidence_boost": 0.1
            })
        
        # Relationship development predictions
        if context.relationship_context:
            trust_level = context.relationship_context.get("trust_level", 0)
            if trust_level > 0.3:
                opportunities.append({
                    "type": "relationship_trajectory",
                    "query_type": "relationship",
                    "context_factors": ["trust_level", "interaction_quality"],
                    "confidence_boost": 0.2
            })
        
        return opportunities
    
    async def _analyze_cross_module_potential(self, context: SharedContext) -> float:
        """Analyze potential for cross-module coherent predictions"""
        potential_score = 0.0
        coherence_factors = []
        
        # Check emotion-memory coherence
        if context.emotional_state and context.memory_context:
            if context.memory_context.get("emotional_memories"):
                potential_score += 0.2
                coherence_factors.append("emotion_memory_alignment")
        
        # Check goal-action coherence
        if context.goal_context and context.action_context:
            active_goals = context.goal_context.get("active_goals", [])
            pending_actions = context.action_context.get("pending_actions", [])
            
            if active_goals and pending_actions:
                # Check if actions align with goals
                aligned_actions = any(
                    action.get("goal_id") in [g.get("id") for g in active_goals]
                    for action in pending_actions
                )
                if aligned_actions:
                    potential_score += 0.25
                    coherence_factors.append("goal_action_alignment")
        
        # Check temporal-pattern coherence
        if context.temporal_context and hasattr(context, "pattern_context"):
            potential_score += 0.15
            coherence_factors.append("temporal_pattern_alignment")
        
        # Store for later use
        if coherence_factors:
            self.cross_module_predictions.append({
                "timestamp": datetime.now().isoformat(),
                "coherence_factors": coherence_factors,
                "potential_score": potential_score
            })
        
        return min(1.0, potential_score)
    
    async def _generate_contextual_prediction(self, opportunity: Dict[str, Any], context: SharedContext):
        """Generate prediction based on contextual opportunity"""
        try:
            # Build specialized context for this opportunity
            specialized_context = await self._build_specialized_context(opportunity, context)
            
            # Extract relevant history
            history = await self._extract_specialized_history(opportunity["type"], context)
            
            # Generate prediction with context boost
            prediction = await self.original_engine.generate_prediction(
                context=specialized_context,
                history=history,
                query_type=opportunity["query_type"]
            )
            
            if prediction:
                # Apply context confidence boost
                prediction.confidence = min(1.0, prediction.confidence + opportunity.get("confidence_boost", 0))
                
                # Track as context-enhanced
                self.context_enhanced_predictions[prediction.prediction_id] = {
                    "opportunity_type": opportunity["type"],
                    "context_factors": opportunity["context_factors"],
                    "confidence_boost": opportunity.get("confidence_boost", 0),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Send notification
                await self.send_context_update(
                    update_type="contextual_prediction_generated",
                    data={
                        "prediction_id": prediction.prediction_id,
                        "prediction_type": opportunity["type"],
                        "confidence": prediction.confidence
                    },
                    priority=ContextPriority.LOW
                )
                
        except Exception as e:
            logger.error(f"Error generating contextual prediction: {e}")
    
    async def _incorporate_pattern_into_predictions(self, pattern_type: str, pattern_data: Dict[str, Any]):
        """Incorporate detected patterns into prediction models"""
        # Update prediction priors based on pattern
        if pattern_type == "behavioral":
            # Update behavioral prediction priors
            pattern_elements = pattern_data.get("elements", [])
            for element in pattern_elements:
                key = f"behavioral_{element}"
                if key not in self.original_engine.prediction_priors["conversation_continuations"]:
                    self.original_engine.prediction_priors["conversation_continuations"][key] = {}
                
                self.original_engine.prediction_priors["conversation_continuations"][key]["pattern_strength"] = pattern_data.get("strength", 0.5)
        
        elif pattern_type == "emotional":
            # Update emotional trajectory priors
            if "emotional_trajectories" in self.original_engine.prediction_priors:
                self.original_engine.prediction_priors["emotional_trajectories"]["latest_pattern"] = pattern_data
        
        # Trigger re-evaluation of active predictions
        await self._reevaluate_active_predictions_with_pattern(pattern_type, pattern_data)
    
    async def _update_emotional_predictions(self, emotional_data: Dict[str, Any]):
        """Update predictions based on emotional state changes"""
        # Find active emotional predictions
        emotional_predictions = [
            (pred_id, pred) for pred_id, pred in self.original_engine.predictions.items()
            if not pred["evaluated"] and pred["result"].get("predicted_emotional_state")
        ]
        
        for pred_id, pred in emotional_predictions:
            # Adjust confidence based on emotional volatility
            emotion_intensity = emotional_data.get("intensity", 0.5)
            if emotion_intensity > 0.8:
                # High intensity emotions are less predictable
                pred["result"]["confidence"] *= 0.8
            
            # Update predicted emotional state if significant change
            dominant_emotion = emotional_data.get("dominant_emotion")
            if dominant_emotion:
                emotion_name, intensity = dominant_emotion
                # Adjust prediction toward current emotional trend
                if pred["result"]["predicted_emotional_state"]:
                    pred["result"]["predicted_emotional_state"][emotion_name] = intensity * 0.7 + pred["result"]["predicted_emotional_state"].get(emotion_name, 0.5) * 0.3
    
    async def _predict_goal_trajectory(self, execution_results: List[Dict[str, Any]], goal_data: Dict[str, Any]):
        """Predict future goal outcomes based on current progress"""
        # Calculate success rate
        success_rate = sum(1 for r in execution_results if r.get("success", False)) / len(execution_results) if execution_results else 0
        
        # Build goal prediction context
        goal_context = {
            "current_success_rate": success_rate,
            "execution_count": len(execution_results),
            "goal_metadata": goal_data
        }
        
        # Predict goal completion likelihood
        completion_prediction = {
            "prediction_type": "goal_completion",
            "likelihood": success_rate * 0.7 + 0.3,  # Baseline optimism
            "estimated_time": self._estimate_goal_completion_time(success_rate, goal_data),
            "confidence": 0.6 + success_rate * 0.3
        }
        
        # Store as informal prediction
        pred_id = f"goal_pred_{datetime.now().timestamp()}"
        self.context_enhanced_predictions[pred_id] = {
            "type": "goal_trajectory",
            "prediction": completion_prediction,
            "context": goal_context,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _adjust_predictions_for_trend(self, trend_type: str, trend_direction: str, trend_data: Dict[str, Any]):
        """Adjust predictions based on detected behavioral trends"""
        # Find predictions that might be affected by the trend
        affected_predictions = []
        
        for pred_id, pred in self.original_engine.predictions.items():
            if not pred["evaluated"]:
                # Check if prediction type matches trend
                if trend_type == "emotional" and pred["result"].get("predicted_emotional_state"):
                    affected_predictions.append(pred_id)
                elif trend_type == "behavioral" and pred["result"].get("predicted_input"):
                    affected_predictions.append(pred_id)
        
        # Adjust affected predictions
        for pred_id in affected_predictions:
            pred = self.original_engine.predictions[pred_id]
            
            # Adjust confidence based on trend consistency
            if trend_direction in ["increasing", "decreasing"]:
                # Clear trends increase confidence
                pred["result"]["confidence"] = min(1.0, pred["result"]["confidence"] + 0.1)
            else:
                # Volatile trends decrease confidence
                pred["result"]["confidence"] *= 0.9
            
            # Note the adjustment
            if pred_id not in self.context_enhanced_predictions:
                self.context_enhanced_predictions[pred_id] = {}
            
            self.context_enhanced_predictions[pred_id]["trend_adjusted"] = {
                "trend_type": trend_type,
                "trend_direction": trend_direction,
                "adjustment_time": datetime.now().isoformat()
            }
    
    async def _process_prediction_feedback(self, prediction_id: str, accuracy: float, feedback_data: Dict[str, Any]):
        """Process feedback on prediction accuracy"""
        # Update context-specific accuracy tracking
        if prediction_id in self.context_enhanced_predictions:
            context_info = self.context_enhanced_predictions[prediction_id]
            context_type = context_info.get("opportunity_type", "general")
            
            if context_type not in self.prediction_accuracy_by_context:
                self.prediction_accuracy_by_context[context_type] = []
            
            self.prediction_accuracy_by_context[context_type].append({
                "prediction_id": prediction_id,
                "accuracy": accuracy,
                "timestamp": datetime.now().isoformat(),
                "context_factors": context_info.get("context_factors", [])
            })
        
        # Use original engine's evaluation if prediction exists there
        if prediction_id in self.original_engine.predictions:
            await self.original_engine.evaluate_prediction(
                prediction_id,
                feedback_data
            )
    
    async def _adjust_prediction_confidence(self, new_stability: float):
        """Adjust confidence of active predictions based on context stability"""
        # Find active (unevaluated) predictions
        active_predictions = [
            (pred_id, pred) for pred_id, pred in self.original_engine.predictions.items()
            if not pred["evaluated"]
        ]
        
        for pred_id, pred in active_predictions:
            # Calculate confidence adjustment
            current_confidence = pred["result"]["confidence"]
            
            if new_stability > self.stability_threshold:
                # High stability increases confidence
                adjustment = min(0.1, (new_stability - self.stability_threshold) * 0.3)
                pred["result"]["confidence"] = min(1.0, current_confidence + adjustment)
            elif new_stability < 0.3:
                # Low stability decreases confidence
                adjustment = (0.3 - new_stability) * 0.3
                pred["result"]["confidence"] = max(0.1, current_confidence - adjustment)
    
    async def _generate_temporal_predictions(self, pattern: str, temporal_data: Dict[str, Any]):
        """Generate predictions based on temporal patterns"""
        # Extract pattern details
        pattern_type = temporal_data.get("pattern_type", "unknown")
        pattern_strength = temporal_data.get("strength", 0.5)
        
        if pattern_type == "daily_cycle":
            # Predict based on time of day patterns
            current_hour = datetime.now().hour
            predicted_state = self._predict_from_daily_pattern(current_hour, temporal_data)
            
            if predicted_state:
                # Create informal prediction
                pred_id = f"temporal_pred_{datetime.now().timestamp()}"
                self.context_enhanced_predictions[pred_id] = {
                    "type": "temporal_pattern",
                    "prediction": predicted_state,
                    "pattern": pattern,
                    "confidence": pattern_strength,
                    "timestamp": datetime.now().isoformat()
                }
        
        elif pattern_type == "weekly_cycle":
            # Predict based on day of week patterns
            current_day = datetime.now().weekday()
            predicted_behavior = self._predict_from_weekly_pattern(current_day, temporal_data)
            
            if predicted_behavior:
                pred_id = f"weekly_pred_{datetime.now().timestamp()}"
                self.context_enhanced_predictions[pred_id] = {
                    "type": "weekly_pattern",
                    "prediction": predicted_behavior,
                    "pattern": pattern,
                    "confidence": pattern_strength * 0.8,  # Weekly patterns are less reliable
                    "timestamp": datetime.now().isoformat()
                }
    
    async def _predict_causal_outcomes(self, causal_chain: List[Dict[str, Any]], causal_data: Dict[str, Any]):
        """Predict outcomes based on causal chains"""
        if not causal_chain:
            return
        
        # Analyze the chain
        chain_strength = causal_data.get("chain_strength", 0.5)
        
        # Predict the next likely element in the chain
        last_element = causal_chain[-1]
        predicted_next = None
        
        # Simple causal prediction based on patterns
        if last_element.get("type") == "action":
            # Action likely leads to reaction
            predicted_next = {
                "type": "reaction",
                "likely_outcome": "user_response",
                "confidence": chain_strength
            }
        elif last_element.get("type") == "emotion":
            # Emotion likely leads to expression
            predicted_next = {
                "type": "expression",
                "likely_outcome": "emotional_communication",
                "confidence": chain_strength * 0.9
            }
        
        if predicted_next:
            # Store causal prediction
            pred_id = f"causal_pred_{datetime.now().timestamp()}"
            self.context_enhanced_predictions[pred_id] = {
                "type": "causal_chain",
                "prediction": predicted_next,
                "chain": causal_chain,
                "chain_strength": chain_strength,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _extract_prediction_query(self, user_input: str) -> Dict[str, Any]:
        """Extract prediction query from user input"""
        query = {
            "requires_prediction": False,
            "query_type": None,
            "explicit_request": False
        }
        
        # Check for explicit prediction requests
        prediction_keywords = ["predict", "forecast", "expect", "anticipate", "will", "going to", "likely"]
        if any(keyword in user_input.lower() for keyword in prediction_keywords):
            query["requires_prediction"] = True
            query["explicit_request"] = True
            
            # Determine query type
            if any(word in user_input.lower() for word in ["feel", "emotion", "mood"]):
                query["query_type"] = "emotional"
            elif any(word in user_input.lower() for word in ["say", "respond", "answer"]):
                query["query_type"] = "response"
            elif any(word in user_input.lower() for word in ["do", "action", "behavior"]):
                query["query_type"] = "behavioral"
            else:
                query["query_type"] = "general"
        
        # Check for implicit prediction needs
        question_words = ["what", "how", "when", "where", "why"]
        future_indicators = ["next", "then", "after", "later", "tomorrow", "future"]
        
        if any(word in user_input.lower() for word in question_words) and \
           any(word in user_input.lower() for word in future_indicators):
            query["requires_prediction"] = True
            query["query_type"] = "contextual"
        
        return query
    
    async def _build_prediction_context(self, context: SharedContext, messages: List[Any]) -> Dict[str, Any]:
        """Build comprehensive context for prediction"""
        prediction_context = {
            "user_id": context.user_id,
            "conversation_id": context.conversation_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add emotional context
        if context.emotional_state:
            prediction_context["emotional_state"] = context.emotional_state
        
        # Add goal context
        if context.goal_context:
            prediction_context["active_goals"] = context.goal_context.get("active_goals", [])
        
        # Add relationship context
        if context.relationship_context:
            prediction_context["relationship_state"] = context.relationship_context
        
        # Add temporal context
        if context.temporal_context:
            prediction_context["temporal_factors"] = context.temporal_context
        
        # Add stability score
        stability = await self._analyze_context_stability(context)
        prediction_context["stability_score"] = stability["overall_stability"]
        
        # Add cross-module insights
        cross_module_factors = []
        for message in messages:
            if message.get("module_name") and message.get("update_type"):
                cross_module_factors.append({
                    "module": message["module_name"],
                    "update": message["update_type"]
                })
        
        prediction_context["cross_module_factors"] = cross_module_factors
        
        return prediction_context
    
    async def _extract_relevant_history(self, context: SharedContext, messages: List[Any]) -> List[Dict[str, Any]]:
        """Extract history relevant for predictions"""
        history = []
        
        # Add conversation history
        if context.session_context and "conversation_history" in context.session_context:
            conv_history = context.session_context["conversation_history"]
            for entry in conv_history[-10:]:  # Last 10 entries
                history.append({
                    "timestamp": entry.get("timestamp", datetime.now().isoformat()),
                    "role": entry.get("role", "unknown"),
                    "text": entry.get("text", ""),
                    "emotional_state": entry.get("emotional_state"),
                    "topic": entry.get("topic")
                })
        
        # Add relevant cross-module events
        for message in messages[-5:]:  # Recent messages
            if message.get("update_type") in ["action_executed", "goal_completed", "emotional_shift"]:
                history.append({
                    "timestamp": message.get("timestamp", datetime.now().isoformat()),
                    "role": "system",
                    "text": f"System event: {message['update_type']}",
                    "event_data": message.get("data", {})
                })
        
        # Sort by timestamp
        history.sort(key=lambda x: x["timestamp"])
        
        return history
    
    async def _check_implicit_prediction_needs(self, context: SharedContext, messages: List[Any]) -> List[Dict[str, Any]]:
        """Check for implicit needs for predictions"""
        implicit_needs = []
        
        # Check for decision points
        if any(m.get("update_type") == "decision_required" for m in messages):
            implicit_needs.append({
                "type": "decision_support",
                "query_type": "outcome",
                "reason": "upcoming_decision"
            })
        
        # Check for uncertainty expressions
        if context.emotional_state:
            primary_emotion = context.emotional_state.get("primary_emotion", {}).get("name")
            if primary_emotion in ["Confusion", "Uncertainty", "Anxiety"]:
                implicit_needs.append({
                    "type": "uncertainty_resolution",
                    "query_type": "clarification",
                    "reason": "emotional_uncertainty"
                })
        
        # Check for goal deadlines
        if context.goal_context:
            for goal in context.goal_context.get("active_goals", []):
                if goal.get("deadline"):
                    deadline = datetime.fromisoformat(goal["deadline"])
                    if (deadline - datetime.now()).days < 7:
                        implicit_needs.append({
                            "type": "goal_deadline",
                            "query_type": "goal",
                            "reason": "approaching_deadline",
                            "goal_id": goal.get("id")
                        })
                        break
        
        return implicit_needs
    
    async def _generate_implicit_prediction(self, implicit_need: Dict[str, Any], context: SharedContext) -> Optional[Any]:
        """Generate prediction for implicit need"""
        # Build specialized context
        specialized_context = {
            "implicit_need": implicit_need["type"],
            "reason": implicit_need["reason"]
        }
        
        # Add relevant context based on need type
        if implicit_need["type"] == "decision_support":
            specialized_context["decision_factors"] = await self._extract_decision_factors(context)
        elif implicit_need["type"] == "uncertainty_resolution":
            specialized_context["uncertainty_sources"] = await self._identify_uncertainty_sources(context)
        elif implicit_need["type"] == "goal_deadline":
            specialized_context["goal_id"] = implicit_need.get("goal_id")
        
        # Generate prediction
        try:
            prediction = await self.original_engine.generate_prediction(
                context=specialized_context,
                history=[],  # Minimal history for implicit predictions
                query_type=implicit_need["query_type"]
            )
            
            if prediction:
                # Mark as implicit
                self.context_enhanced_predictions[prediction.prediction_id] = {
                    "implicit": True,
                    "need_type": implicit_need["type"],
                    "reason": implicit_need["reason"],
                    "timestamp": datetime.now().isoformat()
                }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating implicit prediction: {e}")
            return None
    
    async def _build_specialized_context(self, opportunity: Dict[str, Any], context: SharedContext) -> Dict[str, Any]:
        """Build specialized context for specific prediction opportunity"""
        specialized = {
            "prediction_type": opportunity["type"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Add factors specific to opportunity type
        for factor in opportunity.get("context_factors", []):
            if factor == "current_emotion" and context.emotional_state:
                specialized["current_emotion"] = context.emotional_state
            elif factor == "emotional_history":
                specialized["emotional_history"] = getattr(self, "emotional_history", [])
            elif factor == "active_goals" and context.goal_context:
                specialized["active_goals"] = context.goal_context.get("active_goals", [])
            elif factor == "trust_level" and context.relationship_context:
                specialized["trust_level"] = context.relationship_context.get("trust_level", 0.5)
        
        return specialized
    
    async def _extract_specialized_history(self, opportunity_type: str, context: SharedContext) -> List[Dict[str, Any]]:
        """Extract history specialized for opportunity type"""
        history = []
        
        if opportunity_type == "emotional_trajectory":
            # Focus on emotional history
            if hasattr(self, "emotional_history"):
                for emotion_record in self.emotional_history[-10:]:
                    history.append({
                        "timestamp": emotion_record.get("timestamp", datetime.now().isoformat()),
                        "emotional_state": emotion_record.get("state"),
                        "trigger": emotion_record.get("trigger")
                    })
        
        elif opportunity_type == "goal_outcomes":
            # Focus on goal-related history
            if context.action_context:
                for action in context.action_context.get("recent_actions", [])[-10:]:
                    if action.get("goal_related"):
                        history.append({
                            "timestamp": action.get("timestamp", datetime.now().isoformat()),
                            "action": action.get("name"),
                            "goal_id": action.get("goal_id"),
                            "success": action.get("success", False)
                        })
        
        elif opportunity_type == "interaction_patterns":
            # Use conversation history
            if context.session_context:
                return context.session_context.get("conversation_history", [])[-15:]
        
        return history
    
    async def _reevaluate_active_predictions_with_pattern(self, pattern_type: str, pattern_data: Dict[str, Any]):
        """Re-evaluate active predictions with new pattern information"""
        for pred_id, pred in self.original_engine.predictions.items():
            if not pred["evaluated"]:
                # Check if prediction is affected by pattern
                affected = False
                
                if pattern_type == "behavioral" and pred["result"].get("predicted_input"):
                    affected = True
                elif pattern_type == "emotional" and pred["result"].get("predicted_emotional_state"):
                    affected = True
                
                if affected:
                    # Adjust confidence based on pattern alignment
                    pattern_strength = pattern_data.get("strength", 0.5)
                    
                    # If pattern supports prediction, increase confidence
                    # This is simplified - real implementation would check actual alignment
                    confidence_adjustment = pattern_strength * 0.1
                    
                    pred["result"]["confidence"] = min(1.0, pred["result"]["confidence"] + confidence_adjustment)
    
    def _estimate_goal_completion_time(self, success_rate: float, goal_data: Dict[str, Any]) -> str:
        """Estimate time to goal completion based on progress"""
        # Simple estimation based on success rate
        if success_rate > 0.8:
            return "within_days"
        elif success_rate > 0.5:
            return "within_week"
        elif success_rate > 0.2:
            return "within_month"
        else:
            return "uncertain"
    
    def _predict_from_daily_pattern(self, current_hour: int, temporal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict based on daily patterns"""
        # Simple time-based predictions
        if 6 <= current_hour < 12:
            return {
                "period": "morning",
                "likely_state": "focused",
                "energy_level": "increasing"
            }
        elif 12 <= current_hour < 17:
            return {
                "period": "afternoon",
                "likely_state": "steady",
                "energy_level": "moderate"
            }
        elif 17 <= current_hour < 22:
            return {
                "period": "evening",
                "likely_state": "relaxed",
                "energy_level": "decreasing"
            }
        else:
            return {
                "period": "night",
                "likely_state": "contemplative",
                "energy_level": "low"
            }
    
    def _predict_from_weekly_pattern(self, current_day: int, temporal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict based on weekly patterns"""
        # Simple day-based predictions
        if current_day < 5:  # Weekday
            return {
                "day_type": "weekday",
                "likely_focus": "productive",
                "interaction_style": "efficient"
            }
        else:  # Weekend
            return {
                "day_type": "weekend",
                "likely_focus": "relaxed",
                "interaction_style": "exploratory"
            }
    
    async def _calculate_cross_module_coherence(self, context: SharedContext) -> float:
        """Calculate coherence across modules"""
        coherence_score = 0.5  # Base score
        
        # Check emotion-action coherence
        if context.emotional_state and context.action_context:
            emotion = context.emotional_state.get("primary_emotion", {}).get("name")
            recent_actions = context.action_context.get("recent_actions", [])
            
            # Simple coherence check
            if emotion == "Joy" and any(a.get("type") == "positive" for a in recent_actions):
                coherence_score += 0.2
            elif emotion == "Frustration" and any(a.get("type") == "corrective" for a in recent_actions):
                coherence_score += 0.2
        
        # Check goal-need coherence
        if context.goal_context and context.needs_context:
            goals_address_needs = any(
                g.get("associated_need") in [n.get("name") for n in context.needs_context.get("active_needs", [])]
                for g in context.goal_context.get("active_goals", [])
            )
            if goals_address_needs:
                coherence_score += 0.2
        
        return min(1.0, coherence_score)
    
    def _assess_stability_level(self, stability_score: float) -> str:
        """Assess stability level from score"""
        if stability_score > 0.8:
            return "highly_stable"
        elif stability_score > 0.6:
            return "stable"
        elif stability_score > 0.4:
            return "moderately_stable"
        elif stability_score > 0.2:
            return "unstable"
        else:
            return "highly_unstable"
    
    def _summarize_context(self, context: SharedContext) -> str:
        """Create summary of context for tracking"""
        elements = []
        
        if context.emotional_state:
            emotion = context.emotional_state.get("primary_emotion", {}).get("name")
            if emotion:
                elements.append(f"emotion:{emotion}")
        
        if context.goal_context:
            goal_count = len(context.goal_context.get("active_goals", []))
            elements.append(f"goals:{goal_count}")
        
        if context.temporal_context:
            time_of_day = context.temporal_context.get("time_of_day")
            if time_of_day:
                elements.append(f"time:{time_of_day}")
        
        return "_".join(elements) if elements else "general_context"
    
    async def _analyze_context_specific_accuracy(self) -> Dict[str, Dict[str, float]]:
        """Analyze prediction accuracy by context type"""
        accuracy_by_type = {}
        
        for context_type, predictions in self.prediction_accuracy_by_context.items():
            if predictions:
                accuracies = [p["accuracy"] for p in predictions]
                accuracy_by_type[context_type] = {
                    "average_accuracy": sum(accuracies) / len(accuracies),
                    "sample_size": len(accuracies),
                    "best_accuracy": max(accuracies),
                    "worst_accuracy": min(accuracies)
                }
        
        return accuracy_by_type
    
    async def _analyze_prediction_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in predictions"""
        patterns = {
            "common_prediction_types": {},
            "confidence_distribution": {},
            "horizon_preferences": {},
            "success_patterns": []
        }
        
        # Analyze prediction types
        for pred_id, enhanced_data in self.context_enhanced_predictions.items():
            pred_type = enhanced_data.get("type", enhanced_data.get("opportunity_type", "unknown"))
            patterns["common_prediction_types"][pred_type] = patterns["common_prediction_types"].get(pred_type, 0) + 1
        
        # Analyze confidence distribution
        for pred_id, pred in self.original_engine.predictions.items():
            confidence = pred["result"]["confidence"]
            confidence_bucket = f"{int(confidence * 10) / 10:.1f}"
            patterns["confidence_distribution"][confidence_bucket] = patterns["confidence_distribution"].get(confidence_bucket, 0) + 1
        
        # Analyze horizon preferences
        for pred_id, pred in self.original_engine.predictions.items():
            horizon = pred["result"]["prediction_horizon"]
            patterns["horizon_preferences"][horizon] = patterns["horizon_preferences"].get(horizon, 0) + 1
        
        # Identify success patterns
        if self.prediction_accuracy_by_context:
            for context_type, predictions in self.prediction_accuracy_by_context.items():
                high_accuracy_preds = [p for p in predictions if p["accuracy"] > 0.8]
                if len(high_accuracy_preds) > 2:
                    patterns["success_patterns"].append({
                        "context_type": context_type,
                        "success_rate": len(high_accuracy_preds) / len(predictions),
                        "common_factors": self._extract_common_factors(high_accuracy_preds)
                    })
        
        return patterns
    
    async def _analyze_cross_module_predictions(self) -> Dict[str, Any]:
        """Analyze success of cross-module predictions"""
        analysis = {
            "total_cross_module": len(self.cross_module_predictions),
            "average_coherence": 0.0,
            "coherence_factors": {},
            "success_correlation": 0.0
        }
        
        if self.cross_module_predictions:
            # Calculate average coherence
            coherence_scores = [p["potential_score"] for p in self.cross_module_predictions]
            analysis["average_coherence"] = sum(coherence_scores) / len(coherence_scores)
            
            # Count coherence factors
            for pred in self.cross_module_predictions:
                for factor in pred["coherence_factors"]:
                    analysis["coherence_factors"][factor] = analysis["coherence_factors"].get(factor, 0) + 1
            
            # TODO: Correlate with actual prediction success
            # This would require tracking which predictions had cross-module support
        
        return analysis
    
    async def _generate_prediction_insights(self, performance: Dict[str, Any], 
                                          context_accuracy: Dict[str, Dict[str, float]], 
                                          patterns: Dict[str, Any]) -> List[str]:
        """Generate insights from prediction analysis"""
        insights = []
        
        # Performance insights
        if performance["accuracy"] > 0.7:
            insights.append("Prediction accuracy is high, indicating good model performance")
        elif performance["accuracy"] < 0.4:
            insights.append("Prediction accuracy needs improvement - consider more context factors")
        
        # Context-specific insights
        if context_accuracy:
            best_context = max(context_accuracy.items(), key=lambda x: x[1]["average_accuracy"])
            insights.append(f"Predictions are most accurate for {best_context[0]} contexts")
        
        # Pattern insights
        if patterns["common_prediction_types"]:
            most_common = max(patterns["common_prediction_types"].items(), key=lambda x: x[1])
            insights.append(f"Most predictions are of type: {most_common[0]}")
        
        # Confidence calibration
        if performance.get("confidence_calibration", 0) > 0.8:
            insights.append("Confidence scores are well-calibrated with actual accuracy")
        elif performance.get("confidence_calibration", 0) < 0.5:
            insights.append("Confidence scores need calibration - tend to be overconfident")
        
        return insights
    
    async def _identify_improvement_areas(self, context: SharedContext) -> List[str]:
        """Identify areas where predictions could be improved"""
        areas = []
        
        # Check for low-accuracy contexts
        context_accuracy = await self._analyze_context_specific_accuracy()
        for context_type, accuracy_data in context_accuracy.items():
            if accuracy_data["average_accuracy"] < 0.5:
                areas.append(f"Improve {context_type} predictions (current accuracy: {accuracy_data['average_accuracy']:.2f})")
        
        # Check for missing prediction types
        all_types = ["emotional", "behavioral", "goal", "relationship", "temporal"]
        used_types = set(self.context_enhanced_predictions.get(pid, {}).get("type") 
                        for pid in self.context_enhanced_predictions)
        
        missing_types = set(all_types) - used_types
        for missing in missing_types:
            areas.append(f"Add {missing} prediction capabilities")
        
        # Check for context stability issues
        if self.context_stability_history:
            recent_stability = [h["stability"] for h in self.context_stability_history[-10:]]
            avg_stability = sum(recent_stability) / len(recent_stability) if recent_stability else 0
            
            if avg_stability < 0.5:
                areas.append("Improve handling of unstable contexts")
        
        return areas
    
    def _calculate_context_enhancement_ratio(self) -> float:
        """Calculate ratio of context-enhanced predictions"""
        if not self.original_engine.predictions:
            return 0.0
        
        enhanced_count = len(self.context_enhanced_predictions)
        total_count = len(self.original_engine.predictions)
        
        return enhanced_count / total_count if total_count > 0 else 0.0
    
    async def _get_contextually_relevant_predictions(self, context: SharedContext) -> List[Tuple[str, Dict[str, Any]]]:
        """Get predictions relevant to current context"""
        relevant = []
        
        for pred_id, pred in self.original_engine.predictions.items():
            if pred["evaluated"]:
                continue
            
            relevance_score = 0.0
            relevance_factors = []
            
            # Check emotional relevance
            if pred["result"].get("predicted_emotional_state") and context.emotional_state:
                relevance_score += 0.3
                relevance_factors.append("emotional_relevance")
            
            # Check temporal relevance
            pred_timestamp = datetime.fromisoformat(pred["timestamp"])
            age_hours = (datetime.now() - pred_timestamp).total_seconds() / 3600
            
            if age_hours < 1:
                relevance_score += 0.3
                relevance_factors.append("recent")
            elif age_hours < 6:
                relevance_score += 0.1
                relevance_factors.append("moderately_recent")
            
            # Check context enhancement
            if pred_id in self.context_enhanced_predictions:
                relevance_score += 0.2
                relevance_factors.append("context_enhanced")
            
            # Check goal relevance
            if context.goal_context and pred["input"].get("context", {}).get("active_goals"):
                relevance_score += 0.2
                relevance_factors.append("goal_relevant")
            
            if relevance_score > 0.3:
                relevant.append((pred_id, {
                    "relevance_score": relevance_score,
                    "context_factors": relevance_factors,
                    "prediction": pred
                }))
        
        # Sort by relevance
        relevant.sort(key=lambda x: x[1]["relevance_score"], reverse=True)
        
        return relevant
    
    def _summarize_prediction(self, prediction: Dict[str, Any]) -> str:
        """Create human-readable summary of prediction"""
        result = prediction["result"]
        
        if result.get("predicted_input"):
            return f"Next input likely: '{result['predicted_input'][:50]}...'"
        elif result.get("predicted_response"):
            return f"Optimal response: '{result['predicted_response'][:50]}...'"
        elif result.get("predicted_emotional_state"):
            emotions = result["predicted_emotional_state"]
            if emotions:
                top_emotion = max(emotions.items(), key=lambda x: x[1])
                return f"Emotional trajectory: {top_emotion[0]} ({top_emotion[1]:.2f})"
        
        return "General prediction available"
    
    def _generate_confidence_summary(self, avg_confidence: float, context: SharedContext) -> str:
        """Generate summary of prediction confidence"""
        stability = self.context_stability_history[-1]["stability"] if self.context_stability_history else 0.5
        
        if avg_confidence > 0.8 and stability > 0.7:
            return "High confidence predictions based on stable context"
        elif avg_confidence > 0.6:
            return "Moderate confidence in predictions with some uncertainty"
        elif stability < 0.4:
            return "Lower confidence due to unstable context"
        else:
            return "Predictions carry significant uncertainty"
    
    async def _generate_action_recommendations(self, predictions: List[Tuple[str, Dict[str, Any]]], 
                                             context: SharedContext, 
                                             messages: List[Any]) -> List[str]:
        """Generate action recommendations based on predictions"""
        recommendations = []
        
        for pred_id, pred_data in predictions[:3]:
            prediction = pred_data["prediction"]
            result = prediction["result"]
            
            # Generate recommendations based on prediction type
            if result.get("predicted_emotional_state"):
                emotions = result["predicted_emotional_state"]
                if emotions:
                    dominant = max(emotions.items(), key=lambda x: x[1])
                    if dominant[0] == "Anxiety" and dominant[1] > 0.6:
                        recommendations.append("Consider calming interactions based on predicted anxiety")
                    elif dominant[0] == "Joy" and dominant[1] > 0.7:
                        recommendations.append("Leverage predicted positive emotional state")
            
            elif result.get("predicted_input"):
                if "question" in result["predicted_input"].lower():
                    recommendations.append("Prepare informative response for likely question")
            
            elif pred_id in self.context_enhanced_predictions:
                enhanced = self.context_enhanced_predictions[pred_id]
                if enhanced.get("type") == "goal_trajectory":
                    recommendations.append("Focus on goal support based on trajectory prediction")
        
        return recommendations[:3]  # Limit recommendations
    
    async def _identify_uncertainty_areas(self, context: SharedContext, 
                                        predictions: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
        """Identify areas of prediction uncertainty"""
        uncertainties = []
        
        # Low confidence predictions
        for pred_id, pred_data in predictions:
            if pred_data["prediction"]["result"]["confidence"] < 0.5:
                horizon = pred_data["prediction"]["result"]["prediction_horizon"]
                uncertainties.append(f"Low confidence in {horizon} predictions")
        
        # Unstable context
        if self.context_stability_history:
            recent_stability = self.context_stability_history[-1]["stability"]
            if recent_stability < 0.4:
                uncertainties.append("Context instability affecting prediction accuracy")
        
        # Missing cross-module data
        if not any(m.get("module_name") == "memory_core" for m in await self.get_cross_module_messages()):
            uncertainties.append("Limited memory context for historical predictions")
        
        # Conflicting patterns
        if hasattr(self, "detected_conflicts"):
            uncertainties.append("Conflicting behavioral patterns detected")
        
        return list(set(uncertainties))[:4]  # Unique uncertainties, max 4
    
    async def _generate_meta_predictions(self, context: SharedContext) -> List[str]:
        """Generate predictions about the prediction process itself"""
        meta_predictions = []
        
        # Predict prediction accuracy trend
        if self.prediction_accuracy_by_context:
            recent_accuracies = []
            for context_type, predictions in self.prediction_accuracy_by_context.items():
                recent = predictions[-5:] if len(predictions) > 5 else predictions
                recent_accuracies.extend([p["accuracy"] for p in recent])
            
            if recent_accuracies:
                avg_recent = sum(recent_accuracies) / len(recent_accuracies)
                if avg_recent > 0.7:
                    meta_predictions.append("Prediction accuracy likely to remain high")
                elif avg_recent < 0.4:
                    meta_predictions.append("Prediction accuracy may need recalibration")
        
        # Predict context stability
        if len(self.context_stability_history) >= 3:
            recent_stability = [h["stability"] for h in self.context_stability_history[-3:]]
            stability_trend = recent_stability[-1] - recent_stability[0]
            
            if stability_trend > 0.2:
                meta_predictions.append("Context becoming more stable - predictions will improve")
            elif stability_trend < -0.2:
                meta_predictions.append("Context becoming less stable - expect prediction variance")
        
        return meta_predictions[:2]  # Limit meta-predictions
    
    async def _extract_decision_factors(self, context: SharedContext) -> List[str]:
        """Extract factors relevant for decision support"""
        factors = []
        
        if context.goal_context:
            factors.append("active_goals")
        
        if context.emotional_state:
            factors.append("emotional_state")
        
        if context.relationship_context:
            factors.append("relationship_dynamics")
        
        if context.temporal_context:
            factors.append("time_constraints")
        
        return factors
    
    async def _identify_uncertainty_sources(self, context: SharedContext) -> List[str]:
        """Identify sources of uncertainty in context"""
        sources = []
        
        # Emotional uncertainty
        if context.emotional_state:
            emotion_blend = context.emotional_state.get("emotion_blend", [])
            if len(emotion_blend) > 2:
                sources.append("complex_emotional_state")
        
        # Goal conflicts
        if context.goal_context:
            goals = context.goal_context.get("active_goals", [])
            if len(goals) > 3:
                sources.append("multiple_competing_goals")
        
        # Context instability
        if self.context_stability_history and self.context_stability_history[-1]["stability"] < 0.4:
            sources.append("unstable_context")
        
        return sources
    
    def _extract_common_factors(self, predictions: List[Dict[str, Any]]) -> List[str]:
        """Extract common factors from successful predictions"""
        if not predictions:
            return []
        
        # Extract all context factors
        all_factors = []
        for pred in predictions:
            factors = pred.get("context_factors", [])
            all_factors.extend(factors)
        
        # Find common ones (appear in at least half)
        factor_counts = {}
        for factor in all_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        threshold = len(predictions) / 2
        common_factors = [f for f, count in factor_counts.items() if count >= threshold]
        
        return common_factors
    
    # Delegate all other methods to the original engine
    def __getattr__(self, name):
        """Delegate any missing methods to the original engine"""
        return getattr(self.original_engine, name)
