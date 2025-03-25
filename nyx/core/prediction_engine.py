# nyx/core/prediction_engine.py

import asyncio
import logging
import datetime
import math
import random
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class PredictionInput(BaseModel):
    context: Dict[str, Any] = Field(..., description="Current context data")
    history: List[Dict[str, Any]] = Field(..., description="Recent interaction history")
    query_type: Optional[str] = Field(None, description="Type of prediction query")

class PredictionResult(BaseModel):
    predicted_input: Optional[str] = Field(None, description="Predicted next user input")
    predicted_response: Optional[str] = Field(None, description="Predicted optimal response")
    predicted_emotional_state: Optional[Dict[str, float]] = Field(None, description="Predicted emotional state")
    confidence: float = Field(..., description="Confidence in prediction (0-1)")
    prediction_horizon: str = Field(..., description="Time horizon for prediction")
    prediction_id: str = Field(..., description="Unique ID for this prediction")

class PredictionEvaluation(BaseModel):
    prediction_id: str = Field(..., description="ID of prediction being evaluated")
    actual_input: Optional[str] = Field(None, description="Actual user input received")
    actual_response: Optional[str] = Field(None, description="Actual response generated")
    prediction_error: float = Field(..., description="Error measurement (0-1, lower is better)")
    error_details: Dict[str, Any] = Field(default_factory=dict, description="Detailed error analysis")

class PredictionEngine:
    """
    Engine for generating predictions about future inputs, optimal responses,
    and expected system states based on current context and history.
    
    Implements a predictive processing approach inspired by neuroscience,
    where the system continuously generates predictions and updates based
    on prediction errors.
    """
    
    def __init__(self):
        # Store predictions and their evaluations
        self.predictions = {}
        self.prediction_evaluations = {}
        
        # Track prediction performance metrics
        self.performance_metrics = {
            "total_predictions": 0,
            "correct_predictions": 0,
            "avg_error": 0.0,
            "confidence_calibration": 0.0
        }
        
        # Prediction parameters
        self.prediction_horizon_options = ["immediate", "short_term", "medium_term", "long_term"]
        self.prediction_types = ["input", "response", "emotional", "contextual"]
        
        # Prior distributions for different prediction scenarios
        self.prediction_priors = {
            "conversation_continuations": {},
            "emotional_trajectories": {},
            "topic_transitions": {}
        }
        
        # Initialize with empty prediction history
        self.prediction_history = []
        self.history_limit = 100
    
    async def generate_prediction(self, 
                              prediction_input: PredictionInput) -> PredictionResult:
        """
        Generate predictions based on current context and history
        
        Args:
            prediction_input: Context and history data for prediction
            
        Returns:
            Prediction result
        """
        # Set prediction horizon based on context
        horizon = self._determine_prediction_horizon(prediction_input)
        
        # Generate unique prediction ID
        prediction_id = f"pred_{datetime.datetime.now().timestamp()}_{horizon}"
        
        # Generate different types of predictions based on context
        predicted_input = await self._predict_next_input(prediction_input, horizon)
        predicted_response = await self._predict_optimal_response(prediction_input, horizon)
        predicted_emotional_state = await self._predict_emotional_state(prediction_input, horizon)
        
        # Calculate confidence based on context stability and history
        confidence = self._calculate_prediction_confidence(prediction_input, horizon)
        
        # Create prediction result
        result = PredictionResult(
            predicted_input=predicted_input,
            predicted_response=predicted_response,
            predicted_emotional_state=predicted_emotional_state,
            confidence=confidence,
            prediction_horizon=horizon,
            prediction_id=prediction_id
        )
        
        # Store the prediction
        self.predictions[prediction_id] = {
            "result": result,
            "input": prediction_input,
            "timestamp": datetime.datetime.now().isoformat(),
            "evaluated": False
        }
        
        # Update metrics
        self.performance_metrics["total_predictions"] += 1
        
        # Add to history
        self.prediction_history.append({
            "prediction_id": prediction_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "context_summary": self._summarize_context(prediction_input.context),
            "confidence": confidence,
            "horizon": horizon
        })
        
        # Trim history if needed
        if len(self.prediction_history) > self.history_limit:
            self.prediction_history = self.prediction_history[-self.history_limit:]
        
        return result
    
    async def evaluate_prediction(self,
                              prediction_id: str,
                              actual_data: Dict[str, Any]) -> PredictionEvaluation:
        """
        Evaluate a previous prediction against actual outcomes
        
        Args:
            prediction_id: ID of the prediction to evaluate
            actual_data: Actual data to compare against the prediction
            
        Returns:
            Evaluation result
        """
        if prediction_id not in self.predictions:
            raise ValueError(f"Prediction {prediction_id} not found")
        
        prediction = self.predictions[prediction_id]
        prediction_result = prediction["result"]
        
        # Extract actual values
        actual_input = actual_data.get("actual_input")
        actual_response = actual_data.get("actual_response")
        actual_emotional_state = actual_data.get("actual_emotional_state")
        
        # Calculate error for each prediction type
        input_error = self._calculate_text_similarity_error(
            prediction_result.predicted_input, actual_input) if actual_input else None
        
        response_error = self._calculate_text_similarity_error(
            prediction_result.predicted_response, actual_response) if actual_response else None
        
        emotional_error = self._calculate_emotional_error(
            prediction_result.predicted_emotional_state, actual_emotional_state) if actual_emotional_state else None
        
        # Calculate overall error
        errors = [e for e in [input_error, response_error, emotional_error] if e is not None]
        overall_error = sum(errors) / len(errors) if errors else 0.0
        
        # Create detailed error analysis
        error_details = {
            "input_error": input_error,
            "response_error": response_error,
            "emotional_error": emotional_error,
            "confidence": prediction_result.confidence,
            "horizon": prediction_result.prediction_horizon
        }
        
        # Create evaluation result
        evaluation = PredictionEvaluation(
            prediction_id=prediction_id,
            actual_input=actual_input,
            actual_response=actual_response,
            prediction_error=overall_error,
            error_details=error_details
        )
        
        # Store the evaluation
        self.prediction_evaluations[prediction_id] = evaluation
        
        # Mark prediction as evaluated
        self.predictions[prediction_id]["evaluated"] = True
        
        # Update performance metrics
        correct_threshold = 0.3  # Error threshold for "correct" prediction
        if overall_error < correct_threshold:
            self.performance_metrics["correct_predictions"] += 1
        
        # Update average error using exponential moving average
        alpha = 0.1
        self.performance_metrics["avg_error"] = (
            (1 - alpha) * self.performance_metrics["avg_error"] + 
            alpha * overall_error
        )
        
        # Update confidence calibration
        # (how well confidence predicts accuracy)
        expected_error = 1.0 - prediction_result.confidence
        calibration_error = abs(overall_error - expected_error)
        self.performance_metrics["confidence_calibration"] = (
            (1 - alpha) * self.performance_metrics["confidence_calibration"] + 
            alpha * (1.0 - calibration_error)
        )
        
        return evaluation
    
    # Private helper methods
    
    def _determine_prediction_horizon(self, prediction_input: PredictionInput) -> str:
        """Determine appropriate prediction horizon based on context"""
        # Default to immediate
        horizon = "immediate"
        
        # Check if query type specifies a horizon
        if prediction_input.query_type in self.prediction_horizon_options:
            return prediction_input.query_type
        
        # If history is substantial, consider longer horizons
        if len(prediction_input.history) > 10:
            # Check context for indicators of longer-term planning
            if "planning" in prediction_input.context:
                horizon = "medium_term"
            elif "long_term_goal" in prediction_input.context:
                horizon = "long_term"
            else:
                horizon = "short_term"
        
        return horizon
    
    async def _predict_next_input(self, 
                             prediction_input: PredictionInput, 
                             horizon: str) -> Optional[str]:
        """Predict the next user input based on context and history"""
        if not prediction_input.history:
            return None
        
        # Extract relevant patterns from history
        context_summary = self._summarize_context(prediction_input.context)
        
        # Check if we have priors for this context
        if context_summary in self.prediction_priors["conversation_continuations"]:
            # Use priors to generate prediction
            return "Predicted user input based on context patterns"
        
        # Fallback to simple history-based prediction
        if len(prediction_input.history) >= 2:
            # Get the pattern of back-and-forth exchanges
            messages = [
                entry.get("text", "") for entry in prediction_input.history[-4:]
            ]
            
            # Very simple prediction
            if messages:
                return f"Predicted response based on dialogue history pattern"
        
        return None
    
    async def _predict_optimal_response(self, 
                                   prediction_input: PredictionInput, 
                                   horizon: str) -> Optional[str]:
        """Predict the optimal response based on context and history"""
        # Similar to predicting input, but focuses on system response
        return "Optimal predicted response based on context"
    
    async def _predict_emotional_state(self, 
                                  prediction_input: PredictionInput, 
                                  horizon: str) -> Optional[Dict[str, float]]:
        """Predict emotional state based on context and history"""
        # Extract current emotional state if available
        current_emotional_state = prediction_input.context.get("emotional_state", {})
        
        if not current_emotional_state:
            return None
        
        # Simple prediction that assumes emotional states evolve gradually
        predicted_state = {}
        
        for emotion, value in current_emotional_state.items():
            # Apply simple decay or intensification based on context
            if "positive_event" in prediction_input.context:
                # Positive emotions intensify, negative ones decay
                if emotion in ["Joy", "Trust", "Anticipation"]:
                    predicted_state[emotion] = min(1.0, value * 1.2)
                elif emotion in ["Sadness", "Fear", "Anger"]:
                    predicted_state[emotion] = max(0.0, value * 0.8)
                else:
                    predicted_state[emotion] = value
            elif "negative_event" in prediction_input.context:
                # Negative emotions intensify, positive ones decay
                if emotion in ["Sadness", "Fear", "Anger"]:
                    predicted_state[emotion] = min(1.0, value * 1.2)
                elif emotion in ["Joy", "Trust", "Anticipation"]:
                    predicted_state[emotion] = max(0.0, value * 0.8)
                else:
                    predicted_state[emotion] = value
            else:
                # General regression to baseline
                predicted_state[emotion] = value * 0.9 + 0.1 * 0.5  # Regress toward 0.5
        
        return predicted_state
    
    def _calculate_prediction_confidence(self, 
                                    prediction_input: PredictionInput, 
                                    horizon: str) -> float:
        """Calculate confidence in the prediction"""
        # Base confidence depends on horizon (longer = less confident)
        base_confidence = {
            "immediate": 0.8,
            "short_term": 0.6,
            "medium_term": 0.4,
            "long_term": 0.2
        }.get(horizon, 0.5)
        
        # Adjust based on history length
        history_factor = min(1.0, len(prediction_input.history) / 10)
        
        # Adjust based on context stability
        context_stability = self._estimate_context_stability(prediction_input.context)
        
        # Calculate final confidence
        confidence = base_confidence * 0.5 + history_factor * 0.3 + context_stability * 0.2
        
        return min(1.0, max(0.0, confidence))
    
    def _estimate_context_stability(self, context: Dict[str, Any]) -> float:
        """Estimate the stability of the current context"""
        # Simplified implementation
        return 0.7  # Default medium-high stability
    
    def _calculate_text_similarity_error(self, 
                                    predicted_text: Optional[str], 
                                    actual_text: Optional[str]) -> float:
        """Calculate error between predicted and actual text"""
        if predicted_text is None or actual_text is None:
            return 1.0  # Maximum error when one is missing
        
        # Very simplified text similarity
        pred_words = set(predicted_text.lower().split())
        actual_words = set(actual_text.lower().split())
        
        if not pred_words or not actual_words:
            return 1.0
            
        # Jaccard similarity
        intersection = len(pred_words.intersection(actual_words))
        union = len(pred_words.union(actual_words))
        
        similarity = intersection / union if union > 0 else 0
        
        # Convert similarity to error (1 - similarity)
        return 1.0 - similarity
    
    def _calculate_emotional_error(self, 
                              predicted_emotional: Optional[Dict[str, float]], 
                              actual_emotional: Optional[Dict[str, float]]) -> float:
        """Calculate error between predicted and actual emotional states"""
        if predicted_emotional is None or actual_emotional is None:
            return 1.0  # Maximum error when one is missing
        
        # Calculate error for each emotion present in both
        total_error = 0.0
        count = 0
        
        # Find common emotions
        common_emotions = set(predicted_emotional.keys()).intersection(set(actual_emotional.keys()))
        
        for emotion in common_emotions:
            # Calculate absolute difference
            diff = abs(predicted_emotional[emotion] - actual_emotional[emotion])
            total_error += diff
            count += 1
        
        # Add penalty for missing emotions
        missing_in_pred = set(actual_emotional.keys()) - set(predicted_emotional.keys())
        missing_in_actual = set(predicted_emotional.keys()) - set(actual_emotional.keys())
        
        # Add error for each missing emotion (as if the value was 0 when missing)
        for emotion in missing_in_pred:
            total_error += actual_emotional[emotion]
            count += 1
        
        for emotion in missing_in_actual:
            total_error += predicted_emotional[emotion]
            count += 1
        
        # Calculate average error
        avg_error = total_error / count if count > 0 else 1.0
        
        return avg_error
    
    def _summarize_context(self, context: Dict[str, Any]) -> str:
        """Generate a simple summary string of the context for use as a key"""
        # Extract key elements of context
        elements = []
        
        # Check for common context elements
        if "emotional_state" in context:
            emotions = context["emotional_state"]
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "neutral"
            elements.append(f"emotion:{dominant_emotion}")
        
        if "topic" in context:
            elements.append(f"topic:{context['topic']}")
        
        if "scenario_type" in context:
            elements.append(f"scenario:{context['scenario_type']}")
        
        if "user_id" in context:
            elements.append(f"user:{context['user_id']}")
        
        # Fallback if no elements found
        if not elements:
            return "generic_context"
            
        # Join elements into a context key
        return "_".join(elements)
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics for predictions"""
        # Calculate accuracy
        accuracy = 0.0
        if self.performance_metrics["total_predictions"] > 0:
            accuracy = (
                self.performance_metrics["correct_predictions"] / 
                self.performance_metrics["total_predictions"]
            )
        
        return {
            "total_predictions": self.performance_metrics["total_predictions"],
            "accuracy": accuracy,
            "avg_error": self.performance_metrics["avg_error"],
            "confidence_calibration": self.performance_metrics["confidence_calibration"],
            "active_predictions": len([p for p in self.predictions.values() if not p["evaluated"]])
        }
    
    async def update_prediction_priors(self, evaluation: PredictionEvaluation) -> None:
        """Update prediction priors based on evaluation results"""
        # Get the original prediction
        if evaluation.prediction_id not in self.predictions:
            return
            
        prediction = self.predictions[evaluation.prediction_id]
        prediction_input = prediction["input"]
        
        # Generate context summary key
        context_summary = self._summarize_context(prediction_input.context)
        
        # Update conversation continuation priors
        if evaluation.actual_input:
            if context_summary not in self.prediction_priors["conversation_continuations"]:
                self.prediction_priors["conversation_continuations"][context_summary] = {}
            
            # Use a simplified update mechanism
            self.prediction_priors["conversation_continuations"][context_summary]["last_input"] = evaluation.actual_input
            self.prediction_priors["conversation_continuations"][context_summary]["error"] = evaluation.prediction_error
        
        # Similarly update other prior types
        if evaluation.error_details.get("emotional_error") is not None:
            if context_summary not in self.prediction_priors["emotional_trajectories"]:
                self.prediction_priors["emotional_trajectories"][context_summary] = {}
            
            self.prediction_priors["emotional_trajectories"][context_summary]["last_error"] = \
                evaluation.error_details["emotional_error"]
