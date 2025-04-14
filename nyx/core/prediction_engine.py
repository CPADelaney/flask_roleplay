# nyx/core/prediction_engine.py

import asyncio
import logging
import datetime
import math
import random
from typing import Dict, List, Any, Optional, Union, Tuple

from agents import Agent, Runner, trace, function_tool, custom_span, handoff, RunContextWrapper, ModelSettings, RunConfig
from agents.tracing.util import gen_trace_id
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Pydantic models for structured I/O
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

class ContextAnalysisOutput(BaseModel):
    """Output from context analysis agent"""
    stability_score: float = Field(..., description="Context stability score (0-1)")
    key_factors: List[str] = Field(..., description="Key factors influencing prediction")
    recommended_horizon: str = Field(..., description="Recommended prediction horizon")
    confidence_modifier: float = Field(..., description="Confidence modifier based on context (-0.3 to 0.3)")

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
        
        # Initialize the agents
        self._init_agents()
        
        logger.info("Prediction Engine initialized with Agents SDK")
    
    def _init_agents(self):
        """Initialize agents used by the prediction engine"""
        # Context Analysis Agent
        self.context_analysis_agent = Agent(
            name="Context Analysis Agent",
            instructions="""You analyze the stability and predictability of the current context.
            Your task is to determine how stable the current context is and what prediction horizon
            would be most appropriate. Consider factors like conversation history consistency,
            topic stability, and emotional state predictability.""",
            tools=[function_tool(self._estimate_context_stability)],
            output_type=ContextAnalysisOutput
        )
        
        # Next Input Prediction Agent
        self.input_prediction_agent = Agent(
            name="Input Prediction Agent",
            instructions="""You predict the user's next input based on conversation history and context.
            Your task is to identify patterns in the user's communication style, preferences, and
            current conversation flow to predict what they might say next.""",
            tools=[function_tool(self._analyze_conversation_patterns)],
            model_settings=ModelSettings(temperature=0.7)
        )
        
        # Response Prediction Agent
        self.response_prediction_agent = Agent(
            name="Response Prediction Agent",
            instructions="""You predict the optimal response to the expected next user input.
            Your task is to anticipate what would be the most effective response given the
            conversation history, user's likely next input, and current context.""",
            tools=[function_tool(self._analyze_response_patterns)],
            model_settings=ModelSettings(temperature=0.7)
        )
        
        # Emotional Prediction Agent
        self.emotional_prediction_agent = Agent(
            name="Emotional Prediction Agent",
            instructions="""You predict how emotional states will evolve based on context.
            Your task is to analyze emotional trajectories and predict future emotional states
            considering the current context and recent history.""",
            tools=[function_tool(self._analyze_emotional_patterns_predict)],
            model_settings=ModelSettings(temperature=0.5)
        )
    
    @function_tool
    async def _estimate_context_stability(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate the stability of the current context"""
        with custom_span("estimate_context_stability"):
            # Extract key elements for stability estimation
            stability_factors = {}
            
            # Topic consistency
            if "topic" in context and context["topic"]:
                stability_factors["topic_defined"] = 0.2
            
            # History length
            history_length = len(context.get("history", []))
            if history_length > 10:
                stability_factors["substantial_history"] = 0.2
            elif history_length > 5:
                stability_factors["moderate_history"] = 0.1
            
            # Emotional state clarity
            if "emotional_state" in context and context["emotional_state"]:
                emotional_state = context["emotional_state"]
                if "primary_emotion" in emotional_state and emotional_state["primary_emotion"]:
                    primary_intensity = emotional_state.get("primary_intensity", 0.5)
                    stability_factors["clear_emotion"] = primary_intensity * 0.2
            
            # Scenario type defined
            if "scenario_type" in context and context["scenario_type"]:
                stability_factors["scenario_defined"] = 0.15
            
            # Calculate base stability
            base_stability = sum(stability_factors.values())
            
            # Calculate volatility factors (lower stability)
            volatility_factors = {}
            
            # Recent context changes
            if context.get("recent_context_change", False):
                volatility_factors["recent_change"] = -0.2
            
            # Inconsistent history
            if context.get("inconsistent_history", False):
                volatility_factors["inconsistency"] = -0.15
            
            # High emotional intensity
            if context.get("emotional_intensity", 0) > 0.8:
                volatility_factors["high_emotions"] = -0.1
            
            # Apply volatility
            volatility = sum(volatility_factors.values())
            
            # Calculate final stability (0-1 range)
            stability = max(0.0, min(1.0, 0.5 + base_stability + volatility))
            
            return {
                "stability_score": stability,
                "stability_factors": stability_factors,
                "volatility_factors": volatility_factors,
                "analysis_time": datetime.datetime.now().isoformat()
            }
    
    @function_tool
    async def _analyze_conversation_patterns(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in conversation history"""
        with custom_span("analyze_conversation_patterns"):
            # Extract patterns from history
            if not history:
                return {
                    "patterns": {},
                    "message": "Insufficient history for pattern analysis"
                }
            
            # Analyze message lengths
            message_lengths = [len(msg.get("text", "")) for msg in history]
            avg_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0
            
            # Analyze response times if available
            response_times = []
            for i in range(1, len(history)):
                if "timestamp" in history[i] and "timestamp" in history[i-1]:
                    try:
                        t1 = datetime.datetime.fromisoformat(history[i-1]["timestamp"].replace("Z", "+00:00"))
                        t2 = datetime.datetime.fromisoformat(history[i]["timestamp"].replace("Z", "+00:00"))
                        response_times.append((t2 - t1).total_seconds())
                    except (ValueError, TypeError):
                        pass
            
            avg_response_time = sum(response_times) / len(response_times) if response_times else None
            
            # Extract topic transitions
            topics = [msg.get("topic", "") for msg in history if "topic" in msg]
            topic_transitions = []
            for i in range(1, len(topics)):
                if topics[i] != topics[i-1]:
                    topic_transitions.append((topics[i-1], topics[i]))
            
            return {
                "avg_message_length": avg_length,
                "avg_response_time": avg_response_time,
                "topic_transitions": topic_transitions,
                "message_count": len(history),
                "patterns": {
                    "consistent_length": max(0, 1 - (sum(abs(l - avg_length) for l in message_lengths) / (avg_length * len(message_lengths)) if avg_length > 0 else 1)),
                    "predictable_timing": max(0, 1 - (sum(abs(t - avg_response_time) for t in response_times) / (avg_response_time * len(response_times)) if avg_response_time and response_times else 1))
                }
            }
    
    @function_tool
    async def _analyze_response_patterns(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in responses"""
        with custom_span("analyze_response_patterns"):
            if len(history) < 2:
                return {
                    "patterns": {},
                    "message": "Insufficient history for response pattern analysis"
                }
            
            # Extract pairs of user input and system response
            pairs = []
            for i in range(1, len(history)):
                if history[i-1].get("role") == "user" and history[i].get("role") == "assistant":
                    pairs.append((history[i-1].get("text", ""), history[i].get("text", "")))
            
            return {
                "input_response_pairs": len(pairs),
                "analysis_time": datetime.datetime.now().isoformat()
            }
    
    @function_tool
    async def _analyze_emotional_patterns_predict(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in emotional states"""
        with custom_span("analyze_emotional_patterns_predict"):
            # Extract emotional states from history
            emotional_states = []
            for entry in history:
                if "emotional_state" in entry:
                    emotional_states.append(entry["emotional_state"])
            
            if not emotional_states:
                return {
                    "patterns": {},
                    "message": "No emotional data in history"
                }
            
            # Track emotion changes over time
            emotion_trends = {}
            for state in emotional_states:
                if "primary_emotion" in state:
                    emotion_name = state["primary_emotion"].get("name", "Neutral")
                    intensity = state["primary_emotion"].get("intensity", 0.5)
                    
                    if emotion_name not in emotion_trends:
                        emotion_trends[emotion_name] = []
                    
                    emotion_trends[emotion_name].append(intensity)
            
            # Analyze trends for each emotion
            patterns = {}
            for emotion, intensities in emotion_trends.items():
                if len(intensities) > 1:
                    # Calculate trend
                    start = intensities[0]
                    end = intensities[-1]
                    change = end - start
                    
                    if abs(change) < 0.1:
                        trend = "stable"
                    elif change > 0:
                        trend = "increasing"
                    else:
                        trend = "decreasing"
                    
                    # Calculate volatility
                    volatility = sum(abs(intensities[i] - intensities[i-1]) 
                                    for i in range(1, len(intensities))) / (len(intensities) - 1)
                    
                    patterns[emotion] = {
                        "trend": trend,
                        "volatility": volatility,
                        "start_intensity": start,
                        "current_intensity": end,
                        "change": change,
                        "occurrences": len(intensities)
                    }
            
            return {
                "patterns": patterns,
                "emotions_tracked": len(emotion_trends),
                "analysis_time": datetime.datetime.now().isoformat()
            }
    
    @function_tool
    async def generate_prediction(self, context: Dict[str, Any], 
                              history: List[Dict[str, Any]], 
                              query_type: Optional[str] = None) -> PredictionResult:
        """
        Generate predictions based on current context and history
        
        Args:
            context: Current context data
            history: Recent interaction history
            query_type: Optional type of prediction query
            
        Returns:
            Prediction result
        """
        prediction_input = PredictionInput(
            context=context,
            history=history,
            query_type=query_type
        )
        
        with trace(workflow_name="Generate Prediction"):
            # First analyze context to determine prediction approach
            context_analysis_prompt = f"""Analyze the stability and predictability of this context:
            
            Context: {context}
            History length: {len(history)} entries
            Query type: {query_type or 'Not specified'}
            
            Determine the appropriate prediction horizon and confidence based on context stability.
            """
            
            context_analysis_result = await Runner.run(
                self.context_analysis_agent,
                context_analysis_prompt,
                context={"prediction_input": prediction_input}
            )
            
            context_analysis = context_analysis_result.final_output_as(ContextAnalysisOutput)
            
            # Set prediction horizon based on context analysis
            horizon = context_analysis.recommended_horizon
            
            # Generate unique prediction ID
            prediction_id = f"pred_{datetime.datetime.now().timestamp()}_{horizon}"
            
            # Prepare prediction parameters based on query type
            prediction_tasks = []
            
            # Input prediction
            if not query_type or query_type in ["input", "all"]:
                input_prediction_task = self._predict_next_input(prediction_input, horizon)
                prediction_tasks.append(("input", input_prediction_task))
            
            # Response prediction
            if not query_type or query_type in ["response", "all"]:
                response_prediction_task = self._predict_optimal_response(prediction_input, horizon)
                prediction_tasks.append(("response", response_prediction_task))
            
            # Emotional prediction
            if not query_type or query_type in ["emotional", "all"]:
                emotional_prediction_task = self._predict_emotional_state(prediction_input, horizon)
                prediction_tasks.append(("emotional", emotional_prediction_task))
            
            # Run predictions in parallel
            prediction_results = {}
            for name, task in prediction_tasks:
                try:
                    result = await task
                    prediction_results[name] = result
                except Exception as e:
                    logger.error(f"Error in {name} prediction: {str(e)}")
                    prediction_results[name] = None
            
            # Calculate confidence based on context analysis and prediction results
            base_confidence = {
                "immediate": 0.8,
                "short_term": 0.6,
                "medium_term": 0.4,
                "long_term": 0.2
            }.get(horizon, 0.5)
            
            # Apply context analysis confidence modifier
            confidence = base_confidence + context_analysis.confidence_modifier
            
            # Adjust based on history length
            history_factor = min(1.0, len(history) / 10)
            confidence = 0.7 * confidence + 0.3 * history_factor
            
            # Ensure confidence is in 0-1 range
            confidence = max(0.0, min(1.0, confidence))
            
            # Create prediction result
            result = PredictionResult(
                predicted_input=prediction_results.get("input"),
                predicted_response=prediction_results.get("response"),
                predicted_emotional_state=prediction_results.get("emotional"),
                confidence=confidence,
                prediction_horizon=horizon,
                prediction_id=prediction_id
            )
            
            # Store the prediction
            self.predictions[prediction_id] = {
                "result": result.model_dump(),
                "input": prediction_input.model_dump(),
                "timestamp": datetime.datetime.now().isoformat(),
                "evaluated": False
            }
            
            # Update metrics
            self.performance_metrics["total_predictions"] += 1
            
            # Add to history
            self.prediction_history.append({
                "prediction_id": prediction_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "context_summary": self._summarize_context(context),
                "confidence": confidence,
                "horizon": horizon
            })
            
            # Trim history if needed
            if len(self.prediction_history) > self.history_limit:
                self.prediction_history = self.prediction_history[-self.history_limit:]
            
            return result
    
    async def _predict_next_input(self, 
                             prediction_input: PredictionInput, 
                             horizon: str) -> Optional[str]:
        """Predict the next user input based on context and history"""
        with custom_span("predict_next_input", {"horizon": horizon}):
            if not prediction_input.history:
                return None
            
            # Prepare prompt for input prediction
            prompt = f"""Predict the user's next input based on this conversation history and context.
            
            Context: {prediction_input.context}
            
            Recent conversation history:
            {self._format_history(prediction_input.history[-5:])}
            
            Prediction horizon: {horizon}
            
            Based on this information, what is the most likely next message from the user?
            Consider patterns in their communication style, the current topic, and their interests.
            """
            
            # Run agent
            result = await Runner.run(
                self.input_prediction_agent,
                prompt,
                context={"prediction_input": prediction_input}
            )
            
            return result.final_output
    
    async def _predict_optimal_response(self, 
                                   prediction_input: PredictionInput, 
                                   horizon: str) -> Optional[str]:
        """Predict the optimal response based on context and history"""
        with custom_span("predict_optimal_response", {"horizon": horizon}):
            # Predict the next user input first
            predicted_input = await self._predict_next_input(prediction_input, horizon)
            
            if not predicted_input:
                return "I don't have enough information to predict an optimal response yet."
            
            # Prepare prompt for response prediction
            prompt = f"""Predict the optimal response to the user's anticipated next message.
            
            Context: {prediction_input.context}
            
            Recent conversation history:
            {self._format_history(prediction_input.history[-5:])}
            
            Anticipated next user message: "{predicted_input}"
            
            Prediction horizon: {horizon}
            
            Based on this information, what would be the most effective response to this anticipated message?
            Consider the user's needs, preferences, and emotional state.
            """
            
            # Run agent
            result = await Runner.run(
                self.response_prediction_agent,
                prompt,
                context={"prediction_input": prediction_input, "predicted_input": predicted_input}
            )
            
            return result.final_output
    
    async def _predict_emotional_state(self, 
                                  prediction_input: PredictionInput, 
                                  horizon: str) -> Optional[Dict[str, float]]:
        """Predict emotional state based on context and history"""
        with custom_span("predict_emotional_state", {"horizon": horizon}):
            # Extract current emotional state if available
            current_emotional_state = prediction_input.context.get("emotional_state", {})
            
            if not current_emotional_state:
                return None
            
            # Prepare prompt for emotional prediction
            prompt = f"""Predict how the emotional state will evolve based on context and history.
            
            Current emotional state: {current_emotional_state}
            
            Recent conversation history:
            {self._format_history(prediction_input.history[-5:])}
            
            Prediction horizon: {horizon}
            
            Based on this information, how will the emotional state likely evolve?
            Provide predictions for primary emotions and their intensities.
            """
            
            # Run agent
            result = await Runner.run(
                self.emotional_prediction_agent,
                prompt,
                context={"prediction_input": prediction_input, "current_emotional_state": current_emotional_state}
            )
            
            # Parse emotional state from text response
            response = result.final_output
            
            # Simple parsing for demonstration - in production would use more robust parsing
            predicted_state = {}
            
            # Extract emotion-value pairs using simple pattern matching
            import re
            emotion_patterns = re.findall(r'(Joy|Sadness|Fear|Anger|Trust|Disgust|Anticipation|Surprise|Love|Frustration|Neutral)\s*:\s*(0\.\d+)', response)
            
            for emotion, value in emotion_patterns:
                predicted_state[emotion] = float(value)
            
            # If no emotions were extracted, create a basic prediction
            if not predicted_state and current_emotional_state:
                # Simple decay model toward neutral
                for emotion, value in current_emotional_state.items():
                    if isinstance(value, (int, float)):
                        # Apply decay based on horizon
                        decay_factor = {
                            "immediate": 0.1,
                            "short_term": 0.3,
                            "medium_term": 0.5,
                            "long_term": 0.7
                        }.get(horizon, 0.3)
                        
                        predicted_state[emotion] = value * (1 - decay_factor) + 0.5 * decay_factor
            
            return predicted_state
    
    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """Format history for agent prompts"""
        formatted = []
        for i, entry in enumerate(history):
            role = entry.get("role", "unknown")
            text = entry.get("text", "")
            formatted.append(f"{role.capitalize()}: {text}")
        
        return "\n".join(formatted)
    
    def _summarize_context(self, context: Dict[str, Any]) -> str:
        """Generate a simple summary string of the context for use as a key"""
        # Extract key elements of context
        elements = []
        
        # Check for common context elements
        if "emotional_state" in context:
            emotions = context["emotional_state"]
            if isinstance(emotions, dict) and emotions:
                items = list(emotions.items())
                if items:
                    dominant_emotion = max(items, key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)[0]
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
    
    @function_tool
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
        with trace(workflow_name="Evaluate Prediction"):
            if prediction_id not in self.predictions:
                raise ValueError(f"Prediction {prediction_id} not found")
            
            prediction = self.predictions[prediction_id]
            prediction_result = PredictionResult(**prediction["result"])
            
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
            self.prediction_evaluations[prediction_id] = evaluation.model_dump()
            
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
            
            # Update prediction priors
            await self.update_prediction_priors(evaluation)
            
            return evaluation
    
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
    
    @function_tool
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics for predictions"""
        with custom_span("get_performance_metrics"):
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
        with custom_span("update_prediction_priors"):
            # Get the original prediction
            if evaluation.prediction_id not in self.predictions:
                return
                
            prediction = self.predictions[evaluation.prediction_id]
            prediction_input = PredictionInput(**prediction["input"])
            
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
