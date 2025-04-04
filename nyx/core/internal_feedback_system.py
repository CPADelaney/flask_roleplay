# nyx/core/internal_feedback_system.py

import asyncio
import json
import logging
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, TYPE_CHECKING
import math
import random
from pydantic import BaseModel
import pathlib

# Import necessary components from the agents SDK
from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_key,
    RunContextWrapper # Needed for the tool function signature
)

# --- Existing Logger Setup ---
logger = logging.getLogger(__name__) # Main logger for the module

# --- Dev Log Configuration ---
DEV_LOG_FILE_PATH = "/dev_log/internal_feedback.log" # Consolidated log file

# --- Setup Dev Logger ---
dev_logger = logging.getLogger("InternalFeedbackLog") # Renamed for broader scope
dev_logger.setLevel(logging.INFO)
dev_logger.propagate = False

try:
    log_dir = pathlib.Path(DEV_LOG_FILE_PATH).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    dev_log_handler = logging.FileHandler(DEV_LOG_FILE_PATH, encoding='utf-8')
    # Added %(levelname)s to the format
    dev_log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(agent_name)s] - %(message)s')
    dev_log_handler.setFormatter(dev_log_formatter)
    if not dev_logger.handlers:
        dev_logger.addHandler(dev_log_handler)
except Exception as e:
    logger.error(f"Failed to set up development log at {DEV_LOG_FILE_PATH}: {e}")
    dev_logger = None

# --- Define the Missing Capability Logging Tool ---
# Use a global reference to the logger, checked within the function
_global_dev_logger = dev_logger

# Define the tool function using the agents SDK decorator
@function_tool
async def log_missing_capability(
    ctx: RunContextWrapper[Any], # Context object (optional but good practice)
    capability_description: str,
    required_for: str,
    agent_name: str = "Unknown Agent" # Add agent name parameter
) -> str:
    """
    Logs a missing capability or functionality identified by an agent.

    Use this tool when you realize you lack a specific ability, tool, access,
    or piece of information needed to perform your task more effectively or accurately.

    Args:
        capability_description: A clear description of the missing capability, tool, or information.
        required_for: Briefly explain why this capability is needed for the current task.
        agent_name: The name of the agent reporting the missing capability.
    """
    if _global_dev_logger:
        log_record_extra = {'agent_name': agent_name}
        # Use LoggerAdapter for contextual info if needed, or pass via extra
        # adapter = logging.LoggerAdapter(_global_dev_logger, log_record_extra)
        try:
            _global_dev_logger.warning( # Log as WARNING to make it stand out
                f"Missing Capability Identified: {capability_description} (Required for: {required_for})",
                extra=log_record_extra
            )
            return f"Successfully logged missing capability: {capability_description}"
        except Exception as e:
            # Log error during logging using the main logger
            logger.error(f"Error writing missing capability to dev log: {e}")
            return f"Error logging missing capability: {e}"
    else:
        logger.warning("Dev logger not configured. Cannot log missing capability.")
        return "Dev logger not available. Could not log missing capability."


# Initialize the OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    set_default_openai_key(api_key)
# --- End Tool Definition ---


class InternalFeedbackSystem:
    """
    System for internal feedback, evaluation, and quality assessment.
    Provides mechanisms for self-monitoring and improvement.
    Logs improvement suggestions and identified missing capabilities to a dev log.
    """

    def __init__(self):
        self.performance_metrics = {}
        self.confidence_tracking = {
            "history": [],
            "calibration": {
                "buckets": [0.0] * 10,
                "correct": [0.0] * 10
            }
        }
        self.evaluation_criteria = {
            "consistency": { "coherence": 0.4, "continuity": 0.3, "context_adherence": 0.3 },
            "effectiveness": { "goal_achievement": 0.5, "comprehensiveness": 0.3, "efficiency": 0.2 },
            "efficiency": { "response_time": 0.4, "resource_utilization": 0.3, "complexity_management": 0.3 }
        }
        self.quality_threshold = 0.7
        self.dev_logger = dev_logger # Use the globally configured logger

        # Initialize the agent-based critic system
        self._init_critic_agents()

    def _log_suggestions(self, source_agent_name: str, suggestions: List[str]):
        """Helper method to log suggestions to the dev log."""
        if self.dev_logger and suggestions:
            log_record_extra = {'agent_name': source_agent_name}
            log_prefix = "[Suggestion]"
            try:
                for suggestion in suggestions:
                    self.dev_logger.info(f"{log_prefix} {suggestion}", extra=log_record_extra)
            except Exception as e:
                logger.error(f"Error writing suggestion to development log: {e}")

    def _init_critic_agents(self):
        """Initialize the agent-based critic system"""
        # --- Output Types (unchanged) ---
        class ConsistencyEvaluation(BaseModel):
            coherence_score: float
            coherence_feedback: str
            continuity_score: float
            continuity_feedback: str
            context_adherence_score: float
            context_adherence_feedback: str
            overall_score: float
            meets_threshold: bool
            suggestions: List[str]

        class EffectivenessEvaluation(BaseModel):
            goal_achievement_score: float
            goal_achievement_feedback: str
            comprehensiveness_score: float
            comprehensiveness_feedback: str
            efficiency_score: float
            efficiency_feedback: str
            overall_score: float
            meets_threshold: bool
            suggestions: List[str]

        class EfficiencyEvaluation(BaseModel):
            response_time_score: float
            response_time_feedback: str
            resource_utilization_score: float
            resource_utilization_feedback: str
            complexity_management_score: float
            complexity_management_feedback: str
            overall_score: float
            meets_threshold: bool
            suggestions: List[str]

        class MetaEvaluation(BaseModel):
            consistency_score: float
            effectiveness_score: float
            efficiency_score: float
            overall_score: float
            quality_level: str
            meets_threshold: bool
            key_strengths: List[str]
            key_weaknesses: List[str]
            improvement_suggestions: List[str]

        # --- Tool List ---
        # Include the newly defined tool
        common_tools = [log_missing_capability]

        # --- Agent Definitions (Instructions updated, tools added) ---
        self.consistency_critic = Agent(
            name="Consistency Critic",
            instructions=(
                "You evaluate content for consistency (Coherence: 40%, Continuity: 30%, Context Adherence: 30%).\n"
                "Score each aspect from 0.0 (Poor) to 1.0 (Excellent). Provide feedback.\n"
                "Calculate a weighted overall score and determine if it meets the threshold.\n"
                "Provide specific improvement suggestions.\n\n"
                "**IMPORTANT:** If you identify a capability, tool, or piece of information that you lack but would need "
                "to perform a more thorough evaluation (e.g., ability to compare against previous versions, access external knowledge bases, run specific code checks), "
                "use the `log_missing_capability` tool to report it. Clearly state what is missing and why it's needed for consistency evaluation. Provide your name as the 'agent_name'."
            ),
            output_type=ConsistencyEvaluation,
            tools=common_tools # Add the tool
        )

        self.effectiveness_critic = Agent(
            name="Effectiveness Critic",
            instructions=(
                "You evaluate content for effectiveness (Goal Achievement: 50%, Comprehensiveness: 30%, Efficiency (goal-related): 20%).\n"
                "Score each aspect from 0.0 (Poor) to 1.0 (Excellent). Provide feedback.\n"
                "Calculate a weighted overall score and determine if it meets the threshold.\n"
                "Provide specific improvement suggestions.\n\n"
                "**IMPORTANT:** If you identify a capability, tool, or piece of information that you lack but would need "
                "to perform a more thorough evaluation (e.g., access to the original goal definition, ability to measure impact, user feedback data), "
                "use the `log_missing_capability` tool to report it. Clearly state what is missing and why it's needed for effectiveness evaluation. Provide your name as the 'agent_name'."
            ),
            output_type=EffectivenessEvaluation,
            tools=common_tools # Add the tool
        )

        self.efficiency_critic = Agent(
            name="Efficiency Critic",
            instructions=(
                "You evaluate content for operational efficiency (Response Time: 40%, Resource Utilization: 30%, Complexity Management: 30%).\n"
                "Score each aspect from 0.0 (Poor) to 1.0 (Excellent). Provide feedback.\n"
                "Calculate a weighted overall score and determine if it meets the threshold.\n"
                "Provide specific improvement suggestions.\n\n"
                "**IMPORTANT:** If you identify a capability, tool, or piece of information that you lack but would need "
                "to perform a more thorough evaluation (e.g., access to performance metrics, resource usage logs, code complexity analysis tools), "
                "use the `log_missing_capability` tool to report it. Clearly state what is missing and why it's needed for efficiency evaluation. Provide your name as the 'agent_name'."
            ),
            output_type=EfficiencyEvaluation,
            tools=common_tools # Add the tool
        )

        self.meta_critic = Agent(
            name="Meta Critic",
            instructions=(
                "You synthesize evaluations from the consistency, effectiveness, and efficiency critics.\n"
                "Calculate a weighted overall score (Consistency: 40%, Effectiveness: 40%, Efficiency: 20%).\n"
                "Determine the quality level (Outstanding, Excellent, Good, Satisfactory, Fair, Needs Improvement, Unsatisfactory).\n"
                "Identify key strengths and weaknesses.\n"
                "Provide prioritized improvement suggestions based on the synthesis.\n\n"
                "**IMPORTANT:** If, during synthesis, you identify a systemic limitation or a required capability that none of the individual critics could address "
                "(e.g., lack of cross-aspect analysis tools, inability to correlate feedback trends), "
                "use the `log_missing_capability` tool to report it. Explain the systemic gap. Provide your name as the 'agent_name'."
            ),
            output_type=MetaEvaluation,
            tools=common_tools # Add the tool
        )

    async def track_performance(self,
                          metric: str,
                          value: float) -> Dict[str, Any]:
        """
        Track a performance metric over time.

        Args:
            metric: Name of the metric
            value: Value of the metric (0.0-1.0)

        Returns:
            Statistics for the metric
        """
        if metric not in self.performance_metrics:
            self.performance_metrics[metric] = {
                "values": [],
                "timestamps": [],
                "mean": value,
                "std_dev": 0.0,
                "min": value,
                "max": value,
                "trend": "stable"
            }

        self.performance_metrics[metric]["values"].append(value)
        self.performance_metrics[metric]["timestamps"].append(datetime.now().isoformat())

        if len(self.performance_metrics[metric]["values"]) > 100:
            self.performance_metrics[metric]["values"].pop(0)
            self.performance_metrics[metric]["timestamps"].pop(0)

        values = self.performance_metrics[metric]["values"]
        self.performance_metrics[metric]["mean"] = sum(values) / len(values) if values else 0.0 # Avoid division by zero

        if len(values) > 1:
            mean = self.performance_metrics[metric]["mean"]
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            self.performance_metrics[metric]["std_dev"] = math.sqrt(variance)
        else:
             self.performance_metrics[metric]["std_dev"] = 0.0


        self.performance_metrics[metric]["min"] = min(values) if values else 0.0
        self.performance_metrics[metric]["max"] = max(values) if values else 0.0

        self.performance_metrics[metric]["trend"] = self._calculate_trend(values)

        if len(values) >= 10:
            self.performance_metrics[metric]["confidence_interval"] = self._calculate_confidence_interval(values)
        elif len(values) > 0: # Calculate even for smaller samples, though less reliable
             self.performance_metrics[metric]["confidence_interval"] = self._calculate_confidence_interval(values)


        return self.performance_metrics[metric]

    async def evaluate_confidence(self,
                            confidence: float,
                            success: bool) -> Dict[str, Any]:
        """
        Evaluate confidence prediction against actual success.

        Args:
            confidence: Predicted confidence (0.0-1.0)
            success: Whether the prediction was correct

        Returns:
            Confidence calibration information
        """
        # Add to history
        self.confidence_tracking["history"].append({
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
            "success": success
        })

        bucket = min(9, int(confidence * 10))
        self.confidence_tracking["calibration"]["buckets"][bucket] += 1
        if success:
            self.confidence_tracking["calibration"]["correct"][bucket] += 1

        calibration = []
        for i in range(10):
            bucket_total = self.confidence_tracking["calibration"]["buckets"][i]
            if bucket_total > 0:
                accuracy = self.confidence_tracking["calibration"]["correct"][i] / bucket_total
                calibration.append({
                    "confidence_range": f"{i/10:.1f}-{(i+1)/10:.1f}",
                    "instances": bucket_total,
                    "accuracy": accuracy,
                    "calibration_error": abs(((i + 0.5) / 10) - accuracy)
                })

        total_instances = sum(self.confidence_tracking["calibration"]["buckets"])
        history_len = len(self.confidence_tracking["history"])
        if total_instances > 0 and history_len > 0:
            overall_accuracy = sum(self.confidence_tracking["calibration"]["correct"]) / total_instances

            brier_score = 0.0
            for entry in self.confidence_tracking["history"]:
                error = entry["confidence"] - (1.0 if entry["success"] else 0.0)
                brier_score += error ** 2
            brier_score /= history_len

            conf_sum = sum(entry["confidence"] for entry in self.confidence_tracking["history"])
            avg_confidence = conf_sum / history_len
            confidence_bias = avg_confidence - overall_accuracy

            confidence_metrics = {
                "overall_accuracy": overall_accuracy,
                "average_confidence": avg_confidence,
                "confidence_bias": confidence_bias,
                "brier_score": brier_score,
                "bias_type": "overconfident" if confidence_bias > 0.05 else
                            "underconfident" if confidence_bias < -0.05 else "well-calibrated"
            }
        else:
            confidence_metrics = {
                "overall_accuracy": 0.0,
                "average_confidence": 0.0,
                "confidence_bias": 0.0,
                "brier_score": 0.0,
                "bias_type": "unknown"
            }

        return {
            "current": { "confidence": confidence, "success": success },
            "calibration": calibration,
            "metrics": confidence_metrics
        }


    async def critic_evaluate(self, aspect: str, content: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate content on a specific aspect using critic agents.
        Logs suggestions and missing capabilities to the dev log.
        Falls back to basic evaluation if agent fails.
        """
        if aspect not in self.evaluation_criteria:
            # Log this attempt?
            if self.dev_logger:
                self.dev_logger.error(f"Attempted evaluation with unknown aspect: {aspect}", extra={'agent_name': 'System'})
            return {
                "error": f"Unknown evaluation aspect: {aspect}",
                "available_aspects": list(self.evaluation_criteria.keys())
            }

        result_data = {}
        agent_to_run: Optional[Agent] = None
        agent_name = f"{aspect.capitalize()} Critic" # Determine agent name for logging

        try:
            if aspect == "consistency": agent_to_run = self.consistency_critic
            elif aspect == "effectiveness": agent_to_run = self.effectiveness_critic
            elif aspect == "efficiency": agent_to_run = self.efficiency_critic
            else: raise ValueError(f"Logic error: Aspect {aspect} check passed but no agent assigned.") # Should not happen

            # Format the content and context
            content_text = str(content.get("content", content)) if isinstance(content, dict) else str(content)
            context_text = "\n".join([f"{k}: {v}" for k, v in context.items()])
            prompt = f"Please evaluate this content on the aspect of {aspect}:\n\nContent:\n{content_text}\n\nContext:\n{context_text}"

            # Run the agent
            # Note: The agent might call the log_missing_capability tool during its run.
            result = await Runner.run(agent_to_run, prompt)
            evaluation = result.final_output_as(agent_to_run.output_type)

            # Log suggestions from the successful agent run
            self._log_suggestions(agent_name, evaluation.suggestions)

            # Format the result
            if aspect == "consistency":
                result_data = {
                    "aspect": aspect, "weighted_score": evaluation.overall_score,
                    "quality_level": self._determine_quality_level(evaluation.overall_score),
                    "meets_threshold": evaluation.meets_threshold,
                    "improvement_suggestions": evaluation.suggestions,
                    "criteria_evaluations": {
                        "coherence": { "score": evaluation.coherence_score, "weight": self.evaluation_criteria[aspect]["coherence"], "description": evaluation.coherence_feedback },
                        "continuity": { "score": evaluation.continuity_score, "weight": self.evaluation_criteria[aspect]["continuity"], "description": evaluation.continuity_feedback },
                        "context_adherence": { "score": evaluation.context_adherence_score, "weight": self.evaluation_criteria[aspect]["context_adherence"], "description": evaluation.context_adherence_feedback }
                    }
                }
            elif aspect == "effectiveness":
                 result_data = {
                    "aspect": aspect, "weighted_score": evaluation.overall_score,
                    "quality_level": self._determine_quality_level(evaluation.overall_score),
                    "meets_threshold": evaluation.meets_threshold,
                    "improvement_suggestions": evaluation.suggestions,
                    "criteria_evaluations": {
                        "goal_achievement": { "score": evaluation.goal_achievement_score, "weight": self.evaluation_criteria[aspect]["goal_achievement"], "description": evaluation.goal_achievement_feedback },
                        "comprehensiveness": { "score": evaluation.comprehensiveness_score, "weight": self.evaluation_criteria[aspect]["comprehensiveness"], "description": evaluation.comprehensiveness_feedback },
                        "efficiency": { "score": evaluation.efficiency_score, "weight": self.evaluation_criteria[aspect]["efficiency"], "description": evaluation.efficiency_feedback }
                    }
                }
            elif aspect == "efficiency":
                result_data = {
                    "aspect": aspect, "weighted_score": evaluation.overall_score,
                    "quality_level": self._determine_quality_level(evaluation.overall_score),
                    "meets_threshold": evaluation.meets_threshold,
                    "improvement_suggestions": evaluation.suggestions,
                    "criteria_evaluations": {
                        "response_time": { "score": evaluation.response_time_score, "weight": self.evaluation_criteria[aspect]["response_time"], "description": evaluation.response_time_feedback },
                        "resource_utilization": { "score": evaluation.resource_utilization_score, "weight": self.evaluation_criteria[aspect]["resource_utilization"], "description": evaluation.resource_utilization_feedback },
                        "complexity_management": { "score": evaluation.complexity_management_score, "weight": self.evaluation_criteria[aspect]["complexity_management"], "description": evaluation.complexity_management_feedback }
                    }
                }
            return result_data # Return agent result if successful

        except Exception as e:
            logger.error(f"Error in agent-based evaluation for {aspect}: {e}. Falling back.", exc_info=True)
            if self.dev_logger:
                self.dev_logger.error(f"Agent execution failed for {aspect}. Falling back to basic evaluation. Error: {e}", extra={'agent_name': agent_name})

            # Fallback Implementation
            criteria = self.evaluation_criteria[aspect]
            fallback_evaluation = {}
            for criterion, weight in criteria.items():
                score = self._evaluate_criterion(aspect, criterion, content, context)
                fallback_evaluation[criterion] = {
                    "score": score, "weight": weight,
                    "description": self._generate_criterion_description(aspect, criterion, score)
                }
            weighted_score = sum(d["score"] * d["weight"] for d in fallback_evaluation.values())
            quality_level = self._determine_quality_level(weighted_score)
            improvements = self._generate_improvement_suggestions(aspect, fallback_evaluation, weighted_score)

            # Log fallback suggestions
            self._log_suggestions(f"{agent_name} Fallback", improvements)

            return {
                "aspect": aspect, "criteria_evaluations": fallback_evaluation,
                "weighted_score": weighted_score, "quality_level": quality_level,
                "meets_threshold": weighted_score >= self.quality_threshold,
                "improvement_suggestions": improvements
            }

    async def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the feedback system.

        Returns:
            Statistics and metrics
        """
        performance_summary = {}
        for metric, data in self.performance_metrics.items():
             performance_summary[metric] = {
                "mean": data.get("mean", 0.0),
                "std_dev": data.get("std_dev", 0.0),
                "trend": data.get("trend", "unknown"),
                "sample_size": len(data.get("values", []))
            }

        confidence_metrics = {}
        if self.confidence_tracking["history"]:
            total_instances = sum(self.confidence_tracking["calibration"]["buckets"])
            history_len = len(self.confidence_tracking["history"])
            if total_instances > 0 and history_len > 0:
                overall_accuracy = sum(self.confidence_tracking["calibration"]["correct"]) / total_instances
                avg_confidence = sum(e["confidence"] for e in self.confidence_tracking["history"]) / history_len
                confidence_metrics = {
                    "overall_accuracy": overall_accuracy,
                    "average_confidence": avg_confidence,
                    "bias": avg_confidence - overall_accuracy,
                    "sample_size": history_len
                }

        return {
            "performance": performance_summary,
            "confidence": confidence_metrics,
            "evaluation_criteria": self.evaluation_criteria,
            "quality_threshold": self.quality_threshold
        }


    async def comprehensive_evaluate(self, content: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a comprehensive evaluation across all aspects using critic agents.
        Logs suggestions and missing capabilities to the dev log.
        """
        content_text = str(content.get("content", content)) if isinstance(content, dict) else str(content)
        context_text = "\n".join([f"{k}: {v}" for k, v in context.items()])
        prompt = f"Please evaluate this content:\n\nContent:\n{content_text}\n\nContext:\n{context_text}"

        try:
            # Run individual critics (they might log missing capabilities during their run)
            tasks = [
                Runner.run(self.consistency_critic, prompt),
                Runner.run(self.effectiveness_critic, prompt),
                Runner.run(self.efficiency_critic, prompt)
            ]
            consistency_result, effectiveness_result, efficiency_result = await asyncio.gather(*tasks)

            consistency_eval = consistency_result.final_output_as(self.consistency_critic.output_type)
            effectiveness_eval = effectiveness_result.final_output_as(self.effectiveness_critic.output_type)
            efficiency_eval = efficiency_result.final_output_as(self.efficiency_critic.output_type)

            # Log suggestions from individual critics
            self._log_suggestions(self.consistency_critic.name, consistency_eval.suggestions)
            self._log_suggestions(self.effectiveness_critic.name, effectiveness_eval.suggestions)
            self._log_suggestions(self.efficiency_critic.name, efficiency_eval.suggestions)

            # Prepare input for meta-critic
            meta_prompt = (
                "Synthesize these evaluations:\n\n"
                f"Consistency Evaluation:\n{json.dumps(consistency_eval.dict())}\n\n"
                f"Effectiveness Evaluation:\n{json.dumps(effectiveness_eval.dict())}\n\n"
                f"Efficiency Evaluation:\n{json.dumps(efficiency_eval.dict())}"
            )

            # Run meta-critic (it might also log missing capabilities)
            meta_result = await Runner.run(self.meta_critic, meta_prompt)
            meta_eval = meta_result.final_output_as(self.meta_critic.output_type)

            # Log meta-suggestions
            self._log_suggestions(self.meta_critic.name, meta_eval.improvement_suggestions)

            # Format the final result
            return {
                "aspect_evaluations": {
                    "consistency": {
                        "score": consistency_eval.overall_score, "meets_threshold": consistency_eval.meets_threshold,
                        "criteria": {
                            "coherence": { "score": consistency_eval.coherence_score, "feedback": consistency_eval.coherence_feedback },
                            "continuity": { "score": consistency_eval.continuity_score, "feedback": consistency_eval.continuity_feedback },
                            "context_adherence": { "score": consistency_eval.context_adherence_score, "feedback": consistency_eval.context_adherence_feedback }
                        },
                        "suggestions": consistency_eval.suggestions
                    },
                    "effectiveness": {
                        "score": effectiveness_eval.overall_score, "meets_threshold": effectiveness_eval.meets_threshold,
                        "criteria": {
                            "goal_achievement": { "score": effectiveness_eval.goal_achievement_score, "feedback": effectiveness_eval.goal_achievement_feedback },
                            "comprehensiveness": { "score": effectiveness_eval.comprehensiveness_score, "feedback": effectiveness_eval.comprehensiveness_feedback },
                            "efficiency": { "score": effectiveness_eval.efficiency_score, "feedback": effectiveness_eval.efficiency_feedback }
                        },
                        "suggestions": effectiveness_eval.suggestions
                    },
                    "efficiency": {
                        "score": efficiency_eval.overall_score, "meets_threshold": efficiency_eval.meets_threshold,
                        "criteria": {
                            "response_time": { "score": efficiency_eval.response_time_score, "feedback": efficiency_eval.response_time_feedback },
                            "resource_utilization": { "score": efficiency_eval.resource_utilization_score, "feedback": efficiency_eval.resource_utilization_feedback },
                            "complexity_management": { "score": efficiency_eval.complexity_management_score, "feedback": efficiency_eval.complexity_management_feedback }
                        },
                        "suggestions": efficiency_eval.suggestions
                    }
                },
                "meta_evaluation": {
                    "overall_score": meta_eval.overall_score, "quality_level": meta_eval.quality_level,
                    "meets_threshold": meta_eval.meets_threshold, "key_strengths": meta_eval.key_strengths,
                    "key_weaknesses": meta_eval.key_weaknesses, "improvement_suggestions": meta_eval.improvement_suggestions
                }
            }

        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {e}", exc_info=True)
            if self.dev_logger:
                self.dev_logger.error(f"Comprehensive evaluation failed: {e}", extra={'agent_name': 'Meta Critic'})
            return {
                "aspect_evaluations": {},
                "meta_evaluation": {
                    "overall_score": 0.0, "quality_level": "unknown", "meets_threshold": False,
                    "key_strengths": [], "key_weaknesses": ["Evaluation failed due to internal error"],
                    "improvement_suggestions": ["Check application logs for details"]
                }
            }

    # Helper methods from the original implementation

    def _calculate_trend(self, values: List[float]) -> str:
        if len(values) < 5: return "stable"
        recent = values[-min(10, len(values)):]
        n = len(recent)
        x = list(range(n))
        y = recent
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        if denominator == 0: return "stable"
        slope = numerator / denominator
        if abs(slope) < 0.01: return "stable"
        elif slope > 0: return "improving" if slope > 0.03 else "slightly improving"
        else: return "declining" if slope < -0.03 else "slightly declining"

    def _calculate_confidence_interval(self, values: List[float]) -> Dict[str, float]:
        n = len(values)
        if n == 0: return {"lower": 0.0, "upper": 0.0, "std_error": 0.0}
        mean = sum(values) / n
        if n < 2: return {"lower": mean, "upper": mean, "std_error": 0.0}
        variance = sum((v - mean) ** 2 for v in values) / (n - 1) # Use sample variance (n-1)
        std_dev = math.sqrt(variance)
        std_error = std_dev / math.sqrt(n)
        # Using scipy for t-value would be more accurate, but approximating for simplicity
        t_value = 2.0 # Rough approximation for 95% CI with moderate sample size
        if n < 5: t_value = 2.776 # df=4
        elif n < 10: t_value = 2.262 # df=9
        elif n < 30: t_value = 2.045 # df=29
        else: t_value = 1.96 # z-value
        margin = t_value * std_error
        return {
            "lower": max(0.0, mean - margin),
            "upper": min(1.0, mean + margin),
            "std_error": std_error
        }

    def _evaluate_criterion(self, aspect: str, criterion: str, content: Any, context: Dict[str, Any]) -> float:
        # Fallback - unchanged from previous version
        text_content = str(content.get("content", content)) if isinstance(content, dict) else str(content)
        content_length_norm = min(1.0, len(text_content) / 1000.0)
        words = text_content.split()
        num_words = len(words)
        content_complexity = min(1.0, (len(set(words)) / num_words) if num_words > 0 else 0.0)
        score = 0.5
        try:
            if aspect == "consistency":
                if criterion == "coherence": score = 0.5 + (content_length_norm * 0.2) + (content_complexity * 0.2)
                elif criterion == "continuity": score = 0.6 + random.uniform(-0.1, 0.1)
                elif criterion == "context_adherence": score = 0.6 + random.uniform(-0.1, 0.1)
            elif aspect == "effectiveness":
                if criterion == "goal_achievement": score = 0.6 + random.uniform(-0.1, 0.1)
                elif criterion == "comprehensiveness": score = 0.4 + (content_length_norm * 0.5)
                elif criterion == "efficiency": score = 0.8 - (content_length_norm * 0.3)
            elif aspect == "efficiency":
                if criterion == "response_time":
                    response_time = context.get("response_time")
                    if isinstance(response_time, (int, float)): score = max(0.0, 1.0 - (response_time / 5.0))
                    else: score = 0.7
                elif criterion == "resource_utilization": score = 0.7 + random.uniform(-0.1, 0.1)
                elif criterion == "complexity_management": score = 0.8 - (content_complexity * 0.3)
        except Exception as e:
            logger.warning(f"Error during fallback criterion evaluation ({aspect}/{criterion}): {e}")
            score = 0.5
        return max(0.0, min(1.0, score))

    def _generate_criterion_description(self, aspect: str, criterion: str, score: float) -> str:
        # Fallback - unchanged from previous version
        if score >= 0.9: quality = "excellent"
        elif score >= 0.8: quality = "very good"
        elif score >= 0.7: quality = "good"
        elif score >= 0.6: quality = "satisfactory"
        elif score >= 0.5: quality = "fair"
        elif score >= 0.4: quality = "needs improvement"
        else: quality = "poor"
        desc = f"{quality.capitalize()} performance on {criterion} ({score:.2f})"
        mapping = {
            "consistency": {"coherence": f"{quality.capitalize()} coherence.", "continuity": f"{quality.capitalize()} continuity.", "context_adherence": f"{quality.capitalize()} context adherence."},
            "effectiveness": {"goal_achievement": f"{quality.capitalize()} goal achievement.", "comprehensiveness": f"{quality.capitalize()} comprehensiveness.", "efficiency": f"{quality.capitalize()} goal efficiency."},
            "efficiency": {"response_time": f"{quality.capitalize()} response time.", "resource_utilization": f"{quality.capitalize()} resource use.", "complexity_management": f"{quality.capitalize()} complexity management."}
        }
        try: desc = mapping[aspect][criterion]
        except KeyError: pass
        return f"{desc} (Score: {score:.2f})"


    def _determine_quality_level(self, score: float) -> str:
        # Unchanged
        if score >= 0.9: return "outstanding"
        elif score >= 0.8: return "excellent"
        elif score >= 0.7: return "good"
        elif score >= 0.6: return "satisfactory"
        elif score >= 0.5: return "fair"
        elif score >= 0.4: return "needs improvement"
        else: return "unsatisfactory"

    def _generate_improvement_suggestions(self, aspect: str, evaluation: Dict[str, Dict[str, Any]], overall_score: float) -> List[str]:
        # Fallback - unchanged from previous version
        suggestions = []
        criteria_scores = sorted([(c, d["score"]) for c, d in evaluation.items()], key=lambda x: x[1])
        suggestions_added = 0
        for criterion, score in criteria_scores:
            if score < 0.7 and suggestions_added < 2:
                suggestion_map = {
                    "consistency": {"coherence": "Improve logical flow.", "continuity": "Ensure consistent theme/tone.", "context_adherence": "Align better with context."},
                    "effectiveness": {"goal_achievement": "Refocus on primary objective.", "comprehensiveness": "Add more relevant details.", "efficiency": "Streamline content/presentation."},
                    "efficiency": {"response_time": "Optimize process for speed.", "resource_utilization": "Investigate resource usage.", "complexity_management": "Simplify structure/vocabulary."}
                }
                try:
                    suggestions.append(f"Improve {criterion}: {suggestion_map[aspect][criterion]}")
                    suggestions_added += 1
                except KeyError:
                    suggestions.append(f"Review {criterion} performance (score: {score:.2f}).")
                    suggestions_added += 1
        if overall_score < self.quality_threshold and not suggestions:
            suggestions.append(f"Overall {aspect} score ({overall_score:.2f}) below threshold ({self.quality_threshold}). Review criteria.")
        elif overall_score < self.quality_threshold and suggestions:
             suggestions.append(f"Address specific points to improve overall {aspect} score ({overall_score:.2f}).")
        if not suggestions and overall_score >= self.quality_threshold:
             suggestions.append(f"Maintain current {aspect} performance (score: {overall_score:.2f}).")
        return suggestions
