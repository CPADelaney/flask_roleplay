# nyx/core/internal_feedback_system.py

import asyncio
import json
import logging
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import math
import random
from pydantic import BaseModel
import pathlib # Added for path handling

from agents import Agent, Runner, function_tool, set_default_openai_key

# --- Existing Logger Setup ---
logger = logging.getLogger(__name__) # Main logger for the module

# --- Dev Log Configuration ---
DEV_LOG_FILE_PATH = "/dev_log/internal_feedback_suggestions.log" # As requested

# --- Setup Dev Logger ---
# Create a specific logger for development suggestions
dev_logger = logging.getLogger("DevSuggestionsLog")
dev_logger.setLevel(logging.INFO) # Log suggestions at INFO level
dev_logger.propagate = False # Prevent suggestions from going to the root logger/console unless configured

# Ensure the log directory exists
try:
    log_dir = pathlib.Path(DEV_LOG_FILE_PATH).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create a file handler for the dev log
    dev_log_handler = logging.FileHandler(DEV_LOG_FILE_PATH, encoding='utf-8')
    dev_log_formatter = logging.Formatter('%(asctime)s - %(message)s') # Simple format: timestamp - message
    dev_log_handler.setFormatter(dev_log_formatter)

    # Add the handler to the dev logger
    if not dev_logger.handlers: # Avoid adding handlers multiple times if module reloads
        dev_logger.addHandler(dev_log_handler)

except Exception as e:
    # Log an error using the main logger if dev log setup fails
    logger.error(f"Failed to set up development suggestions log at {DEV_LOG_FILE_PATH}: {e}")
    # Assign None to prevent logging attempts later if setup failed
    dev_logger = None


# Initialize the OpenAI API key from environment variable (or you can set it directly)
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    set_default_openai_key(api_key)

class InternalFeedbackSystem:
    """
    System for internal feedback, evaluation, and quality assessment.
    Provides mechanisms for self-monitoring and improvement.
    Logs improvement suggestions to a development log file.
    """

    def __init__(self):
        self.performance_metrics = {}
        self.confidence_tracking = {
            "history": [],
            "calibration": {
                "buckets": [0.0] * 10,  # 10 confidence buckets (0.0-1.0)
                "correct": [0.0] * 10   # Correct predictions in each bucket
            }
        }
        self.evaluation_criteria = {
            "consistency": {
                "coherence": 0.4,
                "continuity": 0.3,
                "context_adherence": 0.3
            },
            "effectiveness": {
                "goal_achievement": 0.5,
                "comprehensiveness": 0.3,
                "efficiency": 0.2
            },
            "efficiency": {
                "response_time": 0.4,
                "resource_utilization": 0.3,
                "complexity_management": 0.3
            }
        }
        self.quality_threshold = 0.7

        # Keep a reference to the dev logger (or None if setup failed)
        self.dev_logger = dev_logger

        # Initialize the agent-based critic system
        self._init_critic_agents()

    def _log_suggestions(self, source: str, suggestions: List[str]):
        """Helper method to log suggestions to the dev log."""
        if self.dev_logger and suggestions:
            log_prefix = f"[{source.upper()} Suggestions]"
            try:
                for suggestion in suggestions:
                    self.dev_logger.info(f"{log_prefix} {suggestion}")
            except Exception as e:
                # Log error during logging using the main logger
                logger.error(f"Error writing to development suggestions log: {e}")


    def _init_critic_agents(self):
        """Initialize the agent-based critic system"""
        # Create output types for each critic agent
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

        # Create the critic agents
        self.consistency_critic = Agent(
            name="Consistency Critic",
            instructions=(
                "You evaluate content for consistency, focusing on these aspects:\n"
                "1. Coherence (40%): How well the content flows logically and maintains internal consistency\n"
                "2. Continuity (30%): How well the content maintains narrative or thematic continuity\n"
                "3. Context Adherence (30%): How well the content adheres to the provided context\n\n"
                "You should analyze the content thoroughly and provide specific feedback for each aspect.\n"
                "Score each aspect on a scale of 0.0 to 1.0, where:\n"
                "- 0.0-0.3: Poor\n"
                "- 0.4-0.6: Fair\n"
                "- 0.7-0.8: Good\n"
                "- 0.9-1.0: Excellent\n\n"
                "Calculate an overall weighted score and determine if it meets the quality threshold.\n"
                "Provide specific suggestions for improvement."
            ),
            output_type=ConsistencyEvaluation
        )

        self.effectiveness_critic = Agent(
            name="Effectiveness Critic",
            instructions=(
                "You evaluate content for effectiveness, focusing on these aspects:\n"
                "1. Goal Achievement (50%): How well the content achieves its intended purpose\n"
                "2. Comprehensiveness (30%): How comprehensive the content is in covering relevant information\n"
                "3. Efficiency (20%): How efficiently the content achieves its goals\n\n"
                "You should analyze the content thoroughly and provide specific feedback for each aspect.\n"
                "Score each aspect on a scale of 0.0 to 1.0, where:\n"
                "- 0.0-0.3: Poor\n"
                "- 0.4-0.6: Fair\n"
                "- 0.7-0.8: Good\n"
                "- 0.9-1.0: Excellent\n\n"
                "Calculate an overall weighted score and determine if it meets the quality threshold.\n"
                "Provide specific suggestions for improvement."
            ),
            output_type=EffectivenessEvaluation
        )

        self.efficiency_critic = Agent(
            name="Efficiency Critic",
            instructions=(
                "You evaluate content for efficiency, focusing on these aspects:\n"
                "1. Response Time (40%): How timely the content is delivered\n"
                "2. Resource Utilization (30%): How efficiently resources are used\n"
                "3. Complexity Management (30%): How well complexity is managed\n\n"
                "You should analyze the content thoroughly and provide specific feedback for each aspect.\n"
                "Score each aspect on a scale of 0.0 to 1.0, where:\n"
                "- 0.0-0.3: Poor\n"
                "- 0.4-0.6: Fair\n"
                "- 0.7-0.8: Good\n"
                "- 0.9-1.0: Excellent\n\n"
                "Calculate an overall weighted score and determine if it meets the quality threshold.\n"
                "Provide specific suggestions for improvement."
            ),
            output_type=EfficiencyEvaluation
        )

        self.meta_critic = Agent(
            name="Meta Critic",
            instructions=(
                "You synthesize evaluations from the consistency, effectiveness, and efficiency critics.\n"
                "Calculate a weighted overall score using these weights:\n"
                "- Consistency: 40%\n"
                "- Effectiveness: 40%\n"
                "- Efficiency: 20%\n\n"
                "Determine the quality level based on the overall score:\n"
                "- 0.9-1.0: Outstanding\n"
                "- 0.8-0.9: Excellent\n"
                "- 0.7-0.8: Good\n"
                "- 0.6-0.7: Satisfactory\n"
                "- 0.5-0.6: Fair\n"
                "- 0.4-0.5: Needs Improvement\n"
                "- 0.0-0.4: Unsatisfactory\n\n"
                "Identify key strengths and weaknesses.\n"
                "Provide prioritized improvement suggestions."
            ),
            output_type=MetaEvaluation
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

        # Add new value
        self.performance_metrics[metric]["values"].append(value)
        self.performance_metrics[metric]["timestamps"].append(datetime.now().isoformat())

        # Keep only recent history (last 100 values)
        if len(self.performance_metrics[metric]["values"]) > 100:
            self.performance_metrics[metric]["values"].pop(0)
            self.performance_metrics[metric]["timestamps"].pop(0)

        # Update statistics
        values = self.performance_metrics[metric]["values"]
        self.performance_metrics[metric]["mean"] = sum(values) / len(values)

        if len(values) > 1:
            mean = self.performance_metrics[metric]["mean"]
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            self.performance_metrics[metric]["std_dev"] = math.sqrt(variance)

        self.performance_metrics[metric]["min"] = min(values)
        self.performance_metrics[metric]["max"] = max(values)

        # Calculate trend
        self.performance_metrics[metric]["trend"] = self._calculate_trend(values)

        # Calculate confidence interval
        if len(values) >= 10:
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

        # Update calibration buckets
        bucket = min(9, int(confidence * 10))  # Map to 0-9 bucket index
        self.confidence_tracking["calibration"]["buckets"][bucket] += 1
        if success:
            self.confidence_tracking["calibration"]["correct"][bucket] += 1

        # Calculate calibration
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

        # Calculate overall metrics
        total_instances = sum(self.confidence_tracking["calibration"]["buckets"])
        if total_instances > 0:
            overall_accuracy = sum(self.confidence_tracking["calibration"]["correct"]) / total_instances

            # Calculate Brier score (mean squared error of confidence)
            brier_score = 0.0
            history_len = len(self.confidence_tracking["history"])
            if history_len > 0:
                for entry in self.confidence_tracking["history"]:
                    error = entry["confidence"] - (1.0 if entry["success"] else 0.0)
                    brier_score += error ** 2
                brier_score /= history_len

            # Calculate overconfidence/underconfidence
            avg_confidence = 0.0
            if history_len > 0:
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
            "current": {
                "confidence": confidence,
                "success": success
            },
            "calibration": calibration,
            "metrics": confidence_metrics
        }

    async def critic_evaluate(self,
                        aspect: str,
                        content: Any,
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate content on a specific aspect using critic agents.
        Logs suggestions to the dev log. Falls back to basic evaluation if agent fails.

        Args:
            aspect: Aspect to evaluate (consistency, effectiveness, efficiency)
            content: Content to evaluate
            context: Context for evaluation

        Returns:
            Evaluation results
        """
        if aspect not in self.evaluation_criteria:
            return {
                "error": f"Unknown evaluation aspect: {aspect}",
                "available_aspects": list(self.evaluation_criteria.keys())
            }

        result_data = {}
        try:
            # Format the content and context
            if isinstance(content, dict) and "content" in content:
                content_text = content["content"]
            else:
                content_text = str(content)

            # Format context
            context_text = "\n".join([f"{k}: {v}" for k, v in context.items()])

            # Prepare the prompt
            prompt = f"Please evaluate this content on the aspect of {aspect}:\n\nContent:\n{content_text}\n\nContext:\n{context_text}"

            # Run the appropriate critic agent
            if aspect == "consistency":
                result = await Runner.run(self.consistency_critic, prompt)
                evaluation = result.final_output_as(result.last_agent.output_type)

                # Log suggestions
                self._log_suggestions(f"{aspect} Critic", evaluation.suggestions)

                # Format the result to match the original API
                result_data = {
                    "aspect": aspect,
                    "criteria_evaluations": {
                        "coherence": {
                            "score": evaluation.coherence_score,
                            "weight": self.evaluation_criteria[aspect]["coherence"],
                            "description": evaluation.coherence_feedback
                        },
                        "continuity": {
                            "score": evaluation.continuity_score,
                            "weight": self.evaluation_criteria[aspect]["continuity"],
                            "description": evaluation.continuity_feedback
                        },
                        "context_adherence": {
                            "score": evaluation.context_adherence_score,
                            "weight": self.evaluation_criteria[aspect]["context_adherence"],
                            "description": evaluation.context_adherence_feedback
                        }
                    },
                    "weighted_score": evaluation.overall_score,
                    "quality_level": self._determine_quality_level(evaluation.overall_score),
                    "meets_threshold": evaluation.meets_threshold,
                    "improvement_suggestions": evaluation.suggestions
                }

            elif aspect == "effectiveness":
                result = await Runner.run(self.effectiveness_critic, prompt)
                evaluation = result.final_output_as(result.last_agent.output_type)

                # Log suggestions
                self._log_suggestions(f"{aspect} Critic", evaluation.suggestions)

                # Format the result to match the original API
                result_data = {
                    "aspect": aspect,
                    "criteria_evaluations": {
                        "goal_achievement": {
                            "score": evaluation.goal_achievement_score,
                            "weight": self.evaluation_criteria[aspect]["goal_achievement"],
                            "description": evaluation.goal_achievement_feedback
                        },
                        "comprehensiveness": {
                            "score": evaluation.comprehensiveness_score,
                            "weight": self.evaluation_criteria[aspect]["comprehensiveness"],
                            "description": evaluation.comprehensiveness_feedback
                        },
                        "efficiency": {
                            "score": evaluation.efficiency_score,
                            "weight": self.evaluation_criteria[aspect]["efficiency"],
                            "description": evaluation.efficiency_feedback
                        }
                    },
                    "weighted_score": evaluation.overall_score,
                    "quality_level": self._determine_quality_level(evaluation.overall_score),
                    "meets_threshold": evaluation.meets_threshold,
                    "improvement_suggestions": evaluation.suggestions
                }

            elif aspect == "efficiency":
                result = await Runner.run(self.efficiency_critic, prompt)
                evaluation = result.final_output_as(result.last_agent.output_type)

                # Log suggestions
                self._log_suggestions(f"{aspect} Critic", evaluation.suggestions)

                # Format the result to match the original API
                result_data = {
                    "aspect": aspect,
                    "criteria_evaluations": {
                        "response_time": {
                            "score": evaluation.response_time_score,
                            "weight": self.evaluation_criteria[aspect]["response_time"],
                            "description": evaluation.response_time_feedback
                        },
                        "resource_utilization": {
                            "score": evaluation.resource_utilization_score,
                            "weight": self.evaluation_criteria[aspect]["resource_utilization"],
                            "description": evaluation.resource_utilization_feedback
                        },
                        "complexity_management": {
                            "score": evaluation.complexity_management_score,
                            "weight": self.evaluation_criteria[aspect]["complexity_management"],
                            "description": evaluation.complexity_management_feedback
                        }
                    },
                    "weighted_score": evaluation.overall_score,
                    "quality_level": self._determine_quality_level(evaluation.overall_score),
                    "meets_threshold": evaluation.meets_threshold,
                    "improvement_suggestions": evaluation.suggestions
                }
            return result_data # Return agent result if successful

        except Exception as e:
            logger.error(f"Error in agent-based evaluation for {aspect}: {e}. Falling back to basic evaluation.")
            # Fallback to the original implementation if the agent-based system fails
            criteria = self.evaluation_criteria[aspect]
            evaluation = {}

            # Evaluate each criterion using the original implementation
            for criterion, weight in criteria.items():
                score = self._evaluate_criterion(aspect, criterion, content, context)
                evaluation[criterion] = {
                    "score": score,
                    "weight": weight,
                    "description": self._generate_criterion_description(aspect, criterion, score)
                }

            # Calculate weighted score
            weighted_score = sum(
                eval_data["score"] * eval_data["weight"]
                for eval_data in evaluation.values()
            )

            # Generate overall assessment
            quality_level = self._determine_quality_level(weighted_score)

            # Generate improvement suggestions
            improvements = self._generate_improvement_suggestions(aspect, evaluation, weighted_score)

            # Log fallback suggestions
            self._log_suggestions(f"{aspect} Fallback", improvements)

            return {
                "aspect": aspect,
                "criteria_evaluations": evaluation,
                "weighted_score": weighted_score,
                "quality_level": quality_level,
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
                "mean": data["mean"],
                "std_dev": data.get("std_dev", 0.0),
                "trend": data["trend"],
                "sample_size": len(data["values"])
            }

        # Calculate confidence calibration metrics
        confidence_metrics = {}
        if self.confidence_tracking["history"]:
            total_instances = sum(self.confidence_tracking["calibration"]["buckets"])
            history_len = len(self.confidence_tracking["history"])
            if total_instances > 0 and history_len > 0:
                overall_accuracy = sum(self.confidence_tracking["calibration"]["correct"]) / total_instances

                # Calculate average confidence
                avg_confidence = sum(entry["confidence"] for entry in self.confidence_tracking["history"]) / history_len

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
        Logs meta-suggestions to the dev log.

        Args:
            content: Content to evaluate
            context: Context for evaluation

        Returns:
            Comprehensive evaluation results
        """
        # Format the content and context
        if isinstance(content, dict) and "content" in content:
            content_text = content["content"]
        else:
            content_text = str(content)

        # Format context
        context_text = "\n".join([f"{k}: {v}" for k, v in context.items()])

        # Prepare the prompt
        prompt = f"Please evaluate this content:\n\nContent:\n{content_text}\n\nContext:\n{context_text}"

        try:
            # Run all three critic agents in parallel
            tasks = [
                Runner.run(self.consistency_critic, prompt),
                Runner.run(self.effectiveness_critic, prompt),
                Runner.run(self.efficiency_critic, prompt)
            ]

            consistency_result, effectiveness_result, efficiency_result = await asyncio.gather(*tasks)

            consistency_eval = consistency_result.final_output_as(consistency_result.last_agent.output_type)
            effectiveness_eval = effectiveness_result.final_output_as(effectiveness_result.last_agent.output_type)
            efficiency_eval = efficiency_result.final_output_as(efficiency_result.last_agent.output_type)

            # Log suggestions from individual critics
            self._log_suggestions("Comprehensive Consistency", consistency_eval.suggestions)
            self._log_suggestions("Comprehensive Effectiveness", effectiveness_eval.suggestions)
            self._log_suggestions("Comprehensive Efficiency", efficiency_eval.suggestions)

            # Format the input for the meta-critic
            meta_prompt = (
                "Synthesize these evaluations:\n\n"
                f"Consistency Evaluation:\n{json.dumps(consistency_eval.dict())}\n\n" # Use dict for robust serialization
                f"Effectiveness Evaluation:\n{json.dumps(effectiveness_eval.dict())}\n\n"
                f"Efficiency Evaluation:\n{json.dumps(efficiency_eval.dict())}"
            )

            # Run meta-critic
            meta_result = await Runner.run(self.meta_critic, meta_prompt)
            meta_eval = meta_result.final_output_as(meta_result.last_agent.output_type)

            # Log meta-suggestions
            self._log_suggestions("Meta Critic", meta_eval.improvement_suggestions)

            # Format the final evaluation result
            return {
                "aspect_evaluations": {
                    "consistency": {
                        "score": consistency_eval.overall_score,
                        "meets_threshold": consistency_eval.meets_threshold,
                        "criteria": {
                            "coherence": {
                                "score": consistency_eval.coherence_score,
                                "feedback": consistency_eval.coherence_feedback
                            },
                            "continuity": {
                                "score": consistency_eval.continuity_score,
                                "feedback": consistency_eval.continuity_feedback
                            },
                            "context_adherence": {
                                "score": consistency_eval.context_adherence_score,
                                "feedback": consistency_eval.context_adherence_feedback
                            }
                        },
                        "suggestions": consistency_eval.suggestions
                    },
                    "effectiveness": {
                        "score": effectiveness_eval.overall_score,
                        "meets_threshold": effectiveness_eval.meets_threshold,
                        "criteria": {
                            "goal_achievement": {
                                "score": effectiveness_eval.goal_achievement_score,
                                "feedback": effectiveness_eval.goal_achievement_feedback
                            },
                            "comprehensiveness": {
                                "score": effectiveness_eval.comprehensiveness_score,
                                "feedback": effectiveness_eval.comprehensiveness_feedback
                            },
                            "efficiency": {
                                "score": effectiveness_eval.efficiency_score,
                                "feedback": effectiveness_eval.efficiency_feedback
                            }
                        },
                        "suggestions": effectiveness_eval.suggestions
                    },
                    "efficiency": {
                        "score": efficiency_eval.overall_score,
                        "meets_threshold": efficiency_eval.meets_threshold,
                        "criteria": {
                            "response_time": {
                                "score": efficiency_eval.response_time_score,
                                "feedback": efficiency_eval.response_time_feedback
                            },
                            "resource_utilization": {
                                "score": efficiency_eval.resource_utilization_score,
                                "feedback": efficiency_eval.resource_utilization_feedback
                            },
                            "complexity_management": {
                                "score": efficiency_eval.complexity_management_score,
                                "feedback": efficiency_eval.complexity_management_feedback
                            }
                        },
                        "suggestions": efficiency_eval.suggestions
                    }
                },
                "meta_evaluation": {
                    "overall_score": meta_eval.overall_score,
                    "quality_level": meta_eval.quality_level,
                    "meets_threshold": meta_eval.meets_threshold,
                    "key_strengths": meta_eval.key_strengths,
                    "key_weaknesses": meta_eval.key_weaknesses,
                    "improvement_suggestions": meta_eval.improvement_suggestions
                }
            }

        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {e}")
            # Log the failure to dev log as well
            if self.dev_logger:
                self.dev_logger.error(f"[META Critic Error] Comprehensive evaluation failed: {e}")
            # Return an empty result if evaluation fails
            return {
                "aspect_evaluations": {},
                "meta_evaluation": {
                    "overall_score": 0.0,
                    "quality_level": "unknown",
                    "meets_threshold": False,
                    "key_strengths": [],
                    "key_weaknesses": ["Evaluation failed due to internal error"],
                    "improvement_suggestions": ["Check application logs for details"]
                }
            }

    # Helper methods from the original implementation

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from recent values"""
        if len(values) < 5:
            return "stable"

        # Use more recent values for trend (last 10 or fewer)
        recent = values[-min(10, len(values)):]

        # Simple linear regression to find slope
        n = len(recent)
        x = list(range(n))
        y = recent

        # Calculate slope
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Determine trend
        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            return "improving" if slope > 0.03 else "slightly improving"
        else:
            return "declining" if slope < -0.03 else "slightly declining"

    def _calculate_confidence_interval(self, values: List[float]) -> Dict[str, float]:
        """Calculate 95% confidence interval for the mean"""
        n = len(values)
        if n == 0:
            return {"lower": 0.0, "upper": 0.0, "std_error": 0.0}
        mean = sum(values) / n

        if n < 2:
            return {
                "lower": mean,
                "upper": mean,
                "std_error": 0.0
            }

        # Calculate standard error
        variance = sum((v - mean) ** 2 for v in values) / n # Population variance if this is all data
        # Use sample variance if treating as sample: variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std_dev = math.sqrt(variance)
        std_error = std_dev / math.sqrt(n)

        # Use t-distribution for small samples, normal for large
        if n < 30:
            # Approximation of t-value for 95% confidence (degrees of freedom = n-1)
            # Rough approximations:
            t_value = 2.262 if n == 10 else 2.045 if n == 29 else 2.0 # Simplified
        else:
            t_value = 1.96  # Normal distribution (z-value)

        margin = t_value * std_error

        return {
            "lower": max(0.0, mean - margin),
            "upper": min(1.0, mean + margin),
            "std_error": std_error
        }

    def _evaluate_criterion(self,
                           aspect: str,
                           criterion: str,
                           content: Any,
                           context: Dict[str, Any]) -> float:
        """Evaluate content on a specific criterion - fallback implementation"""
        # Check for content type
        if isinstance(content, dict) and "content" in content:
            # Use content field if available
            text_content = str(content["content"])
        else:
            # Convert to string
            text_content = str(content)

        # Get base features
        content_length_raw = len(text_content)
        content_length_norm = min(1.0, content_length_raw / 1000.0) # Normalize, cap at 1000 chars
        words = text_content.split()
        num_words = len(words)
        num_unique_words = len(set(words))
        content_complexity = min(1.0, (num_unique_words / num_words) if num_words > 0 else 0.0) # Lexical density

        # Evaluate based on aspect and criterion
        score = 0.5 # Default score

        try:
            if aspect == "consistency":
                if criterion == "coherence":
                    # Simple heuristic: longer, less repetitive text might be more coherent
                    score = 0.5 + (content_length_norm * 0.2) + (content_complexity * 0.2)
                elif criterion == "continuity":
                    # Hard to measure simply; use default with slight variation
                    score = 0.6 + random.uniform(-0.1, 0.1)
                elif criterion == "context_adherence":
                    # Basic check if context keys are mentioned? Too simplistic. Use default.
                    score = 0.6 + random.uniform(-0.1, 0.1)

            elif aspect == "effectiveness":
                if criterion == "goal_achievement":
                    # Very context-dependent, hard to guess. Use default.
                    score = 0.6 + random.uniform(-0.1, 0.1)
                elif criterion == "comprehensiveness":
                    # Use content length as a proxy
                    score = 0.4 + (content_length_norm * 0.5)
                elif criterion == "efficiency":
                    # Shorter might be more efficient? Inverse length?
                    score = 0.8 - (content_length_norm * 0.3)

            elif aspect == "efficiency":
                if criterion == "response_time":
                    # Check response time if available in context
                    response_time = context.get("response_time") # Expect float in seconds
                    if isinstance(response_time, (int, float)):
                         # Lower response time is better. Score = 1 for 0s, 0 for 5s+.
                        score = max(0.0, 1.0 - (response_time / 5.0))
                    else:
                        score = 0.7 # Default if not provided/invalid
                elif criterion == "resource_utilization":
                    # Impossible to know without metrics. Use default.
                    score = 0.7 + random.uniform(-0.1, 0.1)
                elif criterion == "complexity_management":
                    # Lower lexical complexity might imply better management?
                    score = 0.8 - (content_complexity * 0.3)

        except Exception as e:
            logger.warning(f"Error during fallback criterion evaluation ({aspect}/{criterion}): {e}")
            score = 0.5 # Default on error

        return max(0.0, min(1.0, score)) # Clamp score between 0.0 and 1.0


    def _generate_criterion_description(self,
                                       aspect: str,
                                       criterion: str,
                                       score: float) -> str:
        """Generate a description for a criterion score - fallback implementation"""
        if score >= 0.9: quality = "excellent"
        elif score >= 0.8: quality = "very good"
        elif score >= 0.7: quality = "good"
        elif score >= 0.6: quality = "satisfactory"
        elif score >= 0.5: quality = "fair"
        elif score >= 0.4: quality = "needs improvement"
        else: quality = "poor"

        # Generate descriptions based on aspect and criterion
        desc = f"{quality.capitalize()} performance on {criterion} ({score:.2f})" # Default desc
        mapping = {
            "consistency": {
                "coherence": f"{quality.capitalize()} coherence in content structure and flow.",
                "continuity": f"{quality.capitalize()} continuity throughout the content.",
                "context_adherence": f"{quality.capitalize()} adherence to the provided context."
            },
            "effectiveness": {
                "goal_achievement": f"{quality.capitalize()} achievement of intended goals.",
                "comprehensiveness": f"{quality.capitalize()} coverage of relevant information.",
                "efficiency": f"{quality.capitalize()} efficiency in achieving goals relative to content."
            },
            "efficiency": {
                "response_time": f"{quality.capitalize()} response time performance.",
                "resource_utilization": f"{quality.capitalize()} assumed utilization of resources.",
                "complexity_management": f"{quality.capitalize()} management of content complexity."
            }
        }
        try:
            desc = mapping[aspect][criterion]
        except KeyError:
            pass # Use default if not found

        return f"{desc} (Score: {score:.2f})"


    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level based on score"""
        if score >= 0.9: return "outstanding"
        elif score >= 0.8: return "excellent"
        elif score >= 0.7: return "good"
        elif score >= 0.6: return "satisfactory"
        elif score >= 0.5: return "fair"
        elif score >= 0.4: return "needs improvement"
        else: return "unsatisfactory"

    def _generate_improvement_suggestions(self,
                                         aspect: str,
                                         evaluation: Dict[str, Dict[str, Any]],
                                         overall_score: float) -> List[str]:
        """Generate improvement suggestions based on evaluation - fallback implementation"""
        suggestions = []

        # Find the weakest criteria (lowest scores)
        criteria_scores = [(criterion, data["score"]) for criterion, data in evaluation.items()]
        criteria_scores.sort(key=lambda x: x[1])

        # Generate suggestions for up to 2 lowest scoring criteria if below 'good' threshold
        suggestions_added = 0
        for criterion, score in criteria_scores:
            if score < 0.7 and suggestions_added < 2: # Suggest improvements for criteria below "good"
                suggestion_map = {
                    "consistency": {
                        "coherence": "Improve logical flow and transitions between ideas.",
                        "continuity": "Ensure consistent theme, tone, and narrative throughout.",
                        "context_adherence": "Review content alignment with provided context details.",
                    },
                    "effectiveness": {
                        "goal_achievement": "Refocus content to more directly address the primary objective.",
                        "comprehensiveness": "Consider adding more relevant details or examples.",
                        "efficiency": "Streamline the content or presentation for better clarity/impact.",
                    },
                    "efficiency": {
                        "response_time": "Optimize underlying process for faster generation/response.",
                        "resource_utilization": "Investigate potential inefficiencies in resource usage (if applicable).",
                        "complexity_management": "Simplify sentence structure or vocabulary where appropriate.",
                    }
                }
                try:
                    suggestions.append(f"Improve {criterion}: {suggestion_map[aspect][criterion]}")
                    suggestions_added += 1
                except KeyError:
                    suggestions.append(f"Review {criterion} performance (score: {score:.2f}).") # Generic fallback
                    suggestions_added += 1

        # Add general suggestion if overall score is low
        if overall_score < self.quality_threshold and not suggestions:
            suggestions.append(f"Overall {aspect} score ({overall_score:.2f}) is below threshold ({self.quality_threshold}). Review all criteria.")
        elif overall_score < self.quality_threshold and suggestions:
             suggestions.append(f"Address specific points above to improve overall {aspect} score ({overall_score:.2f}).")

        if not suggestions and overall_score >= self.quality_threshold:
             suggestions.append(f"Maintain current {aspect} performance (score: {overall_score:.2f}).")

        return suggestions
