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

logger = logging.getLogger(__name__)

class InternalFeedbackSystem:
    """
    System for internal feedback, evaluation, and quality assessment.
    Provides mechanisms for self-monitoring and improvement.
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
            for entry in self.confidence_tracking["history"]:
                error = entry["confidence"] - (1.0 if entry["success"] else 0.0)
                brier_score += error ** 2
            brier_score /= len(self.confidence_tracking["history"])
            
            # Calculate overconfidence/underconfidence
            conf_sum = sum(entry["confidence"] for entry in self.confidence_tracking["history"])
            avg_confidence = conf_sum / len(self.confidence_tracking["history"])
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
        Evaluate content on a specific aspect.
        
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
        
        criteria = self.evaluation_criteria[aspect]
        evaluation = {}
        
        # Evaluate each criterion
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
            if total_instances > 0:
                overall_accuracy = sum(self.confidence_tracking["calibration"]["correct"]) / total_instances
                
                # Calculate average confidence
                avg_confidence = sum(entry["confidence"] for entry in self.confidence_tracking["history"]) / len(self.confidence_tracking["history"])
                
                confidence_metrics = {
                    "overall_accuracy": overall_accuracy,
                    "average_confidence": avg_confidence,
                    "bias": avg_confidence - overall_accuracy,
                    "sample_size": len(self.confidence_tracking["history"])
                }
        
        return {
            "performance": performance_summary,
            "confidence": confidence_metrics,
            "evaluation_criteria": self.evaluation_criteria,
            "quality_threshold": self.quality_threshold
        }
    
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
        mean = sum(values) / n
        
        if n < 2:
            return {
                "lower": mean,
                "upper": mean
            }
        
        # Calculate standard error
        variance = sum((v - mean) ** 2 for v in values) / n
        std_dev = math.sqrt(variance)
        std_error = std_dev / math.sqrt(n)
        
        # Use t-distribution for small samples, normal for large
        if n < 30:
            # Approximation of t-value for 95% confidence
            t_value = 2.0 if n < 10 else 1.96
        else:
            t_value = 1.96  # Normal distribution
        
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
        """Evaluate content on a specific criterion"""
        # This is a simplified implementation
        # In a real system, this would use more sophisticated evaluation methods
        
        # Check for content type
        if isinstance(content, dict) and "content" in content:
            # Use content field if available
            text_content = str(content["content"])
        else:
            # Convert to string
            text_content = str(content)
        
        # Get base features
        content_length = min(1.0, len(text_content) / 1000.0)  # Normalize, cap at 1000 chars
        content_complexity = min(1.0, len(set(text_content.split())) / 100.0)  # Lexical variety
        
        # Evaluate based on aspect and criterion
        if aspect == "consistency":
            if criterion == "coherence":
                # Check sentence structure, paragraph coherence
                # For simplicity, use a fixed score with slight randomness
                base_score = 0.75
                return base_score + random.uniform(-0.1, 0.1)
                
            elif criterion == "continuity":
                # Check for narrative continuity
                # For simplicity, use a fixed score with slight randomness
                base_score = 0.8
                return base_score + random.uniform(-0.1, 0.1)
                
            elif criterion == "context_adherence":
                # Check if content adheres to provided context
                # For simplicity, use a fixed score with slight randomness
                base_score = 0.7
                return base_score + random.uniform(-0.1, 0.1)
                
        elif aspect == "effectiveness":
            if criterion == "goal_achievement":
                # Check if the content achieves its goal
                # For simplicity, use a fixed score with slight randomness
                base_score = 0.7
                return base_score + random.uniform(-0.1, 0.1)
                
            elif criterion == "comprehensiveness":
                # Check if the content is comprehensive
                # Use content length as a proxy
                return 0.5 + (content_length * 0.4)
                
            elif criterion == "efficiency":
                # Check how efficiently the content achieves its goal
                # For simplicity, use a fixed score with slight randomness
                base_score = 0.75
                return base_score + random.uniform(-0.1, 0.1)
                
        elif aspect == "efficiency":
            if criterion == "response_time":
                # Check response time
                response_time = context.get("response_time", 0.5)
                # Lower response time is better
                return max(0.0, 1.0 - (response_time / 5.0))  # Cap at 5 seconds
                
            elif criterion == "resource_utilization":
                # Check resource utilization
                # For simplicity, use a fixed score with slight randomness
                base_score = 0.8
                return base_score + random.uniform(-0.1, 0.1)
                
            elif criterion == "complexity_management":
                # Check how well complexity is managed
                # Use inverse of content complexity
                return 0.9 - (content_complexity * 0.4)
        
        # Default score for unknown criteria
        return 0.5
    
    def _generate_criterion_description(self, 
                                      aspect: str, 
                                      criterion: str, 
                                      score: float) -> str:
        """Generate a description for a criterion score"""
        if score >= 0.8:
            quality = "excellent"
        elif score >= 0.6:
            quality = "good"
        elif score >= 0.4:
            quality = "fair"
        else:
            quality = "poor"
        
        # Generate descriptions based on aspect and criterion
        if aspect == "consistency":
            if criterion == "coherence":
                return f"{quality.capitalize()} coherence in content structure and flow"
            elif criterion == "continuity":
                return f"{quality.capitalize()} continuity throughout the content"
            elif criterion == "context_adherence":
                return f"{quality.capitalize()} adherence to the provided context"
                
        elif aspect == "effectiveness":
            if criterion == "goal_achievement":
                return f"{quality.capitalize()} achievement of intended goals"
            elif criterion == "comprehensiveness":
                return f"{quality.capitalize()} coverage of relevant information"
            elif criterion == "efficiency":
                return f"{quality.capitalize()} efficiency in achieving goals"
                
        elif aspect == "efficiency":
            if criterion == "response_time":
                return f"{quality.capitalize()} response time performance"
            elif criterion == "resource_utilization":
                return f"{quality.capitalize()} utilization of available resources"
            elif criterion == "complexity_management":
                return f"{quality.capitalize()} management of content complexity"
        
        return f"{quality.capitalize()} performance on {criterion}"
    
    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level based on score"""
        if score >= 0.9:
            return "outstanding"
        elif score >= 0.8:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.6:
            return "satisfactory"
        elif score >= 0.5:
            return "fair"
        elif score >= 0.4:
            return "needs improvement"
        else:
            return "unsatisfactory"
    
    def _generate_improvement_suggestions(self, 
                                        aspect: str, 
                                        evaluation: Dict[str, Dict[str, Any]], 
                                        overall_score: float) -> List[str]:
        """Generate improvement suggestions based on evaluation"""
        suggestions = []
        
        # Find the weakest criteria (lowest scores)
        criteria_scores = [(criterion, data["score"]) for criterion, data in evaluation.items()]
        criteria_scores.sort(key=lambda x: x[1])
        
        # Generate suggestions for up to 2 lowest scoring criteria
        for criterion, score in criteria_scores[:2]:
            if score < 0.7:  # Only suggest improvements for criteria below "good"
                if aspect == "consistency":
                    if criterion == "coherence":
                        suggestions.append("Improve logical flow between sentences and paragraphs")
                    elif criterion == "continuity":
                        suggestions.append("Ensure consistent narrative thread throughout the content")
                    elif criterion == "context_adherence":
                        suggestions.append("Align content more closely with the provided context")
                        
                elif aspect == "effectiveness":
                    if criterion == "goal_achievement":
                        suggestions.append("Focus more directly on achieving the primary objective")
                    elif criterion == "comprehensiveness":
                        suggestions.append("Include more relevant details and cover additional aspects")
                    elif criterion == "efficiency":
                        suggestions.append("Streamline content to achieve goals more efficiently")
                        
                elif aspect == "efficiency":
                    if criterion == "response_time":
                        suggestions.append("Optimize processing to improve response time")
                    elif criterion == "resource_utilization":
                        suggestions.append("Use resources more efficiently to achieve objectives")
                    elif criterion == "complexity_management":
                        suggestions.append("Simplify complex elements while preserving quality")
        
        # Add general suggestion if overall score is below threshold
        if overall_score < self.quality_threshold:
            suggestions.append(f"General {aspect} needs improvement to meet quality standards")
        
        return suggestions
