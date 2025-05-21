# nyx/core/tools/evaluator.py

import asyncio
import time
import json
import logging
import datetime
from typing import Dict, Any, List, Optional, Union, TypeVar, Generic
from dataclasses import dataclass, field
from pydantic import BaseModel

from agents import Agent, Runner, ModelSettings, function_tool, trace

logger = logging.getLogger(__name__)

class EvaluationDimension(BaseModel):
    """A dimension for evaluating agent performance."""
    name: str
    prompt: str
    weight: float = 1.0
    description: Optional[str] = None

class EvaluationMetric(BaseModel):
    """A metric used to evaluate agent performance."""
    dimension: str
    score: float
    explanation: str
    
class EvaluationSummary(BaseModel):
    """Summary of an agent evaluation."""
    agent_name: str
    timestamp: str
    metrics: Dict[str, EvaluationMetric]
    average_score: float
    input_type: str
    total_tokens: int = 0
    evaluation_time: float = 0
    metadata: Dict[str, Any] = {}

class AgentEvaluator:
    """Evaluates agent performance and provides improvement feedback."""
    
    DEFAULT_DIMENSIONS = {
        "coherence": EvaluationDimension(
            name="coherence",
            prompt="Evaluate the coherence of this agent response. Consider logical flow, clarity, and organization. Score on a scale of 1-10.",
            description="How well the response flows logically and is organized"
        ),
        "relevance": EvaluationDimension(
            name="relevance",
            prompt="Evaluate how relevant this response is to the user's request. Score on a scale of 1-10.",
            description="How directly the response addresses the user's query"
        ),
        "accuracy": EvaluationDimension(
            name="accuracy",
            prompt="Evaluate the factual accuracy of this response. Score on a scale of 1-10.",
            description="The factual correctness of information provided"
        ),
        "creativity": EvaluationDimension(
            name="creativity",
            prompt="Evaluate the creativity and thoughtfulness of this response. Score on a scale of 1-10.",
            description="How original and insightful the response is"
        ),
        "completeness": EvaluationDimension(
            name="completeness",
            prompt="Evaluate how completely this response addresses all aspects of the user's query. Score on a scale of 1-10.",
            description="How thoroughly the response covers all aspects of the query"
        )
    }
    
    def __init__(self, metrics_collector=None, dimensions: Optional[Dict[str, EvaluationDimension]] = None):
        """
        Initialize the agent evaluator.
        
        Args:
            metrics_collector: Optional metrics collector to use for tracking evaluations
            dimensions: Custom evaluation dimensions to use instead of defaults
        """
        self.metrics_collector = metrics_collector
        self.evaluation_dimensions = dimensions or self.DEFAULT_DIMENSIONS
        self.evaluation_history = []  # Stores historical evaluations
        self.evaluator_agent = self._create_evaluation_agent()
        self.dimension_scores = {dim: [] for dim in self.evaluation_dimensions}
    
    def _create_evaluation_agent(self) -> Agent:
        """Create an agent for evaluating responses."""
        return Agent(
            name="EvaluationAgent",
            instructions="""You are an objective evaluator of AI agent responses. Your task is to evaluate responses across specific dimensions.

For each evaluation, you will receive:
1. The original user input
2. The agent's response
3. The specific dimension to evaluate
4. The evaluation prompt for that dimension

Provide a fair and balanced assessment, with a numerical score (1-10) and explanation. Be specific about strengths and weaknesses.

Always return your evaluation as a JSON object with:
- "score": A numerical value from 1-10 where 1 is worst and 10 is best
- "explanation": A concise explanation of your rating (1-3 sentences)

Focus ONLY on the requested dimension in each evaluation.
""",
            model="gpt-4.1-nano",  # Using a smaller model for cost efficiency
            model_settings=ModelSettings(
                temperature=0.2,  # Low temperature for consistent evaluations
            ),
            tools=[
                self._get_evaluation_criteria
            ]
        )
    
    @function_tool
    async def _get_evaluation_criteria(self, dimension: str) -> Dict[str, Any]:
        """Get detailed criteria for a specific evaluation dimension."""
        if dimension not in self.evaluation_dimensions:
            return {
                "dimension": dimension,
                "criteria": "General quality assessment",
                "score_guidance": {
                    "1-3": "Poor quality",
                    "4-6": "Average quality",
                    "7-9": "Good quality",
                    "10": "Exceptional quality"
                }
            }
        
        dim = self.evaluation_dimensions[dimension]
        return {
            "dimension": dimension,
            "description": dim.description or "No specific description available",
            "criteria": dim.prompt,
            "score_guidance": {
                "1-3": "Poor performance in this dimension",
                "4-6": "Average performance in this dimension",
                "7-9": "Good performance in this dimension",
                "10": "Exceptional performance in this dimension"
            }
        }
    
    async def evaluate_response(self, 
                              agent_name: str, 
                              user_input: str, 
                              agent_output: Any,
                              dimensions: Optional[List[str]] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> EvaluationSummary:
        """
        Evaluate an agent's response across multiple dimensions.
        
        Args:
            agent_name: Name of the agent being evaluated
            user_input: The original user input
            agent_output: The agent's response output
            dimensions: Specific dimensions to evaluate (defaults to all)
            metadata: Additional metadata to include in the evaluation
            
        Returns:
            Evaluation summary with scores across dimensions
        """
        start_time = time.time()
        
        with trace(workflow_name="AgentEvaluation", trace_metadata={"agent_name": agent_name}):
            # Determine which dimensions to evaluate
            eval_dimensions = dimensions or list(self.evaluation_dimensions.keys())
            
            # Convert agent output to string if needed
            output_str = json.dumps(agent_output) if not isinstance(agent_output, str) else agent_output
            
            # Determine input type
            input_type = "text" if isinstance(user_input, str) else "structured"
            
            # Track token usage
            total_tokens = 0
            
            # Evaluate each dimension in parallel
            async def evaluate_dimension(dimension):
                if dimension not in self.evaluation_dimensions:
                    logger.warning(f"Unknown evaluation dimension: {dimension}")
                    return dimension, None
                
                dimension_info = self.evaluation_dimensions[dimension]
                
                # Run evaluation
                evaluation_result = await Runner.run(
                    self.evaluator_agent,
                    {
                        "evaluation_dimension": dimension,
                        "evaluation_prompt": dimension_info.prompt,
                        "user_input": user_input,
                        "agent_output": output_str
                    }
                )
                
                # Track tokens
                nonlocal total_tokens
                total_tokens += evaluation_result.usage.total_tokens
                
                # Parse result
                try:
                    result = evaluation_result.final_output
                    if isinstance(result, dict):
                        score = float(result.get("score", 5.0))
                        explanation = result.get("explanation", "No explanation provided")
                    else:
                        # Handle string output
                        logger.warning(f"Unexpected evaluation output format: {type(result)}")
                        score = 5.0  # Default score
                        explanation = "Error parsing evaluation result"
                    
                    # Create evaluation metric
                    metric = EvaluationMetric(
                        dimension=dimension,
                        score=score,
                        explanation=explanation
                    )
                    
                    # Add to dimension history
                    self.dimension_scores[dimension].append(score)
                    
                    return dimension, metric
                except Exception as e:
                    logger.error(f"Error processing evaluation for dimension {dimension}: {e}")
                    return dimension, None
            
            # Run all evaluations in parallel
            eval_tasks = [evaluate_dimension(dim) for dim in eval_dimensions]
            results = await asyncio.gather(*eval_tasks)
            
            # Process results
            metrics = {}
            for dimension, metric in results:
                if metric:
                    metrics[dimension] = metric
            
            # Calculate weighted average score
            if metrics:
                weighted_sum = 0
                total_weight = 0
                
                for dim_name, metric in metrics.items():
                    dimension = self.evaluation_dimensions.get(dim_name)
                    weight = dimension.weight if dimension else 1.0
                    weighted_sum += metric.score * weight
                    total_weight += weight
                
                average_score = weighted_sum / total_weight if total_weight > 0 else 0
            else:
                average_score = 0
            
            # Create summary
            summary = EvaluationSummary(
                agent_name=agent_name,
                timestamp=datetime.datetime.now().isoformat(),
                metrics=metrics,
                average_score=average_score,
                input_type=input_type,
                total_tokens=total_tokens,
                evaluation_time=time.time() - start_time,
                metadata=metadata or {}
            )
            
            # Store in history
            self.evaluation_history.append(summary)
            
            # Update metrics collector if available
            if self.metrics_collector:
                try:
                    await self.metrics_collector.record_evaluation_metrics(
                        agent_name=agent_name,
                        evaluation_summary=summary
                    )
                except Exception as e:
                    logger.error(f"Error recording evaluation metrics: {e}")
            
            return summary
    
    def get_agent_performance_trend(self, agent_name: str, dimension: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance trend for a specific agent over time.
        
        Args:
            agent_name: Name of the agent to analyze
            dimension: Optional specific dimension to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        agent_evals = [e for e in self.evaluation_history if e.agent_name == agent_name]
        
        if not agent_evals:
            return {"agent": agent_name, "evaluations": 0, "message": "No evaluations found for this agent"}
        
        if dimension:
            # Analyze trend for specific dimension
            scores = [e.metrics[dimension].score for e in agent_evals if dimension in e.metrics]
            if not scores:
                return {"agent": agent_name, "dimension": dimension, "evaluations": 0, "message": "No evaluations found for this dimension"}
                
            return {
                "agent": agent_name,
                "dimension": dimension,
                "evaluations": len(scores),
                "current_score": scores[-1],
                "average_score": sum(scores) / len(scores),
                "trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "declining" if len(scores) > 1 and scores[-1] < scores[0] else "stable",
                "min_score": min(scores),
                "max_score": max(scores)
            }
        else:
            # Analyze overall trend
            scores = [e.average_score for e in agent_evals]
            return {
                "agent": agent_name,
                "evaluations": len(scores),
                "current_score": scores[-1],
                "average_score": sum(scores) / len(scores),
                "trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "declining" if len(scores) > 1 and scores[-1] < scores[0] else "stable",
                "min_score": min(scores),
                "max_score": max(scores),
                "dimensions": {dim: self.get_agent_performance_trend(agent_name, dim) for dim in self.evaluation_dimensions}
            }
    
    def get_benchmark_report(self) -> Dict[str, Any]:
        """
        Get a benchmark report comparing all evaluated agents.
        
        Returns:
            Dictionary with comparison data
        """
        if not self.evaluation_history:
            return {"agents": 0, "message": "No evaluations available"}
        
        # Get unique agent names
        agent_names = set(e.agent_name for e in self.evaluation_history)
        
        # Generate per-agent statistics
        agent_stats = {}
        for agent in agent_names:
            agent_stats[agent] = self.get_agent_performance_trend(agent)
        
        # Generate per-dimension statistics
        dimension_stats = {}
        for dim in self.evaluation_dimensions:
            dim_scores = self.dimension_scores.get(dim, [])
            if dim_scores:
                dimension_stats[dim] = {
                    "average_score": sum(dim_scores) / len(dim_scores),
                    "evaluations": len(dim_scores),
                    "min_score": min(dim_scores),
                    "max_score": max(dim_scores),
                    "top_agent": max([(agent, self.get_agent_performance_trend(agent, dim).get("average_score", 0)) 
                                   for agent in agent_names], 
                                   key=lambda x: x[1])[0] if agent_names else None
                }
        
        return {
            "agents": len(agent_names),
            "evaluations": len(self.evaluation_history),
            "dimensions": list(self.evaluation_dimensions.keys()),
            "agent_stats": agent_stats,
            "dimension_stats": dimension_stats,
            "top_agent": max([(agent, stats.get("average_score", 0)) 
                          for agent, stats in agent_stats.items()], 
                          key=lambda x: x[1])[0] if agent_stats else None
        }

# Example usage
async def evaluator_example():
    evaluator = AgentEvaluator()
    
    # Example agent response
    agent_name = "TestAgent"
    user_input = "Explain quantum computing to a high school student"
    agent_output = """
    Quantum computing is like having a super powerful calculator that works in a completely different way than regular computers.
    
    Regular computers use bits (0s and 1s) to process information. Quantum computers use something called qubits, which can be both 0 and 1 at the same time! This is because of quantum physics, where tiny particles can exist in multiple states simultaneously.
    
    This special property allows quantum computers to solve certain problems much faster than regular computers. For example, they could potentially break encryption codes or simulate complex molecules for discovering new medicines.
    
    However, quantum computers are still being developed and aren't ready to replace your laptop yet. They're very sensitive to their environment and need to be kept extremely cold to work properly.
    """
    
    # Evaluate the response
    evaluation = await evaluator.evaluate_response(
        agent_name=agent_name,
        user_input=user_input,
        agent_output=agent_output,
        metadata={"purpose": "example"}
    )
    
    # Print the evaluation summary
    print(f"Evaluation of {agent_name}:")
    print(f"Average score: {evaluation.average_score:.2f}/10")
    print("Dimension scores:")
    for dim_name, metric in evaluation.metrics.items():
        print(f"  {dim_name}: {metric.score}/10 - {metric.explanation}")
    
    # Get performance trend
    trend = evaluator.get_agent_performance_trend(agent_name)
    print(f"\nPerformance trend: {trend['trend']}")
    
    # Get benchmark report
    benchmark = evaluator.get_benchmark_report()
    print(f"\nBenchmark report:")
    print(f"Total agents evaluated: {benchmark['agents']}")
    print(f"Total evaluations: {benchmark['evaluations']}")
