# nyx/api/thinking_tools.py

import logging
import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from agents import function_tool, RunContextWrapper

logger = logging.getLogger(__name__)

# Pydantic models for function tool inputs/outputs
class ThinkingContext(BaseModel):
    """Context information for thinking tools"""
    previous_response: Optional[Dict[str, float]] = None
    # Add other fields as needed based on actual usage

class DecisionFactors(BaseModel):
    """Factors used in deciding whether to use extended thinking"""
    query_complexity: float
    contains_reasoning_keywords: bool
    explicit_request: bool
    critical_domain: bool
    previous_low_confidence: bool
    ambiguous_query: bool
    reasoning_keywords_found: Optional[List[str]] = None

class ExtendedThinkingDecision(BaseModel):
    """Result of should_use_extended_thinking"""
    should_think: bool
    thinking_level: int
    decision_factors: DecisionFactors

class ThinkingStep(BaseModel):
    """A single step in the thinking process"""
    step: str
    content: str

class ThinkingResult(BaseModel):
    """Result of think_before_responding"""
    reasoning_complete: bool
    thinking_steps: List[ThinkingStep]
    thinking_level: int
    confidence: float
    reasoned_output: Optional[str] = None

class ReasonedResponse(BaseModel):
    """Result of generate_reasoned_response"""
    message: str
    response_type: str
    thinking_applied: bool
    thinking_level: int
    thinking_steps: List[ThinkingStep]
    confidence: float

@function_tool
async def should_use_extended_thinking(ctx, user_input: str, 
                                  context: Optional[ThinkingContext] = None) -> ExtendedThinkingDecision:
    """
    Determine if the query requires extended thinking before responding
    
    Args:
        user_input: User's message text
        context: Optional additional context
    
    Returns:
        Decision details with reasoning
    """
    brain = ctx.context
    
    # Initialize decision factors
    decision_factors = {
        "query_complexity": 0.0,
        "contains_reasoning_keywords": False,
        "explicit_request": False,
        "critical_domain": False,
        "previous_low_confidence": False,
        "ambiguous_query": False
    }
    
    # 1. Query complexity (length, structure, multiple questions)
    words = user_input.split()
    query_length = len(words)
    question_count = user_input.count("?")
    
    # Calculate complexity score based on length and questions
    length_score = min(1.0, query_length / 40)  # Normalize for queries up to 40 words
    question_score = min(1.0, question_count / 3)  # Normalize for up to 3 questions
    
    complexity_score = (length_score * 0.6) + (question_score * 0.4)
    decision_factors["query_complexity"] = complexity_score
    
    # 2. Contains reasoning keywords
    reasoning_keywords = [
        "explain", "why", "how come", "reason", "analyze", "compare", 
        "evaluate", "think", "consider", "what if", "implications",
        "consequences", "tradeoffs", "pros and cons"
    ]
    
    matching_keywords = [keyword for keyword in reasoning_keywords if keyword in user_input.lower()]
    decision_factors["contains_reasoning_keywords"] = len(matching_keywords) > 0
    decision_factors["reasoning_keywords_found"] = matching_keywords
    
    # 3. User has explicitly requested thinking
    explicit_thinking_phrases = [
        "think about this", "think carefully", "think through",
        "reasoning", "extended thinking", "think before answering"
    ]
    
    explicit_request = any(phrase in user_input.lower() for phrase in explicit_thinking_phrases)
    decision_factors["explicit_request"] = explicit_request
    
    # 4. Previous response lacked confidence (if available in context)
    if context and context.previous_response:
        confidence = context.previous_response.get("confidence", 1.0)
        decision_factors["previous_low_confidence"] = confidence < 0.5
    
    # 5. Query involves critical domains
    critical_domains = [
        "ethics", "safety", "complex decision", "medical", "legal", "financial", 
        "important decision", "significant consequence"
    ]
    
    decision_factors["critical_domain"] = any(domain in user_input.lower() for domain in critical_domains)
    
    # Make the final decision
    should_think = (
        explicit_request or 
        decision_factors["critical_domain"] or
        (complexity_score > 0.6 and decision_factors["contains_reasoning_keywords"]) or
        (decision_factors["previous_low_confidence"] and decision_factors["ambiguous_query"])
    )
    
    # Determine thinking level (1-3, with 3 being most intensive)
    thinking_level = 1  # Default basic thinking
    
    if explicit_request or decision_factors["critical_domain"]:
        thinking_level = 3  # Deep thinking
    elif complexity_score > 0.7:
        thinking_level = 2  # Moderate thinking
    
    return ExtendedThinkingDecision(
        should_think=should_think,
        thinking_level=thinking_level,
        decision_factors=DecisionFactors(**decision_factors)
    )

@function_tool
async def think_before_responding(ctx, user_input: str, 
                             thinking_level: int = 1, 
                             context: Optional[ThinkingContext] = None) -> ThinkingResult:
    """
    Engage in extended reasoning before generating a response
    
    Args:
        user_input: User's message text
        thinking_level: Intensity of thinking (1-3)
        context: Optional additional context
    
    Returns:
        Thinking results with reasoning steps and improved response
    """
    brain = ctx.context
    
    # Different thinking approaches based on level
    if thinking_level == 3 and hasattr(brain, 'reasoning_core'):
        # Deep thinking - use reasoning agents when available
        return await _deep_reasoning_thinking(brain, user_input, context)
    elif thinking_level == 2 and hasattr(brain, 'reasoning_core'):
        # Moderate thinking - use standard reasoning
        return await _standard_reasoning_thinking(brain, user_input, context)
    else:
        # Basic thinking or no reasoning core available
        return await _basic_thinking(brain, user_input, context)

@function_tool
async def generate_reasoned_response(ctx, user_input: str, 
                                 thinking_result: ThinkingResult,
                                 context: Optional[ThinkingContext] = None) -> ReasonedResponse:
    """
    Generate a response after thinking process
    
    Args:
        user_input: User's message text
        thinking_result: Results from thinking process
        context: Optional additional context
    
    Returns:
        Complete response with thinking incorporated
    """
    brain = ctx.context
    
    # If reasoned output was directly generated, use it
    if thinking_result.reasoned_output:
        message = thinking_result.reasoned_output
        confidence = thinking_result.confidence
    else:
        # Generate response using standard method but with thinking context
        enhanced_context = context.model_dump() if context else {}
        enhanced_context["thinking_steps"] = [step.model_dump() for step in thinking_result.thinking_steps]
        
        base_response = await brain.generate_response(user_input, enhanced_context)
        message = base_response.get("message", "")
        confidence = base_response.get("confidence", 0.5)
    
    thinking_level = thinking_result.thinking_level
    
    # Update response type based on thinking level
    response_type_map = {
        1: "basic_reasoned",
        2: "moderately_reasoned", 
        3: "deeply_reasoned"
    }
    response_type = response_type_map.get(thinking_level, "reasoned")
    
    return ReasonedResponse(
        message=message,
        response_type=response_type,
        thinking_applied=True,
        thinking_level=thinking_level,
        thinking_steps=thinking_result.thinking_steps,
        confidence=max(confidence, thinking_result.confidence)
    )

# Helper functions for different thinking approaches

async def _basic_thinking(brain, user_input: str, context: Optional[ThinkingContext]) -> ThinkingResult:
    """Basic structured thinking process"""
    thinking_steps = [
        ThinkingStep(step="Understand query", content=f"Analyzing: {user_input[:100]}..."),
        ThinkingStep(step="Identify key elements", content="Extracting core components of the request"),
        ThinkingStep(step="Retrieve relevant information", content="Accessing knowledge and contextual information"),
        ThinkingStep(step="Formulate response", content="Developing clear and helpful response")
    ]
    
    # Use internal feedback if available for critique
    if hasattr(brain, 'internal_feedback'):
        try:
            critique = await brain.internal_feedback.critic_evaluate(
                aspect="effectiveness",
                content=user_input,
                context=context.model_dump() if context else {}
            )
            
            # Add critique step
            if critique:
                thinking_steps.append(ThinkingStep(
                    step="Self-evaluate response", 
                    content=f"Quality assessment: {critique.get('weighted_score', 0.5):.2f}"
                ))
        except Exception as e:
            logger.error(f"Error in feedback-based thinking: {str(e)}")
    
    return ThinkingResult(
        reasoning_complete=True,
        thinking_steps=thinking_steps,
        thinking_level=1,
        confidence=0.6
    )

async def _standard_reasoning_thinking(brain, user_input: str, context: Optional[ThinkingContext]) -> ThinkingResult:
    """Standard reasoning using reasoning agents"""
    try:
        # Use reasoning triage agent if available
        from agents import Runner
        
        # Create a thinking-oriented prompt
        thinking_prompt = f"""I need to carefully think through this query before responding:
        
        {user_input}
        
        Please help me reason step-by-step about how to best answer this query."""
        
        # Run the reasoning agent
        reasoning_result = await Runner.run(
            brain.reasoning_core,
            thinking_prompt
        )
        
        # Extract the output
        reasoned_output = reasoning_result.final_output
        
        # Create structured thinking steps
        thinking_steps = [
            ThinkingStep(step="Initial analysis", content="Analyzing query for reasoning requirements"),
            ThinkingStep(step="Structured reasoning", content="Applying systematic reasoning approach"),
            ThinkingStep(step="Generate reasoned response", content="Producing well-reasoned answer")
        ]
        
        return ThinkingResult(
            reasoning_complete=True,
            thinking_steps=thinking_steps,
            reasoned_output=reasoned_output,
            thinking_level=2,
            confidence=0.7
        )
    except Exception as e:
        logger.error(f"Error in reasoning agent thinking: {str(e)}")
        
        # Fallback to basic thinking
        return await _basic_thinking(brain, user_input, context)

async def _deep_reasoning_thinking(brain, user_input: str, context: Optional[ThinkingContext]) -> ThinkingResult:
    """Deep thinking with thorough reasoning"""
    try:
        from agents import Runner
        
        # Create a detailed thinking prompt
        thinking_prompt = f"""I need to engage in deep, thorough reasoning about this query:
        
        {user_input}
        
        Please help me analyze this systematically:
        1. What are the explicit and implicit requirements?
        2. What domain knowledge is required?
        3. What are potential approaches to answering this?
        4. What nuances or complexities should I be aware of?
        5. What are potential pitfalls or misunderstandings to avoid?
        6. What's the most helpful and accurate way to respond?
        
        Let's think step-by-step to ensure a complete, accurate, and helpful response."""
        
        # Run the reasoning agent
        reasoning_result = await Runner.run(
            brain.reasoning_core,
            thinking_prompt
        )
        
        # Extract the output
        reasoned_output = reasoning_result.final_output
        
        # Create detailed thinking steps
        thinking_steps = [
            ThinkingStep(step="Comprehensive analysis", content="Thoroughly analyzing query requirements and implications"),
            ThinkingStep(step="Identify domains", content="Determining knowledge domains and expertise needed"),
            ThinkingStep(step="Consider approaches", content="Evaluating multiple potential response strategies"),
            ThinkingStep(step="Analyze nuances", content="Examining subtleties and complexities of the question"),
            ThinkingStep(step="Identify pitfalls", content="Recognizing potential misunderstandings or problematic areas"),
            ThinkingStep(step="Generate optimal response", content="Creating most helpful and accurate response")
        ]
        
        return ThinkingResult(
            reasoning_complete=True,
            thinking_steps=thinking_steps,
            reasoned_output=reasoned_output,
            thinking_level=3,
            confidence=0.8
        )
    except Exception as e:
        logger.error(f"Error in deep reasoning thinking: {str(e)}")
        
        # Fallback to standard thinking
        return await _standard_reasoning_thinking(brain, user_input, context)
