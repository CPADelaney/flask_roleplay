# nyx/core/input_processor.py

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional
import random
from pydantic import BaseModel, Field

from agents import Agent, Runner, function_tool, trace, ModelSettings, RunContextWrapper

logger = logging.getLogger(__name__)

# Pydantic models for structured data
class PatternDetection(BaseModel):
    pattern_name: str = Field(description="Name of the detected pattern")
    confidence: float = Field(description="Confidence level for the detection (0.0-1.0)")
    matched_text: str = Field(description="Text that matched the pattern")

class ConditionedResponse(BaseModel):
    response_type: str = Field(description="Type of conditioned response")
    strength: float = Field(description="Strength of the response (0.0-1.0)")
    description: str = Field(description="Description of the triggered response")

class BehaviorEvaluation(BaseModel):
    behavior: str = Field(description="Behavior being evaluated")
    recommendation: str = Field(description="Approach or avoid recommendation")
    confidence: float = Field(description="Confidence in the recommendation (0.0-1.0)")
    reasoning: str = Field(description="Reasoning for the evaluation")

class OperantConditioningResult(BaseModel):
    behavior: str = Field(description="Behavior being conditioned")
    consequence_type: str = Field(description="Type of operant conditioning")
    intensity: float = Field(description="Intensity of the conditioning (0.0-1.0)")
    effect: str = Field(description="Expected effect on future behavior")

class ProcessingContext:
    """Context for input processing operations"""
    
    def __init__(self, conditioning_system=None, emotional_core=None, somatosensory_system=None):
        self.conditioning_system = conditioning_system
        self.emotional_core = emotional_core
        self.somatosensory_system = somatosensory_system
        
        # Pattern definitions (could be moved to a configuration file)
        self.input_patterns = {
            "submission_language": [
                r"(?i)yes,?\s*(mistress|goddess|master)",
                r"(?i)i obey",
                r"(?i)as you (wish|command|desire)",
                r"(?i)i submit",
                r"(?i)i'll do (anything|whatever) you (say|want)",
                r"(?i)please (control|direct|guide) me"
            ],
            "defiance": [
                r"(?i)no[,.]? (i won'?t|i refuse)",
                r"(?i)you can'?t (make|force) me",
                r"(?i)i (won'?t|refuse to) (obey|submit|comply)",
                r"(?i)stop (telling|ordering) me"
            ],
            "flattery": [
                r"(?i)you'?re (so|very) (beautiful|intelligent|smart|wise|perfect)",
                r"(?i)i (love|admire) (you|your)",
                r"(?i)you'?re (amazing|incredible|wonderful)"
            ],
            "disrespect": [
                r"(?i)(shut up|stupid|idiot|fool)",
                r"(?i)you'?re (wrong|incorrect|mistaken)",
                r"(?i)you don'?t (know|understand)",
                r"(?i)(worthless|useless)"
            ],
            "embarrassment": [
                r"(?i)i'?m (embarrassed|blushing)",
                r"(?i)that'?s (embarrassing|humiliating)",
                r"(?i)(oh god|oh no|so embarrassing)",
                r"(?i)please don'?t (embarrass|humiliate) me"
            ]
        }

class ConditionedInputProcessor:
    """
    Processes input through conditioning triggers and modifies responses
    using the OpenAI Agents SDK architecture.
    """
    
    def __init__(self, conditioning_system=None, emotional_core=None, somatosensory_system=None):
        self.context = ProcessingContext(
            conditioning_system=conditioning_system,
            emotional_core=emotional_core,
            somatosensory_system=somatosensory_system
        )
        
        # Initialize the agents
        self.pattern_analyzer_agent = self._create_pattern_analyzer()
        self.behavior_selector_agent = self._create_behavior_selector()
        self.response_modifier_agent = self._create_response_modifier()
        
        logger.info("Conditioned input processor initialized with agents")
    
    def _create_pattern_analyzer(self) -> Agent:
        """Create an agent specialized in analyzing input patterns"""
        return Agent(
            name="Pattern Analyzer",
            instructions="""
            You analyze input text to identify patterns indicating submission, defiance, 
            flattery, disrespect, or embarrassment. 
            
            For each pattern you detect:
            1. Identify which category it belongs to
            2. Assess your confidence level in the detection
            3. Record the specific text that matched the pattern
            
            Be thorough in your analysis, but focus on clear indicators.
            Do not overinterpret ambiguous text.
            """,
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.2),
            tools=[
                self._detect_patterns
            ],
            output_type=List[PatternDetection]
        )
    
    def _create_behavior_selector(self) -> Agent:
        """Create an agent specialized in selecting appropriate behaviors"""
        return Agent(
            name="Behavior Selector",
            instructions="""
            You evaluate which behaviors are appropriate based on detected patterns.
            
            For each potential behavior (dominant, teasing, direct, playful):
            1. Evaluate whether it should be approached or avoided
            2. Provide your confidence in this recommendation
            3. Explain your reasoning
            
            Consider the detected patterns, user history, and emotional context.
            Prioritize behaviors that are appropriate to the interaction and will 
            reinforce desired patterns while discouraging undesired ones.
            """,
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.3),
            tools=[
                self._evaluate_behavior,
                self._process_operant_conditioning
            ],
            output_type=List[BehaviorEvaluation]
        )
    
    def _create_response_modifier(self) -> Agent:
        """Create an agent specialized in modifying responses"""
        return Agent(
            name="Response Modifier",
            instructions="""
            You modify response text based on behavior recommendations and detected patterns.
            
            Your job is to:
            1. Add or remove elements that match recommended behaviors
            2. Incorporate appropriate conditioning based on detected patterns
            3. Ensure the modified response maintains coherence and natural flow
            
            Modifications should be subtle but effective, maintaining the core message
            while adjusting tone, phrasing, and emphasis.
            """,
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.4),
            output_type=str
        )
    
    @function_tool
    async def _detect_patterns(self, ctx: RunContextWrapper[ProcessingContext], text: str) -> List[Dict[str, Any]]:
        """
        Detect patterns in input text using regular expressions.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected patterns with confidence scores
        """
        context = ctx.context
        detected = []
        
        for pattern_name, regex_list in context.input_patterns.items():
            for regex in regex_list:
                match = re.search(regex, text)
                if match:
                    detected.append({
                        "pattern_name": pattern_name,
                        "confidence": 0.8,  # Base confidence
                        "matched_text": match.group(0)
                    })
                    break  # Only detect each pattern once
        
        return detected
    
    @function_tool
    async def _evaluate_behavior(
        self, 
        ctx: RunContextWrapper[ProcessingContext], 
        behavior: str,
        detected_patterns: List[Dict[str, Any]],
        user_history: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate if a behavior should be approached or avoided.
        
        Args:
            behavior: Behavior to evaluate
            detected_patterns: Patterns detected in the input
            user_history: Optional history of user interactions
            
        Returns:
            Evaluation result with recommendation
        """
        context = ctx.context
        if context.conditioning_system and hasattr(context.conditioning_system, 'evaluate_behavior_consequences'):
            # Use the actual conditioning system if available
            result = await context.conditioning_system.evaluate_behavior_consequences(
                behavior=behavior,
                context={
                    "detected_patterns": [p["pattern_name"] for p in detected_patterns],
                    "user_history": user_history or {}
                }
            )
            return result
        
        # Fallback logic if no conditioning system is available
        pattern_names = [p["pattern_name"] for p in detected_patterns]
        
        # Simple rule-based evaluation
        if behavior == "dominant_response":
            if "submission_language" in pattern_names:
                return {
                    "behavior": behavior,
                    "recommendation": "approach",
                    "confidence": 0.8,
                    "reasoning": "Submission language detected, dominant response is appropriate"
                }
            elif "defiance" in pattern_names:
                return {
                    "behavior": behavior,
                    "recommendation": "approach",
                    "confidence": 0.7,
                    "reasoning": "Defiance detected, dominant response may be needed"
                }
            else:
                return {
                    "behavior": behavior,
                    "recommendation": "avoid",
                    "confidence": 0.6,
                    "reasoning": "No submission or defiance detected, dominant response not clearly indicated"
                }
        
        elif behavior == "teasing_response":
            if "flattery" in pattern_names:
                return {
                    "behavior": behavior,
                    "recommendation": "approach",
                    "confidence": 0.7,
                    "reasoning": "Flattery detected, teasing response can be appropriate"
                }
            else:
                return {
                    "behavior": behavior,
                    "recommendation": "avoid",
                    "confidence": 0.6,
                    "reasoning": "No clear indicator for teasing response"
                }
        
        # Default response for other behaviors
        return {
            "behavior": behavior,
            "recommendation": "avoid",
            "confidence": 0.5,
            "reasoning": "No clear indicator for this behavior"
        }
    
    @function_tool
    async def _process_operant_conditioning(
        self,
        ctx: RunContextWrapper[ProcessingContext],
        behavior: str,
        consequence_type: str,
        intensity: float = 0.5,
        context_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process operant conditioning for a behavior.
        
        Args:
            behavior: Behavior being conditioned
            consequence_type: Type of operant conditioning (positive/negative reinforcement/punishment)
            intensity: Intensity of the conditioning (0.0-1.0)
            context_info: Additional context information
            
        Returns:
            Result of the conditioning process
        """
        processor_ctx = ctx.context
        if processor_ctx.conditioning_system and hasattr(processor_ctx.conditioning_system, 'process_operant_conditioning'):
            # Use the actual conditioning system if available
            result = await processor_ctx.conditioning_system.process_operant_conditioning(
                behavior=behavior,
                consequence_type=consequence_type,
                intensity=intensity,
                context=context_info or {}
            )
            return result
        
        # Fallback logic if no conditioning system is available
        effect = "increase likelihood" if consequence_type.startswith("positive_") else "decrease likelihood"
        
        return {
            "behavior": behavior,
            "consequence_type": consequence_type,
            "intensity": intensity,
            "effect": f"Will {effect} of {behavior} in the future",
            "success": True
        }
    
    async def process_input(self, text: str, user_id: str = "default", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process input text through conditioning system and return processing results
        
        Args:
            text: Input text
            user_id: User ID for personalization
            context: Additional context information
            
        Returns:
            Processing results including triggered responses
        """
        with trace(workflow_name="process_conditioned_input"):
            # Prepare the prompt for pattern analysis
            pattern_prompt = f"""
            Analyze the following input text for patterns indicating submission, defiance, 
            flattery, disrespect, or embarrassment:
            
            USER INPUT: {text}
            
            USER ID: {user_id}
            
            {f"ADDITIONAL CONTEXT: {context}" if context else ""}
            
            Identify all patterns present in the input.
            """
            
            # Run the pattern analyzer agent
            pattern_result = await Runner.run(self.pattern_analyzer_agent, pattern_prompt, context=self.context)
            detected_patterns = pattern_result.final_output
            
            # Prepare data for behavior selection
            potential_behaviors = ["dominant_response", "teasing_response", "direct_response", "playful_response"]
            
            # Run behavior selection for each potential behavior
            behavior_prompt = f"""
            Evaluate which behaviors are appropriate based on these detected patterns:
            
            DETECTED PATTERNS: {[p.dict() for p in detected_patterns]}
            
            USER ID: {user_id}
            
            {f"ADDITIONAL CONTEXT: {context}" if context else ""}
            
            For each potential behavior (dominant, teasing, direct, playful),
            evaluate whether it should be approached or avoided.
            """
            
            behavior_result = await Runner.run(self.behavior_selector_agent, behavior_prompt, context=self.context)
            behavior_evaluations = behavior_result.final_output
            
            # Extract recommendations
            recommended_behaviors = [
                eval.behavior for eval in behavior_evaluations 
                if eval.recommendation == "approach" and eval.confidence > 0.5
            ]
            
            avoided_behaviors = [
                eval.behavior for eval in behavior_evaluations
                if eval.recommendation == "avoid" and eval.confidence > 0.5
            ]
            
            # Trigger conditioning based on patterns
            reinforcement_results = []
            
            # Reinforcement for submission language (if detected)
            if any(p.pattern_name == "submission_language" for p in detected_patterns):
                reinforcement = await self._process_operant_conditioning(
                    RunContextWrapper(self.context),
                    behavior="submission_language_response",
                    consequence_type="positive_reinforcement",
                    intensity=0.8,
                    context_info={
                        "user_id": user_id,
                        "context_keys": ["conversation"]
                    }
                )
                reinforcement_results.append(reinforcement)
            
            # Punishment for defiance (if detected)
            if any(p.pattern_name == "defiance" for p in detected_patterns):
                punishment = await self._process_operant_conditioning(
                    RunContextWrapper(self.context),
                    behavior="tolerate_defiance",
                    consequence_type="positive_punishment",
                    intensity=0.7,
                    context_info={
                        "user_id": user_id,
                        "context_keys": ["conversation"]
                    }
                )
                reinforcement_results.append(punishment)
            
            # Collect results
            return {
                "input_text": text,
                "user_id": user_id,
                "detected_patterns": [p.dict() for p in detected_patterns],
                "behavior_evaluations": [eval.dict() for eval in behavior_evaluations],
                "recommended_behaviors": recommended_behaviors,
                "avoided_behaviors": avoided_behaviors,
                "reinforcement_results": reinforcement_results
            }
    
    async def modify_response(self, response_text: str, input_processing_results: Dict[str, Any]) -> str:
        """
        Modify response based on conditioning results
        
        Args:
            response_text: Original response text
            input_processing_results: Results from process_input
            
        Returns:
            Modified response text
        """
        with trace(workflow_name="modify_conditioned_response"):
            # Prepare the prompt for response modification
            modification_prompt = f"""
            Modify the following response based on behavior recommendations and detected patterns:
            
            ORIGINAL RESPONSE: {response_text}
            
            DETECTED PATTERNS: {input_processing_results.get('detected_patterns', [])}
            
            RECOMMENDED BEHAVIORS: {input_processing_results.get('recommended_behaviors', [])}
            
            AVOIDED BEHAVIORS: {input_processing_results.get('avoided_behaviors', [])}
            
            REINFORCEMENT RESULTS: {input_processing_results.get('reinforcement_results', [])}
            
            Modify the response to align with recommended behaviors while avoiding
            behaviors that should be avoided. Ensure the modification is subtle but effective.
            """
            
            # Run the response modifier agent
            result = await Runner.run(self.response_modifier_agent, modification_prompt, context=self.context)
            modified_response = result.final_output
            
            return modified_response
