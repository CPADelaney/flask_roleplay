# nyx/core/reflexive_system.py

import asyncio
import logging
import time
import random
import math
import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable

from agents import Agent, Runner, trace, function_tool, custom_span, RunContextWrapper, ModelSettings, RunConfig, handoff
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# =============== Base Models for Structured Data ===============

class NumericRange(BaseModel):
    """Numeric range with min/max values"""
    min: float = Field(description="Minimum value")
    max: float = Field(description="Maximum value")
    optimal: Optional[float] = Field(None, description="Optimal value")
    weight: Optional[float] = Field(None, description="Feature weight")

class StringPattern(BaseModel):
    """String pattern with variants"""
    primary: str = Field(description="Primary string value")
    variants: Optional[List[str]] = Field(None, description="String variants")
    weight: Optional[float] = Field(None, description="Feature weight")

class KeyValuePair(BaseModel):
    """A key-value pair for dynamic data"""
    key: str = Field(description="Key name")
    value: str = Field(description="String value")

class KeyNumberPair(BaseModel):
    """A key-number pair for dynamic numeric data"""
    key: str = Field(description="Key name")
    value: float = Field(description="Numeric value")

class KeyBooleanPair(BaseModel):
    """A key-boolean pair for dynamic boolean data"""
    key: str = Field(description="Key name")
    value: bool = Field(description="Boolean value")

class KeyListPair(BaseModel):
    """A key-list pair for dynamic list data"""
    key: str = Field(description="Key name")
    value: List[str] = Field(description="List of string values")

class PatternValue(BaseModel):
    """Generic pattern value that can be various types"""
    key: str = Field(description="Pattern key")
    type: str = Field(description="Type of value: string, number, range, list, boolean")
    string_value: Optional[str] = Field(None, description="String value")
    number_value: Optional[float] = Field(None, description="Numeric value")
    range_value: Optional[NumericRange] = Field(None, description="Numeric range")
    list_value: Optional[List[str]] = Field(None, description="List of strings")
    boolean_value: Optional[bool] = Field(None, description="Boolean value")
    required: Optional[bool] = Field(False, description="Whether this field is required")
    weight: Optional[float] = Field(None, description="Feature weight")

class PatternDict(BaseModel):
    """Dictionary of pattern values"""
    values: List[PatternValue] = Field(default_factory=list, description="Pattern values")

class TimingInfo(BaseModel):
    """Timing information for gaming stimuli"""
    frame_number: int = Field(description="Current frame number")
    time_to_impact: int = Field(description="Time to impact in milliseconds")
    frame_window: int = Field(description="Frame window for reaction")

class OpponentState(BaseModel):
    """Opponent state information"""
    state: str = Field(description="Opponent state")
    position: str = Field(description="Opponent position")
    orientation: str = Field(description="Opponent orientation")

class DistractorInfo(BaseModel):
    """Distractor information"""
    type: str = Field(description="Type of distractor")
    intensity: float = Field(description="Distractor intensity")

class StimulusDict(BaseModel):
    """Dictionary of stimulus values"""
    text: Optional[str] = Field(None, description="Text content")
    visual_clarity: Optional[float] = Field(None, description="Visual clarity score")
    timing: Optional[TimingInfo] = Field(None, description="Timing information")
    opponent: Optional[OpponentState] = Field(None, description="Opponent state")
    distractors: Optional[List[DistractorInfo]] = Field(None, description="Distractor elements")
    # Generic fields for other stimulus data
    string_fields: Optional[List[KeyValuePair]] = Field(None, description="Additional string fields")
    number_fields: Optional[List[KeyNumberPair]] = Field(None, description="Additional numeric fields")
    list_fields: Optional[List[KeyListPair]] = Field(None, description="Additional list fields")

class ContextDict(BaseModel):
    """Context dictionary"""
    game: Optional[str] = Field(None, description="Game name")
    action_type: Optional[str] = Field(None, description="Action type")
    gaming_reflex: Optional[bool] = Field(None, description="Whether this is a gaming reflex")
    simulation: Optional[bool] = Field(None, description="Whether this is a simulation")
    difficulty: Optional[float] = Field(None, description="Difficulty level")
    domain: Optional[str] = Field(None, description="Domain")
    # Generic fields for additional context
    string_fields: Optional[List[KeyValuePair]] = Field(None, description="Additional string fields")
    number_fields: Optional[List[KeyNumberPair]] = Field(None, description="Additional numeric fields")
    boolean_fields: Optional[List[KeyBooleanPair]] = Field(None, description="Additional boolean fields")

class ExecutionResult(BaseModel):
    """Result from procedure execution"""
    success: bool = Field(description="Whether execution was successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    data: Optional[List[KeyValuePair]] = Field(None, description="Result data")

class ReflexStats(BaseModel):
    """Statistics for a reflex"""
    execution_count: int = Field(description="Number of executions")
    success_rate: float = Field(description="Success rate")
    avg_response_time_ms: float = Field(description="Average response time in ms")
    last_executed: Optional[str] = Field(None, description="Last execution timestamp")

class ReflexInfo(BaseModel):
    """Information about a reflex"""
    name: str = Field(description="Reflex name")
    procedure_name: str = Field(description="Procedure name")
    threshold: float = Field(description="Matching threshold")
    priority: int = Field(description="Priority level")
    stats: ReflexStats = Field(description="Reflex statistics")

# =============== Pydantic Models ===============

class ReflexPatternData(BaseModel):
    """Pattern data for a reflex pattern"""
    name: str = Field(description="Unique name for this pattern")
    pattern_data: PatternDict = Field(description="Pattern definition")
    procedure_name: str = Field(description="Name of procedure to execute when triggered")
    threshold: float = Field(0.7, description="Matching threshold (0.0-1.0)")
    priority: int = Field(1, description="Priority level (higher values take precedence)")
    context_template: ContextDict = Field(default_factory=ContextDict, description="Template for context to pass to procedure")
    domain: Optional[str] = Field(None, description="Optional domain for specialized responses")

class StimulusInput(BaseModel):
    """Input for stimulus processing"""
    stimulus: StimulusDict = Field(description="The stimulus data requiring reaction")
    domain: Optional[str] = Field(None, description="Optional domain to limit reflex patterns")
    context: Optional[ContextDict] = Field(None, description="Additional context information")

class ReactionResult(BaseModel):
    """Result of a reaction to stimulus"""
    success: bool = Field(description="Whether the reaction was successful")
    pattern_name: Optional[str] = Field(None, description="Name of the matched pattern")
    reaction_time_ms: float = Field(description="Reaction time in milliseconds")
    match_score: Optional[float] = Field(None, description="Match score for the pattern")
    error: Optional[str] = Field(None, description="Error message if reaction failed")
    procedure_result: Optional[ExecutionResult] = Field(None, description="Result from the executed procedure")

class PatternMatchResult(BaseModel):
    """Result of pattern matching"""
    matched: bool = Field(description="Whether a pattern was matched")
    pattern_name: Optional[str] = Field(None, description="Name of the matched pattern")
    match_score: Optional[float] = Field(None, description="Match score for the pattern")
    priority: Optional[int] = Field(None, description="Priority of the matched pattern")

class PatternRecognitionInput(BaseModel):
    """Input for pattern recognition"""
    stimulus: StimulusDict = Field(description="The stimulus data")
    patterns: List[PatternDict] = Field(description="Patterns to match against")
    domain: Optional[str] = Field(None, description="Optional domain to limit matching")

class GamingReflexInput(BaseModel):
    """Input for creating a gaming reflex"""
    game_name: str = Field(description="Name of the game")
    action_type: str = Field(description="Type of action (e.g., 'block', 'attack', 'dodge')")
    trigger_pattern: PatternDict = Field(description="Pattern to recognize")
    response_procedure: str = Field(description="Procedure to execute")
    reaction_threshold: float = Field(0.7, description="Recognition threshold")

class CurrentStats(BaseModel):
    """Current statistics for a pattern"""
    avg_response_time: float = Field(description="Average response time")
    success_rate: float = Field(description="Success rate")

class ImprovementData(BaseModel):
    """Improvement data for a pattern"""
    response_time_improvement_ms: float = Field(description="Response time improvement in ms")
    success_rate_improvement: float = Field(description="Success rate improvement")
    current_stats: CurrentStats = Field(description="Current statistics")

class PatternImprovement(BaseModel):
    """Improvement for a specific pattern"""
    pattern_name: str = Field(description="Pattern name")
    improvement: ImprovementData = Field(description="Improvement data")

class TrainingResult(BaseModel):
    """Result of reflex training"""
    success: bool = Field(description="Whether training was successful")
    iterations: int = Field(description="Number of training iterations performed")
    improvements: List[PatternImprovement] = Field(description="Improvements by pattern")
    training_accuracy: float = Field(description="Overall training accuracy")
    avg_reaction_time: float = Field(description="Average reaction time in milliseconds")

class ScenarioResult(BaseModel):
    """Result of a single scenario"""
    scenario: int = Field(description="Scenario number")
    pattern_tested: str = Field(description="Pattern that was tested")
    reaction_time_ms: float = Field(description="Reaction time in milliseconds")
    success: bool = Field(description="Whether the scenario was successful")
    matched_pattern: Optional[str] = Field(None, description="Pattern that was matched")

class SimulationResult(BaseModel):
    """Result of gaming scenario simulation"""
    success: bool = Field(description="Whether simulation was successful")
    game: str = Field(description="Game that was simulated")
    scenarios_run: int = Field(description="Number of scenarios run")
    success_rate: float = Field(description="Success rate of the simulations")
    avg_reaction_time_ms: float = Field(description="Average reaction time in milliseconds")
    results: List[ScenarioResult] = Field(description="Detailed results of each scenario")

class DomainCount(BaseModel):
    """Count for a specific domain"""
    domain: str = Field(description="Domain name")
    count: int = Field(description="Pattern count")

class ReflexiveSystemStats(BaseModel):
    """Statistics about the reflexive system"""
    total_patterns: int = Field(description="Total number of reflex patterns")
    domain_counts: List[DomainCount] = Field(description="Pattern counts by domain")
    response_mode: str = Field(description="Current response mode")
    overall_avg_reaction_time_ms: float = Field(description="Overall average reaction time in ms")
    min_reaction_time_ms: float = Field(description="Minimum reaction time in ms")
    max_reaction_time_ms: float = Field(description="Maximum reaction time in ms")
    active_status: bool = Field(description="Whether the system is active")
    top_patterns: List[ReflexInfo] = Field(description="Top patterns by execution count")

# =============== New Models for Function Tools ===============

class RegisterReflexResult(BaseModel):
    """Result of registering a reflex"""
    success: bool = Field(description="Whether registration was successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    reflex: Optional[ReflexInfo] = Field(None, description="Registered reflex details")

class ReactionInfo(BaseModel):
    """Reaction information"""
    pattern_name: str = Field(description="Pattern name")
    reaction_time_ms: float = Field(description="Reaction time in milliseconds")
    match_score: float = Field(description="Match score")

class ProcessStimulusResult(BaseModel):
    """Result of processing a stimulus"""
    success: bool = Field(description="Whether processing was successful")
    reason: Optional[str] = Field(None, description="Reason for failure")
    error: Optional[str] = Field(None, description="Error message if failed")
    pattern_name: Optional[str] = Field(None, description="Name of matched pattern")
    reaction_time_ms: Optional[float] = Field(None, description="Reaction time in milliseconds")
    match_score: Optional[float] = Field(None, description="Pattern match score")
    reaction: Optional[ReactionInfo] = Field(None, description="Reaction metadata")

class SetResponseModeResult(BaseModel):
    """Result of setting response mode"""
    success: bool = Field(description="Whether mode was set successfully")
    error: Optional[str] = Field(None, description="Error message if failed")
    mode: Optional[str] = Field(None, description="New response mode")

class OptimizationInfo(BaseModel):
    """Information about a pattern optimization"""
    status: str = Field(description="Optimization status")
    original_success_rate: Optional[float] = Field(None, description="Original success rate")
    original_avg_response_time: Optional[float] = Field(None, description="Original avg response time")
    optimization_type: Optional[str] = Field(None, description="Type of optimization")
    old_threshold: Optional[float] = Field(None, description="Old threshold")
    new_threshold: Optional[float] = Field(None, description="New threshold")

class PatternOptimization(BaseModel):
    """Optimization for a specific pattern"""
    pattern_name: str = Field(description="Pattern name")
    optimization: OptimizationInfo = Field(description="Optimization details")

class OptimizeReflexesResult(BaseModel):
    """Result of optimizing reflexes"""
    success: bool = Field(description="Whether optimization was successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    optimizations: List[PatternOptimization] = Field(default_factory=list, description="Optimization details")
    patterns_examined: Optional[int] = Field(None, description="Number of patterns examined")
    patterns_optimized: Optional[int] = Field(None, description="Number of patterns optimized")

# =============== Helper Functions ===============

def pattern_dict_to_dict(pattern: PatternDict) -> Dict[str, Any]:
    """Convert PatternDict to regular dict for backward compatibility"""
    result = {}
    for value in pattern.values:
        key = value.key
        if value.type == "string":
            result[key] = value.string_value
        elif value.type == "number":
            result[key] = value.number_value
        elif value.type == "range":
            result[key] = {"min": value.range_value.min, "max": value.range_value.max}
            if value.range_value.optimal is not None:
                result[key]["optimal"] = value.range_value.optimal
            if value.range_value.weight is not None:
                result[key]["weight"] = value.range_value.weight
        elif value.type == "list":
            result[key] = value.list_value
        elif value.type == "boolean":
            result[key] = value.boolean_value
        
        if value.required:
            if isinstance(result[key], dict):
                result[key]["required"] = True
            else:
                result[key] = {"value": result[key], "required": True}
    
    return result

def dict_to_pattern_dict(data: Dict[str, Any]) -> PatternDict:
    """Convert regular dict to PatternDict"""
    values = []
    for key, value in data.items():
        if isinstance(value, str):
            values.append(PatternValue(key=key, type="string", string_value=value))
        elif isinstance(value, (int, float)):
            values.append(PatternValue(key=key, type="number", number_value=float(value)))
        elif isinstance(value, bool):
            values.append(PatternValue(key=key, type="boolean", boolean_value=value))
        elif isinstance(value, list):
            values.append(PatternValue(key=key, type="list", list_value=[str(v) for v in value]))
        elif isinstance(value, dict):
            if "min" in value and "max" in value:
                values.append(PatternValue(
                    key=key,
                    type="range",
                    range_value=NumericRange(
                        min=value["min"],
                        max=value["max"],
                        optimal=value.get("optimal"),
                        weight=value.get("weight")
                    ),
                    required=value.get("required", False)
                ))
            elif "primary" in value and "variants" in value:
                values.append(PatternValue(
                    key=key,
                    type="string",
                    string_value=value["primary"],
                    weight=value.get("weight")
                ))
            else:
                # Generic handling - convert to string
                values.append(PatternValue(key=key, type="string", string_value=str(value)))
    
    return PatternDict(values=values)

def stimulus_dict_to_dict(stimulus: StimulusDict) -> Dict[str, Any]:
    """Convert StimulusDict to regular dict"""
    result = {}
    if stimulus.text is not None:
        result["text"] = stimulus.text
    if stimulus.visual_clarity is not None:
        result["visual_clarity"] = stimulus.visual_clarity
    if stimulus.timing is not None:
        result["timing"] = {
            "frame_number": stimulus.timing.frame_number,
            "time_to_impact": stimulus.timing.time_to_impact,
            "frame_window": stimulus.timing.frame_window
        }
    if stimulus.opponent is not None:
        result["opponent"] = {
            "state": stimulus.opponent.state,
            "position": stimulus.opponent.position,
            "orientation": stimulus.opponent.orientation
        }
    if stimulus.distractors is not None:
        result["distractors"] = [{"type": d.type, "intensity": d.intensity} for d in stimulus.distractors]
    
    # Add generic fields
    if stimulus.string_fields:
        for kv in stimulus.string_fields:
            result[kv.key] = kv.value
    if stimulus.number_fields:
        for kv in stimulus.number_fields:
            result[kv.key] = kv.value
    if stimulus.list_fields:
        for kv in stimulus.list_fields:
            result[kv.key] = kv.value
    
    return result

def dict_to_stimulus_dict(data: Dict[str, Any]) -> StimulusDict:
    """Convert regular dict to StimulusDict"""
    stimulus = StimulusDict()
    
    if "text" in data:
        stimulus.text = str(data["text"])
    if "visual_clarity" in data:
        stimulus.visual_clarity = float(data["visual_clarity"])
    if "timing" in data and isinstance(data["timing"], dict):
        stimulus.timing = TimingInfo(**data["timing"])
    if "opponent" in data and isinstance(data["opponent"], dict):
        stimulus.opponent = OpponentState(**data["opponent"])
    if "distractors" in data and isinstance(data["distractors"], list):
        stimulus.distractors = [DistractorInfo(**d) if isinstance(d, dict) else DistractorInfo(type=str(d), intensity=0.5) for d in data["distractors"]]
    
    # Handle other fields
    string_fields = []
    number_fields = []
    list_fields = []
    
    for key, value in data.items():
        if key not in ["text", "visual_clarity", "timing", "opponent", "distractors"]:
            if isinstance(value, str):
                string_fields.append(KeyValuePair(key=key, value=value))
            elif isinstance(value, (int, float)):
                number_fields.append(KeyNumberPair(key=key, value=float(value)))
            elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                list_fields.append(KeyListPair(key=key, value=value))
    
    if string_fields:
        stimulus.string_fields = string_fields
    if number_fields:
        stimulus.number_fields = number_fields
    if list_fields:
        stimulus.list_fields = list_fields
    
    return stimulus

# =============== Pattern Recognition ===============

class ReflexPattern:
    """Pattern that can trigger a reflexive response"""
    
    def __init__(self, 
                 name: str,
                 pattern_data: Dict[str, Any],
                 procedure_name: str,
                 threshold: float = 0.7,
                 priority: int = 1,
                 context_template: Dict[str, Any] = None):
        """
        Initialize a reflex pattern
        
        Args:
            name: Unique name for this pattern
            pattern_data: Pattern definition (features that should trigger this reflex)
            procedure_name: Name of procedure to execute when triggered
            threshold: Matching threshold (0.0-1.0)
            priority: Priority level (higher values take precedence)
            context_template: Template for context to pass to procedure
        """
        self.name = name
        self.pattern_data = pattern_data
        self.procedure_name = procedure_name
        self.threshold = threshold
        self.priority = priority          
        self.context_template = context_template or {}
        
        # Performance tracking
        self.execution_count = 0
        self.success_count = 0
        self.avg_execution_time = 0.0
        self.last_executed = None
        self.response_times = []  # Store last 10 response times in ms
        
    def track_execution(self, success: bool, execution_time: float):
        """Track execution statistics"""
        self.execution_count += 1
        if success:
            self.success_count += 1
        
        # Update average execution time
        if self.execution_count == 1:
            self.avg_execution_time = execution_time
        else:
            self.avg_execution_time = (self.avg_execution_time * (self.execution_count - 1) + execution_time) / self.execution_count
        
        self.last_executed = datetime.datetime.now()
        
        # Store response time (in milliseconds)
        self.response_times.append(execution_time * 1000)
        if len(self.response_times) > 10:
            self.response_times.pop(0)
    
    def get_success_rate(self) -> float:
        """Get success rate of this reflex pattern"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count
    
    def get_avg_response_time(self) -> float:
        """Get average response time in milliseconds"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        stats = ReflexStats(
            execution_count=self.execution_count,
            success_rate=self.get_success_rate(),
            avg_response_time_ms=self.get_avg_response_time(),
            last_executed=self.last_executed.isoformat() if self.last_executed else None
        )
        
        return {
            "name": self.name,
            "procedure_name": self.procedure_name,
            "threshold": self.threshold,
            "priority": self.priority,
            "stats": stats
        }

# =============== Pattern Recognition Agent and Tools ===============

class FastMatchInput(BaseModel):
    """Input for fast pattern matching"""
    stimulus: StimulusDict = Field(description="Stimulus data")
    pattern: PatternDict = Field(description="Pattern to match against")

@function_tool
async def fast_match(ctx: RunContextWrapper[Any], input_data: FastMatchInput) -> float:
    """
    Perform fast pattern matching
    
    Args:
        input_data: Contains stimulus and pattern data
        
    Returns:
        Match score (0.0-1.0)
    """
    stimulus = stimulus_dict_to_dict(input_data.stimulus)
    pattern = pattern_dict_to_dict(input_data.pattern)
    
    with custom_span("fast_match"):
        # Handle empty cases
        if not stimulus or not pattern:
            return 0.0
        
        # Get keys present in both
        common_keys = set(stimulus.keys()) & set(pattern.keys())
        if not common_keys:
            return 0.0
        
        # Calculate match score
        match_points = 0
        total_points = 0
        
        for key in common_keys:
            # Skip metadata keys
            if key.startswith('_'):
                continue
                
            total_points += 1
            
            # Simple equality check for strings
            if isinstance(pattern[key], str) and isinstance(stimulus[key], str):
                if stimulus[key].lower() == pattern[key].lower():
                    match_points += 1
                elif pattern[key].lower() in stimulus[key].lower():
                    match_points += 0.7
                elif stimulus[key].lower() in pattern[key].lower():
                    match_points += 0.5
            
            # Numeric range check
            elif isinstance(pattern[key], dict) and 'min' in pattern[key] and 'max' in pattern[key] and isinstance(stimulus[key], (int, float)):
                min_val = pattern[key]['min']
                max_val = pattern[key]['max']
                
                if min_val <= stimulus[key] <= max_val:
                    match_points += 1
                else:
                    # Partial match based on distance to range
                    distance = min(abs(stimulus[key] - min_val), abs(stimulus[key] - max_val))
                    range_size = max_val - min_val
                    if range_size > 0:
                        match_points += max(0, 1 - (distance / range_size))
            
            # Array/list contains check
            elif isinstance(pattern[key], list) and isinstance(stimulus[key], list):
                # Calculate overlap
                pattern_set = set(pattern[key])
                stimulus_set = set(stimulus[key])
                
                if pattern_set and stimulus_set:
                    overlap = len(pattern_set & stimulus_set)
                    match_points += overlap / max(len(pattern_set), len(stimulus_set))
            
            # Simple equality for other types
            elif pattern[key] == stimulus[key]:
                match_points += 1
        
        # Add special handling for required keys
        required_keys = {k for k, v in pattern.items() if isinstance(v, dict) and v.get('required', False)}
        if required_keys:
            for key in required_keys:
                if key not in stimulus:
                    return 0.0  # Required key missing, no match
        
        # If we have significant keys to check
        if total_points > 0:
            return match_points / total_points
        
        return 0.0

class GenerateSimilarStimulusInput(BaseModel):
    """Input for generating similar stimulus"""
    pattern: PatternDict = Field(description="Pattern to generate similar stimulus for")

class GeneratedStimulus(BaseModel):
    """Generated stimulus data"""
    data: StimulusDict = Field(description="Generated stimulus data")

@function_tool
async def generate_similar_stimulus(ctx: RunContextWrapper[Any], input_data: GenerateSimilarStimulusInput) -> GeneratedStimulus:
    """
    Generate a stimulus similar to the given pattern for training
    
    Args:
        input_data: Contains pattern to generate similar stimulus for
        
    Returns:
        Generated stimulus
    """
    pattern = pattern_dict_to_dict(input_data.pattern)
    
    with custom_span("generate_similar_stimulus"):
        stimulus = {}
        
        for key, value in pattern.items():
            # Skip metadata keys
            if key.startswith('_'):
                continue
            
            # String handling
            if isinstance(value, str):
                if random.random() < 0.8:  # 80% keep the same
                    stimulus[key] = value
                else:
                    # Generate variation
                    stimulus[key] = value + " (variant)"
            
            # Numeric range handling
            elif isinstance(value, dict) and 'min' in value and 'max' in value:
                min_val = value['min']
                max_val = value['max']
                
                # Generate random value in range
                stimulus[key] = min_val + random.random() * (max_val - min_val)
            
            # List handling
            elif isinstance(value, list):
                # Use subset of list items
                use_count = max(1, len(value) - random.randint(0, min(2, len(value))))
                stimulus[key] = random.sample(value, use_count)
            
            # Other types
            else:
                stimulus[key] = value
        
        return GeneratedStimulus(data=dict_to_stimulus_dict(stimulus))

class GenerateGamingStimulusInput(BaseModel):
    """Input for generating gaming stimulus"""
    pattern: PatternDict = Field(description="Base pattern")
    difficulty: float = Field(0.5, description="Difficulty level (0.0-1.0)")

@function_tool
async def generate_gaming_stimulus(ctx: RunContextWrapper[Any], 
                                  input_data: GenerateGamingStimulusInput) -> GeneratedStimulus:
    """
    Generate a gaming-specific stimulus with appropriate timing and visual elements
    
    Args:
        input_data: Contains pattern and difficulty level
        
    Returns:
        Gaming-specific stimulus
    """
    pattern = input_data.pattern
    difficulty = input_data.difficulty
    
    with custom_span("generate_gaming_stimulus"):
        # Start with base stimulus
        base_input = GenerateSimilarStimulusInput(pattern=pattern)
        base_result = await generate_similar_stimulus(ctx, base_input)
        stimulus_dict = base_result.data
        
        # Add gaming-specific elements
        stimulus_dict.timing = TimingInfo(
            frame_number=random.randint(1, 1000),
            time_to_impact=max(100, 500 - int(300 * difficulty)),
            frame_window=max(3, 10 - int(difficulty * 7))
        )
        
        # Add visual elements
        visual_noise = difficulty * 0.5
        stimulus_dict.visual_clarity = max(0.2, 1.0 - visual_noise)
        
        # Add opponent state
        stimulus_dict.opponent = OpponentState(
            state=random.choice(["attacking", "defending", "neutral", "special"]),
            position=random.choice(["close", "mid", "far"]),
            orientation=random.choice(["facing", "side", "away"])
        )
        
        return GeneratedStimulus(data=stimulus_dict)

class OptimizePatternInput(BaseModel):
    """Input for optimizing a pattern"""
    pattern: PatternDict = Field(description="Pattern to optimize")

class OptimizedPattern(BaseModel):
    """Optimized pattern data"""
    data: PatternDict = Field(description="Optimized pattern data")

@function_tool
async def optimize_pattern(ctx: RunContextWrapper[Any], input_data: OptimizePatternInput) -> OptimizedPattern:
    """
    Optimize a pattern for better recognition
    
    Args:
        input_data: Contains pattern to optimize
        
    Returns:
        Optimized pattern
    """
    pattern = pattern_dict_to_dict(input_data.pattern)
    
    with custom_span("optimize_pattern"):
        optimized = pattern.copy()
        
        # Find most distinctive features
        for key, value in optimized.items():
            # Skip metadata keys
            if key.startswith('_'):
                continue
            
            # Enhance string matching with variants
            if isinstance(value, str) and len(value) > 5:
                # Add variations or keywords for more robust matching
                optimized[key] = {
                    "primary": value,
                    "variants": [value.lower(), value.upper(), value.capitalize()],
                    "weight": 1.2  # Give this feature more weight
                }
            
            # Enhance numeric ranges
            elif isinstance(value, dict) and 'min' in value and 'max' in value:
                # Expand range slightly for better matching
                min_val = value['min']
                max_val = value['max']
                range_size = max_val - min_val
                
                optimized[key] = {
                    'min': min_val - (range_size * 0.1),
                    'max': max_val + (range_size * 0.1),
                    'optimal': (min_val + max_val) / 2,
                    'weight': 1.5
                }
        
        return OptimizedPattern(data=dict_to_pattern_dict(optimized))

class SimplifyPatternInput(BaseModel):
    """Input for simplifying a pattern"""
    pattern: PatternDict = Field(description="Pattern to simplify")

class SimplifiedPattern(BaseModel):
    """Simplified pattern data"""
    data: PatternDict = Field(description="Simplified pattern data")

@function_tool
async def simplify_pattern(ctx: RunContextWrapper[Any], input_data: SimplifyPatternInput) -> SimplifiedPattern:
    """
    Simplify a pattern for faster recognition
    
    Args:
        input_data: Contains pattern to simplify
        
    Returns:
        Simplified pattern
    """
    pattern = pattern_dict_to_dict(input_data.pattern)
    
    with custom_span("simplify_pattern"):
        simplified = {}
        
        # Find most important features
        feature_scores = {}
        for key, value in pattern.items():
            # Skip metadata keys
            if key.startswith('_'):
                continue
            
            # Score features by expected distinctiveness
            if isinstance(value, dict) and value.get('weight', 0) > 0:
                feature_scores[key] = value.get('weight', 1)
            elif isinstance(value, dict) and ('min' in value or 'max' in value):
                feature_scores[key] = 1.2  # Numeric ranges are distinctive
            elif isinstance(value, str) and len(value) > 3:
                feature_scores[key] = 1.0  # Strings are useful
            elif isinstance(value, list) and len(value) > 0:
                feature_scores[key] = 0.8  # Lists are somewhat useful
            else:
                feature_scores[key] = 0.5  # Other features
        
        # Keep only top 3 most distinctive features
        top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for key, _ in top_features:
            simplified[key] = pattern[key]
        
        # Ensure we keep metadata
        for key, value in pattern.items():
            if key.startswith('_'):
                simplified[key] = value
        
        return SimplifiedPattern(data=dict_to_pattern_dict(simplified))

# =============== Reflexive System Class ===============

class ReflexiveSystem:
    """
    System for rapid, reflexive responses without deliberate thinking.
    
    This system manages stimulus-response patterns that can trigger immediate
    execution of procedures, bypassing the normal thinking process for extremely
    fast reactions - similar to muscle memory or reflexes in humans.
    """
    
    def __init__(self, agent_enhanced_memory):
        """
        Initialize the reflexive system
        
        Args:
            agent_enhanced_memory: Reference to the agent enhanced memory system
        """
        self.memory_manager = agent_enhanced_memory
        self.reflex_patterns = {}  # name -> ReflexPattern
        
        # Reaction time tracking
        self.reaction_times = []  # Track last 100 reaction times in ms
        self._is_active = True
        
        # Specialized reaction libraries (domain-specific)
        self.domain_libraries = {
            "gaming": {},
            "conversation": {},
            "physical": {},
            "decision": {}
        }
        
        # Response modes
        self.response_mode = "normal"  # normal, hyper, relaxed, learning
        
        # For training
        self.training_data = []  # Store stimulus-response pairs for training
        self.training_in_progress = False
        
        # Initialize trace group ID
        self.trace_group_id = f"nyx-reflexive-{random.randint(10000, 99999)}"
        
        # Initialize the agents
        self._init_agents()
        self.decision_system = self.decision_system_agent
        
        logger.info("Reflexive system initialized with Agents SDK")
    
    def _init_agents(self):
        """Initialize the agents for the reflexive system"""
        # Pattern Recognition Agent
        self.pattern_recognition_agent = Agent(
            name="Pattern Recognition Agent",
            instructions="""You are a specialized agent for pattern recognition in the reflexive system.
            Your job is to quickly identify patterns in stimuli and match them to known patterns.
            Focus on speed and accuracy in pattern matching.""",
            tools=[
                fast_match,
                generate_similar_stimulus,
                generate_gaming_stimulus
            ],
            model_settings=ModelSettings(temperature=0.4),
            model="gpt-4.1-nano"
        )
        
        # Pattern Optimization Agent
        self.pattern_optimization_agent = Agent(
            name="Pattern Optimization Agent",
            instructions="""You are a specialized agent for optimizing and refining patterns in the reflexive system.
            Your job is to improve pattern definitions for better matching accuracy and performance.
            Focus on identifying the most distinctive features and creating efficient patterns.""",
            tools=[
                optimize_pattern,
                simplify_pattern
            ],
            model_settings=ModelSettings(temperature=0.4),
            model="gpt-4.1-nano"
        )
        
        # Gaming Reflex Agent
        self.gaming_reflex_agent = Agent(
            name="Gaming Reflex Agent",
            instructions="""You are a specialized agent for gaming reflexes in the reflexive system.
            Your job is to create and optimize reflex patterns specifically for gaming scenarios.
            Focus on timing-critical reactions and accurate pattern recognition for game actions.""",
            tools=[
                generate_gaming_stimulus,
                fast_match
            ],
            model_settings=ModelSettings(temperature=0.4),
            model="gpt-4.1-nano"
        )
        
        # Decision System Agent
        self.decision_system_agent = Agent(
            name="Decision System Agent",
            instructions="""You are a specialized agent for the reflex decision system.
            Your job is to decide when to use reflexes vs. deliberate thinking based on context.
            Focus on analyzing stimuli and contexts to make appropriate reflex usage decisions.""",
            model_settings=ModelSettings(temperature=0.5),
            model="gpt-4.1-nano"
        )
        
        # Main Reflexive Agent
        self.main_agent = Agent(
            name="Reflexive System Coordinator",
            instructions="""You are the main coordinator for the reflexive system.
            Your job is to coordinate among the specialized reflex agents and manage the overall system.
            Focus on efficient processing of stimuli and directing them to the appropriate agent.""",
            handoffs=[
                handoff(self.pattern_recognition_agent, 
                       tool_name_override="recognize_pattern",
                       tool_description_override="Recognize patterns in stimuli"),
                handoff(self.pattern_optimization_agent,
                       tool_name_override="optimize_pattern",
                       tool_description_override="Optimize patterns for better recognition"),
                handoff(self.gaming_reflex_agent,
                       tool_name_override="handle_gaming_reflex",
                       tool_description_override="Handle gaming-specific reflexes"),
                handoff(self.decision_system_agent,
                       tool_name_override="make_reflex_decision",
                       tool_description_override="Decide whether to use reflexes or deliberate thinking")
            ],
            model_settings=ModelSettings(temperature=0.4),
            model="gpt-4.1-nano"
        )

    async def should_use_reflex(self, stimulus: Dict[str, Any], context: Dict[str, Any] = None, pattern: Any = None) -> Tuple[bool, float]:
        """
        Determine whether to use reflexive response or deliberate thinking
        
        Args:
            stimulus: The stimulus data
            context: Additional context information
            pattern: Optional pattern information (not used in this implementation)
            
        Returns:
            Tuple of (should_use_reflex, confidence)
        """
        if not self._is_active:
            return False, 0.0
        
        # Quick checks for reflexive indicators
        confidence = 0.0
        
        # Check if we have any matching patterns
        patterns_to_check = self.reflex_patterns
        if context and "domain" in context and context["domain"] in self.domain_libraries:
            patterns_to_check = self.domain_libraries[context["domain"]]
        
        if not patterns_to_check:
            return False, 0.0
        
        # Quick pattern scan
        highest_match = 0.0
        for name, pattern in patterns_to_check.items():
            # Do a quick similarity check without full matching
            pattern_keys = set(pattern.pattern_data.keys())
            stimulus_keys = set(stimulus.get("text", {}).keys()) if isinstance(stimulus.get("text"), dict) else set()
            
            # If stimulus is just text, check for key phrases
            if isinstance(stimulus.get("text"), str):
                text = stimulus["text"].lower()
                # Check for common reflex triggers
                reflex_triggers = ["quick", "fast", "urgent", "immediately", "now", "react", "reflex"]
                if any(trigger in text for trigger in reflex_triggers):
                    confidence = max(confidence, 0.7)
            
            # Key overlap check for dict stimuli
            if pattern_keys and stimulus_keys:
                overlap = len(pattern_keys & stimulus_keys) / len(pattern_keys)
                if overlap > 0.5:
                    highest_match = max(highest_match, overlap)
        
        # Combine factors
        confidence = max(confidence, highest_match * 0.8)
        
        # Check response mode
        if self.response_mode == "hyper":
            confidence = min(1.0, confidence * 1.2)  # Increase confidence in hyper mode
        elif self.response_mode == "relaxed":
            confidence = confidence * 0.8  # Decrease confidence in relaxed mode
        
        # Decision threshold
        should_use = confidence > 0.5
        
        return should_use, confidence
    
    class RegisterReflexInput(BaseModel):
        """Input for registering a reflex"""
        name: str = Field(description="Unique name for this pattern")
        pattern_data: PatternDict = Field(description="Pattern definition")
        procedure_name: str = Field(description="Name of procedure to execute")
        threshold: float = Field(0.7, description="Matching threshold (0.0-1.0)")
        priority: int = Field(1, description="Priority level")
        domain: Optional[str] = Field(None, description="Optional domain")
        context_template: Optional[ContextDict] = Field(None, description="Context template")
    
    @function_tool
    async def register_reflex(self, input_data: RegisterReflexInput) -> RegisterReflexResult:
        """
        Register a new reflex pattern
        
        Args:
            input_data: Registration parameters
            
        Returns:
            Registration result
        """
        with trace(workflow_name="Register Reflex", group_id=self.trace_group_id):
            # Validate procedure exists
            procedures = await self.memory_manager.list_procedures()
            if input_data.procedure_name not in [p["name"] for p in procedures]:
                return RegisterReflexResult(
                    success=False,
                    error=f"Procedure '{input_data.procedure_name}' not found"
                )
            
            # Convert PatternDict to regular dict for ReflexPattern
            pattern_data = pattern_dict_to_dict(input_data.pattern_data)
            context_template = {}
            if input_data.context_template is not None:
                if input_data.context_template.game:
                    context_template["game"] = input_data.context_template.game
                if input_data.context_template.action_type:
                    context_template["action_type"] = input_data.context_template.action_type
                if input_data.context_template.gaming_reflex is not None:
                    context_template["gaming_reflex"] = input_data.context_template.gaming_reflex
                if input_data.context_template.string_fields:
                    for kv in input_data.context_template.string_fields:
                        context_template[kv.key] = kv.value
                if input_data.context_template.number_fields:
                    for kv in input_data.context_template.number_fields:
                        context_template[kv.key] = kv.value
                if input_data.context_template.boolean_fields:
                    for kv in input_data.context_template.boolean_fields:
                        context_template[kv.key] = kv.value
            
            # Create reflex pattern
            reflex = ReflexPattern(
                name=input_data.name,
                pattern_data=pattern_data,
                procedure_name=input_data.procedure_name,
                threshold=input_data.threshold,
                priority=input_data.priority,
                context_template=context_template
            )
            
            # Store in main registry
            self.reflex_patterns[input_data.name] = reflex
            
            # If domain specified, add to domain library
            if input_data.domain and input_data.domain in self.domain_libraries:
                self.domain_libraries[input_data.domain][input_data.name] = reflex
            
            logger.info(f"Registered reflex pattern '{input_data.name}' for procedure '{input_data.procedure_name}'")
            
            return RegisterReflexResult(
                success=True,
                reflex=ReflexInfo(
                    name=reflex.name,
                    procedure_name=reflex.procedure_name,
                    threshold=reflex.threshold,
                    priority=reflex.priority,
                    stats=ReflexStats(
                        execution_count=reflex.execution_count,
                        success_rate=reflex.get_success_rate(),
                        avg_response_time_ms=reflex.get_avg_response_time(),
                        last_executed=reflex.last_executed.isoformat() if reflex.last_executed else None
                    )
                )
            )
    
    class ProcessStimulusInput(BaseModel):
        """Input for processing stimulus"""
        stimulus: StimulusDict = Field(description="The stimulus data")
        domain: Optional[str] = Field(None, description="Optional domain")
        context: Optional[ContextDict] = Field(None, description="Additional context")
    
    @function_tool
    async def process_stimulus_fast(self, input_data: ProcessStimulusInput) -> ProcessStimulusResult:
        """
        Process stimulus with minimal overhead for fastest possible reaction
        
        This method uses a streamlined matching process for absolute minimum latency.
        
        Args:
            input_data: Stimulus processing parameters
            
        Returns:
            Reaction result
        """
        with trace(workflow_name="Process Stimulus Fast", group_id=self.trace_group_id):
            if not self._is_active:
                return ProcessStimulusResult(success=False, reason="reflexive_system_inactive")
            
            start_time = time.time()
            matched_pattern = None
            highest_match = 0.0
            highest_priority = -1
            
            # Determine patterns to check
            patterns_to_check = self.reflex_patterns
            if input_data.domain and input_data.domain in self.domain_libraries:
                patterns_to_check = self.domain_libraries[input_data.domain]
            
            # Create context for function tools
            tool_context = {
                "stimulus": input_data.stimulus,
                "domain": input_data.domain,
                "context": input_data.context or ContextDict()
            }
            
            # Fast pattern matching
            for name, pattern in patterns_to_check.items():
                match_input = FastMatchInput(
                    stimulus=input_data.stimulus,
                    pattern=dict_to_pattern_dict(pattern.pattern_data)
                )
                match_score = await fast_match(
                    RunContextWrapper(tool_context),
                    match_input
                )
                
                if match_score >= pattern.threshold:
                    # Check if this is higher priority or better match
                    if (pattern.priority > highest_priority or 
                        (pattern.priority == highest_priority and match_score > highest_match)):
                        matched_pattern = pattern
                        highest_match = match_score
                        highest_priority = pattern.priority
            
            # If no match found, return quickly
            if not matched_pattern:
                end_time = time.time()
                self.reaction_times.append((end_time - start_time) * 1000)
                if len(self.reaction_times) > 100:
                    self.reaction_times.pop(0)
                    
                return ProcessStimulusResult(
                    success=False,
                    reason="no_matching_pattern",
                    reaction_time_ms=(end_time - start_time) * 1000
                )
            
            # Prepare execution context
            execution_context = matched_pattern.context_template.copy() if matched_pattern.context_template else {}
            if input_data.context:
                context_dict = {}
                if input_data.context.string_fields:
                    for kv in input_data.context.string_fields:
                        context_dict[kv.key] = kv.value
                if input_data.context.number_fields:
                    for kv in input_data.context.number_fields:
                        context_dict[kv.key] = kv.value
                if input_data.context.boolean_fields:
                    for kv in input_data.context.boolean_fields:
                        context_dict[kv.key] = kv.value
                if input_data.context.simulation is not None:
                    context_dict["simulation"] = input_data.context.simulation
                if input_data.context.difficulty is not None:
                    context_dict["difficulty"] = input_data.context.difficulty
                execution_context.update(context_dict)
            
            # Add stimulus data and match details
            execution_context["stimulus"] = stimulus_dict_to_dict(input_data.stimulus)
            execution_context["match_score"] = highest_match
            execution_context["reflexive"] = True
            execution_context["reaction_start_time"] = start_time
            
            # Execute procedure with minimal overhead
            try:
                result = await self.memory_manager.execute_procedure(
                    matched_pattern.procedure_name,
                    context=execution_context,
                    force_conscious=False,  # Ensure non-conscious execution
                    priority="critical"  # Set highest priority
                )
                
                # Track execution time
                end_time = time.time()
                execution_time = end_time - start_time
                matched_pattern.track_execution(
                    success=result.get("success", False),
                    execution_time=execution_time
                )
                
                # Record reaction time
                reaction_time_ms = execution_time * 1000
                self.reaction_times.append(reaction_time_ms)
                if len(self.reaction_times) > 100:
                    self.reaction_times.pop(0)
                
                # For training purposes
                if self.response_mode == "learning":
                    self.training_data.append({
                        "stimulus": stimulus_dict_to_dict(input_data.stimulus),
                        "pattern_name": matched_pattern.name,
                        "reaction_time_ms": reaction_time_ms,
                        "success": result.get("success", False)
                    })
                
                # Build result
                return ProcessStimulusResult(
                    success=result.get("success", False),
                    pattern_name=matched_pattern.name,
                    reaction_time_ms=reaction_time_ms,
                    match_score=highest_match,
                    reaction=ReactionInfo(
                        pattern_name=matched_pattern.name,
                        reaction_time_ms=reaction_time_ms,
                        match_score=highest_match
                    )
                )
                
            except Exception as e:
                logger.error(f"Error executing reflex pattern '{matched_pattern.name}': {e}")
                
                # Still track execution for failed attempts
                end_time = time.time()
                matched_pattern.track_execution(
                    success=False,
                    execution_time=end_time - start_time
                )
                
                return ProcessStimulusResult(
                    success=False,
                    error=str(e),
                    pattern_name=matched_pattern.name,
                    reaction_time_ms=(end_time - start_time) * 1000
                )
    
    class TrainReflexesInput(BaseModel):
        """Input for training reflexes"""
        training_iterations: int = Field(100, description="Number of training iterations")
        domain: Optional[str] = Field(None, description="Optional domain to limit training")
    
    @function_tool
    async def train_reflexes(self, input_data: TrainReflexesInput) -> TrainingResult:
        """
        Train reflexes to improve response time and accuracy
        
        Args:
            input_data: Training parameters
            
        Returns:
            Training results
        """
        with trace(workflow_name="Train Reflexes", group_id=self.trace_group_id):
            if self.training_in_progress:
                return TrainingResult(
                    success=False,
                    iterations=0,
                    improvements=[],
                    training_accuracy=0.0,
                    avg_reaction_time=0.0
                )
            
            self.training_in_progress = True
            
            try:
                # Get patterns to train
                patterns_to_train = self.reflex_patterns
                if input_data.domain and input_data.domain in self.domain_libraries:
                    patterns_to_train = self.domain_libraries[input_data.domain]
                
                if not patterns_to_train:
                    return TrainingResult(
                        success=False,
                        iterations=0,
                        improvements=[],
                        training_accuracy=0.0,
                        avg_reaction_time=0.0
                    )
                
                # Track improvements
                original_stats = {}
                for name, pattern in patterns_to_train.items():
                    original_stats[name] = {
                        "avg_response_time": pattern.get_avg_response_time(),
                        "success_rate": pattern.get_success_rate()
                    }
                
                # Create context for function tools
                tool_context = {
                    "domain": input_data.domain,
                    "iterations": input_data.training_iterations,
                    "patterns": {name: pattern.pattern_data for name, pattern in patterns_to_train.items()}
                }
                
                # Generate training stimuli
                training_stimuli = []
                for _ in range(input_data.training_iterations):
                    # Generate stimulus similar to patterns we're training
                    pattern = random.choice(list(patterns_to_train.values()))
                    gen_input = GenerateSimilarStimulusInput(pattern=dict_to_pattern_dict(pattern.pattern_data))
                    gen_result = await generate_similar_stimulus(
                        RunContextWrapper(tool_context),
                        gen_input
                    )
                    training_stimuli.append({
                        "stimulus": gen_result.data,
                        "target_pattern": pattern.name
                    })
                
                # Run training iterations
                results = []
                original_mode = self.response_mode
                self.response_mode = "learning"
                
                for training_item in training_stimuli:
                    stimulus = training_item["stimulus"]
                    target_pattern = training_item["target_pattern"]
                    
                    # Process stimulus
                    process_input = ProcessStimulusInput(stimulus=stimulus)
                    result = await self.process_stimulus_fast(process_input)
                    
                    # Record result
                    results.append({
                        "target_pattern": target_pattern,
                        "matched_pattern": result.pattern_name,
                        "reaction_time_ms": result.reaction_time_ms,
                        "success": result.success
                    })
                
                # Restore original mode
                self.response_mode = original_mode
                
                # Calculate improvements
                improvements = []
                for name, pattern in patterns_to_train.items():
                    current_stats = CurrentStats(
                        avg_response_time=pattern.get_avg_response_time(),
                        success_rate=pattern.get_success_rate()
                    )
                    
                    # Calculate improvements
                    if name in original_stats:
                        time_improvement = original_stats[name]["avg_response_time"] - current_stats.avg_response_time
                        success_improvement = current_stats.success_rate - original_stats[name]["success_rate"]
                        
                        improvements.append(PatternImprovement(
                            pattern_name=name,
                            improvement=ImprovementData(
                                response_time_improvement_ms=time_improvement,
                                success_rate_improvement=success_improvement,
                                current_stats=current_stats
                            )
                        ))
                
                # Apply automatic threshold adjustments based on training
                for name, pattern in patterns_to_train.items():
                    # Adjust threshold based on success rate
                    if pattern.execution_count >= 10:
                        success_rate = pattern.get_success_rate()
                        if success_rate < 0.6 and pattern.threshold > 0.6:
                            # Lower threshold slightly if success rate is low
                            pattern.threshold = max(0.6, pattern.threshold - 0.05)
                        elif success_rate > 0.9 and pattern.threshold < 0.9:
                            # Increase threshold if success rate is very high
                            pattern.threshold = min(0.9, pattern.threshold + 0.03)
                
                # Calculate final results
                training_accuracy = sum(1 for r in results if r["target_pattern"] == r["matched_pattern"]) / len(results) if results else 0
                avg_reaction_time = sum(r["reaction_time_ms"] for r in results) / len(results) if results else 0
                
                return TrainingResult(
                    success=True,
                    iterations=input_data.training_iterations,
                    improvements=improvements,
                    training_accuracy=training_accuracy,
                    avg_reaction_time=avg_reaction_time
                )
                
            except Exception as e:
                logger.error(f"Error during reflex training: {e}")
                return TrainingResult(
                    success=False,
                    iterations=0,
                    improvements=[],
                    training_accuracy=0.0,
                    avg_reaction_time=0.0
                )
                
            finally:
                self.training_in_progress = False
    
    class AddGamingReflexInput(BaseModel):
        """Input for adding a gaming reflex"""
        game_name: str = Field(description="Name of the game")
        action_type: str = Field(description="Type of action")
        trigger_pattern: PatternDict = Field(description="Pattern to recognize")
        response_procedure: str = Field(description="Procedure to execute")
        reaction_threshold: float = Field(0.7, description="Recognition threshold")
    
    @function_tool
    async def add_gaming_reflex(self, input_data: AddGamingReflexInput) -> RegisterReflexResult:
        """
        Add a specialized gaming reflex
        
        Args:
            input_data: Gaming reflex parameters
            
        Returns:
            Registration result
        """
        with trace(workflow_name="Add Gaming Reflex", group_id=self.trace_group_id):
            reflex_name = f"gaming_{input_data.game_name}_{input_data.action_type}_{len(self.domain_libraries['gaming'])+1}"
            
            # Add game-specific context to template
            context_template = ContextDict(
                game=input_data.game_name,
                action_type=input_data.action_type,
                gaming_reflex=True
            )
            
            # Gaming reflexes get high priority
            priority = 3
            
            # Create context for the gaming reflex agent
            agent_context = {
                "game_name": input_data.game_name,
                "action_type": input_data.action_type,
                "trigger_pattern": pattern_dict_to_dict(input_data.trigger_pattern),
                "response_procedure": input_data.response_procedure,
                "reaction_threshold": input_data.reaction_threshold
            }
            
            # Run the gaming reflex agent to optimize the pattern
            gaming_agent_prompt = f"""Create an optimized gaming reflex pattern for game '{input_data.game_name}' and action '{input_data.action_type}'.
            The pattern should be optimized for quick recognition in gaming scenarios with appropriate timing elements.
            Consider adding timing information, visual elements, and game-specific attributes for better recognition.
            """
            
            run_config = RunConfig(
                workflow_name=f"Gaming Reflex Creation - {input_data.game_name}",
                trace_metadata={
                    "game": input_data.game_name,
                    "action_type": input_data.action_type
                }
            )
            
            try:
                result = await Runner.run(
                    self.gaming_reflex_agent,
                    gaming_agent_prompt,
                    context=agent_context,
                    run_config=run_config
                )
                
                # Check if we got an optimized pattern
                optimized_pattern = input_data.trigger_pattern
                if isinstance(result.final_output, dict) and "optimized_pattern" in result.final_output:
                    optimized_pattern = dict_to_pattern_dict(result.final_output["optimized_pattern"])
                
                # Register the reflex
                reg_input = self.RegisterReflexInput(
                    name=reflex_name,
                    pattern_data=optimized_pattern,
                    procedure_name=input_data.response_procedure,
                    threshold=input_data.reaction_threshold,
                    priority=priority,
                    domain="gaming",
                    context_template=context_template
                )
                registration_result = await self.register_reflex(reg_input)
                
                return registration_result
                
            except Exception as e:
                logger.error(f"Error creating gaming reflex: {e}")
                return RegisterReflexResult(
                    success=False,
                    error=f"Failed to create gaming reflex: {str(e)}"
                )
    
    class SimulateGamingScenariosInput(BaseModel):
        """Input for simulating gaming scenarios"""
        game_name: str = Field(description="Game to simulate")
        scenario_count: int = Field(10, description="Number of scenarios to run")
    
    @function_tool
    async def simulate_gaming_scenarios(self, input_data: SimulateGamingScenariosInput) -> SimulationResult:
        """
        Simulate gaming scenarios to test and improve reaction time
        
        Args:
            input_data: Simulation parameters
            
        Returns:
            Simulation results
        """
        with trace(workflow_name=f"Simulate Gaming Scenarios - {input_data.game_name}", group_id=self.trace_group_id):
            # Get game-specific patterns
            game_patterns = {
                name: pattern for name, pattern in self.domain_libraries.get("gaming", {}).items()
                if pattern.context_template.get("game") == input_data.game_name
            }
            
            if not game_patterns:
                return SimulationResult(
                    success=False,
                    game=input_data.game_name,
                    scenarios_run=0,
                    success_rate=0.0,
                    avg_reaction_time_ms=0.0,
                    results=[]
                )
            
            # Create context for function tools
            tool_context = {
                "game_name": input_data.game_name,
                "scenario_count": input_data.scenario_count,
                "game_patterns": {name: pattern.pattern_data for name, pattern in game_patterns.items()}
            }
            
            # Run simulations
            results = []
            
            for i in range(input_data.scenario_count):
                # Select a random pattern to test
                pattern_name = random.choice(list(game_patterns.keys()))
                pattern = game_patterns[pattern_name]
                
                # Generate a stimulus based on the pattern
                difficulty = min(1.0, 0.5 + (i/input_data.scenario_count))  # Increase difficulty gradually
                
                gen_input = GenerateGamingStimulusInput(
                    pattern=dict_to_pattern_dict(pattern.pattern_data),
                    difficulty=difficulty
                )
                gen_result = await generate_gaming_stimulus(
                    RunContextWrapper(tool_context),
                    gen_input
                )
                stimulus = gen_result.data
                
                # Add some distractor elements
                if random.random() < 0.3:
                    stimulus.distractors = [
                        DistractorInfo(type="visual", intensity=random.random()),
                        DistractorInfo(type="audio", intensity=random.random() * 0.5)
                    ]
                
                # Process stimulus
                start_time = time.time()
                process_context = ContextDict(simulation=True, difficulty=difficulty)
                process_input = ProcessStimulusInput(
                    stimulus=stimulus,
                    domain="gaming",
                    context=process_context
                )
                reaction = await self.process_stimulus_fast(process_input)
                reaction_time = (time.time() - start_time) * 1000  # ms
                
                results.append(ScenarioResult(
                    scenario=i+1,
                    pattern_tested=pattern_name,
                    reaction_time_ms=reaction_time,
                    success=reaction.success,
                    matched_pattern=reaction.pattern_name
                ))
            
            # Calculate statistics
            success_count = sum(1 for r in results if r.success)
            avg_reaction_time = sum(r.reaction_time_ms for r in results) / len(results) if results else 0
            
            return SimulationResult(
                success=True,
                game=input_data.game_name,
                scenarios_run=input_data.scenario_count,
                success_rate=success_count / input_data.scenario_count if input_data.scenario_count > 0 else 0.0,
                avg_reaction_time_ms=avg_reaction_time,
                results=results
            )
    
    class SetResponseModeInput(BaseModel):
        """Input for setting response mode"""
        mode: str = Field(description="Response mode (normal, hyper, relaxed, learning)")
    
    @function_tool
    def set_response_mode(self, input_data: SetResponseModeInput) -> SetResponseModeResult:
        """
        Set the response mode
        
        Args:
            input_data: Mode setting parameters
            
        Returns:
            Mode change result
        """
        valid_modes = ["normal", "hyper", "relaxed", "learning"]
        
        if input_data.mode not in valid_modes:
            return SetResponseModeResult(
                success=False,
                error=f"Invalid mode. Valid modes are: {', '.join(valid_modes)}"
            )
        
        # Apply mode-specific adjustments
        if input_data.mode == "hyper":
            # Hyper mode lowers thresholds for quicker reactions
            for pattern in self.reflex_patterns.values():
                pattern._original_threshold = pattern.threshold
                pattern.threshold = max(0.5, pattern.threshold - 0.15)
        elif input_data.mode == "relaxed":
            # Relaxed mode increases thresholds for more deliberate reactions
            for pattern in self.reflex_patterns.values():
                pattern._original_threshold = pattern.threshold
                pattern.threshold = min(0.95, pattern.threshold + 0.15)
        elif input_data.mode == "normal" and hasattr(self, "_original_thresholds"):
            # Restore original thresholds
            for name, pattern in self.reflex_patterns.items():
                if hasattr(pattern, "_original_threshold"):
                    pattern.threshold = pattern._original_threshold
                    delattr(pattern, "_original_threshold")
        
        # Set mode
        self.response_mode = input_data.mode
        
        return SetResponseModeResult(
            success=True,
            mode=input_data.mode
        )
    
    @function_tool
    async def get_reflexive_stats(self) -> ReflexiveSystemStats:
        """Get statistics about reflexive system performance"""
        with custom_span("get_reflexive_stats"):
            # Calculate domain counts
            domain_counts = []
            for domain, patterns in self.domain_libraries.items():
                if patterns:
                    domain_counts.append(DomainCount(domain=domain, count=len(patterns)))
            
            # Get reaction time stats
            avg_reaction_time = sum(self.reaction_times) / len(self.reaction_times) if self.reaction_times else 0
            min_reaction_time = min(self.reaction_times) if self.reaction_times else 0
            max_reaction_time = max(self.reaction_times) if self.reaction_times else 0
            
            # Get top patterns by execution count
            top_patterns = sorted(
                self.reflex_patterns.values(),
                key=lambda p: p.execution_count,
                reverse=True
            )[:5]
            
            return ReflexiveSystemStats(
                total_patterns=len(self.reflex_patterns),
                domain_counts=domain_counts,
                response_mode=self.response_mode,
                overall_avg_reaction_time_ms=avg_reaction_time,
                min_reaction_time_ms=min_reaction_time,
                max_reaction_time_ms=max_reaction_time,
                active_status=self._is_active,
                top_patterns=[ReflexInfo(
                    name=p.name,
                    procedure_name=p.procedure_name,
                    threshold=p.threshold,
                    priority=p.priority,
                    stats=ReflexStats(
                        execution_count=p.execution_count,
                        success_rate=p.get_success_rate(),
                        avg_response_time_ms=p.get_avg_response_time(),
                        last_executed=p.last_executed.isoformat() if p.last_executed else None
                    )
                ) for p in top_patterns]
            )
    
    @function_tool
    async def optimize_reflexes(self) -> OptimizeReflexesResult:
        """
        Optimize reflexes through analysis and pattern refinement
        
        Returns:
            Optimization results
        """
        with trace(workflow_name="Optimize Reflexes", group_id=self.trace_group_id):
            if not self.reflex_patterns:
                return OptimizeReflexesResult(success=False, error="No reflex patterns to optimize")
            
            optimization_results = []
            
            # Sort patterns by execution count to focus on most used
            sorted_patterns = sorted(
                self.reflex_patterns.values(),
                key=lambda p: p.execution_count,
                reverse=True
            )
            
            # Create context for function tools
            tool_context = {
                "patterns": {p.name: p.pattern_data for p in sorted_patterns},
                "execution_counts": {p.name: p.execution_count for p in sorted_patterns},
                "success_rates": {p.name: p.get_success_rate() for p in sorted_patterns},
                "response_times": {p.name: p.get_avg_response_time() for p in sorted_patterns}
            }
            
            # Process patterns for optimization
            for pattern in sorted_patterns:
                if pattern.execution_count < 5:
                    continue  # Skip patterns without enough data
                
                # Calculate success rate
                success_rate = pattern.get_success_rate()
                
                try:
                    # If success rate is low, try to improve pattern
                    if success_rate < 0.7:
                        # Check if we can optimize the pattern
                        can_optimize = len(pattern.pattern_data) >= 3
                        
                        if can_optimize:
                            # Try to optimize pattern
                            old_pattern = pattern.pattern_data.copy()
                            opt_input = OptimizePatternInput(pattern=dict_to_pattern_dict(pattern.pattern_data))
                            opt_result = await optimize_pattern(
                                RunContextWrapper(tool_context),
                                opt_input
                            )
                            
                            if opt_result:
                                pattern.pattern_data = pattern_dict_to_dict(opt_result.data)
                                optimization_results.append(PatternOptimization(
                                    pattern_name=pattern.name,
                                    optimization=OptimizationInfo(
                                        status="optimized",
                                        original_success_rate=success_rate,
                                        optimization_type="pattern_refinement"
                                    )
                                ))
                                
                                logger.info(f"Optimized pattern '{pattern.name}' with success rate {success_rate:.2f}")
                        else:
                            # If pattern can't be optimized further, adjust threshold
                            old_threshold = pattern.threshold
                            if success_rate < 0.5 and pattern.threshold > 0.6:
                                pattern.threshold = max(0.6, pattern.threshold - 0.1)
                            elif success_rate > 0.9 and pattern.threshold < 0.9:
                                pattern.threshold = min(0.9, pattern.threshold + 0.05)
                            
                            if old_threshold != pattern.threshold:
                                optimization_results.append(PatternOptimization(
                                    pattern_name=pattern.name,
                                    optimization=OptimizationInfo(
                                        status="threshold_adjusted",
                                        original_success_rate=success_rate,
                                        old_threshold=old_threshold,
                                        new_threshold=pattern.threshold
                                    )
                                ))
                                
                                logger.info(f"Adjusted threshold for pattern '{pattern.name}' from {old_threshold:.2f} to {pattern.threshold:.2f}")
                    
                    # For high success rate but slow reaction time, try simplifying pattern
                    elif success_rate > 0.8 and pattern.get_avg_response_time() > 50:  # 50ms threshold
                        # Check if we can simplify the pattern
                        can_simplify = len(pattern.pattern_data) > 3
                        
                        if can_simplify:
                            old_pattern = pattern.pattern_data.copy()
                            simp_input = SimplifyPatternInput(pattern=dict_to_pattern_dict(pattern.pattern_data))
                            simp_result = await simplify_pattern(
                                RunContextWrapper(tool_context),
                                simp_input
                            )
                            
                            if simp_result and pattern_dict_to_dict(simp_result.data) != old_pattern:
                                pattern.pattern_data = pattern_dict_to_dict(simp_result.data)
                                optimization_results.append(PatternOptimization(
                                    pattern_name=pattern.name,
                                    optimization=OptimizationInfo(
                                        status="simplified",
                                        original_avg_response_time=pattern.get_avg_response_time(),
                                        optimization_type="pattern_simplification"
                                    )
                                ))
                                
                                logger.info(f"Simplified pattern '{pattern.name}' for faster reaction time")
                except Exception as e:
                    logger.error(f"Error optimizing pattern '{pattern.name}': {e}")
            
            return OptimizeReflexesResult(
                success=True,
                optimizations=optimization_results,
                patterns_examined=len(sorted_patterns),
                patterns_optimized=len(optimization_results)
            )
