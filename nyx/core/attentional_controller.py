# nyx/core/attentional_controller.py

import logging
import math
import random
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from collections import defaultdict

from agents import Agent, Runner, function_tool, RunContextWrapper, trace, ModelSettings, handoff, InputGuardrail

class AttentionalFocus(BaseModel):
    """Schema for current attentional focus"""
    target: str = Field(..., description="Target of attention (modality, concept, task, etc.)")
    strength: float = Field(..., description="Strength of attention (0.0-1.0)", ge=0.0, le=1.0)
    duration_ms: int = Field(..., description="Duration in milliseconds")
    source: str = Field(..., description="Source triggering this attention")
    timestamp: str = Field(..., description="When this focus was established")

class AttentionalControl(BaseModel):
    """Schema for attentional control signal"""
    target: str = Field(..., description="Target to attend to")
    priority: float = Field(..., description="Priority level (0.0-1.0)", ge=0.0, le=1.0)
    duration_ms: int = Field(..., description="Requested duration in milliseconds")
    source: str = Field(..., description="Source requesting attention")
    action: str = Field("focus", description="Action: focus, inhibit, maintain")

class SaliencyConfig(BaseModel):
    """Configuration for saliency calculations"""
    novelty_weight: float = Field(0.3, description="Weight for novelty")
    intensity_weight: float = Field(0.2, description="Weight for intensity")
    emotional_weight: float = Field(0.2, description="Weight for emotional relevance")
    goal_weight: float = Field(0.3, description="Weight for goal relevance")

class AttentionDecisionOutput(BaseModel):
    """Output schema for attention decisions"""
    focus_targets: List[Dict[str, Any]] = Field(..., description="Targets to focus attention on")
    inhibit_targets: List[Dict[str, Any]] = Field(..., description="Targets to inhibit attention from")
    reasoning: str = Field(..., description="Reasoning for attention decisions")
    resources_allocated: float = Field(..., description="Percentage of attentional resources allocated")

class InvalidInputGuardrailOutput(BaseModel):
    """Output schema for input validation"""
    is_valid: bool = Field(..., description="Whether the input is valid")
    reason: str = Field(..., description="Reason for invalid input")

# Define an AttentionContext for strong typing
class AttentionContext:
    """Context for the attention controller system"""
    def __init__(self, emotional_core=None):
        self.emotional_core = emotional_core
        self.trace_id = f"attention_{time.time()}"

class AttentionalController:
    """
    Controls attention across all system components by determining
    what information to prioritize based on saliency and current goals.
    """
    
    def __init__(self, max_foci: int = 3, emotional_core = None):
        self.max_foci = max_foci  # Maximum number of simultaneous attentional foci
        self.emotional_core = emotional_core  # Reference to emotional system for affective attention
        
        # Current attentional state
        self.current_foci = []  # Current active attentional foci
        self.inhibited_targets = {}  # Targets currently inhibited with expiry times
        self.attentional_history = []  # History of attentional shifts
        self.max_history = 50  # Maximum history size
        
        # Attentional resources and capacity
        self.total_attentional_capacity = 1.0  # Total available attention
        self.attentional_resources = 1.0  # Current available attentional resources
        self.resource_recovery_rate = 0.1  # Rate of resource recovery per second
        self.last_recovery_time = time.time()
        
        # Attentional control requests queue
        self.control_requests = []  # Pending requests
        
        # Salience configuration
        self.saliency_config = SaliencyConfig()
        
        # Attentional bias (can be modified by learning)
        self.attention_biases = defaultdict(float)  # target -> bias
        
        # Performance monitoring
        self.miss_count = 0  # Attention misses
        self.shift_count = 0  # Attention shifts
        
        self.logger = logging.getLogger(__name__)
        
        # Create shared context for the agents
        self.attention_context = AttentionContext(emotional_core=emotional_core)
        
        # Initialize agent system
        self._initialize_agents()
        
        # Trace ID for linking traces
        self.trace_group_id = f"attention_{time.time()}"
    
    def _initialize_agents(self):
        """Initialize all agents needed for the attention system"""
        # First create specialized agents
        self.saliency_agent = self._create_saliency_agent()
        self.focus_agent = self._create_focus_agent()
        self.inhibition_agent = self._create_inhibition_agent()
        
        # Create input validation guardrail
        self.input_validation = self._create_input_validation()
        
        # Now create the main agent that depends on the specialized agents
        self.attention_agent = self._create_attention_agent()
    
    def _create_attention_agent(self) -> Agent[AttentionContext]:
        """Create agent for attention allocation decisions"""
        return Agent[AttentionContext](
            name="Attention_Allocator",
            instructions="""
            You are the attention allocation system for Nyx.
            Your role is to determine what information deserves focus based on:
            1. Saliency scores and novelty
            2. Goal relevance and current priorities
            3. Emotional significance of stimuli
            4. Current attentional capacity and resources
            
            Prioritize information that's most relevant to current goals while
            balancing exploration of novel stimuli. Consider the emotional context
            when making attention decisions, and ensure efficient use of limited
            attentional resources.
            
            Make decisions about:
            - Which items to focus attention on
            - Which items to inhibit attention from
            - How to allocate attentional resources
            - How long to maintain focus on specific targets
            
            Your decisions should be explainable and based on the available data.
            """,
            tools=[
                function_tool(self._calculate_saliency),
                function_tool(self._focus_attention),
                function_tool(self._inhibit_attention),
                function_tool(self._maintain_attention),
                function_tool(self._calculate_attention_weight),
                function_tool(self._recover_attentional_resources),
                function_tool(self._get_current_attentional_state)
            ],
            handoffs=[
                handoff(self.saliency_agent),
                handoff(self.focus_agent),
                handoff(self.inhibition_agent)
            ],
            output_type=AttentionDecisionOutput,
            input_guardrails=[
                InputGuardrail(guardrail_function=self.input_validation)
            ],
            model="gpt-4.1-nano.1-nano",
            model_settings=ModelSettings(
                temperature=0.3,
            )
        )
    
    def _create_saliency_agent(self) -> Agent[AttentionContext]:
        """Create specialized agent for calculating saliency"""
        return Agent[AttentionContext](
            name="Saliency_Calculator",
            instructions="""
            You are specialized in calculating the saliency (attention-worthiness) of items.
            Consider:
            - Novelty: How new or unexpected is this stimulus?
            - Intensity: How strong is the signal?
            - Emotional relevance: How emotionally significant is this stimulus?
            - Goal relevance: How relevant is this to current goals or tasks?
            
            Calculate a saliency score between 0.0 (not salient) and 1.0 (extremely salient).
            """,
            tools=[
                function_tool(self._calculate_emotional_impact),
                function_tool(self._calculate_goal_relevance)
            ],
            model="gpt-4.1-nano.1-nano-mini"
        )
    
    def _create_focus_agent(self) -> Agent[AttentionContext]:
        """Create specialized agent for focus management"""
        return Agent[AttentionContext](
            name="Focus_Manager",
            instructions="""
            You are specialized in managing attentional focus.
            Your job is to:
            - Decide which items deserve focus
            - Determine the appropriate strength of attention
            - Set duration of attentional focus
            - Ensure resources are allocated efficiently
            
            Prioritize focus based on saliency, goals, and available resources.
            """,
            tools=[
                function_tool(self._focus_attention),
                function_tool(self._maintain_attention)
            ],
            model="gpt-4.1-nano.1-nano-mini"
        )
    
    def _create_inhibition_agent(self) -> Agent[AttentionContext]:
        """Create specialized agent for inhibition management"""
        return Agent[AttentionContext](
            name="Inhibition_Manager",
            instructions="""
            You are specialized in managing attentional inhibition.
            Your job is to:
            - Identify targets that should be inhibited from attention
            - Determine inhibition duration
            - Release inhibition when appropriate
            
            Inhibit targets that are distracting, irrelevant, or have been processed sufficiently.
            """,
            tools=[
                function_tool(self._inhibit_attention)
            ],
            model="gpt-4.1-nano.1-nano-mini"
        )
    
    async def _input_validation(self, 
                              ctx: RunContextWrapper[AttentionContext], 
                              agent: Agent[AttentionContext], 
                              input_data: str | List[Any]) -> dict:
        """Validate input for the attention system"""
        try:
            # Parse the input
            if isinstance(input_data, str):
                data = json.loads(input_data)
            else:
                data = input_data
                
            # Check for required fields
            if "salient_items" not in data:
                return {
                    "is_valid": False,
                    "reason": "Input must contain 'salient_items' field"
                }
                
            # Check that salient_items is a list
            if not isinstance(data["salient_items"], list):
                return {
                    "is_valid": False,
                    "reason": "salient_items must be a list"
                }
                
            # Input is valid
            return {
                "is_valid": True,
                "reason": ""
            }
        except Exception as e:
            return {
                "is_valid": False,
                "reason": f"Invalid input: {str(e)}"
            }

    def _create_input_validation(self) -> InputGuardrail:
        """Create input validation guardrail"""
        return InputGuardrail(guardrail_function=self._input_validation)
    
    async def update_attention(self, 
                             salient_items: List[Dict[str, Any]] = None,
                             control_signals: List[AttentionalControl] = None) -> List[AttentionalFocus]:
        """
        Update attentional focus based on salient items and control signals
        
        Args:
            salient_items: Items detected as salient with their properties
            control_signals: Explicit control signals requesting attention
            
        Returns:
            Current attentional foci after update
        """
        with trace(workflow_name="Attention_Update", group_id=self.trace_group_id):
            # 1. Update attentional resources
            await self._recover_attentional_resources(RunContextWrapper(context=self.attention_context))
            
            # 2. Process any control signals (top-down attention)
            # This is mandatory processing, so we handle these before agent decisions
            if control_signals:
                for signal in control_signals:
                    await self._process_control_signal(signal)
            
            # 3. Add any pending control requests
            for request in self.control_requests:
                if request.action == "focus":
                    await self._focus_attention(
                        RunContextWrapper(context=self.attention_context), 
                        request.target, 
                        request.priority, 
                        request.duration_ms, 
                        request.source
                    )
                elif request.action == "inhibit":
                    await self._inhibit_attention(
                        RunContextWrapper(context=self.attention_context), 
                        request.target, 
                        request.duration_ms
                    )
                elif request.action == "maintain":
                    await self._maintain_attention(
                        RunContextWrapper(context=self.attention_context), 
                        request.target, 
                        request.duration_ms
                    )
            
            # Clear processed requests
            self.control_requests = []
            
            # 4. Run the attention agent to process salient items and make decisions
            if salient_items:
                # Get current attentional state for agent context
                current_state = await self._get_current_attentional_state(
                    RunContextWrapper(context=self.attention_context)
                )
                
                # Run the agent to make attention decisions
                result = await Runner.run(
                    self.attention_agent,
                    {
                        "salient_items": salient_items,
                        "current_state": current_state,
                        "available_resources": self.attentional_resources,
                        "max_foci": self.max_foci
                    },
                    context=self.attention_context
                )
                
                # Process agent decisions
                decisions = result.final_output
                
                # Apply focus decisions
                for focus_target in decisions.focus_targets:
                    await self._focus_attention(
                        RunContextWrapper(context=self.attention_context),
                        focus_target["target"],
                        focus_target.get("strength", 0.7),
                        focus_target.get("duration_ms", 2000),
                        focus_target.get("source", "attention_agent")
                    )
                
                # Apply inhibit decisions
                for inhibit_target in decisions.inhibit_targets:
                    await self._inhibit_attention(
                        RunContextWrapper(context=self.attention_context),
                        inhibit_target["target"],
                        inhibit_target.get("duration_ms", 5000)
                    )
                
                # Log reasoning for audit/debugging
                self.logger.debug(f"Attention allocation reasoning: {decisions.reasoning}")
            
            # 5. Update and expire old foci
            await self._expire_old_foci()
            
            # 6. Update attentional history
            self._update_history()
            
            return self.current_foci

    @staticmethod
    @function_tool
    async def _recover_attentional_resources(ctx: RunContextWrapper[AttentionContext]) -> Dict[str, float]:
        """
        Recover attentional resources over time
        
        Returns:
            Updated attentional resources
        """
        current_time = time.time()
        elapsed_seconds = current_time - self.last_recovery_time
        
        if elapsed_seconds > 0:
            # Calculate recovery
            recovery_amount = self.resource_recovery_rate * elapsed_seconds
            
            # Update resources (capped at max capacity)
            self.attentional_resources = min(self.total_attentional_capacity, 
                                          self.attentional_resources + recovery_amount)
            
            # Update last recovery time
            self.last_recovery_time = current_time
        
        return {
            "current_resources": self.attentional_resources,
            "total_capacity": self.total_attentional_capacity,
            "recovery_rate": self.resource_recovery_rate
        }
    
    async def _process_control_signal(self, signal: AttentionalControl) -> bool:
        """Process an attentional control signal"""
        # Add to request queue
        self.control_requests.append(signal)
        return True

    @staticmethod
    @function_tool
    async def _focus_attention(ctx: RunContextWrapper[AttentionContext],
                            target: str, 
                            strength: float, 
                            duration_ms: int,
                            source: str) -> Dict[str, Any]:
        """
        Focus attention on a specific target
        
        Args:
            target: Target to attend to
            strength: Strength of attention (0.0-1.0)
            duration_ms: Duration in milliseconds
            source: Source requesting attention
            
        Returns:
            Result of focus operation
        """
        # Check if already focused
        for focus in self.current_foci:
            if focus.target == target:
                # Update existing focus
                focus.strength = max(focus.strength, strength)
                focus.duration_ms = max(focus.duration_ms, duration_ms)
                focus.source = f"{focus.source}, {source}"
                return {
                    "success": True,
                    "message": "Updated existing focus",
                    "target": target,
                    "strength": focus.strength
                }
        
        # Check if we have capacity for new focus
        if len(self.current_foci) >= self.max_foci:
            # Remove weakest focus if needed
            self.current_foci.sort(key=lambda x: x.strength)
            if self.current_foci[0].strength < strength:
                removed = self.current_foci.pop(0)
                self.shift_count += 1
                
                # Create new focus
                new_focus = AttentionalFocus(
                    target=target,
                    strength=strength,
                    duration_ms=duration_ms,
                    source=source,
                    timestamp=str(time.time())
                )
                
                # Add to current foci
                self.current_foci.append(new_focus)
                
                # Consume attentional resources
                self.attentional_resources -= strength * 0.2  # Scale resource consumption
                self.attentional_resources = max(0, self.attentional_resources)  # Ensure non-negative
                
                return {
                    "success": True,
                    "message": "Replaced weaker focus",
                    "replaced": removed.target,
                    "target": target,
                    "strength": strength
                }
            else:
                # Can't focus on this target - attention miss
                self.miss_count += 1
                return {
                    "success": False,
                    "message": "Insufficient capacity, target too weak compared to current foci",
                    "target": target,
                    "strength": strength
                }
        else:
            # Create new focus
            new_focus = AttentionalFocus(
                target=target,
                strength=strength,
                duration_ms=duration_ms,
                source=source,
                timestamp=str(time.time())
            )
            
            # Add to current foci
            self.current_foci.append(new_focus)
            
            # Consume attentional resources
            self.attentional_resources -= strength * 0.2  # Scale resource consumption
            self.attentional_resources = max(0, self.attentional_resources)  # Ensure non-negative
            
            return {
                "success": True,
                "message": "Created new focus",
                "target": target,
                "strength": strength
            }

    @staticmethod
    @function_tool
    async def _inhibit_attention(ctx: RunContextWrapper[AttentionContext], 
                              target: str, 
                              duration_ms: int) -> Dict[str, Any]:
        """
        Inhibit attention to a specific target for a duration
        
        Args:
            target: Target to inhibit
            duration_ms: Duration of inhibition in milliseconds
            
        Returns:
            Result of inhibit operation
        """
        # Remove any current focus on this target
        removed = False
        for focus in list(self.current_foci):
            if focus.target == target:
                self.current_foci.remove(focus)
                removed = True
        
        # Add to inhibited targets
        expiry_time = time.time() + (duration_ms / 1000)
        self.inhibited_targets[target] = expiry_time
        
        return {
            "success": True, 
            "target": target,
            "focus_removed": removed,
            "inhibited_until": expiry_time
        }

    @staticmethod
    @function_tool
    async def _maintain_attention(ctx: RunContextWrapper[AttentionContext], 
                               target: str, 
                               duration_ms: int) -> Dict[str, Any]:
        """
        Maintain attention on a currently focused target
        
        Args:
            target: Target to maintain focus on
            duration_ms: Additional duration in milliseconds
            
        Returns:
            Result of maintain operation
        """
        maintained = False
        for focus in self.current_foci:
            if focus.target == target:
                # Extend duration
                focus.duration_ms += duration_ms
                maintained = True
                
                return {
                    "success": True,
                    "target": target,
                    "new_duration_ms": focus.duration_ms
                }
        
        if not maintained:
            return {
                "success": False,
                "message": "Target not currently in focus",
                "target": target
            }
    
    async def _expire_old_foci(self):
        """Remove expired attentional foci and inhibitions"""
        current_time = time.time()
        
        # Expire foci
        active_foci = []
        for focus in self.current_foci:
            # Check if focus has expired
            focus_end_time = float(focus.timestamp) + (focus.duration_ms / 1000)
            
            if current_time < focus_end_time:
                # Still active
                active_foci.append(focus)
            else:
                # Expired - free up resources
                self.attentional_resources += focus.strength * 0.1  # Partial resource recovery
        
        # Update current foci
        self.current_foci = active_foci
        
        # Expire inhibitions
        to_remove = []
        for target, expiry_time in self.inhibited_targets.items():
            if current_time > expiry_time:
                to_remove.append(target)
                
        # Remove expired inhibitions
        for target in to_remove:
            del self.inhibited_targets[target]

    @staticmethod
    @function_tool
    async def _calculate_saliency(ctx: RunContextWrapper[AttentionContext], 
                               item: Dict[str, Any]) -> float:
        """
        Calculate saliency score for an item
        
        Args:
            item: Item to evaluate saliency for
            
        Returns:
            Saliency score (0.0-1.0)
        """
        # Extract features
        novelty = item.get("novelty", 0.5)
        intensity = item.get("intensity", 0.5)
        emotional_impact = item.get("emotional_impact", 0.5)
        goal_relevance = item.get("goal_relevance", 0.5)
        
        # Get attentional bias for this target
        target = item.get("target", item.get("id", "unknown"))
        bias = self.attention_biases[target]
        
        # Calculate weighted saliency
        config = self.saliency_config
        saliency = (
            novelty * config.novelty_weight +
            intensity * config.intensity_weight +
            emotional_impact * config.emotional_weight +
            goal_relevance * config.goal_weight +
            bias  # Add bias directly
        )
        
        # Check emotional core for additional affective influence
        if ctx.context.emotional_core:
            try:
                emotional_state = ctx.context.emotional_core.get_emotional_state()
                arousal = ctx.context.emotional_core.get_emotional_arousal()
                
                # High arousal amplifies saliency
                if arousal > 0.6:
                    saliency *= 1.2
                elif arousal < 0.3:
                    saliency *= 0.8
                    
                # Check valence influence if strong emotion is present
                strongest_emotion, strength = ctx.context.emotional_core.get_dominant_emotion()
                if strength > 0.6:
                    # Check if emotion matches item
                    if "emotion" in item and item["emotion"] == strongest_emotion:
                        saliency *= 1.3  # Boost for emotional congruence
            except Exception as e:
                self.logger.error(f"Error applying emotional influence to saliency: {e}")
        
        # Normalize saliency to 0-1 range
        return max(0.0, min(1.0, saliency))
    
    # New helper functions to support specialized agents

    @staticmethod
    @function_tool
    async def _calculate_emotional_impact(ctx: RunContextWrapper[AttentionContext], 
                                      item: Dict[str, Any]) -> float:
        """
        Calculate emotional impact of an item
        
        Args:
            item: Item to evaluate
            
        Returns:
            Emotional impact score (0.0-1.0)
        """
        # Default emotional impact
        impact = item.get("emotional_impact", 0.5)
        
        # If emotional core is available, do more sophisticated calculation
        if ctx.context.emotional_core:
            try:
                # Get current emotional state
                emotional_state = ctx.context.emotional_core.get_emotional_state()
                
                # Check for emotion matching
                if "emotion" in item:
                    item_emotion = item["emotion"]
                    for emotion, level in emotional_state.items():
                        if emotion.lower() == item_emotion.lower():
                            # Boost impact for matching emotions
                            impact = max(impact, level * 1.2)
                
                # Adjust based on overall arousal
                arousal = ctx.context.emotional_core.get_emotional_arousal()
                impact *= 0.7 + (arousal * 0.6)  # Scale impact with arousal
                
            except Exception as e:
                self.logger.error(f"Error calculating emotional impact: {e}")
        
        # Ensure result is in valid range
        return max(0.0, min(1.0, impact))

    @staticmethod
    @function_tool
    async def _calculate_goal_relevance(ctx: RunContextWrapper[AttentionContext], 
                                    item: Dict[str, Any]) -> float:
        """
        Calculate goal relevance of an item
        
        Args:
            item: Item to evaluate
            
        Returns:
            Goal relevance score (0.0-1.0)
        """
        # Default relevance from item
        relevance = item.get("goal_relevance", 0.5)
        
        # If item relates to a current focus, increase relevance
        target = item.get("target", item.get("id", "unknown"))
        for focus in self.current_foci:
            if focus.target == target or target.startswith(focus.target):
                # Item relates to current focus, increase relevance
                relevance = max(relevance, focus.strength * 1.1)
        
        # Ensure result is in valid range
        return max(0.0, min(1.0, relevance))
    
    def _update_history(self):
        """Update attentional history with current focus"""
        # Add current foci to history
        for focus in self.current_foci:
            self.attentional_history.append({
                "target": focus.target,
                "strength": focus.strength,
                "source": focus.source,
                "timestamp": focus.timestamp
            })
            
        # Trim history
        if len(self.attentional_history) > self.max_history:
            self.attentional_history = self.attentional_history[-self.max_history:]
    
    async def request_attention(self, control: AttentionalControl) -> bool:
        """Request attention focus, inhibition, or maintenance"""
        self.control_requests.append(control)
        return True

    @staticmethod
    @function_tool
    async def _calculate_attention_weight(ctx: RunContextWrapper[AttentionContext],
                                     item: Any, 
                                     modality: str = None) -> float:
        """
        Calculate attention weight for an item based on current attentional focus
        
        Args:
            item: Item to calculate attention for
            modality: Optional modality of the item
            
        Returns:
            Attention weight (0.0-1.0)
        """
        # Get target identifier
        if hasattr(item, "id"):
            target = item.id
        elif hasattr(item, "target"):
            target = item.target
        elif isinstance(item, dict) and "id" in item:
            target = item["id"]
        elif isinstance(item, dict) and "target" in item:
            target = item["target"]
        elif modality:
            target = modality
        else:
            target = "unknown"
        
        # Check if target is currently inhibited
        if target in self.inhibited_targets:
            return 0.1  # Minimal attention to inhibited targets
        
        # Check if target is currently in focus
        for focus in self.current_foci:
            if focus.target == target or (modality and focus.target == modality):
                return focus.strength  # Full attention weight
        
        # If not focused but modality is in focus, give partial attention
        if modality:
            for focus in self.current_foci:
                if focus.target == modality:
                    return focus.strength * 0.7  # Partial attention weight
        
        # Default moderate attention if not inhibited and resources available
        if self.attentional_resources > 0.5:
            return 0.5
        else:
            return 0.3  # Reduced attention when resources are low
    
    async def update_attention_bias(self, target: str, adjustment: float):
        """Update attention bias for a target based on learning"""
        current_bias = self.attention_biases[target]
        
        # Apply adjustment with constraints to keep in reasonable range
        new_bias = current_bias + adjustment
        new_bias = max(-0.3, min(0.3, new_bias))  # Limit bias to -0.3 to 0.3
        
        self.attention_biases[target] = new_bias

    @staticmethod
    @function_tool
    async def _get_current_attentional_state(ctx: RunContextWrapper[AttentionContext]) -> Dict[str, Any]:
        """
        Get the current attentional state
        
        Returns:
            Current attentional state information
        """
        current_foci = []
        for focus in self.current_foci:
            current_foci.append({
                "target": focus.target,
                "strength": focus.strength,
                "duration_ms": focus.duration_ms,
                "source": focus.source,
                "timestamp": focus.timestamp
            })
        
        inhibited = []
        for target, expiry_time in self.inhibited_targets.items():
            inhibited.append({
                "target": target,
                "expires_at": expiry_time
            })
        
        return {
            "current_foci": current_foci,
            "inhibited_targets": inhibited,
            "attentional_resources": self.attentional_resources,
            "total_capacity": self.total_attentional_capacity,
            "shift_count": self.shift_count,
            "miss_count": self.miss_count
        }
    
    async def get_attention_statistics(self) -> Dict[str, Any]:
        """Get statistics about attentional system performance"""
        return {
            "current_foci_count": len(self.current_foci),
            "attentional_resources": self.attentional_resources,
            "inhibited_targets_count": len(self.inhibited_targets),
            "attention_shifts": self.shift_count,
            "attention_misses": self.miss_count,
            "most_focused": self._get_most_focused_targets(5),
            "miss_rate": self.miss_count / max(1, self.miss_count + self.shift_count)
        }
    
    def _get_most_focused_targets(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the most frequently focused targets"""
        target_counts = defaultdict(int)
        
        for entry in self.attentional_history:
            target_counts[entry["target"]] += 1
            
        # Sort by count
        sorted_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N
        return [{"target": t, "focus_count": c} for t, c in sorted_targets[:limit]]
