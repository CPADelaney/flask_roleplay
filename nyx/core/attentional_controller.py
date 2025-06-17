# nyx/core/attentional_controller.py

import logging
import math
import random
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from collections import defaultdict

from agents import Agent, Runner, function_tool, RunContextWrapper, trace, ModelSettings, handoff, InputGuardrail, GuardrailFunctionOutput

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

class AttentionalStateOutput(BaseModel):
    """
    Strict DTO returned by `_get_current_attentional_state`.
    Making this a model (and setting extra='forbid') keeps the Agents SDK happy.
    """
    current_foci:        List[AttentionalFocus]
    inhibited_targets:   List[Dict[str, Any]]
    attentional_resources: float
    total_capacity:        float
    shift_count:           int
    miss_count:            int

    model_config = {"extra": "forbid"}   # 👈 turn off additionalProperties

# Define an AttentionContext for strong typing
class AttentionContext:
    """
    Context object shared across attention‑related agents/tools.
    """
    def __init__(self,
                 controller: "AttentionalController",
                 emotional_core=None):
        self.controller = controller
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
        self.attention_context = AttentionContext(
            controller=self,
            emotional_core=emotional_core
        )
        
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
                self._calculate_saliency,
                self._focus_attention,
                self._inhibit_attention,
                self._maintain_attention,
                self._calculate_attention_weight,
                self._recover_attentional_resources,
                self._get_current_attentional_state
            ],
            handoffs=[
                handoff(self.saliency_agent),
                handoff(self.focus_agent),
                handoff(self.inhibition_agent)
            ],
            output_type=AttentionDecisionOutput,
            input_guardrails=[
                self.input_validation  # This is already an InputGuardrail object
            ],
            model="gpt-4.1-nano",
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
                self._calculate_emotional_impact,
                self._calculate_goal_relevance
            ],
            model="gpt-4.1-nano"
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
                self._focus_attention,
                self._maintain_attention
            ],
            model="gpt-4.1-nano"
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
                self._inhibit_attention
            ],
            model="gpt-4.1-nano"
        )
    
    async def _input_validation(self, 
                              ctx: RunContextWrapper[AttentionContext], 
                              agent: Agent[AttentionContext], 
                              input_data: str | List[Any]) -> GuardrailFunctionOutput:
        """Validate input for the attention system"""
        try:
            # Parse the input
            if isinstance(input_data, str):
                data = json.loads(input_data)
            else:
                data = input_data
                
            # Check for required fields
            if "salient_items" not in data:
                return GuardrailFunctionOutput(
                    output_info={
                        "is_valid": False,
                        "reason": "Input must contain 'salient_items' field"
                    },
                    tripwire_triggered=True  # Trigger tripwire for invalid input
                )
                
            # Check that salient_items is a list
            if not isinstance(data["salient_items"], list):
                return GuardrailFunctionOutput(
                    output_info={
                        "is_valid": False,
                        "reason": "salient_items must be a list"
                    },
                    tripwire_triggered=True  # Trigger tripwire for invalid input
                )
                
            # Input is valid
            return GuardrailFunctionOutput(
                output_info={
                    "is_valid": True,
                    "reason": ""
                },
                tripwire_triggered=False  # Don't trigger tripwire for valid input
            )
        except Exception as e:
            return GuardrailFunctionOutput(
                output_info={
                    "is_valid": False,
                    "reason": f"Invalid input: {str(e)}"
                },
                tripwire_triggered=True  # Trigger tripwire for exceptions
            )

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
    async def _recover_attentional_resources(
            ctx: RunContextWrapper[AttentionContext]
    ) -> dict:
        ctl = ctx.context.controller              # ← grab the controller
        now = time.time()
        elapsed = now - ctl.last_recovery_time

        if elapsed > 0:
            ctl.attentional_resources = min(
                ctl.total_attentional_capacity,
                ctl.attentional_resources + elapsed * ctl.resource_recovery_rate
            )
            ctl.last_recovery_time = now

        return {
            "current_resources": ctl.attentional_resources,
            "total_capacity": ctl.total_attentional_capacity,
            "recovery_rate": ctl.resource_recovery_rate,
        }
    
    async def _process_control_signal(self, signal: AttentionalControl) -> bool:
        """Process an attentional control signal"""
        # Add to request queue
        self.control_requests.append(signal)
        return True


    @staticmethod
    @function_tool
    async def _focus_attention(
        ctx: "RunContextWrapper[AttentionContext]",
        target: str,
        strength: float,
        duration_ms: int,
        source: str,
    ) -> dict:
        """Create or update an attentional focus on *target*."""
        ctl = ctx.context.controller

        # If already focused, just update/extend.
        for focus in ctl.current_foci:
            if focus.target == target:
                focus.strength = max(focus.strength, strength)
                focus.duration_ms = max(focus.duration_ms, duration_ms)
                focus.source = f"{focus.source}, {source}"
                return {
                    "success": True,
                    "message": "Updated existing focus",
                    "target": target,
                    "strength": focus.strength,
                }

        # Need a new focus – check capacity.
        if len(ctl.current_foci) >= ctl.max_foci:
            ctl.current_foci.sort(key=lambda x: x.strength)
            if ctl.current_foci[0].strength < strength:
                removed = ctl.current_foci.pop(0)
                ctl.shift_count += 1
            else:
                ctl.miss_count += 1
                return {
                    "success": False,
                    "message": "Insufficient capacity – target weaker than current foci",
                    "target": target,
                    "strength": strength,
                }
        # Create and insert new focus
        new_focus = AttentionalFocus(
            target=target,
            strength=strength,
            duration_ms=duration_ms,
            source=source,
            timestamp=str(time.time()),
        )
        ctl.current_foci.append(new_focus)
        ctl.attentional_resources = max(0, ctl.attentional_resources - strength * 0.2)
        return {
            "success": True,
            "message": "Created new focus",
            "target": target,
            "strength": strength,
        }


    @staticmethod
    @function_tool
    async def _inhibit_attention(
        ctx: "RunContextWrapper[AttentionContext]",
        target: str,
        duration_ms: int,
    ) -> dict:
        """Temporarily inhibit attention on *target*."""
        ctl = ctx.context.controller

        # Remove existing focus if present.
        removed = False
        for focus in list(ctl.current_foci):
            if focus.target == target:
                ctl.current_foci.remove(focus)
                removed = True

        ctl.inhibited_targets[target] = time.time() + duration_ms / 1000
        return {
            "success": True,
            "target": target,
            "focus_removed": removed,
            "inhibited_until": ctl.inhibited_targets[target],
        }


    @staticmethod
    @function_tool
    async def _maintain_attention(
        ctx: "RunContextWrapper[AttentionContext]",
        target: str,
        duration_ms: int,
    ) -> dict:
        """Extend the lifetime of an existing focus."""
        ctl = ctx.context.controller
        for focus in ctl.current_foci:
            if focus.target == target:
                focus.duration_ms += duration_ms
                return {
                    "success": True,
                    "target": target,
                    "new_duration_ms": focus.duration_ms,
                }
        return {
            "success": False,
            "message": "Target not currently in focus",
            "target": target,
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
                self.attentional_resources = min(
                    self.total_attentional_capacity,
                    self.attentional_resources + focus.strength * 0.1
                )
        
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
    async def _calculate_saliency(
        ctx: RunContextWrapper[AttentionContext],
        item: Any,          # <- was dict
    ) -> float:
        ctl = ctx.context.controller
        novelty = item.get("novelty", 0.5)
        intensity = item.get("intensity", 0.5)
        emotional_impact = item.get("emotional_impact", 0.5)
        goal_relevance = item.get("goal_relevance", 0.5)
        target = item.get("target", item.get("id", "unknown"))
        bias = ctl.attention_biases[target]
        cfg = ctl.saliency_config
        saliency = (
            novelty * cfg.novelty_weight
            + intensity * cfg.intensity_weight
            + emotional_impact * cfg.emotional_weight
            + goal_relevance * cfg.goal_weight
            + bias
        )
        if ctx.context.emotional_core:
            try:
                arousal = ctx.context.emotional_core.get_emotional_arousal()
                saliency *= 1.2 if arousal > 0.6 else 0.8 if arousal < 0.3 else 1.0
                strongest, strength = ctx.context.emotional_core.get_dominant_emotion()
                if strength > 0.6 and item.get("emotion") == strongest:
                    saliency *= 1.3
            except Exception as e:
                ctl.logger.error(f"Emotional influence error: {e}")
        return max(0.0, min(1.0, saliency))

    
    # New helper functions to support specialized agents


    @staticmethod
    @function_tool
    async def _calculate_emotional_impact(
        ctx: RunContextWrapper[AttentionContext],
        item: Any,
    ) -> float:
        ctl = ctx.context.controller
        impact = item.get("emotional_impact", 0.5)
        if ctx.context.emotional_core:
            try:
                state = ctx.context.emotional_core.get_emotional_state()
                if (em := item.get("emotion")):
                    impact = max(impact, state.get(em.lower(), 0) * 1.2)
                arousal = ctx.context.emotional_core.get_emotional_arousal()
                impact *= 0.7 + arousal * 0.6
            except Exception as e:
                ctl.logger.error(f"Emotional impact calculation error: {e}")
        return max(0.0, min(1.0, impact))



    @staticmethod
    @function_tool
    async def _calculate_goal_relevance(
        ctx: RunContextWrapper[AttentionContext],
        item: Any,
    ) -> float:
        ctl = ctx.context.controller
        relevance = item.get("goal_relevance", 0.5)
        target = item.get("target", item.get("id", "unknown"))
        for focus in ctl.current_foci:
            if focus.target == target or target.startswith(focus.target):
                relevance = max(relevance, focus.strength * 1.1)
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
    async def _calculate_attention_weight(
        ctx: RunContextWrapper[AttentionContext],
        item: Any,
        modality: str | None = None,
    ) -> float:
        ctl = ctx.context.controller
        # Determine target identifier
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

        # Inhibited?
        if target in ctl.inhibited_targets:
            return 0.1
        # Focused?
        for focus in ctl.current_foci:
            if focus.target == target or (modality and focus.target == modality):
                return focus.strength
        # Modality partially focused?
        if modality:
            for focus in ctl.current_foci:
                if focus.target == modality:
                    return focus.strength * 0.7
        # Default weight depends on resources
        return 0.5 if ctl.attentional_resources > 0.5 else 0.3

    
    async def update_attention_bias(self, target: str, adjustment: float):
        """Update attention bias for a target based on learning"""
        current_bias = self.attention_biases[target]
        
        # Apply adjustment with constraints to keep in reasonable range
        new_bias = current_bias + adjustment
        new_bias = max(-0.3, min(0.3, new_bias))  # Limit bias to -0.3 to 0.3
        
        self.attention_biases[target] = new_bias

    
    @staticmethod
    @function_tool
    async def _get_current_attentional_state(
        ctx: "RunContextWrapper[AttentionContext]",
    ) -> AttentionalStateOutput:
        """
        Return a strict snapshot of the controller’s attentional state.
        """
        ctl = ctx.context.controller
    
        # Convert `AttentionalFocus` objects into plain dicts so they survive JSON
        foci_payload = [
            AttentionalFocus(
                target=f.target,
                strength=f.strength,
                duration_ms=f.duration_ms,
                source=f.source,
                timestamp=f.timestamp,
            )
            for f in ctl.current_foci
        ]
    
        inhibited_payload = [
            {"target": t, "expires_at": exp}
            for t, exp in ctl.inhibited_targets.items()
        ]
    
        return AttentionalStateOutput(
            current_foci=foci_payload,
            inhibited_targets=inhibited_payload,
            attentional_resources=ctl.attentional_resources,
            total_capacity=ctl.total_attentional_capacity,
            shift_count=ctl.shift_count,
            miss_count=ctl.miss_count,
        )

    
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
