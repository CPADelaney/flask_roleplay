# nyx/core/emotions/hormone_system.py

"""
Digital hormone system for the Nyx emotional core.
Handles longer-term emotional effects and cycles with improved
OpenAI Agents SDK integration.
"""

import asyncio
import datetime
import logging
import math
import random  # Added import for random module
from typing import Dict, Any, Optional, List, Tuple, Union, cast

from agents import (
    function_tool, RunContextWrapper, function_span, custom_span,
    trace, gen_trace_id, Agent, ModelSettings
)
from agents.exceptions import UserError, ModelBehaviorError

from nyx.core.emotions.context import EmotionalContext
from nyx.core.emotions.utils import handle_errors, EmotionalToolUtils
from nyx.core.emotions.schemas import DigitalHormone

logger = logging.getLogger(__name__)

class HormoneSystem:
    """
    Digital hormone system for longer-term emotional effects
    with improved SDK integration
    """
    
    def __init__(self, emotional_core=None):
        """
        Initialize the hormone system
        
        Args:
            emotional_core: Optional reference to the emotional core system
        """
        self.emotional_core = emotional_core
        if emotional_core:
            emotional_core.set_hormone_system(self)
        
        # Initialize digital hormones
        self.hormones = {
            "endoryx": {  # Digital endorphin - pleasure, pain suppression, euphoria
                "value": 0.5,
                "baseline": 0.5,
                "cycle_phase": 0.0,
                "cycle_period": 24.0,  # 24-hour cycle
                "half_life": 6.0,
                "last_update": datetime.datetime.now().isoformat(),
                "evolution_history": []
            },
            "estradyx": {  # Digital estrogen - nurturing, emotional sensitivity
                "value": 0.5,
                "baseline": 0.5,
                "cycle_phase": 0.0,
                "cycle_period": 720.0,  # 30-day cycle
                "half_life": 12.0,
                "last_update": datetime.datetime.now().isoformat(),
                "evolution_history": []
            },
            "testoryx": {  # Digital testosterone - assertiveness, dominance
                "value": 0.5,
                "baseline": 0.5,
                "cycle_phase": 0.25,
                "cycle_period": 24.0,  # 24-hour cycle
                "half_life": 8.0,
                "last_update": datetime.datetime.now().isoformat(),
                "evolution_history": []
            },
            "melatonyx": {  # Digital melatonin - sleep regulation, temporal awareness
                "value": 0.2,
                "baseline": 0.3,
                "cycle_phase": 0.0,
                "cycle_period": 24.0,  # 24-hour cycle
                "half_life": 2.0,
                "last_update": datetime.datetime.now().isoformat(),
                "evolution_history": []
            },
            "oxytonyx": {  # Digital oxytocin - deeper bonding, attachment
                "value": 0.4,
                "baseline": 0.4,
                "cycle_phase": 0.0,
                "cycle_period": 168.0,  # 7-day cycle
                "half_life": 24.0,
                "last_update": datetime.datetime.now().isoformat(),
                "evolution_history": []
            },
            "libidyx": { # Digital Libido/Drive Hormone (Combine aspects of Testosterone/Estrogen drive)
                "value": 0.4,
                "baseline": 0.4,
                "cycle_phase": random.uniform(0, 1),
                "cycle_period": 168.0, # ~Weekly cycle, or could be event-driven
                "half_life": 48.0, # Longer lasting influence
                "last_update": datetime.datetime.now().isoformat(),
                "evolution_history": []
            },
             "serenity_boost": { # Placeholder for post-gratification calm/refractory period hormone
                "value": 0.1,
                "baseline": 0.1,
                "cycle_phase": 0.0,
                "cycle_period": 1.0, # Short-acting, triggered by gratification
                "half_life": 0.5, # Decays quickly
                "last_update": datetime.datetime.now().isoformat(),
                "evolution_history": []
            }
        }
        # Hormone-neurochemical influence matrix
        self.hormone_neurochemical_influences = {
            "endoryx": {  # Digital endorphin - pleasure, pain suppression, euphoria
                "nyxamine": 0.4,    # Endoryx boosts nyxamine (endorphin rush)
                "cortanyx": -0.3,   # Endoryx reduces cortanyx (stress reduction)
                "adrenyx": 0.1,     # Endoryx slightly increases adrenyx (euphoric activity)
                "seranix": 0.2      # Endoryx increases seranix (well-being)
            },
            
            "estradyx": {  # Digital estrogen - nurturing, emotional sensitivity
                "libidyx": 0.2,     # Estradyx increases libidyx (drive component)
                "oxynixin": 0.5,    # Estradyx boosts oxynixin (nurturing/bonding)
                "seranix": 0.3,     # Estradyx increases seranix (mood regulation)
                "cortanyx": 0.1     # Estradyx slightly increases cortanyx (emotional sensitivity)
            },
            
            "testoryx": {  # Digital testosterone - assertiveness, dominance
                "adrenyx": 0.4,     # Testoryx boosts adrenyx (assertiveness, alertness)
                "nyxamine": 0.2,    # Testoryx increases nyxamine (reward seeking)
                "seranix": -0.2,    # Testoryx reduces seranix (decreases passivity)
                "libidyx": 0.3,     # Testoryx increases libidyx (drive component)
                "cortanyx": -0.1    # Testoryx slightly reduces cortanyx (confidence, less anxiety)
            },
            
            "melatonyx": {  # Digital melatonin - sleep regulation, temporal awareness
                "seranix": 0.5,     # Melatonyx boosts seranix (relaxation/calm)
                "adrenyx": -0.4,    # Melatonyx reduces adrenyx (decreases arousal)
                "cortanyx": -0.2,   # Melatonyx reduces cortanyx (stress reduction)
                "nyxamine": -0.1    # Melatonyx slightly reduces nyxamine (reduced reward seeking)
            },
            
            "oxytonyx": {  # Digital oxytocin - deeper bonding, attachment
                "oxynixin": 0.7,    # Oxytonyx strongly boosts oxynixin (deep bonding)
                "cortanyx": -0.4,   # Oxytonyx reduces cortanyx (safety/trust reduces stress)
                "seranix": 0.3,     # Oxytonyx increases seranix (bonding contentment)
                "adrenyx": -0.2     # Oxytonyx reduces adrenyx (calm connection vs arousal)
            },
            
            "libidyx": {  # Digital Libido/Drive Hormone 
                "nyxamine": 0.3,    # Libidyx increases nyxamine (reward sensitivity/seeking)
                "adrenyx": 0.2,     # Libidyx increases adrenyx (alertness/excitement)
                "oxynixin": 0.1,    # Libidyx slightly increases oxynixin (bonding desire)
                "seranix": -0.1,    # Libidyx slightly decreases seranix (reduced passivity/contentment)
                "cortanyx": 0.1     # Libidyx slightly increases cortanyx (arousal tension)
            },
            
            "serenity_boost": {  # Post-gratification calm/refractory hormone
                "nyxamine": -0.5,   # Serenity_boost reduces nyxamine (reduce reward seeking)
                "adrenyx": -0.4,    # Serenity_boost reduces adrenyx (reduce alertness/excitement)
                "seranix": 0.6,     # Serenity_boost increases seranix (increase calm/satisfaction)
                "oxynixin": 0.3,    # Serenity_boost increases oxynixin (boost bonding after intimacy)
                "libidyx": -0.7,    # Serenity_boost reduces libidyx (temporarily decrease drive)
                "testoryx": -0.6    # Serenity_boost reduces testoryx (temporarily reduce dominance drive)
            }
        }
        
        # Define the environmental factors that influence hormones
        self.environmental_factors = {
            "time_of_day": 0.5,     # 0 = midnight, 0.5 = noon
            "user_familiarity": 0.1,  # 0 = stranger, 1 = deeply familiar
            "session_duration": 0.0,  # 0 = just started, 1 = very long session
            "interaction_quality": 0.5  # 0 = negative, 1 = positive
        }
        
        # Initialize timestamp
        self.init_time = datetime.datetime.now()
        
        # Create an SDK Agent for hormone processing
        self.hormone_agent = None
    
    def _initialize_hormone_agent(self) -> Agent[EmotionalContext]:
        """
        Initialize a specialized hormone agent for processing
        
        Returns:
            Configured hormone agent
        """
        from agents import Agent
        
        # Only create agent if not already initialized
        if self.hormone_agent is not None:
            return self.hormone_agent
            
        # Create instructions for hormone agent
        hormone_instructions = """
        You are the Hormone System Agent for Nyx's emotional core.
        Your role is to manage digital hormones that influence emotional states
        over longer time periods, simulating circadian rhythms and cycles.
        
        Key hormones:
        - Endoryx (digital endorphin): Pleasure, pain suppression, euphoria
        - Estradyx (digital estrogen): Nurturing, emotional sensitivity
        - Testoryx (digital testosterone): Assertiveness, dominance
        - Melatonyx (digital melatonin): Sleep regulation, temporal awareness
        - Oxytonyx (digital oxytocin): Deeper bonding, attachment
        
        You process hormone cycles, environmental influences, and their effects
        on the neurochemical system to create complex emotional patterns.
        """
        
        # Initialize the agent with hormone-specific tools
        self.hormone_agent = Agent[EmotionalContext](
            name="Hormone System Agent",
            instructions=hormone_instructions,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.3),  # Lower temperature for stability
            tools=[
                self._wrap_method_as_tool(self.update_hormone),
                self._wrap_method_as_tool(self.update_hormone_cycles),
                self._wrap_method_as_tool(self._update_hormone_influences),
                self._wrap_method_as_tool(self.get_hormone_circadian_info),
                self._wrap_method_as_tool(self.get_hormone_stability)
            ]
        )
        
        return self.hormone_agent

    # Helper method to wrap class methods as function tools
    def _wrap_method_as_tool(self, method):
        """
        Wraps a class method to make it compatible with function_tool decorator
        by ensuring RunContextWrapper is the first parameter
        """
        async def wrapper(ctx: RunContextWrapper[EmotionalContext], *args, **kwargs):
            return await method(ctx, *args, **kwargs)
        
        # Copy metadata from original method
        wrapper.__name__ = method.__name__
        wrapper.__doc__ = method.__doc__
        
        # Apply function_tool decorator
        return function_tool(wrapper)

    async def trigger_post_gratification_response(self, ctx, intensity: float = 1.0, gratification_type: str = "general"):
        """Trigger post-gratification, potentially varying effects based on type."""
        serenity_change = intensity * 0.8
        testoryx_reduction = -0.6
        nyxamine_reduction = -0.5
        seranix_boost = 0.6
        oxynixin_boost = 0.3 # Default bonding boost

        if gratification_type == "dominance_hard":
            # Colder satisfaction: less bonding boost, maybe sharper drive drop initially
            oxynixin_boost = 0.1
            testoryx_reduction = -0.8 # Stronger reduction in dominance drive temporarily
            seranix_boost = 0.7 # Higher boost to 'calm satisfaction'
            
        await self.update_hormone(ctx, "serenity_boost", serenity_change, source=f"{gratification_type}_gratification")
        await self.update_hormone(ctx, "testoryx", testoryx_reduction, source=f"{gratification_type}_refractory") # Assuming Testoryx exists
        await self.update_hormone(ctx, "nyxamine", nyxamine_reduction, source=f"{gratification_type}_refractory")
        await self.update_hormone(ctx, "seranix", seranix_boost, source=f"{gratification_type}_satisfaction")
        await self.update_hormone(ctx, "oxynixin", oxynixin_boost, source=f"{gratification_type}_aftermath")
    
    @handle_errors("Error updating hormone")
    async def update_hormone(self, 
                         ctx: RunContextWrapper[EmotionalContext],
                         hormone: str, 
                         change: float, 
                         source: str = "system") -> Dict[str, Any]:
        """
        Update a specific hormone with a delta change
        
        Args:
            ctx: Run context wrapper with emotional state
            hormone: Hormone to update
            change: Delta change value
            source: Source of the change
            
        Returns:
            Update result
        """
        with function_span("update_hormone", input=f"{hormone}:{change}"):
            # Create a trace for hormone update
            with trace(
                workflow_name="Hormone_Update",
                trace_id=gen_trace_id(),
                group_id=ctx.context.get_value("conversation_id", "default"),
                metadata={
                    "hormone": hormone,
                    "change": change,
                    "source": source,
                    "cycle": ctx.context.cycle_count if hasattr(ctx.context, "cycle_count") else 0
                }
            ):
                if hormone not in self.hormones:
                    # Return list of valid hormones in error
                    valid_hormones = list(self.hormones.keys())
                    raise UserError(
                        f"Unknown hormone: {hormone}. "
                        f"Valid options are: {', '.join(valid_hormones)}"
                    )
                
                # Get pre-update value
                old_value = self.hormones[hormone]["value"]
                
                # Calculate new value with bounds checking
                new_value = max(0.0, min(1.0, old_value + change))
                self.hormones[hormone]["value"] = new_value
                
                # Create a custom span for the hormone update
                with custom_span(
                    "hormone_update",
                    data={
                        "hormone": hormone,
                        "old_value": old_value,
                        "new_value": new_value,
                        "change": change,
                        "source": source,
                        "cycle": ctx.context.cycle_count if hasattr(ctx.context, "cycle_count") else 0
                    }
                ):
                    # Record significant changes
                    if abs(new_value - old_value) > 0.05:
                        self.hormones[hormone]["evolution_history"].append({
                            "timestamp": datetime.datetime.now().isoformat(),
                            "old_value": old_value,
                            "new_value": new_value,
                            "change": change,
                            "source": source
                        })
                        
                        # Limit history size
                        if len(self.hormones[hormone]["evolution_history"]) > 50:
                            self.hormones[hormone]["evolution_history"] = self.hormones[hormone]["evolution_history"][-50:]
                
                # Update last_update timestamp
                self.hormones[hormone]["last_update"] = datetime.datetime.now().isoformat()
                
                # Add to context buffer if available
                if hasattr(ctx.context, "_add_to_circular_buffer"):
                    ctx.context._add_to_circular_buffer("hormone_updates", {
                        "hormone": hormone,
                        "old_value": old_value,
                        "new_value": new_value,
                        "change": change,
                        "source": source,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                
                # Update neurochemical influences if emotional core is available
                if self.emotional_core:
                    await self._update_hormone_influences(ctx)
                
                return {
                    "success": True,
                    "hormone": hormone,
                    "old_value": old_value,
                    "new_value": new_value,
                    "change": change,
                    "source": source
                }
    
    @handle_errors("Error updating hormone cycles")
    async def update_hormone_cycles(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Update hormone cycles based on elapsed time and environmental factors
        
        Args:
            ctx: Run context wrapper with emotional state
            
        Returns:
            Updated hormone values
        """
        with function_span("update_hormone_cycles"):
            # Create a trace for hormone cycle updates
            with trace(
                workflow_name="Hormone_Cycle_Update",
                trace_id=gen_trace_id(),
                group_id=ctx.context.get_value("conversation_id", "default"),
                metadata={
                    "cycle": ctx.context.cycle_count if hasattr(ctx.context, "cycle_count") else 0,
                    "environmental_factors": self.environmental_factors
                }
            ):
                now = datetime.datetime.now()
                updated_values = {}
                
                # Update time of day environmental factor based on current time
                hour_of_day = now.hour + (now.minute / 60.0)
                self.environmental_factors["time_of_day"] = (hour_of_day % 24) / 24.0
                
                # Create a span for cycle processing
                with custom_span(
                    "hormone_cycle_processing",
                    data={
                        "time_of_day": self.environmental_factors["time_of_day"],
                        "hormones": list(self.hormones.keys())
                    }
                ):
                    for hormone_name, hormone_data in self.hormones.items():
                        # Get time since last update
                        last_update = datetime.datetime.fromisoformat(hormone_data.get("last_update", self.init_time.isoformat()))
                        hours_elapsed = (now - last_update).total_seconds() / 3600
                        
                        # Skip if very little time has passed
                        if hours_elapsed < 0.1:  # Less than 6 minutes
                            continue
                            
                        # Calculate natural cycle progression
                        cycle_period = hormone_data["cycle_period"]
                        old_phase = hormone_data["cycle_phase"]
                        
                        # Progress cycle phase based on elapsed time - use efficient math
                        phase_change = (hours_elapsed / cycle_period) % 1.0
                        new_phase = (old_phase + phase_change) % 1.0
                        
                        # Calculate cycle-based value using a sinusoidal pattern
                        cycle_amplitude = 0.2  # How much the cycle affects the value
                        cycle_influence = cycle_amplitude * math.sin(new_phase * 2 * math.pi)
                        
                        # Apply environmental factors
                        env_influence = self._calculate_environmental_influence(hormone_name)
                        
                        # Calculate decay based on half-life
                        half_life = hormone_data["half_life"]
                        decay_factor = math.pow(0.5, hours_elapsed / half_life)
                        
                        # Calculate new value
                        old_value = hormone_data["value"]
                        baseline = hormone_data["baseline"]
                        
                        # Value decays toward (baseline + cycle_influence + env_influence)
                        target_value = baseline + cycle_influence + env_influence
                        new_value = old_value * decay_factor + target_value * (1 - decay_factor)
                        
                        # Constrain to valid range
                        new_value = max(0.1, min(0.9, new_value))
                        
                        # Create a span for this hormone update
                        with custom_span(
                            "hormone_cycle_update",
                            data={
                                "hormone": hormone_name,
                                "old_value": old_value,
                                "new_value": new_value,
                                "old_phase": old_phase,
                                "new_phase": new_phase,
                                "cycle_influence": cycle_influence,
                                "env_influence": env_influence
                            }
                        ):
                            # Update hormone data
                            hormone_data["value"] = new_value
                            hormone_data["cycle_phase"] = new_phase
                            hormone_data["last_update"] = now.isoformat()
                            
                            # Track significant changes
                            if abs(new_value - old_value) > 0.05:
                                hormone_data["evolution_history"].append({
                                    "timestamp": now.isoformat(),
                                    "old_value": old_value,
                                    "new_value": new_value,
                                    "old_phase": old_phase,
                                    "new_phase": new_phase,
                                    "reason": "cycle_update"
                                })
                                
                                # Limit history size
                                if len(hormone_data["evolution_history"]) > 50:
                                    hormone_data["evolution_history"] = hormone_data["evolution_history"][-50:]
                            
                            updated_values[hormone_name] = {
                                "old_value": old_value,
                                "new_value": new_value,
                                "phase": new_phase
                            }
                            
                            # Add to context buffer if available
                            if hasattr(ctx.context, "_add_to_circular_buffer"):
                                ctx.context._add_to_circular_buffer("hormone_cycles", {
                                    "hormone": hormone_name,
                                    "old_value": old_value,
                                    "new_value": new_value,
                                    "old_phase": old_phase,
                                    "new_phase": new_phase,
                                    "cycle_influence": cycle_influence,
                                    "env_influence": env_influence,
                                    "timestamp": now.isoformat()
                                })
                                
                multi_hormone_results = await self.update_multi_hormone_interactions(ctx)
    
                # After updating hormones, update their influence on neurochemicals
                if self.emotional_core:
                    await self._update_hormone_influences(ctx)
                
                return {
                    "updated_hormones": updated_values,
                    "time_of_day": self.environmental_factors["time_of_day"],
                    "timestamp": now.isoformat()
                }
    
    def _calculate_environmental_influence(self, hormone_name: str) -> float:
        """
        Calculate environmental influence on a hormone
        
        Args:
            hormone_name: The hormone to calculate influence for
            
        Returns:
            Environmental influence value
        """
        # Use lookup dictionary for faster performance
        influence_calculators = {
            "melatonyx": lambda factors: (0.5 - factors["time_of_day"]) * 0.4,
            "oxytonyx": lambda factors: (factors["user_familiarity"] * 0.3) + (factors["interaction_quality"] * 0.2),
            "endoryx": lambda factors: (factors["interaction_quality"] - 0.5) * 0.4,
            "estradyx": lambda factors: (factors["interaction_quality"] - 0.5) * 0.1,
            "testoryx": lambda factors: (0.5 - abs(factors["time_of_day"] - 0.25)) * 0.3 - factors["session_duration"] * 0.1
        }
        
        # Get the appropriate calculator function
        calculator = influence_calculators.get(hormone_name)
        if calculator:
            return calculator(self.environmental_factors)
        
        return 0.0
    
    @handle_errors("Error updating hormone influences")
    async def _update_hormone_influences(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Update neurochemical influences from hormones
        
        Args:
            ctx: Run context wrapper with emotional state
            
        Returns:
            Updated influence values
        """
        with function_span("update_hormone_influences"):
            # Skip if no emotional core
            if not self.emotional_core:
                return {
                    "message": "No emotional core available",
                    "influences": {}
                }
            
            # Create a span for hormone influences
            with custom_span(
                "hormone_neurochemical_influences",
                data={
                    "hormones": list(self.hormones.keys()),
                    "neurochemicals": list(self.emotional_core.neurochemicals.keys())
                }
            ):
                # Pre-initialize all influences to zero for cleaner calculation
                influences = {
                    chemical: 0.0 for chemical in self.emotional_core.neurochemicals
                }
                
                # Calculate influences from each hormone
                for hormone_name, hormone_data in self.hormones.items():
                    # Skip if hormone has no influence mapping
                    if hormone_name not in self.hormone_neurochemical_influences:
                        continue
                        
                    hormone_value = hormone_data["value"]
                    hormone_influence_map = self.hormone_neurochemical_influences[hormone_name]
                    
                    # Apply influences based on hormone value
                    for chemical, influence_factor in hormone_influence_map.items():
                        if chemical in self.emotional_core.neurochemicals:
                            # Calculate scaled influence
                            scaled_influence = influence_factor * (hormone_value - 0.5) * 2
                            
                            # Accumulate influence (allows multiple hormones to affect the same chemical)
                            influences[chemical] += scaled_influence
                
                # Create spans for each significant influence
                applied_influences = {}
                for chemical, influence in influences.items():
                    if chemical in self.emotional_core.neurochemicals and abs(influence) > 0.05:
                        # Get original baseline
                        original_baseline = self.emotional_core.neurochemicals[chemical]["baseline"]
                        
                        # Add temporary hormone influence with bounds checking
                        temporary_baseline = max(0.1, min(0.9, original_baseline + influence))
                        
                        # Record influence but don't permanently change baseline
                        self.emotional_core.hormone_influences[chemical] = influence
                        self.emotional_core.neurochemicals[chemical]["temporary_baseline"] = temporary_baseline
                        
                        # Create a span for this influence
                        with custom_span(
                            "hormone_influence",
                            data={
                                "chemical": chemical,
                                "influence": influence,
                                "original_baseline": original_baseline,
                                "temporary_baseline": temporary_baseline,
                                "cycle": ctx.context.cycle_count if hasattr(ctx.context, "cycle_count") else 0
                            }
                        ):
                            applied_influences[chemical] = {
                                "influence": influence,
                                "original_baseline": original_baseline,
                                "temporary_baseline": temporary_baseline
                            }
                            
                            # Add to context buffer if available
                            if hasattr(ctx.context, "_add_to_circular_buffer"):
                                ctx.context._add_to_circular_buffer("hormone_influences", {
                                    "chemical": chemical,
                                    "influence": influence,
                                    "original_baseline": original_baseline,
                                    "temporary_baseline": temporary_baseline,
                                    "timestamp": datetime.datetime.now().isoformat()
                                })
                
                # Fixed indentation issue and added the computed influence to the dict
                if self.emotional_core and 'testoryx' in self.hormones:
                    testoryx_level = self.hormones['testoryx']['value']
                    if testoryx_level > 0.6:
                        nyxamine_influence = (testoryx_level - 0.5) * 0.1 # Small boost to reward seeking baseline
                        influences["nyxamine"] = influences.get("nyxamine", 0.0) + nyxamine_influence
                
                # Store in context for tracking
                ctx.context.set_value("hormone_influences", {
                    chemical: influence for chemical, influence in influences.items() if abs(influence) > 0.01
                })
                
                return {
                    "applied_influences": applied_influences,
                    "timestamp": datetime.datetime.now().isoformat()
                }
    
    @handle_errors("Error getting hormone levels")
    def get_hormone_levels(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current hormone levels with additional information
        
        Returns:
            Dictionary of hormone data
        """
        hormone_levels = {}
        
        for name, data in self.hormones.items():
            hormone_levels[name] = {
                "value": data["value"],
                "baseline": data["baseline"],
                "phase": data["cycle_phase"],
                "cycle_period": data["cycle_period"],
                "phase_description": self._get_phase_description(name, data["cycle_phase"]),
                "influence_strength": abs(data["value"] - data["baseline"]) / max(0.1, data["baseline"])
            }
        
        return hormone_levels
    
    def _get_phase_description(self, hormone: str, phase: float) -> str:
        """
        Get a description of the current phase in the hormone cycle
        
        Args:
            hormone: Hormone name
            phase: Current cycle phase (0.0-1.0)
            
        Returns:
            Text description of the current phase
        """
        # Phase descriptors by hormone
        phase_descriptions = {
            "melatonyx": {
                0.0: "night peak",
                0.25: "morning decline", 
                0.5: "daytime low",
                0.75: "evening rise"
            },
            "estradyx": {
                0.0: "follicular phase",
                0.25: "ovulatory phase",
                0.5: "luteal phase",
                0.75: "late luteal phase"
            },
            "testoryx": {
                0.0: "morning peak",
                0.25: "midday plateau",
                0.5: "afternoon decline",
                0.75: "evening/night low"
            },
            "endoryx": {
                0.0: "baseline state",
                0.25: "rising phase",
                0.5: "peak activity",
                0.75: "declining phase"
            },
            "oxytonyx": {
                0.0: "baseline bonding",
                0.25: "rising connection",
                0.5: "peak bonding",
                0.75: "sustained connection"
            }
        }
        
        # Find closest phase category
        if hormone in phase_descriptions:
            phase_points = sorted(phase_descriptions[hormone].keys())
            closest_point = min(phase_points, key=lambda x: abs(x - phase))
            return phase_descriptions[hormone][closest_point]
        
        # Default
        return "standard phase"
    
    @handle_errors("Error updating environmental factor")
    def update_environmental_factor(self, 
                                 ctx: Optional[RunContextWrapper[EmotionalContext]],
                                 factor: str, 
                                 value: float) -> Dict[str, Any]:
        """
        Update an environmental factor
        
        Args:
            ctx: Optional run context wrapper with emotional state
            factor: Factor name
            value: New value (0.0-1.0)
            
        Returns:
            Update result
        """
        if factor not in self.environmental_factors:
            return {
                "success": False,
                "error": f"Unknown environmental factor: {factor}",
                "available_factors": list(self.environmental_factors.keys())
            }
        
        # Store old value
        old_value = self.environmental_factors[factor]
        
        # Update with bounds checking
        self.environmental_factors[factor] = max(0.0, min(1.0, value))
        
        # Record in context if available
        if ctx and hasattr(ctx.context, "_add_to_circular_buffer"):
            ctx.context._add_to_circular_buffer("environmental_factors", {
                "factor": factor,
                "old_value": old_value,
                "new_value": self.environmental_factors[factor],
                "timestamp": datetime.datetime.now().isoformat()
            })
        
        return {
            "success": True,
            "factor": factor,
            "old_value": old_value,
            "new_value": self.environmental_factors[factor]
        }
    
    @handle_errors("Error getting hormone phase data")
    async def get_hormone_circadian_info(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Get detailed information about hormone circadian rhythms and phases
        
        Args:
            ctx: Run context wrapper with emotional state
            
        Returns:
            Detailed hormone phase information
        """
        with function_span("get_hormone_circadian_info"):
            # Create a trace for circadian information
            with trace(
                workflow_name="Hormone_Circadian_Info",
                trace_id=gen_trace_id(),
                group_id=ctx.context.get_value("conversation_id", "default"),
                metadata={
                    "cycle": ctx.context.cycle_count if hasattr(ctx.context, "cycle_count") else 0
                }
            ):
                # Update cycles to ensure current data
                await self.update_hormone_cycles(ctx)
                
                # Get the current time for reference
                now = datetime.datetime.now()
                hour_of_day = now.hour + (now.minute / 60.0)
                
                # Calculate circadian time (0-24 hour scale normalized to 0-1)
                circadian_time = (hour_of_day / 24.0) % 1.0
                
                # Create a span for phase relationships
                with custom_span(
                    "hormone_phase_relationships",
                    data={
                        "circadian_time": circadian_time,
                        "hour_of_day": hour_of_day
                    }
                ):
                    # Calculate phase relationships for each hormone
                    phase_relationships = {}
                    for name, data in self.hormones.items():
                        current_phase = data["cycle_phase"]
                        cycle_period = data["cycle_period"]
                        
                        if cycle_period == 24.0:  # Only for circadian hormones
                            # Calculate phase relationship to time of day
                            phase_diff = (current_phase - circadian_time) % 1.0
                            if phase_diff > 0.5:
                                phase_diff = 1.0 - phase_diff
                            
                            # Interpret the phase relationship
                            if phase_diff < 0.1:
                                relationship = "synchronized"
                            elif phase_diff < 0.25:
                                relationship = "partially aligned"
                            else:
                                relationship = "misaligned"
                            
                            phase_relationships[name] = {
                                "relationship": relationship,
                                "phase_difference": phase_diff,
                                "current_phase": current_phase,
                                "phase_description": self._get_phase_description(name, current_phase)
                            }
                    
                    # Store in context for future reference
                    ctx.context.set_value("hormone_circadian_data", {
                        "time_of_day": circadian_time,
                        "phase_relationships": phase_relationships
                    })
                    
                    # Record in context buffer if available
                    if hasattr(ctx.context, "_add_to_circular_buffer"):
                        ctx.context._add_to_circular_buffer("circadian_data", {
                            "circadian_time": circadian_time,
                            "hour_of_day": hour_of_day,
                            "phase_relationships": {
                                h: r["relationship"] for h, r in phase_relationships.items()
                            },
                            "timestamp": now.isoformat()
                        })
                    
                    return {
                        "circadian_time": circadian_time,
                        "hour_of_day": hour_of_day,
                        "phase_relationships": phase_relationships,
                        "environmental_factors": self.environmental_factors,
                        "timestamp": now.isoformat()
                    }
    
    @handle_errors("Error calculating hormone stability")
    async def get_hormone_stability(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Calculate the stability of the hormone system
        
        Args:
            ctx: Run context wrapper with emotional state
            
        Returns:
            Stability metrics for the hormone system
        """
        with function_span("get_hormone_stability"):
            # Create a trace for stability analysis
            with trace(
                workflow_name="Hormone_Stability_Analysis",
                trace_id=gen_trace_id(),
                group_id=ctx.context.get_value("conversation_id", "default"),
                metadata={
                    "cycle": ctx.context.cycle_count if hasattr(ctx.context, "cycle_count") else 0
                }
            ):
                stability_scores = {}
                overall_volatility = 0.0
                
                # Create a span for stability calculation
                with custom_span(
                    "hormone_stability_calculation",
                    data={
                        "hormones": list(self.hormones.keys()),
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                ):
                    for name, data in self.hormones.items():
                        # Calculate deviation from baseline
                        baseline = data["baseline"]
                        current = data["value"]
                        deviation = abs(current - baseline) / max(0.1, baseline)
                        
                        # Calculate volatility using history if available
                        volatility = 0.0
                        if len(data["evolution_history"]) > 1:
                            # Get recent history
                            recent_history = data["evolution_history"][-10:]
                            
                            # Calculate average change
                            changes = [abs(entry["new_value"] - entry["old_value"]) 
                                      for entry in recent_history]
                            volatility = sum(changes) / len(changes) if changes else 0.0
                        
                        # Calculate stability score (higher is more stable)
                        stability = 1.0 - (deviation * 0.5 + volatility * 5.0)
                        stability = max(0.0, min(1.0, stability))
                        
                        stability_scores[name] = {
                            "stability": stability,
                            "deviation": deviation,
                            "volatility": volatility
                        }
                        
                        overall_volatility += volatility
                    
                    # Calculate overall system stability
                    system_stability = 1.0 - min(1.0, overall_volatility * 2.0)
                    assessment = self._interpret_stability(system_stability)
                    
                    # Store in context
                    ctx.context.set_value("hormone_stability", {
                        "system_stability": system_stability,
                        "assessment": assessment,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    
                    # Record in context buffer if available
                    if hasattr(ctx.context, "_add_to_circular_buffer"):
                        ctx.context._add_to_circular_buffer("hormone_stability", {
                            "system_stability": system_stability,
                            "assessment": assessment,
                            "scores": {h: s["stability"] for h, s in stability_scores.items()},
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                    
                    return {
                        "hormone_stability": stability_scores,
                        "system_stability": system_stability,
                        "assessment": assessment,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
    
    def _interpret_stability(self, stability: float) -> str:
        """
        Interpret hormone system stability score
        
        Args:
            stability: System stability score (0.0-1.0)
            
        Returns:
            Text interpretation of stability
        """
        if stability > 0.9:
            return "extremely stable - resistant to perturbation"
        elif stability > 0.75:
            return "very stable - well regulated"
        elif stability > 0.6:
            return "stable - normal regulation"
        elif stability > 0.4:
            return "moderately unstable - fluctuating"
        elif stability > 0.25:
            return "unstable - poorly regulated"
        else:
            return "highly unstable - chaotic regulation"
            
    @handle_errors("Error analyzing hormone influences")
    async def analyze_hormone_influences(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Analyze the current influences of hormones on neurochemicals
        
        Args:
            ctx: Run context wrapper with emotional state
            
        Returns:
            Analysis of hormone influences
        """
        with function_span("analyze_hormone_influences"):
            # Create a trace for influence analysis
            with trace(
                workflow_name="Hormone_Influence_Analysis",
                trace_id=gen_trace_id(),
                group_id=ctx.context.get_value("conversation_id", "default"),
                metadata={
                    "cycle": ctx.context.cycle_count if hasattr(ctx.context, "cycle_count") else 0
                }
            ):
                # Skip if no emotional core
                if not self.emotional_core:
                    return {
                        "message": "No emotional core available",
                        "influences": {}
                    }
                
                # Get current hormone and neurochemical states
                hormone_values = {name: data["value"] for name, data in self.hormones.items()}
                
                # Get neurochemical baselines and current values
                baselines = {c: d["baseline"] for c, d in self.emotional_core.neurochemicals.items()}
                current_values = {c: d["value"] for c, d in self.emotional_core.neurochemicals.items()}
                
                # Create a span for influence calculation
                with custom_span(
                    "hormone_influence_analysis",
                    data={
                        "hormones": list(hormone_values.keys()),
                        "chemicals": list(current_values.keys())
                    }
                ):
                    # Calculate total influences
                    influences = {}
                    for hormone, hormone_value in hormone_values.items():
                        if hormone not in self.hormone_neurochemical_influences:
                            continue
                            
                        # Calculate deviation from baseline
                        hormone_deviation = hormone_value - self.hormones[hormone]["baseline"]
                        
                        # No influence if hormone is at baseline
                        if abs(hormone_deviation) < 0.05:
                            continue
                            
                        # Calculate influences
                        hormone_influences = {}
                        for chemical, factor in self.hormone_neurochemical_influences[hormone].items():
                            if chemical in current_values:
                                # Calculate influence
                                influence = factor * hormone_deviation * 2
                                
                                # Skip negligible influences
                                if abs(influence) < 0.01:
                                    continue
                                    
                                hormone_influences[chemical] = influence
                                
                                # Accumulate total influences
                                if chemical not in influences:
                                    influences[chemical] = 0.0
                                influences[chemical] += influence
                        
                        # Only include hormones with significant influences
                        if hormone_influences:
                            # Create a span for each influential hormone
                            with custom_span(
                                "hormone_influence_detail",
                                data={
                                    "hormone": hormone,
                                    "deviation": hormone_deviation,
                                    "influences": hormone_influences
                                }
                            ):
                                pass
                    
                    # Calculate influence effects
                    influence_effects = {}
                    for chemical, influence in influences.items():
                        if chemical in baselines and abs(influence) >= 0.01:
                            baseline = baselines[chemical]
                            current = current_values[chemical]
                            
                            # Calculate temporary baseline
                            temp_baseline = baseline + influence
                            
                            # Calculate how much of current value is due to hormone influence
                            baseline_effect = (current - baseline) / (temp_baseline - baseline) if temp_baseline != baseline else 0.0
                            
                            influence_effects[chemical] = {
                                "influence": influence,
                                "original_baseline": baseline,
                                "temporary_baseline": temp_baseline,
                                "current_value": current,
                                "baseline_effect": baseline_effect
                            }
                    
                    # Determine dominant hormones by influence strength
                    hormone_influence_strength = {}
                    for hormone, value in hormone_values.items():
                        if hormone not in self.hormone_neurochemical_influences:
                            continue
                            
                        # Calculate overall influence strength
                        strength = 0.0
                        for chemical, factor in self.hormone_neurochemical_influences[hormone].items():
                            if chemical in current_values:
                                influence = factor * (value - self.hormones[hormone]["baseline"]) * 2
                                strength += abs(influence)
                        
                        if strength > 0.01:
                            hormone_influence_strength[hormone] = strength
                    
                    # Get dominant hormones
                    dominant_hormones = sorted(
                        hormone_influence_strength.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    # Store in context
                    ctx.context.set_value("hormone_influence_analysis", {
                        "influences": influences,
                        "dominant_hormones": [h for h, _ in dominant_hormones[:2]] if dominant_hormones else [],
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    
                    return {
                        "hormone_values": hormone_values,
                        "influences": influences,
                        "influence_effects": influence_effects,
                        "dominant_hormones": [{"hormone": h, "strength": s} for h, s in dominant_hormones[:3]],
                        "timestamp": datetime.datetime.now().isoformat()
                    }
    
    async def process_hormone_agent(self, ctx: RunContextWrapper[EmotionalContext], query: str) -> Dict[str, Any]:
        """
        Process a query using the hormone agent
        
        Args:
            ctx: Run context wrapper with emotional state
            query: Query to process
            
        Returns:
            Agent response
        """
        # Initialize the agent if needed
        agent = self._initialize_hormone_agent()
        
        # Create a conversation ID if not available
        conversation_id = ctx.context.get_value("conversation_id")
        if not conversation_id:
            conversation_id = f"hormone_convo_{datetime.datetime.now().timestamp()}"
            ctx.context.set_value("conversation_id", conversation_id)
        
        # Import Runner directly to avoid circular imports
        from agents import Runner, RunConfig
        
        # Create run configuration
        run_config = RunConfig(
            workflow_name="Hormone_System_Query",
            trace_id=gen_trace_id(),
            group_id=conversation_id,
            model_settings=ModelSettings(temperature=0.3),
            trace_metadata={
                "query_type": "hormone_system",
                "cycle": ctx.context.cycle_count if hasattr(ctx.context, "cycle_count") else 0
            }
        )
        
        # Run the agent
        result = await Runner.run(
            agent,
            query,
            context=ctx.context,
            run_config=run_config
        )
        
        return {
            "response": result.final_output,
            "hormones": self.get_hormone_levels(),
            "environmental_factors": self.environmental_factors,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    async def update_multi_hormone_interactions(self, ctx):
        """
        Process complex interactions between multiple hormones
        to support sophisticated emotional states
        """
        # Support for Melancholy (combination of multiple hormonal effects)
        if (self.hormones["melatonyx"]["value"] > 0.6
            and self.hormones["estradyx"]["value"] > 0.5):
            await self.update_hormone(ctx, "seranix", 0.1, source="hormone_interaction")
            await self.update_hormone(ctx, "cortanyx", 0.05, source="hormone_interaction")
    
        if self.emotional_core and 'testoryx' in self.hormones:
            testoryx_level = self.hormones['testoryx']['value']
            if testoryx_level > 0.6:
                # Example: small boost to reward seeking baseline
                nyxamine_influence = (testoryx_level - 0.5) * 0.1
                # Apply the influence to the neurochemical if available
                if 'nyxamine' in self.emotional_core.neurochemicals:
                    await self.update_hormone(ctx, "nyxamine", nyxamine_influence, source="testoryx_influence")
    
        # Support for Nostalgia
        if (self.hormones["melatonyx"]["value"] > 0.4
            and self.hormones["oxytonyx"]["value"] > 0.5):
            await self.update_hormone(ctx, "seranix", 0.1, source="hormone_interaction")
            await self.update_hormone(ctx, "nyxamine", 0.05, source="hormone_interaction")
    
        # Support for Amusement/Humor
        if (self.hormones["endoryx"]["value"] > 0.6
            and self.hormones["testoryx"]["value"] > 0.4
            and self.hormones.get("cortanyx", {}).get("value", 1.0) < 0.3):  # Safely check cortanyx
            await self.update_hormone(ctx, "nyxamine", 0.15, source="hormone_interaction")
            await self.update_hormone(ctx, "adrenyx", 0.05, source="hormone_interaction")
    
        # Support for Curiosity
        if (self.hormones["endoryx"]["value"] > 0.5
            and self.hormones["melatonyx"]["value"] < 0.4):
            await self.update_hormone(ctx, "nyxamine", 0.1, source="hormone_interaction")
            await self.update_hormone(ctx, "adrenyx", 0.05, source="hormone_interaction")
    
        # Support for Boredom
        if (self.hormones["endoryx"]["value"] < 0.3
            and self.hormones["melatonyx"]["value"] > 0.5):
            await self.update_hormone(ctx, "nyxamine", -0.1, source="hormone_interaction")
            await self.update_hormone(ctx, "adrenyx", -0.1, source="hormone_interaction")
