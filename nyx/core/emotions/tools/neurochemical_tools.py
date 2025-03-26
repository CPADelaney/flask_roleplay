# nyx/core/emotions/tools/neurochemical_tools.py

"""
Enhanced function tools for neurochemical system operations.
These tools handle updating and processing neurochemicals with improved error handling.
"""

import datetime
import logging
from typing import Dict, Any, Optional

from agents import function_tool, RunContextWrapper, function_span
from agents.exceptions import UserError

from nyx.core.emotions.context import EmotionalContext
from nyx.core.emotions.schemas import (
    EmotionUpdateInput, EmotionUpdateResult, 
    ChemicalDecayOutput, NeurochemicalInteractionOutput
)

logger = logging.getLogger(__name__)

# Custom error handler for neurochemical tools
async def neurochemical_error_handler(ctx: RunContextWrapper[Any], error: Exception) -> str:
    """Custom error handler for neurochemical tools"""
    error_type = type(error).__name__
    
    if isinstance(error, UserError):
        # For user errors, provide helpful guidance
        logger.warning(f"User error in neurochemical tool: {error}")
        return (f"There was an issue with the neurochemical operation: {error}. "
                f"Please check the chemical name and ensure values are between -1.0 and 1.0.")
    else:
        # For system errors, log but provide less detail to user
        logger.error(f"System error in neurochemical tool: {error}", exc_info=True)
        
        # Record error in context for analysis
        if ctx and hasattr(ctx, "context"):
            errors = ctx.context.get_value("system_errors", [])
            errors.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "error_type": error_type,
                "message": str(error)
            })
            ctx.context.set_value("system_errors", errors)
            
        return ("The neurochemical system encountered an internal error. "
                "The default behavior will be used instead.")

class NeurochemicalTools:
    """Enhanced function tools for managing the neurochemical state with improved error handling"""
    
    def __init__(self, neurochemical_system):
        """
        Initialize with reference to the neurochemical system
        
        Args:
            neurochemical_system: The neurochemical system to interact with
        """
        self.neurochemicals = neurochemical_system.neurochemicals
        self.chemical_interactions = neurochemical_system.chemical_interactions
        self.derive_emotional_state = neurochemical_system._derive_emotional_state
        self.last_update = neurochemical_system.last_update
    
    @function_tool(failure_error_function=neurochemical_error_handler)
    async def update_neurochemical(self, ctx: RunContextWrapper[EmotionalContext], 
                           update_data: EmotionUpdateInput) -> EmotionUpdateResult:
        """
        Update a specific neurochemical with a delta change
        
        Args:
            update_data: The update information including chemical, value and source
            
        Returns:
            Update result with neurochemical and emotion changes
        """
        with function_span("update_neurochemical", input=f"{update_data.chemical}:{update_data.value}"):
            # Validation
            if not -1.0 <= update_data.value <= 1.0:
                raise UserError(f"Value must be between -1.0 and 1.0, got {update_data.value}")
            
            if update_data.chemical not in self.neurochemicals:
                raise UserError(f"Unknown neurochemical: {update_data.chemical}")
                
            chemical = update_data.chemical
            value = update_data.value
            
            # Get pre-update value
            old_value = self.neurochemicals[chemical]["value"]
            
            # Update neurochemical
            self.neurochemicals[chemical]["value"] = max(0, min(1, old_value + value))
            
            # Process chemical interactions
            await self.process_chemical_interactions(ctx, source_chemical=chemical, source_delta=value)
            
            # Derive emotions from updated neurochemical state
            emotional_state = await self.derive_emotional_state(ctx)
            
            # Update timestamp
            self.last_update = datetime.datetime.now()
            
            # Track in context
            if ctx.context:
                ctx.context.last_emotions = emotional_state
                ctx.context.record_neurochemical_values({
                    c: d["value"] for c, d in self.neurochemicals.items()
                })
                
                # Track the update in context history
                ctx.context._add_to_circular_buffer("chemical_updates", {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "chemical": chemical,
                    "old_value": old_value,
                    "new_value": self.neurochemicals[chemical]["value"],
                    "change": value,
                    "source": update_data.source
                })
            
            return EmotionUpdateResult(
                success=True,
                updated_chemical=chemical,
                old_value=old_value,
                new_value=self.neurochemicals[chemical]["value"],
                derived_emotions=emotional_state
            )
    
    @function_tool(failure_error_function=neurochemical_error_handler)
    async def apply_chemical_decay(self, ctx: RunContextWrapper[EmotionalContext]) -> ChemicalDecayOutput:
        """
        Apply decay to all neurochemicals based on time elapsed and decay rates
        
        Returns:
            Updated neurochemical state after decay
        """
        with function_span("apply_chemical_decay"):
            now = datetime.datetime.now()
            time_delta = (now - self.last_update).total_seconds() / 3600  # hours
            
            # Don't decay if less than a minute has passed
            if time_delta < 0.016:  # about 1 minute in hours
                return ChemicalDecayOutput(
                    decay_applied=False,
                    neurochemical_state={c: d["value"] for c, d in self.neurochemicals.items()},
                    derived_emotions={},
                    time_elapsed_hours=time_delta,
                    last_update=self.last_update.isoformat()
                )
            
            # Apply decay to each neurochemical using comprehension for efficiency
            original_values = {c: d["value"] for c, d in self.neurochemicals.items()}
            
            for chemical, data in self.neurochemicals.items():
                decay_rate = data["decay_rate"]
                
                # Get baseline, accounting for temporary hormone influences
                if "temporary_baseline" in data:
                    baseline = data["temporary_baseline"]
                else:
                    baseline = data["baseline"]
                    
                current = data["value"]
                
                # Calculate decay based on time passed
                decay_amount = decay_rate * time_delta
                
                # Decay toward baseline
                if current > baseline:
                    self.neurochemicals[chemical]["value"] = max(baseline, current - decay_amount)
                elif current < baseline:
                    self.neurochemicals[chemical]["value"] = min(baseline, current + decay_amount)
            
            # Update timestamp
            self.last_update = now
            
            # Cache the results for future calls
            new_values = {c: d["value"] for c, d in self.neurochemicals.items()}
            ctx.context.record_neurochemical_values(new_values)
            ctx.context.set_value("last_neurochemical_update", now)
            
            # Track significant decay events
            for chemical, new_value in new_values.items():
                old_value = original_values[chemical]
                if abs(new_value - old_value) > 0.05:  # Only track significant changes
                    ctx.context._add_to_circular_buffer("decay_events", {
                        "timestamp": now.isoformat(),
                        "chemical": chemical,
                        "old_value": old_value,
                        "new_value": new_value,
                        "decay_amount": old_value - new_value,
                        "time_delta": time_delta
                    })
            
            # Derive new emotional state after decay
            emotional_state = await self.derive_emotional_state(ctx)
            
            return ChemicalDecayOutput(
                decay_applied=True,
                neurochemical_state=new_values,
                derived_emotions=emotional_state,
                time_elapsed_hours=time_delta,
                last_update=self.last_update.isoformat()
            )
    
    @function_tool(failure_error_function=neurochemical_error_handler)
    async def process_chemical_interactions(
        self, 
        ctx: RunContextWrapper[EmotionalContext],
        source_chemical: str,
        source_delta: float
    ) -> NeurochemicalInteractionOutput:
        """
        Process interactions between neurochemicals when one changes
        
        Args:
            source_chemical: The neurochemical that changed
            source_delta: The amount it changed by
            
        Returns:
            Interaction results
        """
        with function_span("process_chemical_interactions", input=f"{source_chemical}:{source_delta}"):
            if source_chemical not in self.chemical_interactions:
                return NeurochemicalInteractionOutput(
                    source_chemical=source_chemical,
                    source_delta=source_delta,
                    changes={}
                )
            
            changes = {}
            
            # Apply interactions to affected chemicals using a more efficient approach
            # Get all target interactions at once for efficiency
            target_interactions = self.chemical_interactions[source_chemical]
            affected_chemicals = [
                (chemical, source_delta * multiplier)
                for chemical, multiplier in target_interactions.items()
                if chemical in self.neurochemicals and abs(source_delta * multiplier) >= 0.01
            ]
            
            # Apply all effects at once
            for chemical, effect in affected_chemicals:
                old_value = self.neurochemicals[chemical]["value"]
                new_value = max(0, min(1, old_value + effect))
                self.neurochemicals[chemical]["value"] = new_value
                
                # Record change
                changes[chemical] = {
                    "old_value": old_value,
                    "new_value": new_value,
                    "change": new_value - old_value
                }
                
                # Track significant interactions in context
                if abs(new_value - old_value) > 0.05:  # Only track significant changes
                    ctx.context._add_to_circular_buffer("chemical_interactions", {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "source_chemical": source_chemical,
                        "target_chemical": chemical,
                        "effect": effect,
                        "old_value": old_value,
                        "new_value": new_value
                    })
            
            # Update cached neurochemical values
            ctx.context.record_neurochemical_values({
                c: d["value"] for c, d in self.neurochemicals.items()
            })
            
            return NeurochemicalInteractionOutput(
                source_chemical=source_chemical,
                source_delta=source_delta,
                changes=changes
            )
    
    @function_tool(failure_error_function=neurochemical_error_handler)
    async def get_neurochemical_state(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Get the current neurochemical state
        
        Returns:
            Current neurochemical state
        """
        with function_span("get_neurochemical_state"):
            # Check if we have a cached state in context for better performance
            cached_state = ctx.context.get_cached_neurochemicals(max_age_seconds=1.0)
            
            if cached_state is not None:
                return {
                    "chemicals": cached_state,
                    "baselines": {c: d["baseline"] for c, d in self.neurochemicals.items()},
                    "decay_rates": {c: d["decay_rate"] for c, d in self.neurochemicals.items()},
                    "hormone_influences": {
                        c: self.neurochemicals[c].get("temporary_baseline", d["baseline"]) - d["baseline"]
                        for c, d in self.neurochemicals.items()
                        if "temporary_baseline" in self.neurochemicals[c]
                    },
                    "timestamp": datetime.datetime.now().isoformat(),
                    "cached": True,
                    "trends": ctx.context.get_neurochemical_trends(limit=5)
                }
            
            # Apply decay before returning state
            await self.apply_chemical_decay(ctx)
            
            # Cache the result for future calls
            state = {c: d["value"] for c, d in self.neurochemicals.items()}
            ctx.context.record_neurochemical_values(state)
            
            return {
                "chemicals": state,
                "baselines": {c: d["baseline"] for c, d in self.neurochemicals.items()},
                "decay_rates": {c: d["decay_rate"] for c, d in self.neurochemicals.items()},
                "hormone_influences": {
                    c: self.neurochemicals[c].get("temporary_baseline", d["baseline"]) - d["baseline"]
                    for c, d in self.neurochemicals.items()
                    if "temporary_baseline" in self.neurochemicals[c]
                },
                "timestamp": datetime.datetime.now().isoformat(),
                "cached": False,
                "trends": ctx.context.get_neurochemical_trends(limit=5)
            }
