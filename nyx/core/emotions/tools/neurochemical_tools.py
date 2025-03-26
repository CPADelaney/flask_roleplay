# nyx/core/emotions/tools/neurochemical_tools.py

"""
Enhanced function tools for neurochemical system operations.
These tools handle updating and processing neurochemicals with improved
error handling and better OpenAI Agents SDK integration.
"""

import datetime
import logging
from typing import Dict, Any, Optional, List, Tuple, Union, cast

from agents import (
    function_tool, RunContextWrapper, function_span, custom_span,
    FunctionTool, ModelSettings, trace, gen_trace_id, Agent
)
from agents.exceptions import UserError, ModelBehaviorError

from nyx.core.emotions.context import EmotionalContext
from nyx.core.emotions.schemas import (
    EmotionUpdateInput, EmotionUpdateResult, 
    ChemicalDecayOutput, NeurochemicalInteractionOutput,
    ChemicalSource
)
from nyx.core.emotions.utils import handle_errors, EmotionalToolUtils

logger = logging.getLogger(__name__)

# Improved error handler for neurochemical tools with SDK integration
async def neurochemical_error_handler(ctx: RunContextWrapper[Any], error: Exception) -> str:
    """
    Custom error handler for neurochemical tools with improved tracing
    
    Args:
        ctx: Run context wrapper
        error: The exception that was raised
        
    Returns:
        Error message for the LLM
    """
    error_type = type(error).__name__
    
    # Create a custom span for error tracing
    with custom_span(
        "neurochemical_error",
        data={
            "error_type": error_type,
            "message": str(error),
            "timestamp": datetime.datetime.now().isoformat(),
            "cycle": ctx.context.cycle_count if hasattr(ctx, "context") and hasattr(ctx.context, "cycle_count") else 0
        }
    ):
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
                
                # Update error counts
                error_counts = ctx.context.get_value("error_counts", {})
                if error_type not in error_counts:
                    error_counts[error_type] = 0
                error_counts[error_type] += 1
                ctx.context.set_value("error_counts", error_counts)
                
            return ("The neurochemical system encountered an internal error. "
                    "The default behavior will be used instead.")

class NeurochemicalTools:
    """
    Enhanced function tools for managing the neurochemical state with
    improved error handling and SDK integration
    """
    
    def __init__(self, neurochemical_system):
        """
        Initialize with reference to the neurochemical system
        
        Args:
            neurochemical_system: The neurochemical system to interact with
        """
        self.neurochemicals = neurochemical_system.neurochemicals
        self.chemical_interactions = neurochemical_system.chemical_interactions
        
        # Store reference to the emotional_state derivation function
        if hasattr(neurochemical_system, "derive_emotional_state"):
            self.derive_emotional_state = neurochemical_system.derive_emotional_state
        elif hasattr(neurochemical_system, "_derive_emotional_state"):
            self.derive_emotional_state = neurochemical_system._derive_emotional_state
        else:
            self.derive_emotional_state = None
            logger.warning("No derive_emotional_state function found")
            
        # Reference to last_update
        self.last_update = neurochemical_system.last_update
    
    @function_tool(
        name_override="update_neurochemical",
        description_override="Update a specific neurochemical with a delta change",
        failure_error_function=neurochemical_error_handler
    )
    async def update_neurochemical(self, ctx: RunContextWrapper[EmotionalContext], 
                            update_data: EmotionUpdateInput) -> EmotionUpdateResult:
        """
        Update a specific neurochemical with a delta change
        
        Args:
            ctx: Run context wrapper with emotional state
            update_data: The update information including chemical, value and source
            
        Returns:
            Update result with neurochemical and emotion changes
        """
        with function_span("update_neurochemical", input=f"{update_data.chemical}:{update_data.value}"):
            # Create a trace for detailed chemical update monitoring
            with trace(
                workflow_name="Neurochemical_Update",
                trace_id=gen_trace_id(),
                group_id=ctx.context.get_value("conversation_id", "default"),
                metadata={
                    "chemical": update_data.chemical,
                    "value": update_data.value,
                    "source": update_data.source,
                    "cycle": ctx.context.cycle_count
                }
            ):
                # Enhanced validation
                if not -1.0 <= update_data.value <= 1.0:
                    raise UserError(f"Value must be between -1.0 and 1.0, got {update_data.value}")
                
                if update_data.chemical not in self.neurochemicals:
                    # Return list of valid chemicals in error message
                    valid_chemicals = list(self.neurochemicals.keys())
                    raise UserError(
                        f"Unknown neurochemical: {update_data.chemical}. "
                        f"Valid options are: {', '.join(valid_chemicals)}"
                    )
                    
                chemical = update_data.chemical
                value = update_data.value
                
                # Get pre-update value
                old_value = self.neurochemicals[chemical]["value"]
                
                # Create a custom span for the chemical update
                with custom_span(
                    "chemical_update",
                    data={
                        "chemical": chemical,
                        "old_value": old_value,
                        "change": value,
                        "source": update_data.source,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "cycle": ctx.context.cycle_count,
                        "type": "chemical_update"  # Type for analytics processor
                    }
                ):
                    # Update neurochemical
                    self.neurochemicals[chemical]["value"] = max(0, min(1, old_value + value))
                    
                    # Process chemical interactions
                    interaction_result = await self.process_chemical_interactions(
                        ctx, source_chemical=chemical, source_delta=value
                    )
                    
                    # Derive emotions from updated neurochemical state
                    if self.derive_emotional_state:
                        emotional_state = await self.derive_emotional_state(ctx)
                    else:
                        # Fallback if derive_emotional_state is not available
                        emotional_state = {"Neutral": 0.5}
                    
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
                    
                    # Update performance metrics if available
                    if hasattr(ctx.context, "get_value"):
                        metrics = ctx.context.get_value("performance_metrics", {})
                        if "update_counts" not in metrics:
                            metrics["update_counts"] = {}
                        
                        if chemical not in metrics["update_counts"]:
                            metrics["update_counts"][chemical] = 0
                            
                        metrics["update_counts"][chemical] += 1
                        ctx.context.set_value("performance_metrics", metrics)
                    
                    # Create the result
                    return EmotionUpdateResult(
                        success=True,
                        updated_chemical=chemical,
                        old_value=old_value,
                        new_value=self.neurochemicals[chemical]["value"],
                        derived_emotions=emotional_state
                    )
    
    @function_tool(
        name_override="apply_chemical_decay",
        description_override="Apply decay to all neurochemicals based on time elapsed and decay rates",
        failure_error_function=neurochemical_error_handler
    )
    async def apply_chemical_decay(self, ctx: RunContextWrapper[EmotionalContext]) -> ChemicalDecayOutput:
        """
        Apply decay to all neurochemicals based on time elapsed and decay rates
        
        Args:
            ctx: Run context wrapper with emotional state
            
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
            
            # Create a span for the decay computation
            with custom_span(
                "chemical_decay",
                data={
                    "time_elapsed_hours": time_delta,
                    "last_update": self.last_update.isoformat(),
                    "cycle": ctx.context.cycle_count
                }
            ):
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
                if self.derive_emotional_state:
                    emotional_state = await self.derive_emotional_state(ctx)
                else:
                    # Fallback if derive_emotional_state is not available
                    emotional_state = {"Neutral": 0.5}
                
                return ChemicalDecayOutput(
                    decay_applied=True,
                    neurochemical_state=new_values,
                    derived_emotions=emotional_state,
                    time_elapsed_hours=time_delta,
                    last_update=self.last_update.isoformat()
                )
    
    @function_tool(
        name_override="process_chemical_interactions",
        description_override="Process interactions between neurochemicals when one changes",
        failure_error_function=neurochemical_error_handler
    )
    async def process_chemical_interactions(
        self, 
        ctx: RunContextWrapper[EmotionalContext],
        source_chemical: str,
        source_delta: float
    ) -> NeurochemicalInteractionOutput:
        """
        Process interactions between neurochemicals when one changes
        
        Args:
            ctx: Run context wrapper with emotional state
            source_chemical: The neurochemical that changed
            source_delta: The amount it changed by
            
        Returns:
            Interaction results
        """
        with function_span("process_chemical_interactions", input=f"{source_chemical}:{source_delta}"):
            # Skip processing if source chemical has no interactions or delta is negligible
            if source_chemical not in self.chemical_interactions or abs(source_delta) < 0.01:
                return NeurochemicalInteractionOutput(
                    source_chemical=source_chemical,
                    source_delta=source_delta,
                    changes={}
                )
            
            # Create a span for chemical interactions
            with custom_span(
                "chemical_interactions",
                data={
                    "source_chemical": source_chemical,
                    "source_delta": source_delta,
                    "cycle": ctx.context.cycle_count
                }
            ):
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
                        # Create a custom span for each significant interaction
                        with custom_span(
                            "chemical_interaction_effect",
                            data={
                                "source": source_chemical,
                                "target": chemical,
                                "effect": effect,
                                "old_value": old_value,
                                "new_value": new_value,
                                "cycle": ctx.context.cycle_count
                            }
                        ):
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
    
    @function_tool(
        name_override="get_neurochemical_state",
        description_override="Get the current neurochemical state",
        failure_error_function=neurochemical_error_handler
    )
    async def get_neurochemical_state(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Get the current neurochemical state
        
        Args:
            ctx: Run context wrapper with emotional state
            
        Returns:
            Current neurochemical state
        """
        with function_span("get_neurochemical_state"):
            # Check if we have a cached state in context for better performance
            cached_state = ctx.context.get_cached_neurochemicals(max_age_seconds=1.0)
            
            if cached_state is not None:
                # Create a span with cached data
                with custom_span(
                    "cached_neurochemical_state",
                    data={
                        "chemicals": {k: round(v, 2) for k, v in cached_state.items()},
                        "cache_age": datetime.datetime.now().timestamp() - ctx.context.temp_data.get("cached_time", 0),
                        "cycle": ctx.context.cycle_count
                    }
                ):
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
            if hasattr(self, "apply_chemical_decay"):
                await self.apply_chemical_decay(ctx)
            
            # Create a span with fresh data
            with custom_span(
                "fresh_neurochemical_state",
                data={
                    "chemicals": {c: round(d["value"], 2) for c, d in self.neurochemicals.items()},
                    "cycle": ctx.context.cycle_count
                }
            ):
                # Cache the result for future calls
                state = {c: d["value"] for c, d in self.neurochemicals.items()}
                ctx.context.record_neurochemical_values(state)
                
                # Add current chemical change activity
                chemical_activity = []
                for chemical, data in self.neurochemicals.items():
                    baseline = data["baseline"]
                    current = data["value"]
                    deviation = current - baseline
                    
                    if abs(deviation) > 0.1:  # Only include significant deviations
                        chemical_activity.append({
                            "chemical": chemical,
                            "deviation": deviation,
                            "direction": "above" if deviation > 0 else "below",
                            "significance": abs(deviation) / baseline if baseline > 0 else abs(deviation)
                        })
                
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
                    "trends": ctx.context.get_neurochemical_trends(limit=5),
                    "chemical_activity": chemical_activity
                }
                
    @function_tool(
        name_override="neurochemical_analysis",
        description_override="Analyze current neurochemical state for patterns and imbalances",
        failure_error_function=neurochemical_error_handler
    )
    async def analyze_neurochemical_state(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Analyze the current neurochemical state for patterns and imbalances
        
        Args:
            ctx: Run context wrapper with emotional state
            
        Returns:
            Analysis of neurochemical state
        """
        with function_span("analyze_neurochemical_state"):
            # Create a trace for the analysis
            with trace(
                workflow_name="Neurochemical_Analysis",
                trace_id=gen_trace_id(),
                group_id=ctx.context.get_value("conversation_id", "default"),
                metadata={
                    "cycle": ctx.context.cycle_count,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            ):
                # Get current state - use cached if available
                current_state = ctx.context.get_cached_neurochemicals()
                if not current_state:
                    current_state = {c: d["value"] for c, d in self.neurochemicals.items()}
                
                baselines = {c: d["baseline"] for c, d in self.neurochemicals.items()}
                
                # Calculate deviations from baseline
                deviations = {
                    c: current_state[c] - baselines[c]
                    for c in current_state
                }
                
                # Identify dominant neurochemicals (highest above baseline)
                dominant_chemicals = sorted(
                    [(c, d) for c, d in deviations.items() if d > 0.1],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Identify suppressed neurochemicals (furthest below baseline)
                suppressed_chemicals = sorted(
                    [(c, d) for c, d in deviations.items() if d < -0.1],
                    key=lambda x: x[1]
                )
                
                # Calculate balance indices
                excitation_index = current_state.get("nyxamine", 0) + current_state.get("adrenyx", 0)
                calm_index = current_state.get("seranix", 0) + current_state.get("oxynixin", 0)
                stress_index = current_state.get("cortanyx", 0) + current_state.get("adrenyx", 0)
                
                # Calculate overall system balance
                if excitation_index > 0 and calm_index > 0:
                    balance_ratio = excitation_index / calm_index
                else:
                    balance_ratio = 1.0
                
                # Determine system state
                if balance_ratio > 1.5:
                    system_state = "overstimulated"
                elif balance_ratio < 0.5:
                    system_state = "subdued"
                else:
                    system_state = "balanced"
                
                # Get recent trends
                trends = ctx.context.get_neurochemical_trends(limit=5)
                
                # Identify patterns
                patterns = {}
                if trends:
                    for chemical, trend_data in trends.items():
                        if len(trend_data) >= 3:
                            # Get last three values
                            recent_values = [point["value"] for point in trend_data[-3:]]
                            
                            # Check for consistent increase
                            if all(recent_values[i] < recent_values[i+1] for i in range(len(recent_values)-1)):
                                patterns[chemical] = "consistently_increasing"
                            
                            # Check for consistent decrease
                            elif all(recent_values[i] > recent_values[i+1] for i in range(len(recent_values)-1)):
                                patterns[chemical] = "consistently_decreasing"
                            
                            # Check for oscillation
                            elif (recent_values[0] > recent_values[1] and recent_values[1] < recent_values[2]) or \
                                 (recent_values[0] < recent_values[1] and recent_values[1] > recent_values[2]):
                                patterns[chemical] = "oscillating"
                            
                            # Check for plateau
                            elif abs(recent_values[0] - recent_values[-1]) < 0.05:
                                patterns[chemical] = "stable"
                            
                            else:
                                patterns[chemical] = "variable"
                
                # Create a custom span for the analysis results
                with custom_span(
                    "neurochemical_analysis_result",
                    data={
                        "dominant_chemicals": dominant_chemicals[:2] if dominant_chemicals else [],
                        "system_state": system_state,
                        "balance_ratio": round(balance_ratio, 2),
                        "stress_index": round(stress_index, 2),
                        "cycle": ctx.context.cycle_count
                    }
                ):
                    return {
                        "dominant_chemicals": [{"chemical": c, "deviation": d} for c, d in dominant_chemicals[:3]] if dominant_chemicals else [],
                        "suppressed_chemicals": [{"chemical": c, "deviation": d} for c, d in suppressed_chemicals[:3]] if suppressed_chemicals else [],
                        "system_state": system_state,
                        "balance_ratio": balance_ratio,
                        "excitation_index": excitation_index,
                        "calm_index": calm_index,
                        "stress_index": stress_index,
                        "patterns": patterns,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    
    @function_tool(
        name_override="reset_neurochemicals",
        description_override="Reset neurochemicals to their baseline values",
        failure_error_function=neurochemical_error_handler
    )
    async def reset_neurochemicals(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Reset all neurochemicals to their baseline values
        
        Args:
            ctx: Run context wrapper with emotional state
            
        Returns:
            Result of reset operation
        """
        with function_span("reset_neurochemicals"):
            # Create a trace for the reset operation
            with trace(
                workflow_name="Neurochemical_Reset",
                trace_id=gen_trace_id(),
                group_id=ctx.context.get_value("conversation_id", "default"),
                metadata={
                    "cycle": ctx.context.cycle_count,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "reason": "manual_reset"
                }
            ):
                # Store original values for reporting
                original_values = {c: d["value"] for c, d in self.neurochemicals.items()}
                
                # Reset each neurochemical to its baseline
                for chemical, data in self.neurochemicals.items():
                    self.neurochemicals[chemical]["value"] = data["baseline"]
                
                # Update cached state
                new_values = {c: d["value"] for c, d in self.neurochemicals.items()}
                ctx.context.record_neurochemical_values(new_values)
                
                # Record the reset in context history
                ctx.context._add_to_circular_buffer("system_events", {
                    "event": "neurochemical_reset",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "original_values": original_values,
                    "new_values": new_values,
                    "cycle": ctx.context.cycle_count
                })
                
                # Update timestamp
                self.last_update = datetime.datetime.now()
                
                # Create a custom span for the reset result
                with custom_span(
                    "neurochemical_reset_result",
                    data={
                        "chemicals_reset": list(self.neurochemicals.keys()),
                        "cycle": ctx.context.cycle_count
                    }
                ):
                    # Derive emotional state after reset
                    if self.derive_emotional_state:
                        emotional_state = await self.derive_emotional_state(ctx)
                    else:
                        emotional_state = {"Neutral": 0.5}
                    
                    # Store the new emotional state
                    ctx.context.last_emotions = emotional_state
                    
                    return {
                        "success": True,
                        "original_values": original_values,
                        "new_values": new_values,
                        "derived_emotions": emotional_state,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
