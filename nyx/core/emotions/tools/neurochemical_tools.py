# nyx/core/emotions/tools/neurochemical_tools.py

import datetime
import logging
from typing import Dict, Any, Optional, List

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

# Import EmotionTools to check its type
from nyx.core.emotions.tools.emotion_tools import EmotionTools

from nyx.core.emotions.utils import handle_errors, EmotionalToolUtils

logger = logging.getLogger(__name__)

async def neurochemical_error_handler(ctx: RunContextWrapper[Any], error: Exception) -> str:
    """
    Custom error handler for neurochemical tools with improved tracing
    (unchanged from your original code).
    """
    error_type = type(error).__name__
    with custom_span(
        "neurochemical_error",
        data={ # Ensure data is stringified
            "error_type": str(error_type),
            "message": str(error),
            "timestamp": str(datetime.datetime.now().isoformat()),
            "cycle": str(ctx.context.cycle_count if hasattr(ctx, "context") and hasattr(ctx.context, "cycle_count") else 0)
        }
    ):
        if isinstance(error, UserError):
            logger.warning(f"User error in neurochemical tool: {error}")
            return (f"There was an issue with the neurochemical operation: {error}. "
                    f"Please check the chemical name and ensure values are between -1.0 and 1.0.")
        else:
            logger.error(f"System error in neurochemical tool: {error}", exc_info=True)
            if ctx and hasattr(ctx, "context") and hasattr(ctx.context, "get_value") and hasattr(ctx.context, "set_value"): # Check for get_value/set_value
                errors = ctx.context.get_value("system_errors", [])
                errors.append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "error_type": error_type,
                    "message": str(error)
                })
                ctx.context.set_value("system_errors", errors)

                error_counts = ctx.context.get_value("error_counts", {})
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                ctx.context.set_value("error_counts", error_counts)

            return ("The neurochemical system encountered an internal error. "
                    "The default behavior will be used instead.")


class NeurochemicalTools:
    """
    Enhanced function tools for managing the neurochemical state with
    improved error handling and SDK integration.
    """

    def __init__(self, neurochemical_system): # Reverted parameter name
        """
        Initialize with reference to the neurochemical system (which is an EmotionalCore instance).
        """
        self.neurochemical_system_ref = neurochemical_system # Store the EmotionalCore instance
        self.neurochemicals = neurochemical_system.neurochemicals
        self.chemical_interactions = neurochemical_system.chemical_interactions
        
        # CORRECTED: Store the actual callable implementation method for derive_emotional_state
        # 'neurochemical_system' parameter is the EmotionalCore instance.
        # EmotionalCore instance has an 'emotion_tools' attribute, which is an EmotionTools instance.
        # EmotionTools instance has an '_derive_emotional_state_impl' method.
        if hasattr(neurochemical_system, 'emotion_tools') and \
           isinstance(neurochemical_system.emotion_tools, EmotionTools) and \
           hasattr(neurochemical_system.emotion_tools, '_derive_emotional_state_impl'):
            self._actual_derive_emotional_state_callable = neurochemical_system.emotion_tools._derive_emotional_state_impl
        else:
            self._actual_derive_emotional_state_callable = None
            logger.error("NeurochemicalTools: Could not correctly initialize _actual_derive_emotional_state_callable. "
                         "The passed 'neurochemical_system' (EmotionalCore instance) or its 'emotion_tools' "
                         "attribute might be missing or misconfigured.")
        
        self.last_update = neurochemical_system.last_update
    
    # -------------------------------------------------------------------------
    # 1) update_neurochemical
    # -------------------------------------------------------------------------
    async def _update_neurochemical_impl(
        self,
        ctx: RunContextWrapper[EmotionalContext],
        chemical: str,
        value: float,
        source: str = "system"
    ) -> Dict[str, Any]:
        """
        The actual implementation with 'self'. 
        """
        with function_span("update_neurochemical", input=f"{chemical}:{value}"):
            with trace(
                workflow_name="Neurochemical_Update",
                trace_id=gen_trace_id(),
                group_id=ctx.context.get_value("conversation_id", "default"),
                metadata={ # Stringified
                    "chemical": str(chemical),
                    "value": str(value),
                    "source": str(source),
                    "cycle": str(ctx.context.cycle_count)
                }
            ):
                if chemical not in self.neurochemicals:
                    valid_chemicals = list(self.neurochemicals.keys())
                    raise UserError(
                        f"Unknown neurochemical: {chemical}. "
                        f"Valid options are: {', '.join(valid_chemicals)}"
                    )
                
                old_value = self.neurochemicals[chemical]["value"]
                new_value = max(0.0, min(1.0, old_value + value))
                self.neurochemicals[chemical]["value"] = new_value

                with custom_span(
                    "chemical_update",
                    data={ # Stringified
                        "chemical": str(chemical),
                        "old_value": str(old_value),
                        "new_value": str(new_value),
                        "change": str(value), # 'value' is the change here
                        "source": str(source),
                        "cycle": str(ctx.context.cycle_count),
                        "type": "chemical_update"
                    }
                ):
                    ctx.context._add_to_circular_buffer("chemical_updates", {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "chemical": chemical,
                        "old_value": old_value,
                        "new_value": new_value,
                        "change": value,
                        "source": source
                    })
                    self.last_update = datetime.datetime.now()

                    ctx.context.record_neurochemical_values({
                        c: d["value"] for c, d in self.neurochemicals.items()
                    })
                
                    interaction_result = await self._process_chemical_interactions_impl(
                        ctx, source_chemical=chemical, source_delta=value
                    )
                
                    emotional_state = {}
                    # CORRECTED: Call the stored callable method
                    if self._actual_derive_emotional_state_callable:
                        # The 'ctx' here is RunContextWrapper[EmotionalContext], which is what 
                        # _derive_emotional_state_impl from EmotionTools expects.
                        emotional_state = await self._actual_derive_emotional_state_callable(ctx)
                    else:
                        logger.warning("NeurochemicalTools: _actual_derive_emotional_state_callable not available. "
                                       "Skipping emotional state derivation in _update_neurochemical_impl.")
                
                    return {
                        "success": True,
                        "updated_chemical": chemical,
                        "old_value": old_value,
                        "new_value": new_value,
                        "change": value,
                        "source": source,
                        "derived_emotions": emotional_state # Ensure this is populated
                    }

    @staticmethod
    @function_tool(
        name_override="update_neurochemical",
        description_override="Update a specific neurochemical with a delta change",
        failure_error_function=neurochemical_error_handler
    )
    async def update_neurochemical(
        ctx: RunContextWrapper[EmotionalContext],
        chemical: str,
        value: float,
        source: str = "system"
    ) -> Dict[str, Any]:
        """
        Update a specific neurochemical with a delta change - restructured for SDK optimization.
        Args:
            ctx: Run context wrapper with emotional state
            chemical: Neurochemical to update
            value: Delta change value
            source: Source of the change
        """
        instance = ctx.context.get_value("neurochemical_tools_instance")
        if not instance:
            # Log and raise, or return an error dict. Raising is often cleaner.
            logger.error("NeurochemicalTools.update_neurochemical: No NeurochemicalTools instance found in context.")
            raise UserError("System configuration error: NeurochemicalTools instance missing.")
        return await instance._update_neurochemical_impl(ctx, chemical, value, source)

    # -------------------------------------------------------------------------
    # 2) apply_chemical_decay
    # -------------------------------------------------------------------------
    async def _apply_chemical_decay_impl(
        self,
        ctx: RunContextWrapper[EmotionalContext]
    ) -> ChemicalDecayOutput:
        with function_span("apply_chemical_decay"):
            now = datetime.datetime.now()
            time_delta = (now - self.last_update).total_seconds() / 3600

            if time_delta < 0.016: # about 1 minute
                return ChemicalDecayOutput(
                    decay_applied=False,
                    neurochemical_state={c: d["value"] for c, d in self.neurochemicals.items()},
                    derived_emotions={},
                    time_elapsed_hours=time_delta,
                    last_update=self.last_update.isoformat()
                )

            with custom_span(
                "chemical_decay",
                data={ # Stringified
                    "time_elapsed_hours": str(time_delta),
                    "last_update": str(self.last_update.isoformat()),
                    "cycle": str(ctx.context.cycle_count)
                }
            ):
                original_values = {c: d["value"] for c, d in self.neurochemicals.items()}
                for chemical, data in self.neurochemicals.items():
                    decay_rate = data["decay_rate"]
                    if "temporary_baseline" in data:
                        baseline = data["temporary_baseline"]
                    else:
                        baseline = data["baseline"]

                    current = data["value"]
                    decay_amount = decay_rate * time_delta

                    if current > baseline:
                        self.neurochemicals[chemical]["value"] = max(baseline, current - decay_amount)
                    elif current < baseline:
                        self.neurochemicals[chemical]["value"] = min(baseline, current + decay_amount)

                self.last_update = now
                new_values = {c: d["value"] for c, d in self.neurochemicals.items()}
                ctx.context.record_neurochemical_values(new_values)
                ctx.context.set_value("last_neurochemical_update", now)

                for chemical, new_value in new_values.items():
                    old_value = original_values[chemical]
                    if abs(new_value - old_value) > 0.05:
                        ctx.context._add_to_circular_buffer("decay_events", {
                            "timestamp": now.isoformat(),
                            "chemical": chemical,
                            "old_value": old_value,
                            "new_value": new_value,
                            "decay_amount": old_value - new_value,
                            "time_delta": time_delta
                        })

                emotional_state = {}
                if self._actual_derive_emotional_state_callable: # Use the stored callable
                    emotional_state = await self._actual_derive_emotional_state_callable(ctx)
                else:
                    emotional_state = {"Neutral": 0.5} # Fallback

                return ChemicalDecayOutput(
                    decay_applied=True,
                    neurochemical_state=new_values,
                    derived_emotions=emotional_state,
                    time_elapsed_hours=time_delta,
                    last_update=self.last_update.isoformat()
                )

    @staticmethod
    @function_tool(
        name_override="apply_chemical_decay",
        description_override="Apply decay to all neurochemicals based on time elapsed and decay rates",
        failure_error_function=neurochemical_error_handler
    )
    async def apply_chemical_decay(
        ctx: RunContextWrapper[EmotionalContext]
    ) -> ChemicalDecayOutput:
        """
        Apply decay to all neurochemicals based on time elapsed and decay rates.

        Args:
            ctx: Run context wrapper with emotional state
        """
        instance = ctx.context.get_value("neurochemical_tools_instance")
        if not instance:
            raise UserError("No NeurochemicalTools instance found in context.")
        return await instance._apply_chemical_decay_impl(ctx)

    # -------------------------------------------------------------------------
    # 3) process_chemical_interactions
    # -------------------------------------------------------------------------
    async def _process_chemical_interactions_impl(
        self,
        ctx: RunContextWrapper[EmotionalContext],
        source_chemical: str,
        source_delta: float
    ) -> NeurochemicalInteractionOutput:
        with function_span("process_chemical_interactions", input=f"{source_chemical}:{source_delta}"):
            if source_chemical not in self.chemical_interactions or abs(source_delta) < 0.01:
                return NeurochemicalInteractionOutput(
                    source_chemical=source_chemical,
                    source_delta=source_delta,
                    changes={}
                )
            with custom_span(
                "chemical_interactions",
                data={ # Stringified
                    "source_chemical": str(source_chemical),
                    "source_delta": str(source_delta),
                    "cycle": str(ctx.context.cycle_count)
                }
            ):
                changes = {}
                target_interactions = self.chemical_interactions[source_chemical]
                affected_chemicals = [
                    (chem, source_delta * multiplier)
                    for chem, multiplier in target_interactions.items()
                    if chem in self.neurochemicals and abs(source_delta * multiplier) >= 0.01
                ]

                for chemical, effect in affected_chemicals:
                    old_value = self.neurochemicals[chemical]["value"]
                    new_value = max(0, min(1, old_value + effect))
                    self.neurochemicals[chemical]["value"] = new_value

                    changes[chemical] = {
                        "old_value": old_value,
                        "new_value": new_value,
                        "change": new_value - old_value
                    }

                    if abs(new_value - old_value) > 0.05:
                        with custom_span(
                            "chemical_interaction_effect",
                            data={ # Stringified
                                "source": str(source_chemical),
                                "target": str(chemical),
                                "effect": str(effect),
                                "old_value": str(old_value),
                                "new_value": str(new_value),
                                "cycle": str(ctx.context.cycle_count)
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

                ctx.context.record_neurochemical_values({
                    c: d["value"] for c, d in self.neurochemicals.items()
                })

                return NeurochemicalInteractionOutput(
                    source_chemical=source_chemical,
                    source_delta=source_delta,
                    changes=changes
                )

    @staticmethod
    @function_tool(
        name_override="process_chemical_interactions",
        description_override="Process interactions between neurochemicals when one changes",
        failure_error_function=neurochemical_error_handler
    )
    async def process_chemical_interactions(
        ctx: RunContextWrapper[EmotionalContext],
        source_chemical: str,
        source_delta: float
    ) -> NeurochemicalInteractionOutput:
        """
        Process interactions between neurochemicals when one changes.

        Args:
            ctx: Run context wrapper with emotional state
            source_chemical: The neurochemical that changed
            source_delta: The amount it changed by
        """
        instance = ctx.context.get_value("neurochemical_tools_instance")
        if not instance:
            raise UserError("No NeurochemicalTools instance found in context.")
        return await instance._process_chemical_interactions_impl(ctx, source_chemical, source_delta)

    # -------------------------------------------------------------------------
    # 4) get_neurochemical_state
    # -------------------------------------------------------------------------
    @staticmethod
    async def _get_neurochemical_state_impl(
        ctx: RunContextWrapper[EmotionalContext]
    ) -> Dict[str, Any]:
        with function_span("get_neurochemical_state"):
            cached_state = ctx.context.get_cached_neurochemicals(max_age_seconds=1.0)
            if cached_state is not None:
                with custom_span(
                    "cached_neurochemical_state",
                    data={ # Stringified
                        "chemicals": json.dumps({k: round(v, 2) for k, v in cached_state.items()}),
                        "cache_age": str(datetime.datetime.now().timestamp() - ctx.context.temp_data.get("cached_time", 0)),
                        "cycle": str(ctx.context.cycle_count)
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

            # Apply decay if needed
            if hasattr(self, "_apply_chemical_decay_impl"): # Check if method exists
                 await self._apply_chemical_decay_impl(ctx)

            with custom_span(
                "fresh_neurochemical_state",
                data={ # Stringified
                    "chemicals": json.dumps({c: round(d["value"], 2) for c, d in self.neurochemicals.items()}),
                    "cycle": str(ctx.context.cycle_count)
                }
            ):
                state = {c: d["value"] for c, d in self.neurochemicals.items()}
                ctx.context.record_neurochemical_values(state)

                chemical_activity = []
                for chemical, data in self.neurochemicals.items():
                    baseline = data["baseline"]
                    current = data["value"]
                    deviation = current - baseline
                    if abs(deviation) > 0.1:
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

    @staticmethod
    @function_tool(
        name_override="get_neurochemical_state",
        description_override="Get the current neurochemical state",
        failure_error_function=neurochemical_error_handler
    )
    async def get_neurochemical_state(
        ctx: RunContextWrapper[EmotionalContext]
    ) -> Dict[str, Any]:
        """
        Get the current neurochemical state.

        Args:
            ctx: Run context wrapper with emotional state
        """
        instance = ctx.context.get_value("neurochemical_tools_instance")
        if not instance:
            raise UserError("No NeurochemicalTools instance found in context.")
        return await instance._get_neurochemical_state_impl(ctx)

    # -------------------------------------------------------------------------
    # 5) analyze_neurochemical_state
    # -------------------------------------------------------------------------
    @staticmethod
    async def _analyze_neurochemical_state_impl(
        ctx: RunContextWrapper[EmotionalContext]
    ) -> Dict[str, Any]:
        with function_span("analyze_neurochemical_state"):
            with trace(
                workflow_name="Neurochemical_Analysis",
                trace_id=gen_trace_id(),
                group_id=ctx.context.get_value("conversation_id", "default"),
                metadata={ # Stringified
                    "cycle": str(ctx.context.cycle_count),
                    "timestamp": str(datetime.datetime.now().isoformat())
                }
            ):
                current_state = ctx.context.get_cached_neurochemicals()
                if not current_state:
                    current_state = {c: d["value"] for c, d in self.neurochemicals.items()}

                baselines = {c: d["baseline"] for c, d in self.neurochemicals.items()}
                deviations = {c: current_state[c] - baselines[c] for c in current_state}

                dominant_chemicals = sorted(
                    [(c, d) for c, d in deviations.items() if d > 0.1],
                    key=lambda x: x[1],
                    reverse=True
                )
                suppressed_chemicals = sorted(
                    [(c, d) for c, d in deviations.items() if d < -0.1],
                    key=lambda x: x[1]
                )

                excitation_index = current_state.get("nyxamine", 0) + current_state.get("adrenyx", 0)
                calm_index = current_state.get("seranix", 0) + current_state.get("oxynixin", 0)
                stress_index = current_state.get("cortanyx", 0) + current_state.get("adrenyx", 0)

                if excitation_index > 0 and calm_index > 0:
                    balance_ratio = excitation_index / calm_index
                else:
                    balance_ratio = 1.0

                if balance_ratio > 1.5:
                    system_state = "overstimulated"
                elif balance_ratio < 0.5:
                    system_state = "subdued"
                else:
                    system_state = "balanced"

                trends = ctx.context.get_neurochemical_trends(limit=5)
                patterns = {}
                if trends:
                    for chemical, trend_data in trends.items():
                        if len(trend_data) >= 3:
                            recent_values = [point["value"] for point in trend_data[-3:]]
                            # consistent increase
                            if all(recent_values[i] < recent_values[i+1] for i in range(len(recent_values)-1)):
                                patterns[chemical] = "consistently_increasing"
                            # consistent decrease
                            elif all(recent_values[i] > recent_values[i+1] for i in range(len(recent_values)-1)):
                                patterns[chemical] = "consistently_decreasing"
                            # oscillation
                            elif ((recent_values[0] > recent_values[1] and recent_values[1] < recent_values[2]) or
                                  (recent_values[0] < recent_values[1] and recent_values[1] > recent_values[2])):
                                patterns[chemical] = "oscillating"
                            # plateau
                            elif abs(recent_values[0] - recent_values[-1]) < 0.05:
                                patterns[chemical] = "stable"
                            else:
                                patterns[chemical] = "variable"

                with custom_span(
                    "neurochemical_analysis_result",
                    data={ # Stringified
                        "dominant_chemicals": json.dumps(dominant_chemicals[:2] if dominant_chemicals else []),
                        "system_state": str(system_state),
                        "balance_ratio": str(round(balance_ratio, 2)),
                        "stress_index": str(round(stress_index, 2)),
                        "cycle": str(ctx.context.cycle_count)
                    }
                ):
                    return {
                        "dominant_chemicals": [
                            {"chemical": c, "deviation": d} for c, d in dominant_chemicals[:3]
                        ] if dominant_chemicals else [],
                        "suppressed_chemicals": [
                            {"chemical": c, "deviation": d} for c, d in suppressed_chemicals[:3]
                        ] if suppressed_chemicals else [],
                        "system_state": system_state,
                        "balance_ratio": balance_ratio,
                        "excitation_index": excitation_index,
                        "calm_index": calm_index,
                        "stress_index": stress_index,
                        "patterns": patterns,
                        "timestamp": datetime.datetime.now().isoformat()
                    }

    @staticmethod
    @function_tool(
        name_override="neurochemical_analysis",
        description_override="Analyze current neurochemical state for patterns and imbalances",
        failure_error_function=neurochemical_error_handler
    )
    async def analyze_neurochemical_state(
        ctx: RunContextWrapper[EmotionalContext]
    ) -> Dict[str, Any]:
        """
        Analyze the current neurochemical state for patterns and imbalances.

        Args:
            ctx: Run context wrapper with emotional state
        """
        instance = ctx.context.get_value("neurochemical_tools_instance")
        if not instance:
            raise UserError("No NeurochemicalTools instance found in context.")
        return await instance._analyze_neurochemical_state_impl(ctx)

    # -------------------------------------------------------------------------
    # 6) reset_neurochemicals
    # -------------------------------------------------------------------------
    @staticmethod
    async def _reset_neurochemicals_impl(
        ctx: RunContextWrapper[EmotionalContext]
    ) -> Dict[str, Any]:
        with function_span("reset_neurochemicals"):
            with trace(
                workflow_name="Neurochemical_Reset",
                trace_id=gen_trace_id(),
                group_id=ctx.context.get_value("conversation_id", "default"),
                metadata={ # Stringified
                    "cycle": str(ctx.context.cycle_count),
                    "timestamp": str(datetime.datetime.now().isoformat()),
                    "reason": "manual_reset"
                }
            ):
                original_values = {c: d["value"] for c, d in self.neurochemicals.items()}
                for chemical, data in self.neurochemicals.items():
                    self.neurochemicals[chemical]["value"] = data["baseline"]

                new_values = {c: d["value"] for c, d in self.neurochemicals.items()}
                ctx.context.record_neurochemical_values(new_values)

                ctx.context._add_to_circular_buffer("system_events", {
                    "event": "neurochemical_reset",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "original_values": original_values,
                    "new_values": new_values,
                    "cycle": ctx.context.cycle_count
                })
                self.last_update = datetime.datetime.now()

                with custom_span(
                    "neurochemical_reset_result",
                    data={ # Stringified
                        "chemicals_reset": json.dumps(list(self.neurochemicals.keys())),
                        "cycle": str(ctx.context.cycle_count)
                    }
                ):
                    emotional_state = {}
                    if self._actual_derive_emotional_state_callable: # Use the stored callable
                        emotional_state = await self._actual_derive_emotional_state_callable(ctx)
                    else:
                        emotional_state = {"Neutral": 0.5}
                    
                    ctx.context.last_emotions = emotional_state
                    
                    return {
                        "success": True,
                        "original_values": original_values,
                        "new_values": new_values,
                        "derived_emotions": emotional_state,
                        "timestamp": datetime.datetime.now().isoformat()
                    }

    @staticmethod
    @function_tool(
        name_override="reset_neurochemicals",
        description_override="Reset neurochemicals to their baseline values",
        failure_error_function=neurochemical_error_handler
    )
    async def reset_neurochemicals(
        ctx: RunContextWrapper[EmotionalContext]
    ) -> Dict[str, Any]:
        """
        Reset all neurochemicals to their baseline values.

        Args:
            ctx: Run context wrapper with emotional state
        """
        instance = ctx.context.get_value("neurochemical_tools_instance")
        if not instance:
            raise UserError("No NeurochemicalTools instance found in context.")
        return await instance._reset_neurochemicals_impl(ctx)
