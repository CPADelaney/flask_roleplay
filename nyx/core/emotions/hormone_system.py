# nyx/core/emotions/hormone_system.py

"""
Digital hormone system for the Nyx emotional core.
Handles longer-term emotional effects and cycles.
"""

import datetime
import logging
import math
from typing import Dict, Any, Optional, List

from agents import function_tool, RunContextWrapper, function_span
from agents.exceptions import UserError

from nyx.core.emotions.context import EmotionalContext
from nyx.core.emotions.utils import handle_errors
from nyx.core.emotions.schemas import DigitalHormone

logger = logging.getLogger(__name__)

class HormoneSystem:
    """Digital hormone system for longer-term emotional effects"""
    
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
            }
        }
        
        # Hormone-neurochemical influence matrix
        self.hormone_neurochemical_influences = {
            "endoryx": {
                "nyxamine": 0.4,    # Endoryx boosts nyxamine
                "cortanyx": -0.3,   # Endoryx reduces cortanyx
            },
            "estradyx": {
                "oxynixin": 0.5,    # Estradyx boosts oxynixin
                "seranix": 0.3,     # Estradyx boosts seranix
            },
            "testoryx": {
                "adrenyx": 0.4,     # Testoryx boosts adrenyx
                "oxynixin": -0.2,   # Testoryx reduces oxynixin
            },
            "melatonyx": {
                "seranix": 0.5,     # Melatonyx boosts seranix
                "adrenyx": -0.4,    # Melatonyx reduces adrenyx
            },
            "oxytonyx": {
                "oxynixin": 0.7,    # Oxytonyx strongly boosts oxynixin
                "cortanyx": -0.4,   # Oxytonyx reduces cortanyx
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
    
    @handle_errors("Error updating hormone")
    async def update_hormone(self, hormone: str, change: float, source: str = "system") -> Dict[str, Any]:
        """
        Update a specific hormone with a delta change
        
        Args:
            hormone: Hormone to update
            change: Delta change value
            source: Source of the change
            
        Returns:
            Update result
        """
        with function_span("update_hormone", input=f"{hormone}:{change}"):
            if hormone not in self.hormones:
                raise UserError(f"Unknown hormone: {hormone}")
            
            # Get pre-update value
            old_value = self.hormones[hormone]["value"]
            
            # Calculate new value with bounds checking
            new_value = max(0.0, min(1.0, old_value + change))
            self.hormones[hormone]["value"] = new_value
            
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
            
            return {
                "success": True,
                "hormone": hormone,
                "old_value": old_value,
                "new_value": new_value,
                "change": change,
                "source": source
            }
    
    @function_tool
    @handle_errors("Error updating hormone cycles")
    async def update_hormone_cycles(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Update hormone cycles based on elapsed time and environmental factors
        
        Returns:
            Updated hormone values
        """
        with function_span("update_hormone_cycles"):
            now = datetime.datetime.now()
            updated_values = {}
            
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
            
            # After updating hormones, update their influence on neurochemicals
            await self._update_hormone_influences(ctx)
            
            return {
                "updated_hormones": updated_values,
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
    
    @function_tool
    @handle_errors("Error updating hormone influences")
    async def _update_hormone_influences(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Update neurochemical influences from hormones
        
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
            
            # Apply the accumulated influences
            for chemical, influence in influences.items():
                if chemical in self.emotional_core.neurochemicals:
                    # Get original baseline
                    original_baseline = self.emotional_core.neurochemicals[chemical]["baseline"]
                    
                    # Add temporary hormone influence with bounds checking
                    temporary_baseline = max(0.1, min(0.9, original_baseline + influence))
                    
                    # Record influence but don't permanently change baseline
                    self.emotional_core.hormone_influences[chemical] = influence
                    self.emotional_core.neurochemicals[chemical]["temporary_baseline"] = temporary_baseline
            
            # Store in context for tracking
            ctx.context.set_value("hormone_influences", {
                chemical: influence for chemical, influence in influences.items() if abs(influence) > 0.01
            })
            
            return {
                "applied_influences": influences
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
    def update_environmental_factor(self, factor: str, value: float) -> Dict[str, Any]:
        """
        Update an environmental factor
        
        Args:
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
        
        return {
            "success": True,
            "factor": factor,
            "old_value": old_value,
            "new_value": self.environmental_factors[factor]
        }
    
    @function_tool
    @handle_errors("Error getting hormone phase data")
    async def get_hormone_circadian_info(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Get detailed information about hormone circadian rhythms and phases
        
        Returns:
            Detailed hormone phase information
        """
        with function_span("get_hormone_circadian_info"):
            # Update cycles to ensure current data
            await self.update_hormone_cycles(ctx)
            
            # Get the current time for reference
            now = datetime.datetime.now()
            hour_of_day = now.hour + (now.minute / 60.0)
            
            # Calculate circadian time (0-24 hour scale normalized to 0-1)
            circadian_time = (hour_of_day / 24.0) % 1.0
            
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
            
            return {
                "circadian_time": circadian_time,
                "hour_of_day": hour_of_day,
                "phase_relationships": phase_relationships,
                "environmental_factors": self.environmental_factors
            }
    
    @function_tool
    @handle_errors("Error calculating hormone stability")
    async def get_hormone_stability(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Calculate the stability of the hormone system
        
        Returns:
            Stability metrics for the hormone system
        """
        with function_span("get_hormone_stability"):
            stability_scores = {}
            overall_volatility = 0.0
            
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
            
            return {
                "hormone_stability": stability_scores,
                "system_stability": system_stability,
                "assessment": self._interpret_stability(system_stability)
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
