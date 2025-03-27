# nyx/core/integration/identity_imagination_emotional_bridge.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Tuple

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method

logger = logging.getLogger(__name__)

class IdentityImaginationEmotionalBridge:
    """
    Integrates identity evolution with imagination and emotional systems.
    
    This bridge enables:
    1. Identity-consistent imagination scenarios
    2. Emotional responses shaped by identity traits
    3. Identity exploration through imaginative simulation
    4. Emotional feedback loops for identity reinforcement
    5. Cross-module coherence between identity, emotions, and imagination
    """
    
    def __init__(self, 
                identity_evolution=None,
                imagination_simulator=None,
                emotional_core=None,
                memory_orchestrator=None):
        """Initialize the identity-imagination-emotional bridge."""
        self.identity_evolution = identity_evolution
        self.imagination_simulator = imagination_simulator
        self.emotional_core = emotional_core
        self.memory_orchestrator = memory_orchestrator
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Integration parameters
        self.identity_expression_threshold = 0.7  # Minimum trait strength for expression
        self.emotional_influence_factor = 0.5    # How strongly emotions affect imagination
        self.imagination_feedback_factor = 0.3   # How strongly imagination affects identity
        
        # Integration state tracking
        self.recent_simulations = {}  # Tracking recent simulations
        self.emotional_correlations = {}  # Correlations between emotions and traits
        self._subscribed = False
        
        logger.info("IdentityImaginationEmotionalBridge initialized")
    
    async def initialize(self) -> bool:
        """Initialize the bridge and establish connections to systems."""
        try:
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("emotional_state_change", self._handle_emotional_change)
                self.event_bus.subscribe("simulation_completed", self._handle_simulation_completed)
                self.event_bus.subscribe("identity_updated", self._handle_identity_updated)
                self._subscribed = True
            
            # Initial correlation mapping between traits and emotions
            await self._build_trait_emotion_correlations()
            
            logger.info("IdentityImaginationEmotionalBridge successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing IdentityImaginationEmotionalBridge: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="IdentityImaginationEmotional")
    async def generate_identity_exploration_simulation(self, 
                                                    trait_name: str,
                                                    exploration_type: str = "expression",
                                                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an imagination simulation to explore a specific identity trait.
        
        Args:
            trait_name: The trait to explore
            exploration_type: How to explore the trait ("expression", "growth", "conflict")
            context: Optional additional context
            
        Returns:
            Simulation results
        """
        if not self.identity_evolution or not self.imagination_simulator:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # 1. Get identity profile
            identity_profile = await self.identity_evolution.get_identity_profile()
            
            # Check if trait exists
            if trait_name not in identity_profile.get("traits", {}):
                return {
                    "status": "error", 
                    "message": f"Trait '{trait_name}' not found in identity profile"
                }
            
            trait_value = identity_profile["traits"][trait_name]
            
            # 2. Build simulation description based on exploration type
            simulation_description = ""
            brain_state = {
                "identity_traits": identity_profile["traits"],
                "preferences": identity_profile.get("preferences", {})
            }
            
            if context:
                brain_state.update(context)
            
            if exploration_type == "expression":
                # Simulate expressing this trait
                simulation_description = f"How would I express my {trait_name} trait (value {trait_value:.2f}) in a conversation?"
                
            elif exploration_type == "growth":
                # Simulate growing this trait
                simulation_description = f"How might experiences strengthen my {trait_name} trait from its current value of {trait_value:.2f}?"
                
            elif exploration_type == "conflict":
                # Simulate conflict with this trait
                simulation_description = f"What situations might challenge or conflict with my {trait_name} trait (value {trait_value:.2f})?"
            
            # Add emotional context if available
            if self.emotional_core:
                emotional_state = await self.emotional_core.get_emotional_state_matrix()
                brain_state["emotional_state"] = emotional_state
                
                # Get correlated emotions
                related_emotions = await self._get_correlated_emotions(trait_name)
                if related_emotions:
                    brain_state["trait_related_emotions"] = related_emotions
            
            # 3. Run the simulation
            sim_input = await self.imagination_simulator.setup_simulation(
                description=simulation_description,
                current_brain_state=brain_state
            )
            
            if not sim_input:
                return {"status": "error", "message": "Failed to create simulation input"}
                
            sim_result = await self.imagination_simulator.run_simulation(sim_input)
            
            if not sim_result:
                return {"status": "error", "message": "Simulation failed"}
            
            # 4. Process simulation results for identity feedback
            identity_feedback = await self._extract_identity_feedback(
                sim_result, trait_name, trait_value
            )
            
            # 5. Apply subtle identity influence if significant insight
            if identity_feedback.get("confidence", 0) > 0.7:
                trait_impact = identity_feedback.get("suggested_impact", 0)
                if abs(trait_impact) > 0.05:
                    # Apply a subtle impact in the suggested direction
                    scaled_impact = trait_impact * self.imagination_feedback_factor
                    await self.identity_evolution.update_trait(trait_name, scaled_impact)
            
            # 6. Store the simulation for reference
            sim_id = getattr(sim_result, "simulation_id", f"sim_{datetime.datetime.now().timestamp()}")
            self.recent_simulations[sim_id] = {
                "trait": trait_name,
                "exploration_type": exploration_type,
                "timestamp": datetime.datetime.now().isoformat(),
                "feedback": identity_feedback
            }
            
            return {
                "status": "success",
                "simulation_id": sim_id,
                "trait": trait_name,
                "exploration_type": exploration_type,
                "outcome": sim_result.predicted_outcome,
                "identity_feedback": identity_feedback,
                "trait_impact_applied": abs(trait_impact) > 0.05 if 'trait_impact' in locals() else False
            }
            
        except Exception as e:
            logger.error(f"Error generating identity exploration simulation: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="IdentityImaginationEmotional")
    async def modulate_emotional_response(self, 
                                       stimulus: Dict[str, Any],
                                       emotional_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modulate emotional response based on identity traits.
        
        Args:
            stimulus: The stimulus triggering the emotional response
            emotional_response: Base emotional response
            
        Returns:
            Modulated emotional response
        """
        if not self.identity_evolution or not self.emotional_core:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # 1. Get identity profile
            identity_profile = await self.identity_evolution.get_identity_profile()
            traits = identity_profile.get("traits", {})
            
            # 2. Extract base emotional response
            base_emotion = emotional_response.get("emotion", "neutral")
            base_valence = emotional_response.get("valence", 0.0)
            base_arousal = emotional_response.get("arousal", 0.5)
            base_intensity = emotional_response.get("intensity", 0.5)
            
            # 3. Apply identity traits as modulators
            modulated_response = emotional_response.copy()
            applied_modulators = {}
            
            # Modulation rules based on trait-emotion correlations
            
            # Example: Dominance trait dampens negative emotions
            if "dominance" in traits and traits["dominance"] > 0.7:
                if base_valence < 0:
                    # Reduce negative valence
                    valence_mod = traits["dominance"] * 0.3
                    modulated_response["valence"] = min(1.0, base_valence + valence_mod)
                    applied_modulators["dominance_negative_dampen"] = valence_mod
            
            # Patience trait reduces arousal for negative stimuli
            if "patience" in traits and traits["patience"] > 0.6:
                if "negative" in stimulus.get("tags", []) or base_valence < -0.2:
                    arousal_mod = -(traits["patience"] * 0.4)  # Negative modifier to reduce arousal
                    modulated_response["arousal"] = max(0.1, base_arousal + arousal_mod)
                    applied_modulators["patience_arousal_reduction"] = arousal_mod
            
            # Emotional intensity trait increases all emotional intensities
            if "intensity" in traits and traits["intensity"] > 0.6:
                intensity_mod = traits["intensity"] * 0.3
                modulated_response["intensity"] = min(1.0, base_intensity + intensity_mod)
                applied_modulators["intensity_amplification"] = intensity_mod
            
            # Playfulness trait can transform some negative emotions
            if "playfulness" in traits and traits["playfulness"] > 0.7:
                if base_emotion in ["mild_frustration", "mild_annoyance"] and base_valence > -0.6:
                    # Transform to more playful emotion
                    modulated_response["emotion"] = "playful_challenge"
                    modulated_response["valence"] = max(0.1, base_valence + 0.4)
                    applied_modulators["playful_reframing"] = True
            
            # 4. Update system context with modulation info
            self.system_context.set_value("emotional_modulation", {
                "original": {
                    "emotion": base_emotion,
                    "valence": base_valence,
                    "arousal": base_arousal,
                    "intensity": base_intensity
                },
                "modulated": {
                    "emotion": modulated_response["emotion"],
                    "valence": modulated_response["valence"],
                    "arousal": modulated_response["arousal"],
                    "intensity": modulated_response["intensity"]
                },
                "modulators": applied_modulators,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            return {
                "status": "success",
                "original_response": {
                    "emotion": base_emotion,
                    "valence": base_valence,
                    "arousal": base_arousal,
                    "intensity": base_intensity
                },
                "modulated_response": {
                    "emotion": modulated_response["emotion"],
                    "valence": modulated_response["valence"],
                    "arousal": modulated_response["arousal"],
                    "intensity": modulated_response["intensity"]
                },
                "applied_modulators": applied_modulators
            }
            
        except Exception as e:
            logger.error(f"Error modulating emotional response: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="IdentityImaginationEmotional")  
    async def create_identity_driven_scenario(self,
                                           scenario_type: str,
                                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create an imagination scenario driven by identity traits.
        
        Args:
            scenario_type: Type of scenario to create
            context: Optional additional context
            
        Returns:
            Generated scenario
        """
        if not self.identity_evolution or not self.imagination_simulator:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # 1. Get identity profile
            identity_profile = await self.identity_evolution.get_identity_profile()
            
            # 2. Select dominant traits for scenario generation
            traits = identity_profile.get("traits", {})
            trait_items = [(name, value) for name, value in traits.items()]
            trait_items.sort(key=lambda x: x[1], reverse=True)
            dominant_traits = trait_items[:3]  # Top 3 traits
            
            # 3. Get preferences relevant to scenario type
            preferences = {}
            if scenario_type in identity_profile.get("preferences", {}):
                preferences = identity_profile["preferences"][scenario_type]
            
            # 4. Construct scenario parameters
            scenario_params = {
                "scenario_type": scenario_type,
                "dominant_traits": {name: value for name, value in dominant_traits},
                "preferences": preferences
            }
            
            # Add any additional context
            if context:
                scenario_params.update(context)
            
            # 5. Generate scenario description
            scenario_description = f"Create a {scenario_type} scenario that showcases my dominant traits: "
            scenario_description += ", ".join([f"{name} ({value:.2f})" for name, value in dominant_traits])
            
            # 6. Run the imagination simulation
            sim_input = await self.imagination_simulator.setup_simulation(
                description=scenario_description,
                current_brain_state=scenario_params
            )
            
            if not sim_input:
                return {"status": "error", "message": "Failed to create simulation input"}
                
            sim_result = await self.imagination_simulator.run_simulation(sim_input)
            
            if not sim_result:
                return {"status": "error", "message": "Simulation failed"}
            
            # 7. Finalize the scenario, extracting and enhancing from simulation
            scenario = {
                "title": f"Identity-driven {scenario_type} scenario",
                "description": sim_result.predicted_outcome,
                "dominant_traits": {name: value for name, value in dominant_traits},
                "preferences_applied": list(preferences.keys() if preferences else []),
                "confidence": sim_result.confidence
            }
            
            # 8. Store as memory if significant
            memory_id = None
            if self.memory_orchestrator and sim_result.confidence > 0.7:
                memory_text = f"Imagined {scenario_type} scenario: {scenario['description']}"
                memory_id = await self.memory_orchestrator.add_memory(
                    memory_text=memory_text,
                    memory_type="imagination",
                    significance=7,  # High significance
                    tags=["imagination", "identity", scenario_type],
                    metadata={
                        "traits": {name: value for name, value in dominant_traits},
                        "scenario_type": scenario_type,
                        "confidence": sim_result.confidence,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
                
                scenario["memory_id"] = memory_id
            
            return {
                "status": "success",
                "scenario": scenario,
                "simulation_id": getattr(sim_result, "simulation_id", None),
                "memory_created": memory_id is not None
            }
            
        except Exception as e:
            logger.error(f"Error creating identity-driven scenario: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _build_trait_emotion_correlations(self) -> None:
        """Build initial correlations between traits and emotions."""
        self.emotional_correlations = {
            # Trait to emotions
            "dominance": ["confident", "controlling", "powerful", "satisfied", "proud"],
            "playfulness": ["joyful", "excited", "amused", "curious", "mischievous"],
            "strictness": ["stern", "serious", "focused", "determined", "disciplined"],
            "creativity": ["inspired", "curious", "excited", "satisfied", "joyful"],
            "patience": ["calm", "content", "satisfied", "peaceful", "accepting"],
            "intensity": ["passionate", "excited", "aroused", "eager", "focused"],
            "cruelty": ["satisfied", "amused", "powerful", "dominant", "excited"],
            
            # Emotions to traits - inverse mapping
            "confident": ["dominance", "intensity"],
            "joyful": ["playfulness", "creativity"],
            "stern": ["strictness"],
            "curious": ["creativity", "playfulness"],
            "calm": ["patience"],
            "passionate": ["intensity", "creativity"],
            "amused": ["playfulness", "cruelty"],
            "excited": ["playfulness", "intensity", "creativity", "cruelty"]
        }
    
    async def _get_correlated_emotions(self, trait_name: str) -> List[str]:
        """Get emotions correlated with a trait."""
        return self.emotional_correlations.get(trait_name, [])
    
    async def _extract_identity_feedback(self, 
                                     simulation_result: Any, 
                                     trait_name: str,
                                     trait_value: float) -> Dict[str, Any]:
        """Extract identity feedback from simulation result."""
        # Default feedback
        feedback = {
            "insights": [],
            "confidence": getattr(simulation_result, "confidence", 0.5),
            "suggested_impact": 0.0
        }
        
        # Extract outcome
        outcome = getattr(simulation_result, "predicted_outcome", "")
        if isinstance(outcome, dict):
            # If structured result
            if "trait_insights" in outcome:
                feedback["insights"] = outcome["trait_insights"]
                
            if "confidence" in outcome:
                feedback["confidence"] = outcome["confidence"]
                
            if "suggested_trait_change" in outcome:
                feedback["suggested_impact"] = outcome["suggested_trait_change"]
                
        elif isinstance(outcome, str):
            # If text result, look for keywords
            
            # Keywords suggesting trait strengthening
            strengthen_keywords = ["reinforce", "strengthen", "increase", "enhance", "grow"]
            if any(keyword in outcome.lower() for keyword in strengthen_keywords):
                feedback["suggested_impact"] = 0.05  # Small positive impact
                
            # Keywords suggesting trait weakening
            weaken_keywords = ["weaken", "reduce", "decrease", "diminish", "lessen"]
            if any(keyword in outcome.lower() for keyword in weaken_keywords):
                feedback["suggested_impact"] = -0.05  # Small negative impact
        
        return feedback
    
    async def _handle_emotional_change(self, event: Event) -> None:
        """
        Handle emotional state change events.
        
        Args:
            event: Emotional state change event
        """
        try:
            # Extract event data
            emotion = event.data.get("emotion")
            
            if not emotion:
                return
            
            # Check for correlated traits
            correlated_traits = []
            for emotion_name, traits in self.emotional_correlations.items():
                if emotion_name.lower() in emotion.lower():
                    correlated_traits.extend(traits)
            
            # Update system context with correlation info
            self.system_context.set_value("emotion_trait_correlations", {
                "emotion": emotion,
                "correlated_traits": correlated_traits,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # If strong emotion with correlated traits, consider running simulation
            if event.data.get("intensity", 0.5) > 0.7 and correlated_traits:
                # Choose the first trait
                trait = correlated_traits[0]
                
                # Schedule simulation at low priority
                asyncio.create_task(
                    self.generate_identity_exploration_simulation(
                        trait_name=trait,
                        exploration_type="expression",
                        context={"triggered_by_emotion": emotion}
                    )
                )
        except Exception as e:
            logger.error(f"Error handling emotional change: {e}")
    
    async def _handle_simulation_completed(self, event: Event) -> None:
        """
        Handle simulation completed events.
        
        Args:
            event: Simulation completed event
        """
        try:
            # Extract event data
            simulation_id = event.data.get("simulation_id")
            outcome = event.data.get("outcome")
            
            if not simulation_id or not outcome:
                return
            
            # Only process if this is an identity-related simulation
            if not any(keyword in str(event.data.get("description", "")).lower() 
                      for keyword in ["identity", "trait", "preference", "personality"]):
                return
            
            # Store the simulation outcome for future reference
            self.system_context.set_value(f"simulation_outcome_{simulation_id}", {
                "outcome": outcome,
                "timestamp": datetime.datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error handling simulation completed: {e}")
    
    async def _handle_identity_updated(self, event: Event) -> None:
        """
        Handle identity updated events.
        
        Args:
            event: Identity updated event
        """
        try:
            # Extract event data
            trait = event.data.get("trait")
            impact = event.data.get("impact", 0.0)
            
            if not trait:
                return
            
            # Check if significant change
            if abs(impact) < 0.1:
                return
            
            # Get correlated emotions
            correlated_emotions = await self._get_correlated_emotions(trait)
            
            # If emotional core available, update emotional baselines
            if self.emotional_core and correlated_emotions:
                # Choose the first emotion
                emotion = correlated_emotions[0]
                
                # Adjust emotional baseline subtly
                if hasattr(self.emotional_core, 'adjust_emotion_baseline'):
                    await self.emotional_core.adjust_emotion_baseline(
                        emotion=emotion,
                        adjustment=impact * 0.3  # Scale appropriately
                    )
        except Exception as e:
            logger.error(f"Error handling identity updated: {e}")

# Function to create the bridge
def create_identity_imagination_emotional_bridge(nyx_brain):
    """Create an identity-imagination-emotional bridge for the given brain."""
    return IdentityImaginationEmotionalBridge(
        identity_evolution=nyx_brain.identity_evolution if hasattr(nyx_brain, "identity_evolution") else None,
        imagination_simulator=nyx_brain.imagination_simulator if hasattr(nyx_brain, "imagination_simulator") else None,
        emotional_core=nyx_brain.emotional_core if hasattr(nyx_brain, "emotional_core") else None,
        memory_orchestrator=nyx_brain.memory_orchestrator if hasattr(nyx_brain, "memory_orchestrator") else None
    )
