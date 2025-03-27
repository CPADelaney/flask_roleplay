# nyx/core/integration/dominance_imagination_decision_bridge.py

import logging
import asyncio
import datetime
import random
from typing import Dict, List, Any, Optional, Tuple

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method

logger = logging.getLogger(__name__)

class DominanceImaginationDecisionBridge:
    """
    Integrates dominance expression with imagination simulation and decision making.
    Allows Nyx to simulate potential dominance scenarios, predict outcomes,
    and make better decisions about when and how to express dominance.
    """
    
    def __init__(self, 
                dominance_system=None,
                imagination_simulator=None,
                theory_of_mind=None,
                relationship_manager=None):
        """Initialize the bridge."""
        self.dominance_system = dominance_system
        self.imagination_simulator = imagination_simulator
        self.theory_of_mind = theory_of_mind
        self.relationship_manager = relationship_manager
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Simulation configuration
        self.min_confidence_threshold = 0.6  # Minimum confidence for decision
        self.dominance_caution_threshold = 0.7  # Higher threshold for risky scenarios
        self.simulation_lookback = 5  # Store last 5 simulations per user
        
        # Store simulation results
        self.user_simulation_history = {}  # user_id -> list of simulation results
        
        # Integration event subscriptions
        self._subscribed = False
        
        logger.info("DominanceImaginationDecisionBridge initialized")
    
    async def initialize(self) -> bool:
        """Initialize the bridge and establish connections to systems."""
        try:
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("dominance_opportunity", self._handle_dominance_opportunity)
                self._subscribed = True
            
            logger.info("DominanceImaginationDecisionBridge successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing DominanceImaginationDecisionBridge: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="DominanceImaginationDecision")
    async def simulate_dominance_outcome(self,
                                      user_id: str,
                                      dominance_action: str,
                                      intensity: float,
                                      relationship_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Simulate the potential outcome of a dominance action using the imagination
        simulator to predict user response.
        
        Args:
            user_id: User ID
            dominance_action: Type of dominance action
            intensity: Dominance intensity (0.0-1.0)
            relationship_context: Additional relationship context
            
        Returns:
            Simulation results with predicted outcome
        """
        if not self.imagination_simulator:
            return {"status": "error", "message": "Imagination simulator not available"}
        
        try:
            # Get relationship data
            relationship_data = {}
            if self.relationship_manager:
                relationship = await self.relationship_manager.get_relationship_state(user_id)
                if relationship:
                    relationship_data = {
                        "trust": relationship.trust,
                        "familiarity": relationship.familiarity,
                        "intimacy": relationship.intimacy,
                        "conflict": relationship.conflict,
                        "dominance_balance": relationship.dominance_balance,
                        "current_dominance_intensity": relationship.current_dominance_intensity,
                        "max_achieved_intensity": relationship.max_achieved_intensity,
                        "user_stated_intensity_preference": relationship.user_stated_intensity_preference
                    }
            
            # Create simulation description
            simulation_description = f"What if Nyx performs a dominance action '{dominance_action}' at intensity level {intensity:.2f} on user {user_id}?"
            
            # Prepare additional context data
            context_data = relationship_data.copy()
            if relationship_context:
                context_data.update(relationship_context)
                
            # Add dominance-specific context
            context_data["dominance_action"] = dominance_action
            context_data["dominance_intensity"] = intensity
            context_data["is_escalation"] = intensity > context_data.get("current_dominance_intensity", 0)
            context_data["escalation_size"] = intensity - context_data.get("current_dominance_intensity", 0)
            
            # Run simulation
            sim_input = await self.imagination_simulator.setup_simulation(
                description=simulation_description,
                current_brain_state=context_data
            )
            
            if not sim_input:
                return {"status": "error", "message": "Failed to create simulation input"}
                
            sim_result = await self.imagination_simulator.run_simulation(sim_input)
            
            # Process simulation result
            if not sim_result:
                return {"status": "error", "message": "Simulation failed"}
                
            # Store result in history
            if user_id not in self.user_simulation_history:
                self.user_simulation_history[user_id] = []
                
            self.user_simulation_history[user_id].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "action": dominance_action,
                "intensity": intensity,
                "outcome": sim_result.predicted_outcome,
                "confidence": sim_result.confidence,
                "emotional_impact": sim_result.emotional_impact
            })
            
            # Limit history size
            if len(self.user_simulation_history[user_id]) > self.simulation_lookback:
                self.user_simulation_history[user_id] = self.user_simulation_history[user_id][-self.simulation_lookback:]
            
            # Analyze the result for decision making
            decision = await self._analyze_simulation_result(
                sim_result=sim_result,
                user_id=user_id,
                dominance_action=dominance_action,
                intensity=intensity,
                relationship_data=relationship_data
            )
            
            return {
                "status": "success",
                "simulation_id": sim_result.simulation_id,
                "predicted_outcome": sim_result.predicted_outcome,
                "confidence": sim_result.confidence,
                "emotional_impact": sim_result.emotional_impact,
                "decision": decision,
                "decision_reasoning": decision.get("reasoning", "")
            }
            
        except Exception as e:
            logger.error(f"Error simulating dominance outcome: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _analyze_simulation_result(self,
                                      sim_result: Any,
                                      user_id: str,
                                      dominance_action: str,
                                      intensity: float,
                                      relationship_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze simulation results to make a decision about the dominance action.
        
        Args:
            sim_result: Simulation result
            user_id: User ID
            dominance_action: Type of dominance action
            intensity: Dominance intensity
            relationship_data: Relationship context
            
        Returns:
            Decision about whether to proceed
        """
        # Default decision
        decision = {
            "proceed": False,
            "reasoning": "Insufficient data to make decision",
            "confidence": 0.0,
            "alternative_suggestion": None
        }
        
        # Extract prediction and confidence
        prediction = sim_result.predicted_outcome
        confidence = sim_result.confidence
        
        # Check if prediction contains negative outcomes
        negative_indicators = ["resist", "refus", "uncomfort", "upset", "boundary", "limit", "distress"]
        negative_outcome = any(indicator in str(prediction).lower() for indicator in negative_indicators)
        
        # Check if prediction contains positive outcomes
        positive_indicators = ["comply", "submit", "obey", "accept", "enjoy", "pleasure", "willing"]
        positive_outcome = any(indicator in str(prediction).lower() for indicator in positive_indicators)
        
        # Calculate baseline confidence threshold
        threshold = self.min_confidence_threshold
        
        # Adjust threshold based on action and intensity
        is_escalation = intensity > relationship_data.get("current_dominance_intensity", 0)
        escalation_size = intensity - relationship_data.get("current_dominance_intensity", 0)
        
        if is_escalation and escalation_size > 0.2:
            # Higher threshold for significant escalations
            threshold = self.dominance_caution_threshold
            
        # Make decision
        if negative_outcome and confidence >= threshold:
            # Predict negative outcome with sufficient confidence
            decision["proceed"] = False
            decision["reasoning"] = f"Simulation predicts negative outcome: {prediction}"
            decision["confidence"] = confidence
            
            # Suggest alternative
            if escalation_size > 0.2:
                # Suggest smaller escalation
                decision["alternative_suggestion"] = {
                    "action": dominance_action,
                    "intensity": relationship_data.get("current_dominance_intensity", 0) + 0.1
                }
        elif positive_outcome and confidence >= threshold:
            # Predict positive outcome with sufficient confidence
            decision["proceed"] = True
            decision["reasoning"] = f"Simulation predicts positive outcome: {prediction}"
            decision["confidence"] = confidence
        elif confidence < threshold:
            # Insufficient confidence
            decision["proceed"] = False
            decision["reasoning"] = f"Insufficient confidence in prediction: {confidence:.2f} < {threshold:.2f}"
            
            # If previously successful at this level, may still proceed
            max_achieved = relationship_data.get("max_achieved_intensity", 0)
            if intensity <= max_achieved:
                decision["proceed"] = True
                decision["reasoning"] = f"Proceeding despite low confidence because intensity {intensity:.2f} is below maximum achieved {max_achieved:.2f}"
                decision["confidence"] = 0.5  # Medium confidence based on history
        else:
            # Ambiguous outcome
            trust_level = relationship_data.get("trust", 0.5)
            
            # More cautious with lower trust
            if trust_level < 0.4:
                decision["proceed"] = False
                decision["reasoning"] = f"Ambiguous outcome prediction with low trust level ({trust_level:.2f})"
            else:
                # Proceed cautiously with medium-high trust
                decision["proceed"] = True
                decision["reasoning"] = f"Proceeding with ambiguous outcome due to adequate trust level ({trust_level:.2f})"
                decision["confidence"] = 0.5  # Medium confidence
        
        return decision
    
    @trace_method(level=TraceLevel.INFO, group_id="DominanceImaginationDecision")
    async def get_dominance_recommendation(self,
                                        user_id: str,
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a recommendation for dominance action based on user model, relationship 
        state, and simulated outcomes.
        
        Args:
            user_id: User ID
            context: Current interaction context
            
        Returns:
            Recommendation for dominance action
        """
        if not self.dominance_system or not self.theory_of_mind:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # Get user model
            user_model = await self.theory_of_mind.get_user_model(user_id)
            
            # Get relationship state
            relationship_state = None
            if self.relationship_manager:
                relationship_state = await self.relationship_manager.get_relationship_state(user_id)
            
            # Prepare context for dominance recommendation
            dominance_context = {
                "user_model": user_model,
                "relationship_state": relationship_state.model_dump() if relationship_state else None,
                "interaction_context": context,
                "recent_simulations": self.user_simulation_history.get(user_id, [])
            }
            
            # Generate dominance ideas
            dominance_ideas = []
            intensity_range = "1-5"  # Default gentle range
            
            # Adjust intensity range based on relationship
            if relationship_state:
                # If user has stated preference, use that
                if relationship_state.user_stated_intensity_preference:
                    pref = relationship_state.user_stated_intensity_preference
                    if isinstance(pref, int):
                        intensity_range = f"{pref}-{min(10, pref + 2)}"
                    elif isinstance(pref, str) and "-" in pref:
                        intensity_range = pref
                # Otherwise base on max achieved
                else:
                    max_achieved = relationship_state.max_achieved_intensity
                    if max_achieved > 0:
                        # Suggest slightly above max achieved (careful escalation)
                        intensity_range = f"{max(1, int(max_achieved * 10) - 1)}-{min(10, int(max_achieved * 10) + 1)}"
            
            # Get dominance ideas from dominance system
            if hasattr(self.dominance_system, "generate_dominance_ideas"):
                ideas_result = await self.dominance_system.generate_dominance_ideas(
                    user_id=user_id,
                    purpose="interaction",
                    intensity_range=intensity_range,
                    hard_mode=False
                )
                
                dominance_ideas = ideas_result.get("ideas", [])
            
            # Filter ideas based on relationship state and simulate outcomes
            recommended_ideas = []
            for idea in dominance_ideas:
                # Simulate outcome
                simulation = await self.simulate_dominance_outcome(
                    user_id=user_id,
                    dominance_action=idea.description if hasattr(idea, "description") else str(idea),
                    intensity=idea.intensity if hasattr(idea, "intensity") else 0.5,
                    relationship_context=context
                )
                
                # Check if recommended
                if simulation.get("decision", {}).get("proceed", False):
                    recommended_ideas.append({
                        "description": idea.description if hasattr(idea, "description") else str(idea),
                        "category": idea.category if hasattr(idea, "category") else "unknown",
                        "intensity": idea.intensity if hasattr(idea, "intensity") else 0.5,
                        "confidence": simulation.get("confidence", 0.5),
                        "predicted_outcome": simulation.get("predicted_outcome", "Unknown"),
                        "reasoning": simulation.get("decision", {}).get("reasoning", "")
                    })
            
            # If no recommended ideas, get one safe suggestion
            if not recommended_ideas and dominance_ideas:
                # Sort by intensity (lowest first) and take first
                sorted_ideas = sorted(dominance_ideas, key=lambda x: x.intensity if hasattr(x, "intensity") else 0.5)
                if sorted_ideas:
                    idea = sorted_ideas[0]
                    recommended_ideas.append({
                        "description": idea.description if hasattr(idea, "description") else str(idea),
                        "category": idea.category if hasattr(idea, "category") else "unknown",
                        "intensity": idea.intensity if hasattr(idea, "intensity") else 0.5,
                        "confidence": 0.5,  # Medium confidence
                        "predicted_outcome": "Unknown but low intensity suggestion",
                        "reasoning": "Fallback suggestion at low intensity"
                    })
            
            # Return recommendation
            return {
                "status": "success",
                "user_id": user_id,
                "has_recommendation": len(recommended_ideas) > 0,
                "recommended_ideas": recommended_ideas,
                "intensity_range_used": intensity_range,
                "ideas_considered": len(dominance_ideas)
            }
            
        except Exception as e:
            logger.error(f"Error getting dominance recommendation: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _handle_dominance_opportunity(self, event: Event) -> None:
        """
        Handle dominance opportunity events from the event bus.
        
        Args:
            event: Dominance opportunity event
        """
        try:
            # Extract event data
            user_id = event.data.get("user_id")
            context = event.data.get("context", {})
            
            if not user_id:
                return
            
            # Generate recommendation
            asyncio.create_task(self.get_dominance_recommendation(
                user_id=user_id,
                context=context
            ))
            
        except Exception as e:
            logger.error(f"Error handling dominance opportunity event: {e}")

# Function to create the bridge
def create_dominance_imagination_decision_bridge(nyx_brain):
    """Create a dominance-imagination-decision bridge for the given brain."""
    return DominanceImaginationDecisionBridge(
        dominance_system=nyx_brain.dominance_system if hasattr(nyx_brain, "dominance_system") else None,
        imagination_simulator=nyx_brain.imagination_simulator if hasattr(nyx_brain, "imagination_simulator") else None,
        theory_of_mind=nyx_brain.theory_of_mind if hasattr(nyx_brain, "theory_of_mind") else None,
        relationship_manager=nyx_brain.relationship_manager if hasattr(nyx_brain, "relationship_manager") else None
    )
