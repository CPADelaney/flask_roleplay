# nyx/core/imagination_simulator.py

import logging
import asyncio
import datetime
import random
import uuid
import json
from typing import Dict, List, Any, Optional, Sequence, Mapping
from pydantic import BaseModel, Field

from agents import Agent, Runner, ModelSettings, trace, function_tool, RunContextWrapper

logger = logging.getLogger(__name__)

class SimulationState(BaseModel):
    """Represents a state within a simulation."""
    step: int
    timestamp: datetime.datetime
    state_variables: Dict[str, Any] = Field(default_factory=dict)  # Key variables being tracked
    emotional_state: Optional[Dict[str, Any]] = None
    reasoning_focus: Optional[str] = None
    last_action: Optional[str] = None

class SimulationResult(BaseModel):
    """Result of running a simulation."""
    simulation_id: str
    success: bool
    termination_reason: str  # e.g., "max_steps", "goal_reached", "stable_state", "error"
    final_state: SimulationState
    trajectory: List[SimulationState] = Field(default_factory=list)
    predicted_outcome: Any = None  # What the simulation predicts will happen
    emotional_impact: Optional[Dict[str, float]] = None  # Predicted emotional consequence
    causal_analysis: Optional[Dict[str, Any]] = None  # Causal links identified in simulation
    confidence: float = Field(0.5, ge=0.0, le=1.0)

class SimulationInput(BaseModel):
    """Input parameters for running a simulation."""
    simulation_id: str = Field(default_factory=lambda: f"sim_{uuid.uuid4().hex[:8]}")
    description: str = "Hypothetical simulation"
    initial_state: Dict[str, Any]  # Starting variables, emotion state, etc.
    hypothetical_event: Optional[Dict[str, Any]] = None  # e.g., {'action': 'say_X', 'params': {}}
    counterfactual_condition: Optional[Dict[str, Any]] = None  # e.g., {'node_id': 'user_trust', 'value': 0.2}
    goal_condition: Optional[Dict[str, Any]] = None  # State to reach for success
    max_steps: int = 10
    focus_variables: List[str] = Field(default_factory=list)  # Variables to primarily track/report

class ImaginationSimulator:
    """
    Simulates hypothetical scenarios, counterfactuals, and potential futures.
    Integrates reasoning, knowledge, emotional prediction, and identity.
    """

    def __init__(self, reasoning_core=None, knowledge_core=None, emotional_core=None, identity_evolution=None):
        self.reasoning_core = reasoning_core
        self.knowledge_core = knowledge_core
        self.emotional_core = emotional_core
        self.identity_evolution = identity_evolution
        self.simulation_history: Dict[str, SimulationResult] = {}
        self.max_history = 50
        self.simulation_agent = self._create_simulation_agent()
        self.trace_group_id = "NyxImagination"
        
        # Initialize simulation counters
        self.simulation_stats = {
            "total_simulations": 0,
            "successful_simulations": 0,
            "failed_simulations": 0,
            "by_category": {}
        }

        logger.info("ImaginationSimulator initialized")

    def _create_simulation_agent(self) -> Optional[Agent]:
        """Creates an agent to help interpret simulation requests and results."""
        try:
            return Agent(
                name="Simulation Analyst",
                instructions="""You analyze requests for simulations and interpret their results for the Nyx AI.
                
                For simulation setup:
                1. Based on a description (e.g., "What if I apologized?"), determine appropriate initial state modifications, events, or counterfactuals.
                2. Define variables to track during the simulation.
                3. Specify any goal conditions that would indicate successful simulation completion.
                
                For simulation analysis:
                1. Examine the simulation trajectory and final state.
                2. Identify key causal relationships and patterns.
                3. Provide a concise summary of what the simulation predicts will happen.
                4. Assess the confidence in this prediction based on simulation coherence.
                5. Note any significant emotional impacts or changes.
                
                Your outputs should be structured JSON matching the expected input or output formats.
                """,
                model="gpt-4o",
                model_settings=ModelSettings(
                    temperature=0.3,
                    response_format={"type": "json_object"}
                ),
                tools=[
                    self.get_causal_model,
                    self.get_current_emotional_state
                ],
                output_type=Dict  # Flexible output for analysis
            )
        except Exception as e:
            logger.error(f"Error creating simulation agent: {e}")
            return None

    @function_tool
    async def get_causal_model(self, domain: str = "general") -> Dict[str, Any]:
        """Gets the causal model for a specific domain from the reasoning core."""
        if not self.reasoning_core:
            return {"status": "error", "message": "No reasoning core available"}
            
        try:
            if hasattr(self.reasoning_core, 'get_causal_model'):
                model = await self.reasoning_core.get_causal_model(domain)
                return {
                    "status": "success",
                    "model": model
                }
            else:
                return {
                    "status": "error", 
                    "message": "Reasoning core does not support causal models"
                }
        except Exception as e:
            logger.error(f"Error getting causal model: {e}")
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }

    @function_tool
    async def get_current_emotional_state(self) -> Dict[str, Any]:
        """Gets the current emotional state from the emotional core."""
        if not self.emotional_core:
            return {"status": "error", "message": "No emotional core available"}
            
        try:
            if hasattr(self.emotional_core, 'get_current_emotion'):
                emotion = await self.emotional_core.get_current_emotion()
                return {
                    "status": "success",
                    "emotion": emotion
                }
            else:
                return {
                    "status": "error", 
                    "message": "Emotional core does not support emotion retrieval"
                }
        except Exception as e:
            logger.error(f"Error getting emotional state: {e}")
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }

    async def setup_simulation(self, description: str, current_brain_state: Dict[str, Any]) -> Optional[SimulationInput]:
        """Uses the agent to interpret a description into SimulationInput."""
        if not self.simulation_agent:
            logger.warning("Simulation agent not available for setup")
            # Basic setup based on description keywords as fallback
            sim_input = SimulationInput(
                initial_state=current_brain_state, 
                description=description
            )
            
            if "what if i" in description.lower():
                 sim_input.hypothetical_event = {"description": description}  # Placeholder
            elif "what if" in description.lower():
                 sim_input.counterfactual_condition = {"description": description}  # Placeholder
                 
            return sim_input

        try:
            with trace(workflow_name="SetupSimulation", group_id=self.trace_group_id):
                # Prepare context for the agent
                context = {
                    "request": description,
                    "current_state_keys": list(current_brain_state.keys()),
                    "current_state_sample": {
                        k: current_brain_state[k] for k in list(current_brain_state.keys())[:5]
                    } if current_brain_state else {}
                }
                
                prompt = json.dumps(context)

                # Run the agent to get simulation setup
                result = await Runner.run(
                    self.simulation_agent,
                    prompt,
                    run_config={
                        "workflow_name": "SimulationSetup",
                        "trace_metadata": {"description": description}
                    }
                )
                
                # Process agent output
                setup_data = result.final_output
                
                # Ensure we have required fields
                if "initial_state" not in setup_data and "description" not in setup_data:
                    raise ValueError("Missing required fields in simulation setup")
                
                # Merge with current state, apply changes
                initial_state = current_brain_state.copy()
                
                # Apply specific state changes if provided
                if "initial_state" in setup_data and isinstance(setup_data["initial_state"], dict):
                    initial_state.update(setup_data["initial_state"])
                
                # Create SimulationInput object
                sim_input = SimulationInput(
                    description=setup_data.get("description", description),
                    initial_state=initial_state,
                    hypothetical_event=setup_data.get("hypothetical_event"),
                    counterfactual_condition=setup_data.get("counterfactual_condition"),
                    goal_condition=setup_data.get("goal_condition"),
                    max_steps=setup_data.get("max_steps", 10),
                    focus_variables=setup_data.get("focus_variables", [])
                )
                
                return sim_input

        except Exception as e:
            logger.exception(f"Error setting up simulation for '{description}': {e}")
            return None
            

    self.simulation_analyst_agent = Agent(
        name="Simulation Analyst",
        instructions="""You analyze the results of simulations to extract insights.
        
        Your role is to:
        1. Examine the simulation trajectory and identify patterns
        2. Extract key causal relationships
        3. Predict likely outcomes based on the simulation data
        4. Assess confidence in these predictions
        5. Identify emotional impacts and key dynamics
        
        Focus on extracting practical, actionable insights from simulation data.
        """,
        model="gpt-4o",
        model_settings=ModelSettings(temperature=0.3),
        tools=[],
        output_type=SimulationInsights  # Define this Pydantic model
    )
    
    # Add better tracing to simulation runs
    async def run_simulation(self, sim_input: SimulationInput) -> SimulationResult:
        """Runs a simulation based on the input parameters."""
        
        with trace(
            workflow_name="RunSimulation", 
            group_id=self.trace_group_id, 
            metadata={
                "sim_id": sim_input.simulation_id, 
                "category": self._determine_simulation_category(sim_input),
                "domain": sim_input.domain if hasattr(sim_input, 'domain') else "general",
                "max_steps": sim_input.max_steps
            }
        ):
        logger.info(f"Starting simulation '{sim_input.simulation_id}': {sim_input.description}")
        trajectory: List[SimulationState] = []
        
        # Setup initial simulation state
        current_sim_state = SimulationState(
            step=0,
            timestamp=datetime.datetime.now(),
            state_variables=sim_input.initial_state.copy()  # Start with initial state
        )
        trajectory.append(current_sim_state)

        termination_reason = "max_steps"
        success = False
        
        # Track simulation category for stats
        simulation_category = "counterfactual" if sim_input.counterfactual_condition else (
            "hypothetical" if sim_input.hypothetical_event else "general"
        )
        
        # Update simulation counters
        self.simulation_stats["total_simulations"] += 1
        if simulation_category not in self.simulation_stats["by_category"]:
            self.simulation_stats["by_category"][simulation_category] = 0
        self.simulation_stats["by_category"][simulation_category] += 1

        with trace(
            workflow_name="RunSimulation", 
            group_id=self.trace_group_id, 
            metadata={"sim_id": sim_input.simulation_id, "category": simulation_category}
        ):
            try:
                # Step 0: Apply initial conditions (counterfactual or hypothetical event)
                if sim_input.counterfactual_condition:
                    # Apply counterfactual condition
                    node_id = sim_input.counterfactual_condition.get("node_id")
                    value = sim_input.counterfactual_condition.get("value")
                    
                    if node_id:
                        current_sim_state.state_variables[node_id] = value
                        logger.debug(f"Sim Step 0: Applied counterfactual {node_id}={value}")
                    else:
                        description = sim_input.counterfactual_condition.get("description", "")
                        logger.warning(f"Counterfactual condition missing node_id: {description}")

                elif sim_input.hypothetical_event:
                    # Apply hypothetical event
                    action = sim_input.hypothetical_event.get("action")
                    params = sim_input.hypothetical_event.get("parameters", {})
                    
                    if action:
                        current_sim_state.last_action = action
                        
                        # Predict immediate effect using reasoning core
                        if self.reasoning_core and hasattr(self.reasoning_core, 'predict_action_effect'):
                            predicted_changes = await self.reasoning_core.predict_action_effect(
                                action, 
                                params, 
                                current_sim_state.state_variables
                            )
                            current_sim_state.state_variables.update(predicted_changes)
                            logger.debug(f"Sim Step 0: Applied hypothetical event '{action}'. Predicted changes: {predicted_changes}")
                        else:
                            # Fallback if no reasoning core: apply simple effects
                            # For example, if action is "apologize", might increase "user_trust" slightly
                            self._apply_simple_event_effects(current_sim_state, action, params)
                    else:
                        description = sim_input.hypothetical_event.get("description", "")
                        logger.warning(f"Hypothetical event missing action: {description}")

                # Simulation loop for steps 1 to max_steps
                for i in range(1, sim_input.max_steps + 1):
                    prev_state = current_sim_state
                    current_sim_state = SimulationState(
                        step=i,
                        timestamp=prev_state.timestamp + datetime.timedelta(seconds=10),  # Arbitrary time step
                        state_variables=prev_state.state_variables.copy()  # Copy previous state
                    )

                    # Predict changes for this step
                    if self.reasoning_core and hasattr(self.reasoning_core, 'predict_next_state'):
                        # Use causal reasoning to predict next state
                        predicted_changes = await self.reasoning_core.predict_next_state(prev_state.state_variables)
                        current_sim_state.state_variables.update(predicted_changes)
                    else:
                        # Fallback: basic simulation with decay/random walk
                        self._simulate_basic_state_changes(current_sim_state)

                    # Predict emotional state changes if possible
                    if self.emotional_core and hasattr(self.emotional_core, 'predict_emotional_state'):
                        current_sim_state.emotional_state = await self.emotional_core.predict_emotional_state(
                            current_sim_state.state_variables
                        )
                    
                    # Add to trajectory
                    trajectory.append(current_sim_state)
                    logger.debug(f"Sim Step {i}: Updated state variables")

                    # Check goal condition if specified
                    if sim_input.goal_condition:
                        goal_met = True
                        for key, value in sim_input.goal_condition.items():
                            current_value = current_sim_state.state_variables.get(key)
                            
                            # Handle numeric comparisons with tolerance
                            if isinstance(value, (int, float)) and isinstance(current_value, (int, float)):
                                if abs(current_value - value) > 0.01:  # Small tolerance
                                    goal_met = False
                                    break
                            # String/boolean/other exact comparison
                            elif current_value != value:
                                goal_met = False
                                break
                                
                        if goal_met:
                            termination_reason = "goal_reached"
                            success = True
                            break

                    # Check for stable state (minimal changes between steps)
                    if i > 1 and self._is_stable(current_sim_state, trajectory[-2]):
                        termination_reason = "stable_state"
                        # Not marking as success/failure since stability is neutral
                        break

            except Exception as e:
                logger.exception(f"Error during simulation '{sim_input.simulation_id}': {e}")
                termination_reason = "error"
                success = False

        # --- Finalize and analyze results ---
        final_state = trajectory[-1]
        
        # Set success flag based on termination reason
        if termination_reason == "goal_reached":
            success = True
        elif termination_reason == "error":
            success = False
        # For "stable_state" or "max_steps", success depends on progress toward goal
        elif sim_input.goal_condition:
            # Check if we made progress toward the goal even if not reached
            success = self._check_goal_progress(trajectory[0], final_state, sim_input.goal_condition)
        
        # Update statistics
        if success:
            self.simulation_stats["successful_simulations"] += 1
        else:
            self.simulation_stats["failed_simulations"] += 1

        # Get predicted outcome and confidence
        analysis_result = await self._analyze_trajectory(
            trajectory, 
            sim_input.goal_condition, 
            sim_input.description
        )
        
        # Create result object
        result = SimulationResult(
            simulation_id=sim_input.simulation_id,
            success=success,
            termination_reason=termination_reason,
            final_state=final_state,
            trajectory=trajectory,
            predicted_outcome=analysis_result.get("predicted_outcome", "unknown"),
            emotional_impact=analysis_result.get("emotional_impact"),
            causal_analysis=analysis_result.get("causal_analysis"),
            confidence=analysis_result.get("confidence", 0.5)
        )

        # Store result in history
        self.simulation_history[result.simulation_id] = result
        if len(self.simulation_history) > self.max_history:
             oldest_id = next(iter(self.simulation_history))
             del self.simulation_history[oldest_id]

        logger.info(f"Simulation '{result.simulation_id}' finished: {termination_reason}. Success: {success}")
        return result
    
    def _simulate_basic_state_changes(self, state: SimulationState) -> None:
        """Apply basic simulation logic when no reasoning core is available."""
        # Apply random walks with regression to mean for numeric values
        for key, value in state.state_variables.items():
            if isinstance(value, (int, float)):
                # Different behavior for different variable types
                if "trust" in key.lower() or "relationship" in key.lower():
                    # Trust/relationship: slow changes with inertia
                    mean = 0.5  # Default neutral value
                    volatility = 0.03  # Low volatility
                    inertia = 0.95  # High inertia
                    state.state_variables[key] = value * inertia + mean * (1 - inertia) + random.uniform(-volatility, volatility)
                elif "emotion" in key.lower() or "mood" in key.lower():
                    # Emotions: faster changes, less inertia
                    mean = 0.0  # Default neutral
                    volatility = 0.08  # Higher volatility
                    inertia = 0.8  # Medium inertia
                    state.state_variables[key] = value * inertia + mean * (1 - inertia) + random.uniform(-volatility, volatility)
                else:
                    # Default values for other numeric variables
                    inertia = 0.9  # Medium-high inertia
                    volatility = 0.05  # Medium volatility
                    state.state_variables[key] = value * inertia + random.uniform(-volatility, volatility)
                
                # Clamp values between expected ranges
                if "trust" in key.lower() or "confidence" in key.lower() or "probability" in key.lower():
                    state.state_variables[key] = max(0.0, min(1.0, state.state_variables[key]))
                elif "valence" in key.lower():
                    state.state_variables[key] = max(-1.0, min(1.0, state.state_variables[key]))
            
            # For dictionaries, recurse into them
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        # Apply similar logic recursively
                        inertia = 0.9
                        volatility = 0.04
                        state.state_variables[key][subkey] = subvalue * inertia + random.uniform(-volatility, volatility)
    
    def _apply_simple_event_effects(self, state: SimulationState, action: str, params: Dict[str, Any]) -> None:
        """Apply simplified effects for events when no reasoning core is available."""
        # Common human interaction events and their typical effects
        action_effects = {
            "apologize": {
                "user_trust": 0.1, 
                "user_anger": -0.2, 
                "relationship_tension": -0.15
            },
            "praise": {
                "user_happiness": 0.2, 
                "user_trust": 0.05, 
                "user_receptivity": 0.1
            },
            "criticize": {
                "user_defensiveness": 0.2, 
                "user_trust": -0.1, 
                "relationship_tension": 0.15
            },
            "share_personal": {
                "intimacy": 0.15, 
                "user_trust": 0.1, 
                "relationship_depth": 0.1
            },
            "disagree": {
                "user_respect": -0.05, 
                "intellectual_engagement": 0.1, 
                "relationship_tension": 0.1
            },
            "express_vulnerability": {
                "intimacy": 0.2, 
                "user_trust": 0.1, 
                "perceived_authenticity": 0.2
            }
        }
        
        # Apply effects if action is known
        normalized_action = action.lower().strip()
        for key, effect_dict in action_effects.items():
            if key in normalized_action:
                for var, change in effect_dict.items():
                    # Initialize if not exists
                    if var not in state.state_variables:
                        state.state_variables[var] = 0.5  # Default starting value
                    
                    # Apply change
                    state.state_variables[var] += change
                    
                    # Clamp between 0-1 for most variables
                    state.state_variables[var] = max(0.0, min(1.0, state.state_variables[var]))
                
                # Mark the action
                state.last_action = action
                return
        
        # If no predefined action matched, apply a small random effect
        # on trust and emotional variables as a fallback
        if "user_trust" in state.state_variables:
            state.state_variables["user_trust"] += random.uniform(-0.05, 0.05)
            state.state_variables["user_trust"] = max(0.0, min(1.0, state.state_variables["user_trust"]))
        
        # Mark the action
        state.last_action = action
    
    def _check_goal_progress(self, initial_state: SimulationState, final_state: SimulationState, 
                            goal_condition: Dict[str, Any]) -> bool:
        """Check if the simulation made significant progress toward the goal condition."""
        progress_score = 0
        goal_vars_count = 0
        
        for key, target_value in goal_condition.items():
            if key in initial_state.state_variables and key in final_state.state_variables:
                goal_vars_count += 1
                initial_value = initial_state.state_variables[key]
                final_value = final_state.state_variables[key]
                
                # For numeric values, check if we moved closer to target
                if isinstance(target_value, (int, float)) and isinstance(initial_value, (int, float)) and isinstance(final_value, (int, float)):
                    initial_distance = abs(target_value - initial_value)
                    final_distance = abs(target_value - final_value)
                    
                    if final_distance < initial_distance:
                        # Made progress toward goal
                        progress_ratio = (initial_distance - final_distance) / initial_distance
                        progress_score += progress_ratio
                
                # For boolean/string/etc., check if we reached the value
                elif final_value == target_value and initial_value != target_value:
                    progress_score += 1.0
        
        # If no goal variables found, consider it not successful
        if goal_vars_count == 0:
            return False
        
        # Calculate average progress across all goal variables
        avg_progress = progress_score / goal_vars_count
        
        # Consider significant progress (>25% toward goal) as success
        return avg_progress > 0.25
    
    def _is_stable(self, current_state: SimulationState, previous_state: SimulationState) -> bool:
        """Check if the simulation has reached a stable state with minimal changes."""
        stable_threshold = 0.01  # Maximum change considered stable
        stable_vars_count = 0
        total_vars_count = 0
        
        for key, current_value in current_state.state_variables.items():
            if key in previous_state.state_variables:
                prev_value = previous_state.state_variables[key]
                total_vars_count += 1
                
                # Check for stability based on variable type
                if isinstance(current_value, (int, float)) and isinstance(prev_value, (int, float)):
                    change = abs(current_value - prev_value)
                    if change <= stable_threshold:
                        stable_vars_count += 1
                elif current_value == prev_value:
                    stable_vars_count += 1
        
        # Consider stable if at least 90% of variables are stable
        return total_vars_count > 0 and (stable_vars_count / total_vars_count) >= 0.9

    async def _analyze_trajectory(self, trajectory: List[SimulationState], 
                                 goal_condition: Optional[Dict[str, Any]],
                                 description: str) -> Dict[str, Any]:
        """Analyze the simulation trajectory to extract insights."""
        if not self.simulation_agent or len(trajectory) < 2:
            # Fallback basic analysis if no agent or too few steps
            return {
                "predicted_outcome": "Cannot determine outcome without simulation agent",
                "confidence": 0.3,
                "causal_analysis": None,
                "emotional_impact": None
            }
        
        try:
            # Prepare trajectory summary (to avoid overwhelming the agent)
            trajectory_summary = []
            
            # Always include first and last states
            trajectory_summary.append(self._format_state_for_analysis(trajectory[0], is_first=True))
            
            # For longer trajectories, sample intermediate states
            if len(trajectory) > 5:
                step_size = len(trajectory) // 3
                for i in range(step_size, len(trajectory) - 1, step_size):
                    trajectory_summary.append(self._format_state_for_analysis(trajectory[i]))
            else:
                # For short trajectories, include all intermediate states
                for i in range(1, len(trajectory) - 1):
                    trajectory_summary.append(self._format_state_for_analysis(trajectory[i]))
            
            # Always include final state
            trajectory_summary.append(self._format_state_for_analysis(trajectory[-1], is_last=True))
            
            # Build analysis request
            analysis_request = {
                "description": description,
                "trajectory": trajectory_summary,
                "goal_condition": goal_condition
            }
            
            # Run agent to analyze trajectory
            result = await Runner.run(
                self.simulation_agent,
                json.dumps(analysis_request),
                run_config={
                    "workflow_name": "SimulationAnalysis",
                    "trace_metadata": {"type": "trajectory_analysis"}
                }
            )
            
            # Process agent output
            analysis = result.final_output
            
            # Ensure we have the expected fields
            if not isinstance(analysis, dict):
                raise ValueError(f"Expected dict from analysis agent, got {type(analysis)}")
            
            return {
                "predicted_outcome": analysis.get("predicted_outcome", "Outcome unclear"),
                "confidence": analysis.get("confidence", 0.5),
                "causal_analysis": analysis.get("causal_analysis"),
                "emotional_impact": analysis.get("emotional_impact")
            }
            
        except Exception as e:
            logger.error(f"Error analyzing simulation trajectory: {e}")
            return {
                "predicted_outcome": "Error analyzing simulation trajectory",
                "confidence": 0.2,
                "causal_analysis": None,
                "emotional_impact": None
            }
    
    def _format_state_for_analysis(self, state: SimulationState, is_first: bool = False, is_last: bool = False) -> Dict[str, Any]:
        """Format a simulation state for analysis, focusing on the most relevant data."""
        # Extract key variables (limit to most important ones to avoid overwhelming the agent)
        key_vars = {}
        for k, v in state.state_variables.items():
            # Include emotional, trust, relationship variables, and a few others
            if (any(term in k.lower() for term in ["trust", "emotion", "mood", "relationship", "confidence"]) or
                (is_first or is_last)):  # Include more details for first/last states
                if isinstance(v, dict) and len(v) > 3:
                    # For complex nested objects, include only a summary
                    key_vars[k] = f"<complex object with {len(v)} keys>"
                else:
                    key_vars[k] = v
        
        return {
            "step": state.step,
            "key_variables": key_vars,
            "emotional_state": state.emotional_state,
            "last_action": state.last_action,
            "is_first": is_first,
            "is_last": is_last
        }

    async def analyze_simulation_result(self, result: SimulationResult) -> Dict[str, Any]:
        """Uses agent to interpret the simulation result with additional context."""
        if not self.simulation_agent: 
            return {"summary": "Analysis agent unavailable"}

        try:
            with trace(workflow_name="AnalyzeSimulation", group_id=self.trace_group_id):
                # Prepare context for the agent
                context = {
                    "simulation_id": result.simulation_id,
                    "description": result.final_state.state_variables.get("description", "Simulation"),
                    "success": result.success,
                    "termination_reason": result.termination_reason,
                    "predicted_outcome": result.predicted_outcome,
                    "confidence": result.confidence,
                    "initial_state": self._format_state_for_analysis(result.trajectory[0], is_first=True),
                    "final_state": self._format_state_for_analysis(result.final_state, is_last=True),
                    "trajectory_length": len(result.trajectory),
                    "key_metrics": self._extract_key_metrics(result)
                }
                
                # Run the agent
                analysis_result = await Runner.run(
                    self.simulation_agent,
                    json.dumps(context),
                    run_config={
                        "workflow_name": "SimulationInterpretation",
                        "trace_metadata": {"sim_id": result.simulation_id}
                    }
                )
                
                # Process and return the analysis
                analysis = analysis_result.final_output
                
                # Add metadata before returning
                if isinstance(analysis, dict):
                    analysis["simulation_id"] = result.simulation_id
                    analysis["analyzed_at"] = datetime.datetime.now().isoformat()
                
                return analysis
                
        except Exception as e:
            logger.error(f"Error analyzing simulation result {result.simulation_id}: {e}")
            return {"summary": "Error during analysis", "error": str(e)}
    
    def _extract_key_metrics(self, result: SimulationResult) -> Dict[str, Any]:
        """Extract key metrics from the simulation for analysis."""
        metrics = {}
        
        # List of important variable patterns to track
        key_patterns = [
            "trust", "emotion", "mood", "relationship", "confidence", 
            "tension", "intimacy", "satisfaction", "agreement"
        ]
        
        # Get initial and final values for important variables
        if result.trajectory:
            initial_state = result.trajectory[0].state_variables
            final_state = result.final_state.state_variables
            
            for key in set(list(initial_state.keys()) + list(final_state.keys())):
                if any(pattern in key.lower() for pattern in key_patterns):
                    initial_value = initial_state.get(key)
                    final_value = final_state.get(key)
                    
                    if initial_value is not None and final_value is not None:
                        metrics[key] = {
                            "initial": initial_value,
                            "final": final_value,
                            "change": final_value - initial_value if isinstance(final_value, (int, float)) and 
                                                                    isinstance(initial_value, (int, float)) else "N/A"
                        }
        
        return metrics
    
    async def get_simulation_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get summaries of recent simulations."""
        # Convert to list and sort by recency (most recent first)
        history = list(self.simulation_history.values())
        history.sort(key=lambda x: x.trajectory[-1].timestamp if x.trajectory else datetime.datetime.min, reverse=True)
        
        # Limit and format results
        results = []
        for sim in history[:limit]:
            results.append({
                "id": sim.simulation_id,
                "description": sim.final_state.state_variables.get("description", "Simulation"),
                "success": sim.success,
                "termination_reason": sim.termination_reason,
                "steps": len(sim.trajectory),
                "predicted_outcome": str(sim.predicted_outcome),
                "confidence": sim.confidence,
                "timestamp": sim.trajectory[-1].timestamp.isoformat() if sim.trajectory else None
            })
            
        return results
    
    async def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get statistics about simulations run so far."""
        return {
            "total_simulations": self.simulation_stats["total_simulations"],
            "successful_simulations": self.simulation_stats["successful_simulations"],
            "failed_simulations": self.simulation_stats["failed_simulations"],
            "success_rate": self.simulation_stats["successful_simulations"] / max(1, self.simulation_stats["total_simulations"]),
            "by_category": self.simulation_stats["by_category"],
            "average_steps": sum(len(sim.trajectory) for sim in self.simulation_history.values()) / max(1, len(self.simulation_history)),
            "current_history_size": len(self.simulation_history)
        }
