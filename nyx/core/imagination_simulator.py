# nyx/core/imagination_simulator.py

import logging
import asyncio
import datetime
import random
import uuid
import json
from typing import Dict, List, Any, Optional, Sequence, Mapping, Union, TypedDict, Literal
from pydantic import BaseModel, Field

# Import OpenAI Agents SDK components
from agents import (
    Agent, 
    Runner, 
    ModelSettings, 
    trace, 
    function_tool, 
    handoff, 
    RunContextWrapper, 
    RunConfig,
    custom_span
)
from agents.tracing.util import gen_trace_id

# Import Nyx core systems
from nyx.core.reasoning_core import ReasoningCore, CausalModel
from nyx.core.reflection_engine import ReflectionEngine

logger = logging.getLogger(__name__)

# =============== Pydantic Models for Structured Output ===============

class SimulationState(BaseModel):
    """Represents a state within a simulation."""
    step: int
    timestamp: datetime.datetime
    state_variables: Dict[str, Any] = Field(default_factory=dict)
    emotional_state: Optional[Dict[str, Any]] = None
    reasoning_focus: Optional[str] = None
    last_action: Optional[str] = None

class SimulationResult(BaseModel):
    """Result of running a simulation."""
    simulation_id: str
    success: bool
    termination_reason: str
    final_state: SimulationState
    trajectory: List[SimulationState] = Field(default_factory=list)
    predicted_outcome: Any = None
    emotional_impact: Optional[Dict[str, float]] = None
    causal_analysis: Optional[Dict[str, Any]] = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    reflection: Optional[str] = None
    abstraction: Optional[Dict[str, Any]] = None

class SimulationInput(BaseModel):
    """Input parameters for running a simulation."""
    simulation_id: str = Field(default_factory=lambda: f"sim_{uuid.uuid4().hex[:8]}")
    description: str = "Hypothetical simulation"
    initial_state: Dict[str, Any]
    hypothetical_event: Optional[Dict[str, Any]] = None
    counterfactual_condition: Optional[Dict[str, Any]] = None
    goal_condition: Optional[Dict[str, Any]] = None
    max_steps: int = 10
    focus_variables: List[str] = Field(default_factory=list)
    domain: str = "general"
    use_reflection: bool = True

class ScenarioGenerationOutput(BaseModel):
    """Output from the scenario generation agent."""
    scenario_description: str
    initial_state_modifications: Dict[str, Any]
    hypothetical_event: Optional[Dict[str, Any]] = None
    counterfactual_condition: Optional[Dict[str, Any]] = None
    goal_condition: Optional[Dict[str, Any]] = None
    focus_variables: List[str] = Field(default_factory=list)
    creative_elements: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(0.5, ge=0.0, le=1.0)

class SimulationAnalysisOutput(BaseModel):
    """Output from the simulation analysis agent."""
    analysis_text: str
    causal_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    emotional_impacts: Dict[str, float] = Field(default_factory=dict)
    key_insights: List[str] = Field(default_factory=list)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    recommended_followup: Optional[str] = None

class CausalModelOutput(BaseModel):
    """
    Strict DTO returned by `get_causal_model_for_simulation`.
    The `model` blob is serialised as JSON to avoid open-ended objects
    in the schema, and `extra = "forbid"` keeps the schema strict.
    """
    status: Literal["cached", "retrieved", "created"]
    model_json: str                     # ← JSON-encoded causal model

    model_config = {"extra": "forbid"}

class SimulationContext:
    """Context object for simulation agents."""
    def __init__(self, 
                reasoning_core=None, 
                reflection_engine=None, 
                knowledge_core=None, 
                emotional_core=None,
                identity_evolution=None):
        self.reasoning_core = reasoning_core
        self.reflection_engine = reflection_engine
        self.knowledge_core = knowledge_core
        self.emotional_core = emotional_core
        self.identity_evolution = identity_evolution
        
        # Runtime state
        self.current_simulation = None
        self.current_state = None
        self.history = []
        self.trace_group_id = "NyxImagination"
        
        # Temporary storage
        self.causal_models = {}
        self.concept_spaces = {}
        self.simulation_stats = {
            "total_simulations": 0,
            "successful_simulations": 0,
            "failed_simulations": 0,
            "by_category": {}
        }

# =============== Function Tools for Agents ===============

@function_tool
async def setup_simulation_from_description(
    ctx: RunContextWrapper[SimulationContext],
    description: str,
    current_brain_state: Dict[str, Any],
    domain: str = "general"
) -> Dict[str, Any]:
    """
    Interpret a simulation description into a structured simulation setup.
    
    Args:
        description: User description of the desired simulation
        current_brain_state: Current state variables
        domain: Domain for the simulation
        
    Returns:
        Structured simulation setup
    """
    with custom_span("setup_simulation_from_description"):
        # Create base simulation input
        sim_input = {
            "simulation_id": f"sim_{uuid.uuid4().hex[:8]}",
            "description": description,
            "initial_state": current_brain_state.copy(),
            "domain": domain,
            "max_steps": 10,
            "focus_variables": []
        }
        
        # Extract simulation type from description
        if "what if i" in description.lower():
            sim_input["hypothetical_event"] = {
                "action": "hypothetical_user_action",
                "description": description
            }
        elif "what if" in description.lower():
            sim_input["counterfactual_condition"] = {
                "description": description
            }
        
        # Add timestamp
        sim_input["timestamp"] = datetime.datetime.now().isoformat()
        
        return sim_input

@function_tool
async def get_causal_model_for_simulation(
    ctx: RunContextWrapper[SimulationContext],
    domain: str = "general",
) -> CausalModelOutput:                   # ←  strict DTO return type
    """
    Retrieve (or create) a causal model for the requested *domain*.

    If we already have one in context cache, use it.  Otherwise try the
    reasoning-core; if that fails, build a small default skeleton.
    """
    with custom_span("get_causal_model_for_simulation"):
        # 1️⃣  Cached?
        if domain in ctx.context.causal_models:
            model = ctx.context.causal_models[domain]
            return CausalModelOutput(status="cached",
                                     model_json=json.dumps(model))

        # 2️⃣  Reasoning core?
        if ctx.context.reasoning_core:
            try:
                model = await ctx.context.reasoning_core.get_causal_model(domain)
                if model:
                    ctx.context.causal_models[domain] = model            # cache
                    return CausalModelOutput(status="retrieved",
                                             model_json=json.dumps(model))
            except Exception as e:
                logger.error(f"[Causal-Model] retrieval error: {e}")

        # 3️⃣  Build a simple default
        default_model = {
            "id": f"model_{uuid.uuid4().hex[:8]}",
            "name": f"Default {domain} model",
            "domain": domain,
            "nodes": {},
            "relations": [],
            "metadata": {
                "created_at": datetime.datetime.now().isoformat(),
                "is_default": True,
            },
        }

        # --- domain-specific scaffolding ------------------------------------
        if domain == "social":
            default_model["nodes"] = {
                "user_trust":            {"type": "variable", "current_value": 0.5},
                "user_satisfaction":     {"type": "variable", "current_value": 0.5},
                "relationship_depth":    {"type": "variable", "current_value": 0.3},
                "communication_quality": {"type": "variable", "current_value": 0.6},
            }
            default_model["relations"] = [
                {"source": "communication_quality", "target": "user_satisfaction", "strength": 0.7},
                {"source": "user_satisfaction",     "target": "user_trust",        "strength": 0.8},
                {"source": "user_trust",            "target": "relationship_depth","strength": 0.6},
            ]

        elif domain == "problem_solving":
            default_model["nodes"] = {
                "problem_complexity":      {"type": "variable", "current_value": 0.5},
                "solution_quality":        {"type": "variable", "current_value": 0.5},
                "user_comprehension":      {"type": "variable", "current_value": 0.6},
                "implementation_feasibility": {"type": "variable", "current_value": 0.7},
            }
            default_model["relations"] = [
                {"source": "problem_complexity", "target": "solution_quality",        "strength": -0.6},
                {"source": "solution_quality",   "target": "implementation_feasibility","strength": 0.7},
                {"source": "user_comprehension", "target": "implementation_feasibility","strength": 0.5},
            ]

        else:  # generic template
            default_model["nodes"] = {
                "user_satisfaction":  {"type": "variable", "current_value": 0.5},
                "system_performance": {"type": "variable", "current_value": 0.7},
                "interaction_quality": {"type": "variable", "current_value": 0.6},
            }
            default_model["relations"] = [
                {"source": "system_performance", "target": "user_satisfaction", "strength": 0.6},
                {"source": "interaction_quality", "target": "user_satisfaction","strength": 0.7},
            ]
        # --------------------------------------------------------------------

        ctx.context.causal_models[domain] = default_model                # cache
        return CausalModelOutput(status="created",
                                 model_json=json.dumps(default_model))

@function_tool
async def predict_simulation_step(
    ctx: RunContextWrapper[SimulationContext],
    current_state: Dict[str, Any],
    causal_model: Dict[str, Any],
    step: int
) -> Dict[str, Any]:
    """
    Predict the next simulation state based on causal model.
    
    Args:
        current_state: Current simulation state
        causal_model: Causal model to use for prediction
        step: Current step number
        
    Returns:
        Updated simulation state
    """
    with custom_span("predict_simulation_step"):
        # Start with a copy of the current state
        next_state = current_state.copy()
        next_state["step"] = step
        next_state["timestamp"] = datetime.datetime.now().isoformat()
        
        # Get the state variables
        state_vars = next_state.get("state_variables", {}).copy()
        
        # Propagate causal effects through the model
        nodes = causal_model.get("nodes", {})
        relations = causal_model.get("relations", [])
        
        # Process each causal relation
        for relation in relations:
            source_id = relation.get("source")
            target_id = relation.get("target")
            relation_type = relation.get("type", "causal")
            strength = relation.get("strength", 0.5)
            
            # Skip if not a causal relation
            if relation_type != "causal":
                continue
                
            # Skip if nodes don't exist in model
            if source_id not in nodes or target_id not in nodes:
                continue
                
            # Skip if source not in state variables
            if source_id not in state_vars:
                continue
                
            # Initialize target if not exists
            if target_id not in state_vars:
                state_vars[target_id] = nodes[target_id].get("current_value", 0.5)
                
            # Get current values
            source_value = state_vars[source_id]
            target_value = state_vars[target_id]
            
            # Apply causal influence with some random variation
            if isinstance(source_value, (int, float)) and isinstance(target_value, (int, float)):
                # Calculate influence
                change = (source_value - 0.5) * strength * 0.2  # Scale the effect
                
                # Add some noise
                noise = random.uniform(-0.05, 0.05)
                
                # Apply change
                new_value = target_value + change + noise
                
                # Clamp between 0 and 1
                new_value = max(0.0, min(1.0, new_value))
                
                # Update state
                state_vars[target_id] = new_value
        
        # Update emotional state based on state variables
        emotional_state = next_state.get("emotional_state", {}).copy()
        
        # Map key variables to emotional impacts
        if "user_satisfaction" in state_vars:
            # Satisfaction impacts valence
            emotional_state["valence"] = state_vars["user_satisfaction"] * 2 - 1  # Map 0-1 to -1 to 1
            
        if "interaction_quality" in state_vars:
            # Interaction quality impacts arousal
            emotional_state["arousal"] = state_vars["interaction_quality"]
            
        # Set a primary emotion based on valence and arousal
        valence = emotional_state.get("valence", 0)
        arousal = emotional_state.get("arousal", 0.5)
        
        # Simple emotion mapping based on valence and arousal
        primary_emotion = "Neutral"
        if valence > 0.3:
            if arousal > 0.6:
                primary_emotion = "Joy"
            else:
                primary_emotion = "Contentment"
        elif valence < -0.3:
            if arousal > 0.6:
                primary_emotion = "Frustration"
            else:
                primary_emotion = "Sadness"
        
        emotional_state["primary_emotion"] = {
            "name": primary_emotion,
            "intensity": abs(valence) * 0.8 + 0.2  # Scale to 0.2-1.0
        }
        
        # Update the next state
        next_state["state_variables"] = state_vars
        next_state["emotional_state"] = emotional_state
        
        return next_state

@function_tool
async def apply_hypothetical_event(
    ctx: RunContextWrapper[SimulationContext],
    initial_state: Dict[str, Any],
    event: Dict[str, Any],
    causal_model: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply a hypothetical event to the initial state.
    
    Args:
        initial_state: Initial simulation state
        event: Hypothetical event to apply
        causal_model: Causal model for simulation
        
    Returns:
        Updated state after applying event
    """
    with custom_span("apply_hypothetical_event"):
        # Start with a copy of the initial state
        updated_state = initial_state.copy()
        state_vars = updated_state.get("state_variables", {}).copy()
        
        # Extract event information
        action = event.get("action", "")
        description = event.get("description", "")
        parameters = event.get("parameters", {})
        
        # Apply event effects based on action type
        if "apologize" in action.lower() or "apologize" in description.lower():
            # Apologizing typically increases trust and decreases negative emotions
            if "user_trust" in state_vars:
                state_vars["user_trust"] = min(1.0, state_vars["user_trust"] + 0.15)
            if "user_anger" in state_vars:
                state_vars["user_anger"] = max(0.0, state_vars["user_anger"] - 0.2)
            if "relationship_tension" in state_vars:
                state_vars["relationship_tension"] = max(0.0, state_vars["relationship_tension"] - 0.15)
        
        elif "explain" in action.lower() or "explain" in description.lower():
            # Explaining typically increases comprehension
            if "user_comprehension" in state_vars:
                state_vars["user_comprehension"] = min(1.0, state_vars["user_comprehension"] + 0.2)
            if "communication_quality" in state_vars:
                state_vars["communication_quality"] = min(1.0, state_vars["communication_quality"] + 0.1)
        
        elif "share" in action.lower() or "share" in description.lower():
            # Sharing typically increases intimacy and trust
            if "intimacy" in state_vars:
                state_vars["intimacy"] = min(1.0, state_vars["intimacy"] + 0.15)
            if "user_trust" in state_vars:
                state_vars["user_trust"] = min(1.0, state_vars["user_trust"] + 0.1)
            if "relationship_depth" in state_vars:
                state_vars["relationship_depth"] = min(1.0, state_vars["relationship_depth"] + 0.1)
        
        elif "disagree" in action.lower() or "disagree" in description.lower():
            # Disagreeing can increase tension but also respect if done well
            if "relationship_tension" in state_vars:
                state_vars["relationship_tension"] = min(1.0, state_vars["relationship_tension"] + 0.15)
            if "intellectual_engagement" in state_vars:
                state_vars["intellectual_engagement"] = min(1.0, state_vars["intellectual_engagement"] + 0.2)
        
        else:
            # Generic event based on description
            # Extract key words from description
            description_lower = description.lower()
            
            # Apply effects based on sentiment
            positive_words = ["help", "improve", "increase", "better", "good", "positive", "success"]
            negative_words = ["harm", "worsen", "decrease", "bad", "negative", "failure"]
            
            # Count positive and negative words
            positive_count = sum(1 for word in positive_words if word in description_lower)
            negative_count = sum(1 for word in negative_words if word in description_lower)
            
            # Determine overall sentiment
            sentiment = positive_count - negative_count
            
            # Apply generic effects based on sentiment
            for var_name in state_vars.keys():
                if isinstance(state_vars[var_name], (int, float)):
                    if "trust" in var_name or "satisfaction" in var_name or "quality" in var_name:
                        # Adjust positively or negatively based on sentiment
                        if sentiment > 0:
                            state_vars[var_name] = min(1.0, state_vars[var_name] + 0.1 * sentiment)
                        elif sentiment < 0:
                            state_vars[var_name] = max(0.0, state_vars[var_name] + 0.1 * sentiment)
        
        # Update the state variables
        updated_state["state_variables"] = state_vars
        updated_state["last_action"] = action or description
        
        return updated_state

@function_tool
async def apply_counterfactual_condition(
    ctx: RunContextWrapper[SimulationContext],
    initial_state: Dict[str, Any],
    counterfactual: Dict[str, Any],
    causal_model: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply a counterfactual condition to the initial state.
    
    Args:
        initial_state: Initial simulation state
        counterfactual: Counterfactual condition to apply
        causal_model: Causal model for simulation
        
    Returns:
        Updated state after applying counterfactual
    """
    with custom_span("apply_counterfactual_condition"):
        # Start with a copy of the initial state
        updated_state = initial_state.copy()
        state_vars = updated_state.get("state_variables", {}).copy()
        
        # Extract counterfactual information
        node_id = counterfactual.get("node_id")
        value = counterfactual.get("value")
        description = counterfactual.get("description", "")
        
        # If node_id and value provided, apply directly
        if node_id and value is not None:
            state_vars[node_id] = value
        
        # Otherwise, interpret from description
        elif description:
            description_lower = description.lower()
            
            # Look for phrases matching "what if X was Y"
            matched = False
            
            # Check nodes in model
            nodes = causal_model.get("nodes", {})
            for node_id, node_data in nodes.items():
                node_name = node_data.get("name", "").lower()
                
                # Check if node name is in description
                if node_name in description_lower:
                    # Look for value indicators
                    if "high" in description_lower or "increase" in description_lower:
                        state_vars[node_id] = min(1.0, (state_vars.get(node_id, 0.5) + 0.3))
                        matched = True
                    elif "low" in description_lower or "decrease" in description_lower:
                        state_vars[node_id] = max(0.0, (state_vars.get(node_id, 0.5) - 0.3))
                        matched = True
                    elif "very high" in description_lower:
                        state_vars[node_id] = 0.9
                        matched = True
                    elif "very low" in description_lower:
                        state_vars[node_id] = 0.1
                        matched = True
            
            # If no match, look for emotional counterfactuals
            if not matched:
                if "user was happy" in description_lower or "user felt good" in description_lower:
                    if "user_satisfaction" in state_vars:
                        state_vars["user_satisfaction"] = 0.8
                    emotional_state = updated_state.get("emotional_state", {}).copy()
                    emotional_state["valence"] = 0.7
                    emotional_state["primary_emotion"] = {"name": "Joy", "intensity": 0.8}
                    updated_state["emotional_state"] = emotional_state
                    
                elif "user was unhappy" in description_lower or "user felt bad" in description_lower:
                    if "user_satisfaction" in state_vars:
                        state_vars["user_satisfaction"] = 0.2
                    emotional_state = updated_state.get("emotional_state", {}).copy()
                    emotional_state["valence"] = -0.6
                    emotional_state["primary_emotion"] = {"name": "Sadness", "intensity": 0.7}
                    updated_state["emotional_state"] = emotional_state
                    
                elif "relationship was stronger" in description_lower:
                    if "relationship_depth" in state_vars:
                        state_vars["relationship_depth"] = 0.8
                    if "user_trust" in state_vars:
                        state_vars["user_trust"] = 0.8
                
                elif "relationship was weaker" in description_lower:
                    if "relationship_depth" in state_vars:
                        state_vars["relationship_depth"] = 0.2
                    if "user_trust" in state_vars:
                        state_vars["user_trust"] = 0.3
        
        # Update the state variables
        updated_state["state_variables"] = state_vars
        updated_state["reasoning_focus"] = f"Counterfactual: {description}"
        
        return updated_state

@function_tool
async def check_goal_condition(
    ctx: RunContextWrapper[SimulationContext],
    current_state: Dict[str, Any],
    goal_condition: Dict[str, Any]
) -> Dict[str, bool]:
    """
    Check if the current state satisfies the goal condition.
    
    Args:
        current_state: Current simulation state
        goal_condition: Goal condition to check
        
    Returns:
        Whether the goal condition is met
    """
    with custom_span("check_goal_condition"):
        # Get state variables
        state_vars = current_state.get("state_variables", {})
        
        # Check each goal condition
        all_conditions_met = True
        conditions_met = {}
        
        for var_name, target_value in goal_condition.items():
            if var_name not in state_vars:
                all_conditions_met = False
                conditions_met[var_name] = False
                continue
                
            current_value = state_vars[var_name]
            
            # Handle numeric comparisons with tolerance
            if isinstance(target_value, (int, float)) and isinstance(current_value, (int, float)):
                is_met = abs(current_value - target_value) <= 0.1  # 10% tolerance
            else:
                is_met = current_value == target_value
                
            conditions_met[var_name] = is_met
            if not is_met:
                all_conditions_met = False
        
        return {
            "goal_met": all_conditions_met,
            "conditions_met": conditions_met
        }

@function_tool
async def evaluate_simulation_stability(
    ctx: RunContextWrapper[SimulationContext],
    current_state: Dict[str, Any],
    previous_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check if the simulation has reached a stable state.
    
    Args:
        current_state: Current simulation state
        previous_state: Previous simulation state
        
    Returns:
        Stability evaluation
    """
    with custom_span("evaluate_simulation_stability"):
        # Get state variables
        current_vars = current_state.get("state_variables", {})
        previous_vars = previous_state.get("state_variables", {})
        
        # Calculate changes in each variable
        changes = {}
        total_change = 0.0
        num_variables = 0
        
        for var_name, current_value in current_vars.items():
            if var_name in previous_vars:
                previous_value = previous_vars[var_name]
                
                # Calculate change for numeric variables
                if isinstance(current_value, (int, float)) and isinstance(previous_value, (int, float)):
                    change = abs(current_value - previous_value)
                    changes[var_name] = change
                    total_change += change
                    num_variables += 1
        
        # Calculate average change
        avg_change = total_change / num_variables if num_variables > 0 else 0.0
        
        # Determine stability
        is_stable = avg_change < 0.05  # Less than 5% average change
        
        return {
            "is_stable": is_stable,
            "avg_change": avg_change,
            "changes": changes,
            "stability_confidence": 1.0 - min(1.0, avg_change * 10)  # Convert change to confidence
        }

@function_tool
async def analyze_simulation_result(
    ctx: RunContextWrapper[SimulationContext],
    simulation_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze the results of a simulation run.
    
    Args:
        simulation_result: Simulation result to analyze
        
    Returns:
        Analysis of simulation results
    """
    with custom_span("analyze_simulation_result"):
        # Extract key information
        simulation_id = simulation_result.get("simulation_id", "unknown")
        success = simulation_result.get("success", False)
        termination_reason = simulation_result.get("termination_reason", "unknown")
        final_state = simulation_result.get("final_state", {})
        trajectory = simulation_result.get("trajectory", [])
        
        # Calculate key metrics
        state_vars = final_state.get("state_variables", {})
        initial_state = trajectory[0].get("state_variables", {}) if trajectory else {}
        
        # Track changes in key variables
        variable_changes = {}
        for var_name, final_value in state_vars.items():
            if var_name in initial_state:
                initial_value = initial_state[var_name]
                if isinstance(final_value, (int, float)) and isinstance(initial_value, (int, float)):
                    change = final_value - initial_value
                    variable_changes[var_name] = {
                        "initial": initial_value,
                        "final": final_value,
                        "change": change,
                        "percent_change": (change / initial_value) * 100 if initial_value != 0 else 0
                    }
        
        # Identify significant changes
        significant_changes = {k: v for k, v in variable_changes.items() if abs(v["change"]) > 0.1}
        
        # Analyze emotional impact
        emotional_impact = {}
        if "emotional_state" in final_state:
            final_emotional = final_state["emotional_state"]
            
            if trajectory and "emotional_state" in trajectory[0]:
                initial_emotional = trajectory[0]["emotional_state"]
                
                # Compare emotional states
                if "valence" in final_emotional and "valence" in initial_emotional:
                    valence_change = final_emotional["valence"] - initial_emotional["valence"]
                    emotional_impact["valence_change"] = valence_change
                
                if "arousal" in final_emotional and "arousal" in initial_emotional:
                    arousal_change = final_emotional["arousal"] - initial_emotional["arousal"]
                    emotional_impact["arousal_change"] = arousal_change
                
                # Compare primary emotions
                if "primary_emotion" in final_emotional and "primary_emotion" in initial_emotional:
                    initial_emotion = initial_emotional["primary_emotion"].get("name") if isinstance(initial_emotional["primary_emotion"], dict) else initial_emotional["primary_emotion"]
                    final_emotion = final_emotional["primary_emotion"].get("name") if isinstance(final_emotional["primary_emotion"], dict) else final_emotional["primary_emotion"]
                    
                    emotional_impact["emotion_transition"] = f"{initial_emotion} -> {final_emotion}"
            
            # Set current emotional state
            if "primary_emotion" in final_emotional:
                if isinstance(final_emotional["primary_emotion"], dict):
                    emotional_impact["final_emotion"] = final_emotional["primary_emotion"].get("name", "Unknown")
                    emotional_impact["emotion_intensity"] = final_emotional["primary_emotion"].get("intensity", 0.5)
                else:
                    emotional_impact["final_emotion"] = final_emotional["primary_emotion"]
                    emotional_impact["emotion_intensity"] = 0.5
        
        # Generate insights
        insights = []
        
        # Insight about success/failure
        if success:
            insights.append(f"The simulation was successful, terminating due to {termination_reason}.")
        else:
            insights.append(f"The simulation was unsuccessful, terminating due to {termination_reason}.")
        
        # Insights about significant changes
        if significant_changes:
            changes_text = ", ".join([f"{k} ({v['change']:.2f})" for k, v in significant_changes.items()])
            insights.append(f"Significant changes occurred in: {changes_text}")
        else:
            insights.append("No significant changes occurred in state variables.")
        
        # Insight about emotional impact
        if emotional_impact:
            if "emotion_transition" in emotional_impact:
                insights.append(f"Emotional transition: {emotional_impact['emotion_transition']}")
            if "valence_change" in emotional_impact:
                valence_desc = "positive" if emotional_impact["valence_change"] > 0 else "negative"
                insights.append(f"Emotional valence shifted in a {valence_desc} direction.")
        
        # Confidence calculation based on trajectory length and termination reason
        confidence_factors = {
            "trajectory_length": min(1.0, len(trajectory) / 10) * 0.3,  # More steps = more confidence
            "termination": 0.0,
            "stability": 0.0
        }
        
        # Adjust for termination reason
        if termination_reason == "goal_reached":
            confidence_factors["termination"] = 0.4
        elif termination_reason == "stable_state":
            confidence_factors["termination"] = 0.3
            confidence_factors["stability"] = 0.2
        elif termination_reason == "max_steps":
            confidence_factors["termination"] = 0.1
        
        # Calculate overall confidence
        confidence = sum(confidence_factors.values()) + 0.2  # Base confidence of 0.2
        
        return {
            "simulation_id": simulation_id,
            "variable_changes": variable_changes,
            "significant_changes": significant_changes,
            "emotional_impact": emotional_impact,
            "insights": insights,
            "confidence": confidence,
            "confidence_factors": confidence_factors
        }

@function_tool
async def generate_simulation_reflection(
    ctx: RunContextWrapper[SimulationContext],
    simulation_result: Dict[str, Any],
    analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate reflection on a simulation using the reflection engine.
    
    Args:
        simulation_result: Simulation result
        analysis: Analysis of simulation result
        
    Returns:
        Reflection on simulation
    """
    with custom_span("generate_simulation_reflection"):
        reflection_text = ""
        confidence = 0.0
        
        # Check if reflection engine is available
        if ctx.context.reflection_engine:
            try:
                # Convert simulation result to memories format for reflection
                memories = []
                
                # Add initial state as memory
                if simulation_result.get("trajectory"):
                    initial_state = simulation_result["trajectory"][0]
                    initial_memory = {
                        "id": f"sim_init_{simulation_result['simulation_id']}",
                        "memory_text": f"Initial state of simulation '{simulation_result.get('description', 'Simulation')}' with variables: {json.dumps(initial_state.get('state_variables', {}))}",
                        "memory_type": "simulation",
                        "metadata": {
                            "emotional_context": initial_state.get("emotional_state", {}),
                            "simulation_step": 0,
                            "simulation_id": simulation_result["simulation_id"]
                        }
                    }
                    memories.append(initial_memory)
                
                # Add final state as memory
                final_state = simulation_result.get("final_state", {})
                final_memory = {
                    "id": f"sim_final_{simulation_result['simulation_id']}",
                    "memory_text": f"Final state of simulation '{simulation_result.get('description', 'Simulation')}' with variables: {json.dumps(final_state.get('state_variables', {}))}",
                    "memory_type": "simulation",
                    "metadata": {
                        "emotional_context": final_state.get("emotional_state", {}),
                        "simulation_step": len(simulation_result.get("trajectory", [])) - 1,
                        "simulation_id": simulation_result["simulation_id"],
                        "termination_reason": simulation_result.get("termination_reason", "unknown")
                    }
                }
                memories.append(final_memory)
                
                # Add key events from trajectory (significant changes) as memories
                trajectory = simulation_result.get("trajectory", [])
                significant_steps = []
                
                # Find steps with significant changes
                for i in range(1, len(trajectory)):
                    prev_state = trajectory[i-1].get("state_variables", {})
                    curr_state = trajectory[i].get("state_variables", {})
                    
                    # Look for significant changes
                    has_significant_change = False
                    for var_name, curr_value in curr_state.items():
                        if var_name in prev_state:
                            prev_value = prev_state[var_name]
                            if isinstance(curr_value, (int, float)) and isinstance(prev_value, (int, float)):
                                if abs(curr_value - prev_value) >= 0.1:  # 10% change threshold
                                    has_significant_change = True
                                    break
                    
                    if has_significant_change:
                        significant_steps.append(i)
                
                # Add memories for significant steps
                for step_idx in significant_steps[:3]:  # Limit to 3 significant steps
                    step = trajectory[step_idx]
                    step_memory = {
                        "id": f"sim_step_{simulation_result['simulation_id']}_{step_idx}",
                        "memory_text": f"At step {step_idx} of simulation '{simulation_result.get('description', 'Simulation')}', variables changed to: {json.dumps(step.get('state_variables', {}))}",
                        "memory_type": "simulation",
                        "metadata": {
                            "emotional_context": step.get("emotional_state", {}),
                            "simulation_step": step_idx,
                            "simulation_id": simulation_result["simulation_id"],
                            "last_action": step.get("last_action", "")
                        }
                    }
                    memories.append(step_memory)
                
                # Get neurochemical state from emotional core or use default
                neurochemical_state = None
                if ctx.context.emotional_core and hasattr(ctx.context.emotional_core, "_get_neurochemical_state"):
                    neurochemical_state = {c: d["value"] for c, d in ctx.context.emotional_core.neurochemicals.items()}
                
                # Generate reflection using reflection engine
                topic = f"Simulation: {simulation_result.get('description', 'Hypothetical scenario')}"
                reflection_text, confidence = await ctx.context.reflection_engine.generate_reflection(
                    memories=memories,
                    topic=topic,
                    neurochemical_state=neurochemical_state
                )
            except Exception as e:
                logger.error(f"Error generating reflection: {str(e)}")
                reflection_text = f"I tried to reflect on this simulation but encountered difficulties: {str(e)}"
                confidence = 0.3
        else:
            # Generate a basic reflection without the reflection engine
            insights = analysis.get("insights", [])
            emotional_impact = analysis.get("emotional_impact", {})
            
            reflection_parts = [
                f"When I imagine this scenario of {simulation_result.get('description', 'this hypothetical situation')}, I notice several interesting patterns.",
                f"The simulation {simulation_result.get('success', False) and 'succeeded' or 'did not succeed'}, ending due to {simulation_result.get('termination_reason', 'unknown reasons')}."
            ]
            
            # Add insights
            for insight in insights[:2]:  # Limit to 2 insights
                reflection_parts.append(insight)
            
            # Add emotional reflection
            if emotional_impact:
                if "emotion_transition" in emotional_impact:
                    reflection_parts.append(f"I observe an emotional shift from {emotional_impact['emotion_transition']}, which suggests this scenario could have significant emotional impact.")
                elif "final_emotion" in emotional_impact:
                    reflection_parts.append(f"This scenario appears to lead to a primary emotion of {emotional_impact['final_emotion']}, which is worth considering.")
            
            # Add conclusion
            reflection_parts.append(f"This simulation helps me understand potential outcomes and prepare for similar situations.")
            
            # Join parts
            reflection_text = " ".join(reflection_parts)
            confidence = 0.5  # Moderate confidence without reflection engine
        
        return {
            "reflection": reflection_text,
            "confidence": confidence,
            "source": "reflection_engine" if ctx.context.reflection_engine else "basic_generation",
            "timestamp": datetime.datetime.now().isoformat()
        }

@function_tool
async def generate_abstraction_from_simulation(
    ctx: RunContextWrapper[SimulationContext],
    simulation_result: Dict[str, Any],
    pattern_type: str = "causal"
) -> Dict[str, Any]:
    """
    Generate an abstraction from the simulation using the reflection engine.
    
    Args:
        simulation_result: Simulation result
        pattern_type: Type of pattern to look for
        
    Returns:
        Abstraction from simulation
    """
    with custom_span("generate_abstraction_from_simulation"):
        # Default response
        result = {
            "abstraction_text": "",
            "pattern_type": pattern_type,
            "confidence": 0.0,
            "supporting_evidence": []
        }
        
        # Check if reflection engine is available
        if ctx.context.reflection_engine and hasattr(ctx.context.reflection_engine, "create_abstraction"):
            try:
                # Convert simulation result to memories format for abstraction
                memories = []
                
                # Create memories from the trajectory
                trajectory = simulation_result.get("trajectory", [])
                for i, state in enumerate(trajectory):
                    memory = {
                        "id": f"sim_state_{simulation_result['simulation_id']}_{i}",
                        "memory_text": f"Step {i} of simulation: {json.dumps(state.get('state_variables', {}))}",
                        "memory_type": "simulation",
                        "significance": 7.0,  # High significance
                        "metadata": {
                            "emotional_context": state.get("emotional_state", {}),
                            "simulation_step": i,
                            "simulation_id": simulation_result["simulation_id"],
                            "last_action": state.get("last_action", "")
                        }
                    }
                    memories.append(memory)
                
                # Get neurochemical state
                neurochemical_state = None
                if ctx.context.emotional_core and hasattr(ctx.context.emotional_core, "_get_neurochemical_state"):
                    neurochemical_state = {c: d["value"] for c, d in ctx.context.emotional_core.neurochemicals.items()}
                
                # Generate abstraction
                abstraction_text, abstraction_data = await ctx.context.reflection_engine.create_abstraction(
                    memories=memories,
                    pattern_type=pattern_type,
                    neurochemical_state=neurochemical_state
                )
                
                # Update result
                result["abstraction_text"] = abstraction_text
                result["confidence"] = abstraction_data.get("confidence", 0.5)
                
                # Add additional data if available
                if "pattern_type" in abstraction_data:
                    result["pattern_type"] = abstraction_data["pattern_type"]
                if "entity_focus" in abstraction_data:
                    result["entity_focus"] = abstraction_data["entity_focus"]
                if "supporting_evidence" in abstraction_data:
                    result["supporting_evidence"] = abstraction_data["supporting_evidence"]
                if "neurochemical_insight" in abstraction_data:
                    result["neurochemical_insight"] = abstraction_data["neurochemical_insight"]
            except Exception as e:
                logger.error(f"Error generating abstraction: {str(e)}")
                result["abstraction_text"] = f"I tried to generate an abstraction but encountered difficulties: {str(e)}"
                result["confidence"] = 0.2
        else:
            # Generate basic abstraction without reflection engine
            trajectory = simulation_result.get("trajectory", [])
            
            # Look for patterns in state changes
            if len(trajectory) >= 3:
                # Track changes in key variables
                var_trends = {}
                
                # Get state variables from trajectory
                for i in range(1, len(trajectory)):
                    prev_state = trajectory[i-1].get("state_variables", {})
                    curr_state = trajectory[i].get("state_variables", {})
                    
                    # Track changes
                    for var_name, curr_value in curr_state.items():
                        if var_name in prev_state and isinstance(curr_value, (int, float)) and isinstance(prev_state[var_name], (int, float)):
                            change = curr_value - prev_state[var_name]
                            
                            if var_name not in var_trends:
                                var_trends[var_name] = []
                            
                            var_trends[var_name].append(change)
                
                # Analyze trends
                trend_patterns = {}
                for var_name, changes in var_trends.items():
                    if len(changes) >= 2:
                        # Check for consistent direction
                        consistent_direction = all(change > 0 for change in changes) or all(change < 0 for change in changes)
                        
                        # Check for acceleration/deceleration
                        if len(changes) >= 3:
                            differences = [abs(changes[i]) - abs(changes[i-1]) for i in range(1, len(changes))]
                            accelerating = all(diff > 0 for diff in differences)
                            decelerating = all(diff < 0 for diff in differences)
                        else:
                            accelerating = False
                            decelerating = False
                        
                        # Check for oscillation
                        oscillating = any(changes[i] * changes[i-1] < 0 for i in range(1, len(changes)))
                        
                        # Record pattern
                        if consistent_direction:
                            direction = "increasing" if changes[0] > 0 else "decreasing"
                            speed = "accelerating" if accelerating else "decelerating" if decelerating else "constant"
                            trend_patterns[var_name] = f"{direction} at {speed} rate"
                        elif oscillating:
                            trend_patterns[var_name] = "oscillating"
                        else:
                            trend_patterns[var_name] = "variable"
                
                # Generate abstraction text
                if trend_patterns:
                    pattern_desc = ", ".join([f"{var_name} ({pattern})" for var_name, pattern in list(trend_patterns.items())[:3]])
                    abstraction_text = f"In this simulation, I notice a pattern where {pattern_desc}. This suggests a {pattern_type} relationship where certain variables follow predictable trajectories under these conditions."
                    
                    # Add causal insights if available
                    final_state = simulation_result.get("final_state", {})
                    if final_state and "emotional_state" in final_state:
                        emotion = final_state["emotional_state"].get("primary_emotion")
                        emotion_name = emotion.get("name") if isinstance(emotion, dict) else emotion
                        if emotion_name:
                            abstraction_text += f" The emotional outcome tends toward {emotion_name}, which appears connected to these variable changes."
                    
                    # Set result
                    result["abstraction_text"] = abstraction_text
                    result["confidence"] = 0.6
                    result["supporting_evidence"] = [f"Trend in {var}: {pattern}" for var, pattern in trend_patterns.items()]
                else:
                    result["abstraction_text"] = "I don't see a clear pattern in this simulation data."
                    result["confidence"] = 0.3
            else:
                result["abstraction_text"] = "This simulation is too short to extract meaningful patterns."
                result["confidence"] = 0.2
        
        return result

# =============== Main Imagination Simulator Class ===============

class ImaginationSimulator:
    """
    Enhanced Imagination Simulator that leverages reasoning and reflection capabilities 
    through the OpenAI Agents SDK.
    """
    
    def __init__(self, reasoning_core=None, reflection_engine=None, knowledge_core=None, 
                 emotional_core=None, identity_evolution=None):
        """
        Initialize the imagination simulator with required components.
        
        Args:
            reasoning_core: Reference to reasoning core system
            reflection_engine: Reference to reflection engine system
            knowledge_core: Reference to knowledge core system
            emotional_core: Reference to emotional core system 
            identity_evolution: Reference to identity evolution system
        """
        # Store references to other systems
        self.reasoning_core = reasoning_core
        self.reflection_engine = reflection_engine
        self.knowledge_core = knowledge_core
        self.emotional_core = emotional_core
        self.identity_evolution = identity_evolution
        
        # Initialize simulation context
        self.context = SimulationContext(
            reasoning_core=reasoning_core,
            reflection_engine=reflection_engine,
            knowledge_core=knowledge_core,
            emotional_core=emotional_core,
            identity_evolution=identity_evolution
        )
        
        # Initialize history
        self.simulation_history = {}
        self.max_history = 50
        
        # Initialize next ID counters
        self.next_simulation_id = 1
        
        # Initialize agents
        self._init_agents()
        
        self.trace_group_id = "NyxImagination"
        
        logger.info("EnhancedImaginationSimulator initialized with Agents SDK integration")
    
    def _init_agents(self):
        """Initialize the specialized agents for the imagination system."""
        # Configure model settings
        base_model_settings = ModelSettings(
            temperature=0.7,
            top_p=0.9
        )
        
        low_temp_settings = ModelSettings(
            temperature=0.4,
            top_p=0.9
        )
        
        # Scenario Generation Agent - creates creative simulation scenarios
        self.scenario_generation_agent = Agent[SimulationContext](
            name="Scenario Generator",
            instructions="""You are an creative scenario generation agent for the Nyx AI system.
            
            Your role is to interpret natural language descriptions into structured simulation
            inputs that can be used by the imagination system. You should:
            
            1. Identify whether this is a hypothetical event or counterfactual condition
            2. Structure the simulation parameters appropriately 
            3. Identify relevant variables to focus on and track
            4. Set appropriate goal conditions when possible
            5. Add creative elements that enrich the simulation
            
            Be specific and detailed in your interpretations, translating vague descriptions
            into concrete simulation parameters. Use your creativity to elaborate on the
            basic scenario in meaningful ways.""",
            model="gpt-4.1-nano",
            model_settings=base_model_settings,
            tools=[
                function_tool(setup_simulation_from_description),
                function_tool(get_causal_model_for_simulation)
            ],
            output_type=ScenarioGenerationOutput
        )
        
        # Simulation Analysis Agent - analyzes simulation results
        self.simulation_analysis_agent = Agent[SimulationContext](
            name="Simulation Analyst",
            instructions="""You are a simulation analysis agent for the Nyx AI system.
            
            Your role is to analyze the results of simulations to extract meaningful insights,
            patterns, and implications. You should:
            
            1. Identify key patterns in the simulation trajectory
            2. Extract causal relationships between variables
            3. Analyze emotional impacts and responses
            4. Provide overall insights about what the simulation reveals
            5. Suggest potential follow-up simulations when appropriate
            
            Focus on extracting actionable insights that help Nyx understand potential
            outcomes and improve decision-making. Consider both the objective changes in
            variables and the subjective emotional responses.""",
            model="gpt-4.1-nano",
            model_settings=low_temp_settings,
            tools=[
                function_tool(analyze_simulation_result),
                function_tool(generate_simulation_reflection),
                function_tool(generate_abstraction_from_simulation)
            ],
            output_type=SimulationAnalysisOutput
        )
        
        # Create the orchestration agent
        self.orchestrator_agent = Agent[SimulationContext](
            name="Imagination Orchestrator",
            instructions="""You are the orchestrator for Nyx's imagination system, coordinating
            various specialized agents to create and analyze simulations of hypothetical scenarios.
            
            Your role is to:
            1. Interpret user requests for simulations
            2. Coordinate between scenario generation and simulation analysis
            3. Manage the overall simulation process
            4. Integrate reasoning and reflection components into simulations
            5. Return comprehensive simulation results
            
            You should ensure that simulations are both creative and grounded in causal reasoning,
            leveraging Nyx's reasoning core and reflection capabilities to provide meaningful
            insights about hypothetical scenarios.""",
            model="gpt-4.1-nano",
            model_settings=low_temp_settings,
            handoffs=[
                handoff(self.scenario_generation_agent,
                       tool_name_override="generate_scenario",
                       tool_description_override="Generate a detailed simulation scenario from a description"),
                handoff(self.simulation_analysis_agent,
                       tool_name_override="analyze_simulation",
                       tool_description_override="Analyze the results of a simulation run")
            ],
            tools=[
                function_tool(setup_simulation_from_description),
                function_tool(get_causal_model_for_simulation),
                function_tool(predict_simulation_step),
                function_tool(apply_hypothetical_event),
                function_tool(apply_counterfactual_condition),
                function_tool(check_goal_condition),
                function_tool(evaluate_simulation_stability)
            ]
        )
    
    async def setup_simulation(self, description: str, current_brain_state: Dict[str, Any]) -> Optional[SimulationInput]:
        """
        Uses the scenario generation agent to interpret a description into a structured SimulationInput.
        
        Args:
            description: Natural language description of the desired simulation
            current_brain_state: Current state variables
            
        Returns:
            Structured SimulationInput object or None if setup fails
        """
        with trace(workflow_name="SetupSimulation", group_id=self.trace_group_id):
            try:
                # Configure the run
                run_config = RunConfig(
                    workflow_name="Simulation Setup",
                    trace_id=f"sim-setup-{gen_trace_id()}",
                    trace_metadata={"description": description}
                )
                
                # Run the orchestrator agent
                result = await Runner.run(
                    self.orchestrator_agent,
                    f"Generate a simulation setup for this description: {description}",
                    context=self.context,
                    run_config=run_config
                )
                
                # Check if we got a ScenarioGenerationOutput
                if hasattr(result.final_output, "model_dump"):
                    # Convert to SimulationInput
                    scenario = result.final_output
                    
                    # Use scenario data to create SimulationInput
                    sim_input = SimulationInput(
                        simulation_id=f"sim_{self.next_simulation_id}",
                        description=description,
                        initial_state=current_brain_state.copy(),  # Start with current state
                        domain="general",
                        max_steps=10,
                        focus_variables=scenario.focus_variables
                    )
                    
                    # Apply initial state modifications
                    for key, value in scenario.initial_state_modifications.items():
                        sim_input.initial_state[key] = value
                    
                    # Set hypothetical event or counterfactual condition
                    if scenario.hypothetical_event:
                        sim_input.hypothetical_event = scenario.hypothetical_event
                    
                    if scenario.counterfactual_condition:
                        sim_input.counterfactual_condition = scenario.counterfactual_condition
                    
                    # Set goal condition if provided
                    if scenario.goal_condition:
                        sim_input.goal_condition = scenario.goal_condition
                    
                    self.next_simulation_id += 1
                    return sim_input
                
                # If we got a dictionary instead
                elif isinstance(result.final_output, dict):
                    # Try to extract SimulationInput fields
                    sim_input = SimulationInput(
                        simulation_id=f"sim_{self.next_simulation_id}",
                        description=description,
                        initial_state=current_brain_state.copy()
                    )
                    
                    # Apply any modifications from the output
                    if "initial_state_modifications" in result.final_output:
                        for key, value in result.final_output["initial_state_modifications"].items():
                            sim_input.initial_state[key] = value
                    
                    # Set hypothetical event if provided
                    if "hypothetical_event" in result.final_output:
                        sim_input.hypothetical_event = result.final_output["hypothetical_event"]
                    
                    # Set counterfactual condition if provided
                    if "counterfactual_condition" in result.final_output:
                        sim_input.counterfactual_condition = result.final_output["counterfactual_condition"]
                    
                    # Set goal condition if provided
                    if "goal_condition" in result.final_output:
                        sim_input.goal_condition = result.final_output["goal_condition"]
                    
                    # Set focus variables if provided
                    if "focus_variables" in result.final_output:
                        sim_input.focus_variables = result.final_output["focus_variables"]
                    
                    self.next_simulation_id += 1
                    return sim_input
                
                return None
            except Exception as e:
                logger.error(f"Error setting up simulation for '{description}': {e}")
                return None
    
    async def run_simulation(self, sim_input: SimulationInput) -> SimulationResult:
        """
        Run a simulation based on the input parameters.
        
        Args:
            sim_input: Input parameters for simulation
            
        Returns:
            Simulation result
        """
        with trace(
            workflow_name="RunSimulation", 
            group_id=self.trace_group_id, 
            metadata={
                "sim_id": sim_input.simulation_id, 
                "description": sim_input.description
            }
        ):
            logger.info(f"Starting simulation '{sim_input.simulation_id}': {sim_input.description}")
            
            # Initialize trajectory
            trajectory = []
            
            # Get causal model
            causal_model_result = await get_causal_model_for_simulation(
                RunContextWrapper(self.context),
                sim_input.domain
            )
            causal_model = causal_model_result["model"]
            
            # Initialize simulation state
            initial_state = {
                "step": 0,
                "timestamp": datetime.datetime.now().isoformat(),
                "state_variables": sim_input.initial_state.copy(),
                "emotional_state": {
                    "valence": 0.0,
                    "arousal": 0.5,
                    "primary_emotion": {"name": "Neutral", "intensity": 0.5}
                }
            }
            
            # Apply initial conditions
            if sim_input.hypothetical_event:
                initial_state = await apply_hypothetical_event(
                    RunContextWrapper(self.context),
                    initial_state,
                    sim_input.hypothetical_event,
                    causal_model
                )
            
            if sim_input.counterfactual_condition:
                initial_state = await apply_counterfactual_condition(
                    RunContextWrapper(self.context),
                    initial_state,
                    sim_input.counterfactual_condition,
                    causal_model
                )
            
            # Add initial state to trajectory
            trajectory.append(SimulationState(**initial_state))
            
            # Initialize result variables
            termination_reason = "max_steps"
            success = False
            
            # Run simulation steps
            try:
                for step in range(1, sim_input.max_steps + 1):
                    # Predict next state
                    next_state = await predict_simulation_step(
                        RunContextWrapper(self.context),
                        trajectory[-1].model_dump(),
                        causal_model,
                        step
                    )
                    
                    # Add to trajectory
                    trajectory.append(SimulationState(**next_state))
                    
                    # Check goal condition if specified
                    if sim_input.goal_condition:
                        goal_check = await check_goal_condition(
                            RunContextWrapper(self.context),
                            next_state,
                            sim_input.goal_condition
                        )
                        
                        if goal_check["goal_met"]:
                            termination_reason = "goal_reached"
                            success = True
                            break
                    
                    # Check for stability
                    if step > 1:
                        stability_check = await evaluate_simulation_stability(
                            RunContextWrapper(self.context),
                            next_state,
                            trajectory[-2].model_dump()
                        )
                        
                        if stability_check["is_stable"]:
                            termination_reason = "stable_state"
                            # Not marking as success/failure since stability is neutral
                            break
            except Exception as e:
                logger.exception(f"Error during simulation '{sim_input.simulation_id}': {e}")
                termination_reason = "error"
                success = False
            
            # Create result object
            result = SimulationResult(
                simulation_id=sim_input.simulation_id,
                success=success,
                termination_reason=termination_reason,
                final_state=trajectory[-1],
                trajectory=trajectory,
                confidence=0.5  # Default confidence, will be updated by analysis
            )
            
            # Analyze simulation result
            if len(trajectory) > 1:
                try:
                    # Configure the run
                    run_config = RunConfig(
                        workflow_name="Simulation Analysis",
                        trace_id=f"sim-analysis-{gen_trace_id()}",
                        trace_metadata={"sim_id": sim_input.simulation_id}
                    )
                    
                    # Convert SimulationResult to dict for analysis
                    result_dict = result.model_dump()
                    
                    # Run the analysis agent
                    analysis_result = await Runner.run(
                        self.simulation_analysis_agent,
                        f"Analyze this simulation result for '{sim_input.description}'",
                        context=self.context,
                        run_config=run_config
                    )
                    
                    # Update result with analysis
                    if hasattr(analysis_result.final_output, "model_dump"):
                        analysis = analysis_result.final_output.model_dump()
                        result.confidence = analysis.get("confidence", result.confidence)
                        
                        # Get causal patterns
                        causal_analysis = {
                            "patterns": analysis.get("causal_patterns", []),
                            "insights": analysis.get("key_insights", [])
                        }
                        result.causal_analysis = causal_analysis
                        
                        # Get emotional impact
                        result.emotional_impact = analysis.get("emotional_impacts", {})
                    elif isinstance(analysis_result.final_output, dict):
                        analysis = analysis_result.final_output
                        result.confidence = analysis.get("confidence", result.confidence)
                        
                        # Get causal patterns
                        causal_analysis = {
                            "patterns": analysis.get("causal_patterns", []),
                            "insights": analysis.get("key_insights", [])
                        }
                        result.causal_analysis = causal_analysis
                        
                        # Get emotional impact
                        result.emotional_impact = analysis.get("emotional_impacts", {})
                except Exception as e:
                    logger.error(f"Error analyzing simulation: {e}")
            
            # Generate reflection if requested
            if sim_input.use_reflection and self.reflection_engine:
                try:
                    # Generate reflection
                    reflection_result = await generate_simulation_reflection(
                        RunContextWrapper(self.context),
                        result.model_dump(),
                        result.causal_analysis or {}
                    )
                    
                    # Update result with reflection
                    result.reflection = reflection_result.get("reflection", "")
                    
                    # Generate abstraction
                    abstraction_result = await generate_abstraction_from_simulation(
                        RunContextWrapper(self.context),
                        result.model_dump(),
                        "causal"
                    )
                    
                    # Update result with abstraction
                    result.abstraction = abstraction_result
                except Exception as e:
                    logger.error(f"Error generating reflection/abstraction: {e}")
            
            # Store in history
            self.simulation_history[result.simulation_id] = result
            if len(self.simulation_history) > self.max_history:
                oldest_id = next(iter(self.simulation_history))
                del self.simulation_history[oldest_id]
            
            # Update statistics
            self.context.simulation_stats["total_simulations"] += 1
            if success:
                self.context.simulation_stats["successful_simulations"] += 1
            else:
                self.context.simulation_stats["failed_simulations"] += 1
            
            return result
    
    async def imagine_scenario(self, description: str, current_brain_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        High-level function to imagine a scenario from a natural language description.
        
        Args:
            description: Natural language description of the scenario
            current_brain_state: Current state variables
            
        Returns:
            Dictionary with simulation results and insights
        """
        # Setup simulation
        sim_input = await self.setup_simulation(description, current_brain_state)
        
        if not sim_input:
            return {
                "success": False,
                "error": "Failed to setup simulation from description",
                "description": description
            }
        
        # Run simulation
        result = await self.run_simulation(sim_input)
        
        # Return formatted result
        return {
            "success": result.success,
            "simulation_id": result.simulation_id,
            "description": sim_input.description,
            "termination_reason": result.termination_reason,
            "confidence": result.confidence,
            "steps": len(result.trajectory),
            "reflection": result.reflection,
            "key_insights": result.causal_analysis.get("insights", []) if result.causal_analysis else [],
            "abstraction": result.abstraction.get("abstraction_text", "") if result.abstraction else "",
            "predicted_outcome": result.predicted_outcome or "Unknown outcome"
        }
    
    async def imagine_counterfactual(self, 
                                 description: str, 
                                 variable_name: str, 
                                 variable_value: Any,
                                 current_brain_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Imagine a counterfactual scenario with a specific variable change.
        
        Args:
            description: Description of the counterfactual scenario
            variable_name: Name of the variable to modify
            variable_value: New value for the variable
            current_brain_state: Current state variables
            
        Returns:
            Dictionary with simulation results and insights
        """
        # Create simulation input
        sim_input = SimulationInput(
            simulation_id=f"sim_{self.next_simulation_id}",
            description=description,
            initial_state=current_brain_state.copy(),
            counterfactual_condition={
                "node_id": variable_name,
                "value": variable_value,
                "description": description
            },
            max_steps=10,
            use_reflection=True
        )
        
        self.next_simulation_id += 1
        
        # Run simulation
        result = await self.run_simulation(sim_input)
        
        # Return formatted result
        return {
            "success": result.success,
            "simulation_id": result.simulation_id,
            "counterfactual": {
                "variable": variable_name,
                "value": variable_value,
                "description": description
            },
            "termination_reason": result.termination_reason,
            "confidence": result.confidence,
            "steps": len(result.trajectory),
            "reflection": result.reflection,
            "key_insights": result.causal_analysis.get("insights", []) if result.causal_analysis else [],
            "predicted_outcome": result.predicted_outcome or "Unknown outcome"
        }
    
    async def imagine_action_outcome(self, 
                                action: str, 
                                parameters: Dict[str, Any],
                                current_brain_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Imagine the outcome of a specific action.
        
        Args:
            action: Description of the action
            parameters: Parameters for the action
            current_brain_state: Current state variables
            
        Returns:
            Dictionary with simulation results and insights
        """
        # Create simulation input
        sim_input = SimulationInput(
            simulation_id=f"sim_{self.next_simulation_id}",
            description=f"Imagining outcome of action: {action}",
            initial_state=current_brain_state.copy(),
            hypothetical_event={
                "action": action,
                "parameters": parameters,
                "description": f"What if I {action}?"
            },
            max_steps=8,
            use_reflection=True
        )
        
        self.next_simulation_id += 1
        
        # Run simulation
        result = await self.run_simulation(sim_input)
        
        # Return formatted result
        return {
            "success": result.success,
            "simulation_id": result.simulation_id,
            "action": {
                "type": action,
                "parameters": parameters
            },
            "termination_reason": result.termination_reason,
            "confidence": result.confidence,
            "steps": len(result.trajectory),
            "reflection": result.reflection,
            "emotional_impact": result.emotional_impact,
            "key_insights": result.causal_analysis.get("insights", []) if result.causal_analysis else [],
            "predicted_outcome": result.predicted_outcome or "Unknown outcome"
        }

    async def get_simulation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about simulations run so far.
        
        Returns:
            Dictionary with simulation statistics
        """
        with custom_span("get_simulation_statistics"):
            # Calculate average steps across all simulations in history
            avg_steps = 0
            if self.simulation_history:
                avg_steps = sum(len(sim.trajectory) for sim in self.simulation_history.values()) / len(self.simulation_history)
            
            # Calculate success rate
            success_rate = 0
            if self.context.simulation_stats["total_simulations"] > 0:
                success_rate = self.context.simulation_stats["successful_simulations"] / self.context.simulation_stats["total_simulations"]
            
            return {
                "total_simulations": self.context.simulation_stats["total_simulations"],
                "successful_simulations": self.context.simulation_stats["successful_simulations"],
                "failed_simulations": self.context.simulation_stats["failed_simulations"],
                "success_rate": success_rate,
                "by_category": self.context.simulation_stats["by_category"],
                "average_steps": avg_steps,
                "current_history_size": len(self.simulation_history),
                "abstraction_count": len(self.context.concept_spaces),
                "causal_models_available": len(self.context.causal_models),
                "reflection_available": self.reflection_engine is not None
            }
    
    async def get_simulation_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get summaries of recent simulations.
        
        Args:
            limit: Maximum number of simulations to return
            
        Returns:
            List of simulation summaries
        """
        # Convert to list and sort by recency
        history = list(self.simulation_history.values())
        history.sort(key=lambda x: x.trajectory[-1].timestamp if x.trajectory else datetime.datetime.min, reverse=True)
        
        # Format results
        results = []
        for sim in history[:limit]:
            results.append({
                "id": sim.simulation_id,
                "description": sim.final_state.state_variables.get("description", "Simulation"),
                "success": sim.success,
                "termination_reason": sim.termination_reason,
                "steps": len(sim.trajectory),
                "confidence": sim.confidence,
                "reflection": sim.reflection,
                "timestamp": sim.trajectory[-1].timestamp.isoformat() if sim.trajectory else None
            })
            
        return results
