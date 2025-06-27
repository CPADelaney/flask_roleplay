# nyx/core/imagination_simulator.py

import logging
import asyncio
import datetime
import random
import uuid
import json
from typing import Dict, List, Any, Optional, Sequence, Mapping, Union, TypedDict, Literal
from pydantic import BaseModel, Field, ConfigDict
from dateutil.parser import isoparse  # External dependency - add python-dateutil to requirements.txt

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

class SimulationAbstractionDTO(BaseModel):
    abstraction_text: str
    pattern_type: str
    confidence: float
    supporting_evidence: List[str] = Field(default_factory=list)

    # optional enrichments that ReflectionEngine might add
    entity_focus: Optional[Any] = None
    neurochemical_insight: Optional[Any] = None

class SimulationReflectionDTO(BaseModel):
    reflection: str
    confidence: float
    source: Literal["reflection_engine", "basic_generation", "error"]
    timestamp: str

class SimulationAnalysisDTO(BaseModel):
    simulation_id: str
    variable_changes_json: str        # json.dumps(variable_changes)
    significant_changes_json: str     # json.dumps(significant_changes)
    emotional_impact_json: str        # json.dumps(emotional_impact)
    insights_json: str                # json.dumps(insights   )
    confidence: float
    confidence_factors_json: str      # json.dumps(confidence_factors)

class StabilityEvaluationDTO(BaseModel):
    is_stable: bool
    avg_change: float
    changes_json: str            # JSON-encoded dict of per-variable deltas
    stability_confidence: float

class GoalCheckResultDTO(BaseModel):
    goal_met: bool
    conditions_met_json: str            # JSON-encoded dict

class SimulationStateDTO(BaseModel):
    """
    Strict DTO used by predict_simulation_step.
    The whole next-state object is JSON-encoded in `state_json`
    so the schema is closed (additionalProperties == false).
    """
    state_json: str

class SimulationState(BaseModel):
    """Represents a state within a simulation."""
    step: int
    timestamp: datetime.datetime
    state_variables: Dict[str, Any] = Field(default_factory=dict)
    emotional_state: Optional[Dict[str, Any]] = None
    reasoning_focus: Optional[str] = None
    last_action: Optional[str] = None
    # Pydantic handles datetime serialization automatically with model_dump_json()

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

class SimulationInputDTO(BaseModel):
    """
    Strict DTO returned by setup_simulation_from_description.
    The full SimulationInput object is JSON-encoded in `simulation_json`
    so that the schema stays closed (additionalProperties == false).
    """
    simulation_json: str                  # JSON string of a SimulationInput

class ScenarioGenerationOutput(BaseModel, extra="forbid"):
    """Output from the scenario generation agent."""
    scenario_description: str
    initial_state_modifications: Dict[str, Any] = Field(default_factory=dict)
    hypothetical_event: Optional[Dict[str, Any]] = None
    counterfactual_condition: Optional[Dict[str, Any]] = None
    goal_condition: Optional[Dict[str, Any]] = None
    focus_variables: List[str] = Field(default_factory=list)
    creative_elements: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(0.5, ge=0.0, le=1.0)

# Fix #6: Allow extra fields for flexibility in analysis output
class SimulationAnalysisOutput(BaseModel, extra="allow"):
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
    in the schema.
    """
    status: Literal["cached", "retrieved", "created"]
    model_json: str                     # ← JSON-encoded causal model

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
    current_brain_state_json: str,          # <<< was Dict[str, Any]
    domain: str = "general",
) -> SimulationInputDTO:                    # <<< strict output
    """
    Interpret a simulation description into a structured SimulationInput.

    Args:
        description:              User's natural-language description.
        current_brain_state_json: JSON string with the current brain state.
        domain:                   (optional) simulation domain.

    Returns:
        SimulationInputDTO whose `simulation_json` field contains the full
        SimulationInput record (JSON-encoded).
    """
    with custom_span("setup_simulation_from_description"):
        # --- Parse the incoming brain state ---------------------------------
        try:
            current_brain_state: Dict[str, Any] = json.loads(
                current_brain_state_json
            )
        except Exception as e:                         # fall back to empty dict
            logger.error(f"Malformed brain-state JSON: {e}")
            current_brain_state = {}

        # --- Build the SimulationInput --------------------------------------
        sim_input = SimulationInput(
            description=description,
            initial_state=current_brain_state,
            domain=domain,
            max_steps=10,
            focus_variables=[],
        )

        # Detect "what if" / counterfactual clues in the prompt
        d_lower = description.lower()
        if "what if i" in d_lower:
            sim_input.hypothetical_event = {
                "action": "hypothetical_user_action",
                "description": description,
            }
        elif "what if" in d_lower:
            sim_input.counterfactual_condition = {"description": description}

        # --- Return as strict DTO -------------------------------------------
        return SimulationInputDTO(simulation_json=sim_input.model_dump_json())

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
    current_state_json: str,           # <<<  JSON strings, not raw dicts
    causal_model_json: str,            # <<<
    step: int,
) -> SimulationStateDTO:               # <<<  strict output
    """
    Predict the next simulation state.

    Args:
        current_state_json: JSON string for the current state.
        causal_model_json:  JSON string for the causal model.
        step:               Step number being simulated.

    Returns:
        SimulationStateDTO whose `state_json` field contains the full
        next-state record in JSON form.
    """
    with custom_span("predict_simulation_step"):
        # ---------- Parse inputs -------------------------------------------
        try:
            next_state: Dict[str, Any] = json.loads(current_state_json)
        except Exception as e:
            logger.error(f"Bad current_state JSON: {e}")
            next_state = {}
        try:
            causal_model: Dict[str, Any] = json.loads(causal_model_json)
        except Exception as e:
            logger.error(f"Bad causal_model JSON: {e}")
            causal_model = {}

        # ---------- Populate base fields -----------------------------------
        next_state = next_state.copy()
        next_state["step"] = step
        next_state["timestamp"] = datetime.datetime.now().isoformat()

        # ---------- Causal propagation -------------------------------------
        state_vars = next_state.get("state_variables", {}).copy()
        for rel in causal_model.get("relations", []):
            if rel.get("type", "causal") != "causal":
                continue
            s, t = rel.get("source"), rel.get("target")
            if s not in state_vars:
                continue
            if t not in state_vars:
                t_node = causal_model.get("nodes", {}).get(t, {})
                state_vars[t] = t_node.get("current_value", 0.5)

            sv, tv = state_vars[s], state_vars[t]
            if isinstance(sv, (int, float)) and isinstance(tv, (int, float)):
                delta = (sv - 0.5) * rel.get("strength", 0.5) * 0.2
                tv = max(0.0, min(1.0, tv + delta + random.uniform(-0.05, 0.05)))
                state_vars[t] = tv

        # ---------- Emotion update -----------------------------------------
        emo = next_state.get("emotional_state", {}).copy()
        if "user_satisfaction" in state_vars:
            emo["valence"] = state_vars["user_satisfaction"] * 2 - 1
        if "interaction_quality" in state_vars:
            emo["arousal"] = state_vars["interaction_quality"]

        val, ar = emo.get("valence", 0), emo.get("arousal", 0.5)
        if val > 0.3:
            emo_name = "Joy" if ar > 0.6 else "Contentment"
        elif val < -0.3:
            emo_name = "Frustration" if ar > 0.6 else "Sadness"
        else:
            emo_name = "Neutral"
        emo["primary_emotion"] = {"name": emo_name, "intensity": abs(val) * 0.8 + 0.2}

        # ---------- Assemble & return --------------------------------------
        next_state["state_variables"] = state_vars
        next_state["emotional_state"] = emo

        return SimulationStateDTO(state_json=json.dumps(next_state, separators=(",", ":")))

@function_tool
async def apply_hypothetical_event(
    ctx: RunContextWrapper[SimulationContext],
    initial_state_json: str,          # <<< JSON strings instead of dicts
    event_json: str,                  # <<<
    causal_model_json: str | None = None,
) -> SimulationStateDTO:              # <<< strict DTO output
    """
    Apply a hypothetical event to the simulation state.

    Args:
        initial_state_json: JSON string of the starting state.
        event_json:         JSON string describing the event.
        causal_model_json:  (Unused here but kept for parity).

    Returns:
        SimulationStateDTO whose `state_json` contains the updated state.
    """
    with custom_span("apply_hypothetical_event"):
        # ---- Parse inputs -------------------------------------------------
        try:
            updated_state: Dict[str, Any] = json.loads(initial_state_json)
        except Exception as e:
            logger.error(f"Bad initial_state JSON: {e}")
            updated_state = {}

        try:
            event: Dict[str, Any] = json.loads(event_json)
        except Exception as e:
            logger.error(f"Bad event JSON: {e}")
            event = {}

        # causal_model not needed here but parse if supplied
        if causal_model_json:
            try:
                _ = json.loads(causal_model_json)
            except Exception:
                pass

        # ---- Local helpers ------------------------------------------------
        def bump(var: str, delta: float, clamp_lo=0.0, clamp_hi=1.0):
            if var in state_vars and isinstance(state_vars[var], (int, float)):
                state_vars[var] = max(clamp_lo, min(clamp_hi, state_vars[var] + delta))

        # ---- Apply event --------------------------------------------------
        state_vars = updated_state.get("state_variables", {}).copy()
        action = event.get("action", "")
        description = event.get("description", "")
        text = f"{action} {description}".lower()

        if "apologize" in text:
            bump("user_trust", 0.15)
            bump("user_anger", -0.20)
            bump("relationship_tension", -0.15)

        elif "explain" in text:
            bump("user_comprehension", 0.20)
            bump("communication_quality", 0.10)

        elif "share" in text:
            bump("intimacy", 0.15)
            bump("user_trust", 0.10)
            bump("relationship_depth", 0.10)

        elif "disagree" in text:
            bump("relationship_tension", 0.15)
            bump("intellectual_engagement", 0.20)

        else:
            # Sentiment heuristic
            pos = sum(w in text for w in ["help", "improve", "increase", "better", "good", "positive", "success"])
            neg = sum(w in text for w in ["harm", "worsen", "decrease", "bad", "negative", "failure"])
            sentiment = pos - neg
            for var in list(state_vars):
                if any(k in var for k in ["trust", "satisfaction", "quality"]) and isinstance(state_vars[var], (int, float)):
                    bump(var, 0.10 * sentiment)

        updated_state["state_variables"] = state_vars
        updated_state["last_action"] = action or description

        # ---- Return strict DTO -------------------------------------------
        return SimulationStateDTO(state_json=json.dumps(updated_state, separators=(",", ":")))

@function_tool
async def apply_counterfactual_condition(
    ctx: RunContextWrapper[SimulationContext],
    initial_state_json: str,                   # <<< JSON strings
    counterfactual_json: str,
    causal_model_json: str | None = None,
) -> SimulationStateDTO:
    """
    Apply a counterfactual ('what-if') condition to the simulation state.

    Args:
        initial_state_json:  JSON of the starting state.
        counterfactual_json: JSON describing the counterfactual.
        causal_model_json:   JSON causal model (optional, only needed for
                             name-matching heuristics).

    Returns:
        SimulationStateDTO with the updated state in `state_json`.
    """
    with custom_span("apply_counterfactual_condition"):
        # ---------- parse inputs ------------------------------------------
        try:
            updated_state: Dict[str, Any] = json.loads(initial_state_json)
        except Exception as e:
            logger.error(f"Bad initial_state JSON: {e}")
            updated_state = {}

        try:
            counterfactual: Dict[str, Any] = json.loads(counterfactual_json)
        except Exception as e:
            logger.error(f"Bad counterfactual JSON: {e}")
            counterfactual = {}

        causal_nodes: Dict[str, Any] = {}
        if causal_model_json:
            try:
                causal_nodes = json.loads(causal_model_json).get("nodes", {})
            except Exception:
                pass

        # ---------- helpers ----------------------------------------------
        state_vars = updated_state.get("state_variables", {}).copy()

        def set_var(var: str, val: float):
            state_vars[var] = max(0.0, min(1.0, val))

        def bump(var: str, delta: float):
            if var in state_vars and isinstance(state_vars[var], (int, float)):
                set_var(var, state_vars[var] + delta)

        # ---------- apply counterfactual ----------------------------------
        node_id = counterfactual.get("node_id")
        val     = counterfactual.get("value")
        description = counterfactual.get("description", "")
        text = description.lower()

        if node_id and val is not None:
            set_var(node_id, float(val))

        else:      # try to infer from description
            matched = False
            for n_id, n_data in causal_nodes.items():
                name = n_data.get("name", "").lower()
                if name and name in text:
                    if any(w in text for w in ["high", "increase"]):
                        bump(n_id, 0.3)
                    elif any(w in text for w in ["low", "decrease"]):
                        bump(n_id, -0.3)
                    elif "very high" in text:
                        set_var(n_id, 0.9)
                    elif "very low"  in text:
                        set_var(n_id, 0.1)
                    matched = True
            if not matched:
                # sentiment / emotion shortcuts
                if "user was happy" in text or "felt good" in text:
                    set_var("user_satisfaction", 0.8)
                    emo = updated_state.get("emotional_state", {}).copy()
                    emo.update({"valence": 0.7,
                                "primary_emotion": {"name": "Joy", "intensity": 0.8}})
                    updated_state["emotional_state"] = emo
                elif "user was unhappy" in text or "felt bad" in text:
                    set_var("user_satisfaction", 0.2)
                    emo = updated_state.get("emotional_state", {}).copy()
                    emo.update({"valence": -0.6,
                                "primary_emotion": {"name": "Sadness", "intensity": 0.7}})
                    updated_state["emotional_state"] = emo
                elif "relationship was stronger" in text:
                    bump("relationship_depth", 0.3)
                    bump("user_trust", 0.3)
                elif "relationship was weaker" in text:
                    bump("relationship_depth", -0.3)
                    bump("user_trust", -0.2)

        updated_state["state_variables"] = state_vars
        updated_state["reasoning_focus"] = f"Counterfactual: {description}"

        # ---------- return strict DTO ------------------------------------
        return SimulationStateDTO(state_json=json.dumps(updated_state, separators=(",", ":")))
        
@function_tool
async def check_goal_condition(
    ctx: RunContextWrapper[SimulationContext],
    current_state_json: str,
    goal_condition_json: str,
) -> GoalCheckResultDTO:
    """
    Determine whether the goal condition is satisfied by the current
    simulation state.

    Args:
        current_state_json:  JSON string of the current state
        goal_condition_json: JSON string of the goal condition
    """
    with custom_span("check_goal_condition"):
        # ---------- parse -------------------------------------------------
        try:
            current_state = json.loads(current_state_json)
        except Exception as e:
            logger.error(f"Bad current_state JSON: {e}")
            current_state = {}

        try:
            goal_condition = json.loads(goal_condition_json)
        except Exception as e:
            logger.error(f"Bad goal_condition JSON: {e}")
            goal_condition = {}

        state_vars = current_state.get("state_variables", {}) or {}

        # ---------- evaluate ---------------------------------------------
        all_met = True
        conditions: Dict[str, bool] = {}

        for var, target in goal_condition.items():
            current_val = state_vars.get(var, None)

            if current_val is None:              # variable missing
                conditions[var] = False
                all_met = False
                continue

            if isinstance(target, (int, float)) and isinstance(current_val, (int, float)):
                met = abs(current_val - target) <= 0.1  # 10 % tolerance
            else:
                met = current_val == target

            conditions[var] = met
            if not met:
                all_met = False

        # ---------- return DTO -------------------------------------------
        return GoalCheckResultDTO(
            goal_met=all_met,
            conditions_met_json=json.dumps(conditions, separators=(",", ":")),
        )

@function_tool
async def evaluate_simulation_stability(
    ctx: RunContextWrapper[SimulationContext],
    current_state_json: str,
    previous_state_json: str,
) -> StabilityEvaluationDTO:
    """
    Determine whether the simulation has reached a stable state by measuring
    the average absolute change in numeric state_variables between successive
    steps.

    Args:
        current_state_json:  JSON string of the *current* SimulationState
        previous_state_json: JSON string of the *previous* SimulationState
    """
    with custom_span("evaluate_simulation_stability"):

        # ---------- parse JSON safely ------------------------------------
        try:
            current_state = json.loads(current_state_json)
        except Exception as e:
            logger.error(f"Bad current_state JSON: {e}")
            current_state = {}

        try:
            previous_state = json.loads(previous_state_json)
        except Exception as e:
            logger.error(f"Bad previous_state JSON: {e}")
            previous_state = {}

        cur_vars = current_state.get("state_variables", {}) or {}
        prev_vars = previous_state.get("state_variables", {}) or {}

        # ---------- compute per-variable deltas --------------------------
        changes: Dict[str, float] = {}
        total_change = 0.0
        counted = 0

        for name, cur_val in cur_vars.items():
            if name in prev_vars:
                prev_val = prev_vars[name]
                if isinstance(cur_val, (int, float)) and isinstance(prev_val, (int, float)):
                    delta = abs(cur_val - prev_val)
                    changes[name] = delta
                    total_change += delta
                    counted += 1

        avg_change = total_change / counted if counted else 0.0
        is_stable  = avg_change < 0.05          # < 5 % average change
        confidence = 1.0 - min(1.0, avg_change * 10)

        # ---------- return DTO -------------------------------------------
        return StabilityEvaluationDTO(
            is_stable=is_stable,
            avg_change=avg_change,
            changes_json=json.dumps(changes, separators=(",", ":")),
            stability_confidence=confidence,
        )

@function_tool
async def analyze_simulation_result(
    ctx: RunContextWrapper[SimulationContext],
    simulation_result_json: str,
) -> SimulationAnalysisDTO:
    """
    Strict version of analyse_simulation_result.

    Args:
        simulation_result_json: JSON-encoded SimulationResult (or similar dict)
    """

    with custom_span("analyze_simulation_result"):
        try:
            simulation_result = json.loads(simulation_result_json)
        except Exception as e:
            logger.error(f"Bad simulation_result JSON: {e}")
            simulation_result = {}

        # ---------- extract basics --------------------------------------
        sim_id   = simulation_result.get("simulation_id", "unknown")
        success  = simulation_result.get("success", False)
        reason   = simulation_result.get("termination_reason", "unknown")
        final    = simulation_result.get("final_state", {})
        traj     = simulation_result.get("trajectory", [])

        state_vars     = final.get("state_variables", {}) or {}
        init_state_vars = (
            traj[0].get("state_variables", {}) if traj else {}
        )

        # ---------- variable deltas -------------------------------------
        variable_changes: Dict[str, Any] = {}
        for k, v_final in state_vars.items():
            v_init = init_state_vars.get(k)
            if isinstance(v_final, (int, float)) and isinstance(v_init, (int, float)):
                delta = v_final - v_init
                variable_changes[k] = {
                    "initial": v_init,
                    "final": v_final,
                    "change": delta,
                    "percent_change": (delta / v_init * 100) if v_init else 0,
                }

        significant_changes = {
            k: v for k, v in variable_changes.items() if abs(v["change"]) > 0.1
        }

        # ---------- emotional impact ------------------------------------
        emotional_impact: Dict[str, Any] = {}
        if "emotional_state" in final:
            final_em = final["emotional_state"]
            if traj and "emotional_state" in traj[0]:
                init_em = traj[0]["emotional_state"]

                if all(x in final_em and x in init_em for x in ("valence", "arousal")):
                    emotional_impact["valence_change"] = final_em["valence"] - init_em["valence"]
                    emotional_impact["arousal_change"] = final_em["arousal"] - init_em["arousal"]

                if "primary_emotion" in final_em and "primary_emotion" in init_em:
                    init_name = init_em["primary_emotion"]["name"] if isinstance(
                        init_em["primary_emotion"], dict
                    ) else init_em["primary_emotion"]
                    fin_name = final_em["primary_emotion"]["name"] if isinstance(
                        final_em["primary_emotion"], dict
                    ) else final_em["primary_emotion"]
                    emotional_impact["emotion_transition"] = f"{init_name} -> {fin_name}"

            pe = final_em.get("primary_emotion")
            if pe:
                if isinstance(pe, dict):
                    emotional_impact["final_emotion"]     = pe.get("name", "Unknown")
                    emotional_impact["emotion_intensity"] = pe.get("intensity", 0.5)
                else:
                    emotional_impact["final_emotion"] = pe
                    emotional_impact["emotion_intensity"] = 0.5

        # ---------- insights -------------------------------------------
        insights: List[str] = []
        insights.append(
            f"The simulation was {'successful' if success else 'unsuccessful'}, "
            f"terminating due to {reason}."
        )
        if significant_changes:
            changes_txt = ", ".join(
                f"{k} ({v['change']:.2f})" for k, v in significant_changes.items()
            )
            insights.append(f"Significant changes occurred in: {changes_txt}")
        else:
            insights.append("No significant changes occurred in state variables.")

        if "emotion_transition" in emotional_impact:
            insights.append(f"Emotional transition: {emotional_impact['emotion_transition']}")
        if "valence_change" in emotional_impact:
            direction = "positive" if emotional_impact["valence_change"] > 0 else "negative"
            insights.append(f"Emotional valence shifted in a {direction} direction.")

        # ---------- confidence -----------------------------------------
        conf_factors = {
            "trajectory_length": min(1.0, len(traj) / 10) * 0.3,
            "termination": 0.0,
            "stability": 0.0,
        }
        if reason == "goal_reached":
            conf_factors["termination"] = 0.4
        elif reason == "stable_state":
            conf_factors["termination"] = 0.3
            conf_factors["stability"] = 0.2
        elif reason == "max_steps":
            conf_factors["termination"] = 0.1

        confidence = sum(conf_factors.values()) + 0.2

        # ---------- return DTO -----------------------------------------
        return SimulationAnalysisDTO(
            simulation_id=sim_id,
            variable_changes_json=json.dumps(variable_changes, separators=(",", ":")),
            significant_changes_json=json.dumps(significant_changes, separators=(",", ":")),
            emotional_impact_json=json.dumps(emotional_impact, separators=(",", ":")),
            insights_json=json.dumps(insights, separators=(",", ":")),
            confidence=confidence,
            confidence_factors_json=json.dumps(conf_factors, separators=(",", ":")),
        )
        
@function_tool
async def generate_simulation_reflection(
    ctx: RunContextWrapper[SimulationContext],
    simulation_result_json: str,
    analysis_json: str,
) -> SimulationReflectionDTO:
    """
    Strict version of generate_simulation_reflection.

    Args:
        simulation_result_json: JSON-encoded SimulationResult dict
        analysis_json:          JSON-encoded analysis dict (from analyse_simulation_result)
    """

    with custom_span("generate_simulation_reflection"):
        try:
            sim = json.loads(simulation_result_json)
        except Exception as e:
            logger.error(f"Bad simulation_result JSON: {e}")
            sim = {}

        try:
            analysis = json.loads(analysis_json)
        except Exception as e:
            logger.error(f"Bad analysis JSON: {e}")
            analysis = {}

        reflection_txt: str = ""
        confidence: float = 0.0
        src: str = "error"

        # ---------- use ReflectionEngine if available ------------------
        if ctx.context.reflection_engine:
            try:
                memories: List[Dict[str, Any]] = []

                # initial state
                if sim.get("trajectory"):
                    init_state = sim["trajectory"][0]
                    memories.append(
                        {
                            "id": f"sim_init_{sim.get('simulation_id','?')}",
                            "memory_text": (
                                f"Initial state of simulation '{sim.get('description','Simulation')}' "
                                f"with variables: {json.dumps(init_state.get('state_variables', {}))}"
                            ),
                            "memory_type": "simulation",
                            "metadata": {
                                "emotional_context": init_state.get("emotional_state", {}),
                                "simulation_step": 0,
                                "simulation_id": sim.get("simulation_id"),
                            },
                        }
                    )

                # final state
                final_state = sim.get("final_state", {})
                memories.append(
                    {
                        "id": f"sim_final_{sim.get('simulation_id','?')}",
                        "memory_text": (
                            f"Final state of simulation '{sim.get('description','Simulation')}' "
                            f"with variables: {json.dumps(final_state.get('state_variables', {}))}"
                        ),
                        "memory_type": "simulation",
                        "metadata": {
                            "emotional_context": final_state.get("emotional_state", {}),
                            "simulation_step": len(sim.get("trajectory", [])) - 1,
                            "simulation_id": sim.get("simulation_id"),
                            "termination_reason": sim.get("termination_reason", "unknown"),
                        },
                    }
                )

                # significant steps (≤3)
                traj = sim.get("trajectory", [])
                sig_idxs: List[int] = []
                for i in range(1, len(traj)):
                    prev = traj[i - 1].get("state_variables", {})
                    cur  = traj[i].get("state_variables", {})
                    if any(
                        isinstance(cur.get(k), (int, float)) and isinstance(prev.get(k), (int, float))
                        and abs(cur[k] - prev[k]) >= 0.1
                        for k in cur
                    ):
                        sig_idxs.append(i)
                for idx in sig_idxs[:3]:
                    step = traj[idx]
                    memories.append(
                        {
                            "id": f"sim_step_{sim.get('simulation_id','?')}_{idx}",
                            "memory_text": (
                                f"At step {idx} of simulation '{sim.get('description','Simulation')}', "
                                f"variables changed to: {json.dumps(step.get('state_variables', {}))}"
                            ),
                            "memory_type": "simulation",
                            "metadata": {
                                "emotional_context": step.get("emotional_state", {}),
                                "simulation_step": idx,
                                "simulation_id": sim.get("simulation_id"),
                                "last_action": step.get("last_action", ""),
                            },
                        }
                    )

                # optional neurochemicals
                neuro = None
                if ctx.context.emotional_core and hasattr(ctx.context.emotional_core, "neurochemicals"):
                    neuro = {c: d["value"] for c, d in ctx.context.emotional_core.neurochemicals.items()}

                topic = f"Simulation: {sim.get('description','Hypothetical scenario')}"
                reflection_txt, confidence = await ctx.context.reflection_engine.generate_reflection(
                    memories=memories,
                    topic=topic,
                    neurochemical_state=neuro,
                )
                src = "reflection_engine"

            except Exception as e:
                logger.error(f"ReflectionEngine failure: {e}")
                reflection_txt = f"I tried to reflect on this simulation but encountered difficulties: {e}"
                confidence = 0.3
                src = "error"

        # ---------- fallback basic reflection --------------------------
        if src == "error" and not reflection_txt:
            insights = analysis.get("insights", [])
            emo_imp  = analysis.get("emotional_impact", {})

            parts: List[str] = [
                f"When I imagine this scenario of {sim.get('description','this hypothetical situation')}, I notice several interesting patterns.",
                f"The simulation {('succeeded' if sim.get('success') else 'did not succeed')}, ending due to {sim.get('termination_reason','unknown reasons')}."
            ]
            parts.extend(insights[:2])
            if emo_imp:
                if "emotion_transition" in emo_imp:
                    parts.append(f"Emotional shift observed: {emo_imp['emotion_transition']}.")
                elif "final_emotion" in emo_imp:
                    parts.append(f"The scenario seems to result in {emo_imp['final_emotion']}.")

            parts.append("This exercise helps me anticipate possible outcomes and prepare for similar situations.")
            reflection_txt = " ".join(parts)
            confidence = 0.5
            src = "basic_generation"

        return SimulationReflectionDTO(
            reflection=reflection_txt,
            confidence=confidence,
            source=src,
            timestamp=datetime.datetime.now().isoformat(),
        )

@function_tool
async def generate_abstraction_from_simulation(
    ctx: RunContextWrapper[SimulationContext],
    simulation_result_json: str,
    pattern_type: str = "causal",
) -> SimulationAbstractionDTO:
    """
    Strict version of generate_abstraction_from_simulation.

    Args:
        simulation_result_json: JSON-encoded SimulationResult dict
        pattern_type:           pattern family to look for ("causal", "temporal", …)
    """

    with custom_span("generate_abstraction_from_simulation"):

        # -------- parse inputs safely ---------------------------------
        try:
            sim = json.loads(simulation_result_json)
        except Exception as e:
            logger.error(f"Bad simulation_result JSON: {e}")
            sim = {}

        result: Dict[str, Any] = {
            "abstraction_text": "",
            "pattern_type": pattern_type,
            "confidence": 0.0,
            "supporting_evidence": [],
        }

        # -------- use ReflectionEngine if we can ----------------------
        if ctx.context.reflection_engine and hasattr(ctx.context.reflection_engine, "create_abstraction"):
            try:
                traj = sim.get("trajectory", [])
                memories: List[Dict[str, Any]] = [
                    {
                        "id": f"sim_state_{sim.get('simulation_id','?')}_{i}",
                        "memory_text": f"Step {i} of simulation: {json.dumps(s.get('state_variables', {}))}",
                        "memory_type": "simulation",
                        "significance": 7.0,
                        "metadata": {
                            "emotional_context": s.get("emotional_state", {}),
                            "simulation_step": i,
                            "simulation_id": sim.get("simulation_id"),
                            "last_action": s.get("last_action", ""),
                        },
                    }
                    for i, s in enumerate(traj)
                ]

                neuro = None
                if ctx.context.emotional_core and hasattr(ctx.context.emotional_core, "neurochemicals"):
                    neuro = {c: d["value"] for c, d in ctx.context.emotional_core.neurochemicals.items()}

                abs_text, abs_data = await ctx.context.reflection_engine.create_abstraction(
                    memories=memories,
                    pattern_type=pattern_type,
                    neurochemical_state=neuro,
                )

                result["abstraction_text"] = abs_text
                result["confidence"] = abs_data.get("confidence", 0.5)
                # copy optional keys if present
                for k in ("pattern_type", "entity_focus", "supporting_evidence", "neurochemical_insight"):
                    if k in abs_data:
                        result[k] = abs_data[k]

            except Exception as e:
                logger.error(f"ReflectionEngine abstraction error: {e}")
                result["abstraction_text"] = f"I tried to generate an abstraction but encountered difficulties: {e}"
                result["confidence"] = 0.2

        # -------- fallback basic heuristic abstraction ----------------
        if not result["abstraction_text"]:
            traj = sim.get("trajectory", [])
            if len(traj) >= 3:
                var_trends: Dict[str, List[float]] = {}
                for i in range(1, len(traj)):
                    prev = traj[i - 1].get("state_variables", {})
                    cur = traj[i].get("state_variables", {})
                    for v, cur_val in cur.items():
                        if v in prev and all(isinstance(x, (int, float)) for x in (cur_val, prev[v])):
                            var_trends.setdefault(v, []).append(cur_val - prev[v])

                trend_patterns: Dict[str, str] = {}
                for v, deltas in var_trends.items():
                    if len(deltas) >= 2:
                        same_sign = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
                        accel = (
                            len(deltas) >= 3
                            and all(abs(deltas[i]) > abs(deltas[i - 1]) for i in range(1, len(deltas)))
                        )
                        decel = (
                            len(deltas) >= 3
                            and all(abs(deltas[i]) < abs(deltas[i - 1]) for i in range(1, len(deltas)))
                        )
                        osc = any(deltas[i] * deltas[i - 1] < 0 for i in range(1, len(deltas)))

                        if same_sign:
                            direction = "increasing" if deltas[0] > 0 else "decreasing"
                            speed = "accelerating" if accel else "decelerating" if decel else "constant"
                            trend_patterns[v] = f"{direction} at {speed} rate"
                        elif osc:
                            trend_patterns[v] = "oscillating"

                if trend_patterns:
                    desc = ", ".join(f"{k} ({p})" for k, p in list(trend_patterns.items())[:3])
                    result["abstraction_text"] = (
                        f"In this simulation I observe that {desc}. "
                        f"This suggests a {pattern_type} relationship under similar conditions."
                    )
                    result["confidence"] = 0.6
                    result["supporting_evidence"] = [f"Trend in {k}: {p}" for k, p in trend_patterns.items()]
                else:
                    result["abstraction_text"] = "I don't see a clear pattern in this simulation data."
                    result["confidence"] = 0.3
            else:
                result["abstraction_text"] = "This simulation is too short to extract meaningful patterns."
                result["confidence"] = 0.2

        # -------- return strict DTO ----------------------------------
        return SimulationAbstractionDTO(**result)
        
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
                setup_simulation_from_description,
                get_causal_model_for_simulation
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
                analyze_simulation_result,
                generate_simulation_reflection,
                generate_abstraction_from_simulation
            ],
            output_type=SimulationAnalysisOutput
        )
        
        # Fix #1: Add explicit reminder about tool arguments
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
            
            IMPORTANT: When you call the tool `setup_simulation_from_description`, you MUST supply
            all three named arguments:
              • description                (string)
              • current_brain_state_json   (stringified JSON, can be "{}")
              • domain                     (string, defaults to "general")
            
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
                setup_simulation_from_description,
                get_causal_model_for_simulation,
                predict_simulation_step,
                apply_hypothetical_event,
                apply_counterfactual_condition,
                check_goal_condition,
                evaluate_simulation_stability
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
        def set_nested_value(d: dict, path: str, value: Any) -> None:
            """Helper to set nested dictionary values using dot notation."""
            keys = path.split('.')
            for key in keys[:-1]:
                if key not in d:
                    d[key] = {}
                d = d[key]
            d[keys[-1]] = value
        
        with trace(workflow_name="SetupSimulation", group_id=self.trace_group_id):
            try:
                # Configure the run
                run_config = RunConfig(
                    workflow_name="Simulation Setup",
                    trace_id=f"sim-setup-{gen_trace_id()}",
                    trace_metadata={"description": description}
                )
                
                # Fix #1: Provide the JSON up-front when invoking the Orchestrator
                result = await Runner.run(
                    self.orchestrator_agent,
                    {
                        "role": "user",
                        "content": (
                            "Generate a simulation setup for this description.\n"
                            f"description: {description}\n"
                            f"current_brain_state_json: {json.dumps(current_brain_state)}\n"
                            "domain: general"
                        ),
                    },
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
                    
                    # Apply initial state modifications with nested path support
                    for key, value in scenario.initial_state_modifications.items():
                        set_nested_value(sim_input.initial_state, key, value)
                    
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
                    
                    # Apply any modifications from the output with nested path support
                    if "initial_state_modifications" in result.final_output:
                        for key, value in result.final_output["initial_state_modifications"].items():
                            set_nested_value(sim_input.initial_state, key, value)
                    
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
            causal_model_dto = await get_causal_model_for_simulation(
                RunContextWrapper(self.context),
                sim_input.domain
            )
            causal_model = json.loads(causal_model_dto.model_json)
            
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
                state_dto = await apply_hypothetical_event(
                    RunContextWrapper(self.context),
                    json.dumps(initial_state),
                    json.dumps(sim_input.hypothetical_event),
                    json.dumps(causal_model)
                )
                initial_state = json.loads(state_dto.state_json)
            
            if sim_input.counterfactual_condition:
                state_dto = await apply_counterfactual_condition(
                    RunContextWrapper(self.context),
                    json.dumps(initial_state),
                    json.dumps(sim_input.counterfactual_condition),
                    json.dumps(causal_model)
                )
                initial_state = json.loads(state_dto.state_json)
            
            # Add initial state to trajectory
            trajectory.append(SimulationState(**initial_state))
            
            # Initialize result variables
            termination_reason = "max_steps"
            success = False
            
            # Run simulation steps
            try:
                for step in range(1, sim_input.max_steps + 1):
                    # Fix #2: Use model_dump_json() instead of json()
                    current_state_json = trajectory[-1].model_dump_json()
                    
                    # Predict next state
                    state_dto = await predict_simulation_step(
                        RunContextWrapper(self.context),
                        current_state_json,
                        json.dumps(causal_model),
                        step
                    )
                    next_state_dict = json.loads(state_dto.state_json)
                    
                    # Add to trajectory
                    trajectory.append(SimulationState(**next_state_dict))
                    
                    # Check goal condition if specified
                    if sim_input.goal_condition:
                        # Fix #5: Ensure we're wrapping with json.dumps
                        goal_dto = await check_goal_condition(
                            RunContextWrapper(self.context),
                            state_dto.state_json,  # Already a JSON string
                            json.dumps(sim_input.goal_condition)
                        )
                        
                        if goal_dto.goal_met:
                            termination_reason = "goal_reached"
                            success = True
                            break
                    
                    # Check for stability
                    if step > 1:
                        stab_dto = await evaluate_simulation_stability(
                            RunContextWrapper(self.context),
                            state_dto.state_json,  # Already a JSON string
                            trajectory[-2].model_dump_json()  # Fix #2: Use model_dump_json()
                        )
                        
                        if stab_dto.is_stable:
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
                    # Fix #5: Use model_dump_json() for proper JSON serialization
                    # Generate reflection
                    reflection_dto = await generate_simulation_reflection(
                        RunContextWrapper(self.context),
                        result.model_dump_json(),  # Fix #5: Use model_dump_json()
                        json.dumps(result.causal_analysis or {})
                    )
                    
                    # Update result with reflection
                    result.reflection = reflection_dto.reflection
                    
                    # Generate abstraction
                    abstraction_dto = await generate_abstraction_from_simulation(
                        RunContextWrapper(self.context),
                        result.model_dump_json(),  # Fix #5: Use model_dump_json()
                        "causal"
                    )
                    
                    # Update result with abstraction
                    result.abstraction = abstraction_dto.model_dump()
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
        # Fix #3: Use isoparse for proper datetime sorting
        history = list(self.simulation_history.values())
        history.sort(key=lambda x: isoparse(x.trajectory[-1].timestamp) if x.trajectory else datetime.datetime.min, reverse=True)
        
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
    
    @staticmethod
    async def _smoke():
        """Quick sanity test to verify fixes are working"""
        sim = ImaginationSimulator()
        brain = {"clarity": 0.0, "awareness": 0.2}
        out = await sim.imagine_scenario(
            "What would happen if everything changed?",
            brain
        )
        print(json.dumps(out, indent=2))

# To run the smoke test:
# asyncio.run(ImaginationSimulator._smoke())
