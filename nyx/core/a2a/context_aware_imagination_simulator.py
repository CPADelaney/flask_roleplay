# nyx/core/a2a/context_aware_imagination_simulator.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareImaginationSimulator(ContextAwareModule):
    """
    Enhanced ImaginationSimulator with full context distribution capabilities
    """
    
    def __init__(self, original_imagination_system):
        super().__init__("imagination_simulator")
        self.original_system = original_imagination_system
        self.context_subscriptions = [
            "goal_context_available", "emotional_state_update", "memory_retrieval_complete",
            "causal_model_update", "prediction_request", "counterfactual_query",
            "planning_request", "identity_state_update", "relationship_state_change",
            "exploration_trigger"
        ]
        
        # Cache for cross-module coordination
        self.active_simulations = {}
        self.simulation_queue = []
        self.causal_model_cache = {}
    
    async def on_context_received(self, context: SharedContext):
        """Initialize imagination processing for this context"""
        logger.debug(f"ImaginationSimulator received context for user: {context.user_id}")
        
        # Analyze context for simulation opportunities
        simulation_opportunities = await self._identify_simulation_opportunities(context)
        
        # Get current brain state for simulations
        brain_state = await self._extract_brain_state_from_context(context)
        
        # Send initial imagination readiness
        await self.send_context_update(
            update_type="imagination_ready",
            data={
                "simulation_opportunities": simulation_opportunities,
                "brain_state_snapshot": brain_state,
                "simulation_capabilities": {
                    "hypothetical": True,
                    "counterfactual": True,
                    "goal_planning": True,
                    "outcome_prediction": True
                }
            },
            priority=ContextPriority.NORMAL
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that trigger imagination"""
        
        if update.update_type == "goal_context_available":
            # Goals can trigger planning simulations
            goal_data = update.data
            await self._process_goal_simulation_request(goal_data)
            
        elif update.update_type == "emotional_state_update":
            # Strong emotions might trigger outcome simulations
            emotional_data = update.data
            await self._process_emotional_simulation_trigger(emotional_data)
            
        elif update.update_type == "memory_retrieval_complete":
            # Retrieved memories can seed simulations
            memory_data = update.data
            await self._process_memory_based_simulation(memory_data)
            
        elif update.update_type == "causal_model_update":
            # Update causal models for better simulations
            model_data = update.data
            await self._update_causal_model_cache(model_data)
            
        elif update.update_type == "prediction_request":
            # Direct request for outcome prediction
            prediction_data = update.data
            await self._process_prediction_request(prediction_data)
            
        elif update.update_type == "counterfactual_query":
            # Counterfactual reasoning request
            counterfactual_data = update.data
            await self._process_counterfactual_query(counterfactual_data)
            
        elif update.update_type == "planning_request":
            # Planning simulation request
            planning_data = update.data
            await self._process_planning_request(planning_data)
            
        elif update.update_type == "identity_state_update":
            # Identity changes might trigger self-simulations
            identity_data = update.data
            await self._process_identity_simulation(identity_data)
            
        elif update.update_type == "relationship_state_change":
            # Relationship changes trigger social simulations
            relationship_data = update.data
            await self._process_relationship_simulation(relationship_data)
            
        elif update.update_type == "exploration_trigger":
            # Curiosity-driven exploration simulations
            exploration_data = update.data
            await self._process_exploration_simulation(exploration_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for imagination triggers"""
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Analyze input for simulation triggers
        simulation_analysis = await self._analyze_input_for_simulation(
            context.user_input, context, messages
        )
        
        # Check if we should run any simulations
        if simulation_analysis.get("triggers_simulation"):
            # Create and queue simulations
            simulations = await self._create_simulations_from_input(
                context, simulation_analysis, messages
            )
            
            # Run high-priority simulations immediately
            immediate_results = []
            for sim in simulations:
                if sim.get("priority", "normal") == "high":
                    result = await self._run_context_aware_simulation(sim, context, messages)
                    immediate_results.append(result)
                else:
                    self.simulation_queue.append(sim)
            
            # Send simulation initiation update
            await self.send_context_update(
                update_type="simulations_initiated",
                data={
                    "immediate_count": len(immediate_results),
                    "queued_count": len(self.simulation_queue),
                    "simulation_types": [s.get("type") for s in simulations]
                }
            )
            
            return {
                "simulation_analysis": simulation_analysis,
                "simulations_created": len(simulations),
                "immediate_results": immediate_results,
                "context_integrated": True
            }
        
        return {
            "simulation_analysis": simulation_analysis,
            "simulations_created": 0,
            "context_integrated": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze simulation opportunities and results"""
        # Get cross-module context
        messages = await self.get_cross_module_messages()
        
        # Analyze active simulations
        active_analysis = await self._analyze_active_simulations()
        
        # Analyze simulation history for patterns
        pattern_analysis = await self._analyze_simulation_patterns(messages)
        
        # Generate simulation insights
        simulation_insights = await self._generate_simulation_insights(context, messages)
        
        # Identify missed opportunities
        missed_opportunities = await self._identify_missed_opportunities(context, messages)
        
        return {
            "active_simulations": active_analysis,
            "pattern_analysis": pattern_analysis,
            "simulation_insights": simulation_insights,
            "missed_opportunities": missed_opportunities,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize simulation results for response generation"""
        # Get all relevant context
        messages = await self.get_cross_module_messages()
        
        # Run any queued simulations if time permits
        synthesis_simulations = await self._run_synthesis_simulations(context, messages)
        
        # Synthesize insights from all simulations
        synthesized_insights = await self._synthesize_simulation_insights(
            synthesis_simulations, context, messages
        )
        
        # Generate predictive guidance
        predictive_guidance = await self._generate_predictive_guidance(
            synthesized_insights, context
        )
        
        # Generate counterfactual awareness
        counterfactual_awareness = await self._generate_counterfactual_awareness(
            context, messages
        )
        
        # Create imagination synthesis
        imagination_synthesis = {
            "simulation_insights": synthesized_insights,
            "predictive_guidance": predictive_guidance,
            "counterfactual_awareness": counterfactual_awareness,
            "imaginative_elements": await self._suggest_imaginative_elements(context),
            "future_projections": await self._project_future_states(context, messages)
        }
        
        # Send synthesis to response generation
        await self.send_context_update(
            update_type="imagination_synthesis_complete",
            data=imagination_synthesis,
            priority=ContextPriority.NORMAL
        )
        
        return {
            "imagination_synthesis": imagination_synthesis,
            "simulations_run": len(synthesis_simulations),
            "synthesis_complete": True
        }
    
    # Enhanced helper methods
    
    async def _identify_simulation_opportunities(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify opportunities for simulations from context"""
        opportunities = []
        
        # Check for goal-based opportunities
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            for goal in active_goals:
                if goal.get("priority", 0) > 0.6:
                    opportunities.append({
                        "type": "goal_planning",
                        "description": f"Plan path to achieve: {goal.get('description', 'goal')}",
                        "priority": goal.get("priority", 0.5)
                    })
        
        # Check for emotional exploration opportunities
        if context.emotional_state:
            emotion_intensity = 0.0
            primary_emotion = context.emotional_state.get("primary_emotion", {})
            if isinstance(primary_emotion, dict):
                emotion_intensity = primary_emotion.get("intensity", 0.0)
            
            if emotion_intensity > 0.7:
                opportunities.append({
                    "type": "emotional_outcome",
                    "description": "Explore emotional trajectory",
                    "priority": emotion_intensity
                })
        
        # Check for relationship opportunities
        if context.relationship_context:
            recent_change = context.relationship_context.get("recent_change", 0)
            if abs(recent_change) > 0.2:
                opportunities.append({
                    "type": "relationship_projection",
                    "description": "Project relationship development",
                    "priority": abs(recent_change)
                })
        
        # Check for uncertainty in input
        if "what if" in context.user_input.lower() or "?" in context.user_input:
            opportunities.append({
                "type": "hypothetical_exploration",
                "description": "Explore hypothetical scenarios",
                "priority": 0.7
            })
        
        return opportunities
    
    async def _extract_brain_state_from_context(self, context: SharedContext) -> Dict[str, Any]:
        """Extract current brain state from context"""
        brain_state = {
            "emotional_valence": 0.0,
            "emotional_arousal": 0.5,
            "goal_activation": 0.5,
            "relationship_depth": 0.5,
            "uncertainty_level": 0.3,
            "exploration_drive": 0.5
        }
        
        # Extract from emotional state
        if context.emotional_state:
            brain_state["emotional_valence"] = context.emotional_state.get("valence", 0.0)
            brain_state["emotional_arousal"] = context.emotional_state.get("arousal", 0.5)
        
        # Extract from goal context
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            if active_goals:
                avg_priority = sum(g.get("priority", 0.5) for g in active_goals) / len(active_goals)
                brain_state["goal_activation"] = avg_priority
        
        # Extract from relationship context
        if context.relationship_context:
            brain_state["relationship_depth"] = context.relationship_context.get("depth", 0.5)
        
        # Extract from session context
        if context.session_context:
            brain_state["uncertainty_level"] = context.session_context.get("uncertainty", 0.3)
            brain_state["exploration_drive"] = context.session_context.get("curiosity", 0.5)
        
        return brain_state
    
    async def _process_goal_simulation_request(self, goal_data: Dict[str, Any]):
        """Process goal-based simulation requests"""
        active_goals = goal_data.get("active_goals", [])
        
        for goal in active_goals:
            if goal.get("priority", 0) > 0.7:
                # Create goal achievement simulation
                simulation = {
                    "id": f"goal_sim_{uuid.uuid4().hex[:8]}",
                    "type": "goal_planning",
                    "description": f"Simulate achieving: {goal.get('description', 'goal')}",
                    "goal_condition": {
                        "goal_id": goal.get("id"),
                        "target_state": goal.get("target_state", {})
                    },
                    "priority": goal.get("priority", 0.5),
                    "max_steps": 15
                }
                
                # Run simulation
                result = await self._run_goal_simulation(simulation, goal)
                
                # Send results to goal manager
                await self.send_context_update(
                    update_type="goal_simulation_complete",
                    data={
                        "goal_id": goal.get("id"),
                        "simulation_result": result,
                        "success_probability": result.get("confidence", 0.5),
                        "key_steps": self._extract_key_steps(result)
                    },
                    target_modules=["goal_manager"],
                    scope=ContextScope.TARGETED
                )
    
    async def _process_emotional_simulation_trigger(self, emotional_data: Dict[str, Any]):
        """Process emotion-triggered simulations"""
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        # Only simulate for strong emotions
        if isinstance(dominant_emotion, tuple) and len(dominant_emotion) >= 2:
            emotion_name, intensity = dominant_emotion[0], dominant_emotion[1]
        elif isinstance(dominant_emotion, dict):
            emotion_name = dominant_emotion.get("name", "")
            intensity = dominant_emotion.get("intensity", 0.0)
        else:
            return
        
        if intensity < 0.7:
            return
        
        # Create emotional trajectory simulation
        simulation = {
            "id": f"emo_sim_{uuid.uuid4().hex[:8]}",
            "type": "emotional_trajectory",
            "description": f"Explore trajectory of {emotion_name}",
            "initial_state": {
                "emotional_state": emotional_data,
                "intensity": intensity
            },
            "max_steps": 10
        }
        
        # Run simulation
        result = await self._run_emotional_simulation(simulation, emotional_data)
        
        # Send insights back to emotional core
        await self.send_context_update(
            update_type="emotional_trajectory_insight",
            data={
                "initial_emotion": emotion_name,
                "predicted_trajectory": result.get("trajectory", []),
                "stable_state": result.get("final_state", {}),
                "regulation_suggestions": self._generate_regulation_suggestions(result)
            },
            target_modules=["emotional_core"],
            scope=ContextScope.TARGETED
        )
    
    async def _process_memory_based_simulation(self, memory_data: Dict[str, Any]):
        """Process simulations based on retrieved memories"""
        memories = memory_data.get("retrieved_memories", [])
        
        # Look for pattern-rich memories
        significant_memories = [m for m in memories if m.get("significance", 0) > 7]
        
        if significant_memories:
            # Create pattern projection simulation
            simulation = {
                "id": f"mem_sim_{uuid.uuid4().hex[:8]}",
                "type": "pattern_projection",
                "description": "Project patterns from past experiences",
                "memory_seeds": significant_memories[:3],  # Use top 3
                "max_steps": 8
            }
            
            # Run simulation
            result = await self._run_memory_simulation(simulation, significant_memories)
            
            # Send pattern insights
            await self.send_context_update(
                update_type="pattern_projection_complete",
                data={
                    "identified_patterns": result.get("patterns", []),
                    "future_projections": result.get("projections", []),
                    "confidence": result.get("confidence", 0.5)
                }
            )
    
    async def _process_prediction_request(self, prediction_data: Dict[str, Any]):
        """Process direct prediction requests"""
        target_variable = prediction_data.get("target_variable", "")
        time_horizon = prediction_data.get("time_horizon", "short")
        context_data = prediction_data.get("context", {})
        
        # Create prediction simulation
        simulation = {
            "id": f"pred_sim_{uuid.uuid4().hex[:8]}",
            "type": "outcome_prediction",
            "description": f"Predict {target_variable} over {time_horizon} term",
            "target": target_variable,
            "initial_state": context_data,
            "max_steps": 5 if time_horizon == "short" else 15
        }
        
        # Run simulation
        result = await self._run_prediction_simulation(simulation, prediction_data)
        
        # Send prediction results
        await self.send_context_update(
            update_type="prediction_complete",
            data={
                "target_variable": target_variable,
                "prediction": result.get("predicted_outcome"),
                "confidence": result.get("confidence", 0.5),
                "key_factors": result.get("causal_analysis", {})
            },
            priority=ContextPriority.HIGH
        )
    
    async def _process_counterfactual_query(self, counterfactual_data: Dict[str, Any]):
        """Process counterfactual reasoning requests"""
        condition = counterfactual_data.get("condition", "")
        context_state = counterfactual_data.get("current_state", {})
        
        # Create counterfactual simulation
        brain_state = await self._get_current_brain_state()
        
        result = await self.original_system.imagine_counterfactual(
            description=condition,
            variable_name=counterfactual_data.get("variable", "general_state"),
            variable_value=counterfactual_data.get("value", 0.8),
            current_brain_state=brain_state
        )
        
        # Send counterfactual insights
        await self.send_context_update(
            update_type="counterfactual_analysis_complete",
            data={
                "condition": condition,
                "outcomes": result.get("predicted_outcome", ""),
                "key_differences": result.get("key_insights", []),
                "confidence": result.get("confidence", 0.5)
            }
        )
    
    async def _process_planning_request(self, planning_data: Dict[str, Any]):
        """Process planning simulation requests"""
        objective = planning_data.get("objective", "")
        constraints = planning_data.get("constraints", [])
        resources = planning_data.get("resources", {})
        
        # Create planning simulation
        simulation = {
            "id": f"plan_sim_{uuid.uuid4().hex[:8]}",
            "type": "strategic_planning",
            "description": f"Plan to achieve: {objective}",
            "goal_condition": {"objective_met": 1.0},
            "constraints": constraints,
            "resources": resources,
            "max_steps": 20
        }
        
        # Run simulation
        result = await self._run_planning_simulation(simulation, planning_data)
        
        # Send planning results
        await self.send_context_update(
            update_type="planning_complete",
            data={
                "objective": objective,
                "recommended_steps": self._extract_plan_steps(result),
                "success_probability": result.get("confidence", 0.5),
                "resource_requirements": self._analyze_resource_needs(result)
            }
        )
    
    async def _process_identity_simulation(self, identity_data: Dict[str, Any]):
        """Process identity-based self-simulations"""
        traits_changed = identity_data.get("traits_changed", [])
        
        if traits_changed:
            # Simulate identity evolution outcomes
            simulation = {
                "id": f"identity_sim_{uuid.uuid4().hex[:8]}",
                "type": "identity_projection",
                "description": "Project identity evolution outcomes",
                "changed_traits": traits_changed,
                "current_identity": identity_data,
                "max_steps": 12
            }
            
            # Run simulation
            result = await self._run_identity_simulation(simulation, identity_data)
            
            # Send identity insights
            await self.send_context_update(
                update_type="identity_projection_complete",
                data={
                    "projected_changes": result.get("trajectory", []),
                    "stable_configuration": result.get("final_state", {}),
                    "coherence_trajectory": self._analyze_coherence_trajectory(result)
                },
                target_modules=["identity_evolution"],
                scope=ContextScope.TARGETED
            )
    
    async def _process_relationship_simulation(self, relationship_data: Dict[str, Any]):
        """Process relationship-based simulations"""
        relationship_state = relationship_data.get("relationship_context", {})
        recent_change = relationship_state.get("recent_change", 0)
        
        if abs(recent_change) > 0.1:
            # Simulate relationship trajectory
            simulation = {
                "id": f"rel_sim_{uuid.uuid4().hex[:8]}",
                "type": "relationship_projection",
                "description": "Project relationship development",
                "initial_relationship": relationship_state,
                "change_momentum": recent_change,
                "max_steps": 15
            }
            
            # Run simulation
            result = await self._run_relationship_simulation(simulation, relationship_data)
            
            # Send relationship insights
            await self.send_context_update(
                update_type="relationship_projection_complete",
                data={
                    "projected_trajectory": result.get("trajectory", []),
                    "milestone_predictions": self._predict_milestones(result),
                    "stability_point": result.get("final_state", {})
                },
                target_modules=["relationship_manager"],
                scope=ContextScope.TARGETED
            )
    
    async def _process_exploration_simulation(self, exploration_data: Dict[str, Any]):
        """Process curiosity-driven exploration simulations"""
        exploration_target = exploration_data.get("target", "unknown")
        curiosity_level = exploration_data.get("curiosity_level", 0.5)
        
        if curiosity_level > 0.6:
            # Create exploration simulation
            brain_state = await self._get_current_brain_state()
            
            result = await self.original_system.imagine_scenario(
                description=f"What if I explored {exploration_target}?",
                current_brain_state=brain_state
            )
            
            # Send exploration insights
            await self.send_context_update(
                update_type="exploration_insights",
                data={
                    "exploration_target": exploration_target,
                    "discovered_possibilities": result.get("key_insights", []),
                    "excitement_potential": self._calculate_excitement_potential(result),
                    "learning_opportunities": self._identify_learning_opportunities(result)
                }
            )
    
    async def _analyze_input_for_simulation(self, user_input: str, 
                                          context: SharedContext, 
                                          messages: Dict) -> Dict[str, Any]:
        """Analyze user input for simulation triggers"""
        analysis = {
            "triggers_simulation": False,
            "simulation_types": [],
            "urgency": "normal",
            "specific_queries": []
        }
        
        input_lower = user_input.lower()
        
        # Check for explicit simulation triggers
        simulation_triggers = {
            "what if": "hypothetical",
            "what would happen": "outcome_prediction",
            "how can i": "planning",
            "imagine": "creative_exploration",
            "suppose": "counterfactual",
            "predict": "prediction",
            "will this": "outcome_prediction"
        }
        
        for trigger, sim_type in simulation_triggers.items():
            if trigger in input_lower:
                analysis["triggers_simulation"] = True
                analysis["simulation_types"].append(sim_type)
                analysis["specific_queries"].append(user_input)
        
        # Check for implicit triggers
        if "?" in user_input and any(word in input_lower for word in ["future", "outcome", "result", "consequence"]):
            analysis["triggers_simulation"] = True
            analysis["simulation_types"].append("outcome_exploration")
        
        # Check cross-module messages for simulation needs
        for module_name, module_messages in messages.items():
            if module_name == "goal_manager":
                for msg in module_messages:
                    if msg["type"] == "planning_needed":
                        analysis["triggers_simulation"] = True
                        analysis["simulation_types"].append("goal_planning")
                        analysis["urgency"] = "high"
        
        # Determine urgency based on context
        if context.emotional_state:
            emotion_intensity = 0.0
            primary_emotion = context.emotional_state.get("primary_emotion", {})
            if isinstance(primary_emotion, dict):
                emotion_intensity = primary_emotion.get("intensity", 0.0)
            
            if emotion_intensity > 0.8:
                analysis["urgency"] = "high"
        
        return analysis
    
    async def _create_simulations_from_input(self, context: SharedContext, 
                                           simulation_analysis: Dict[str, Any], 
                                           messages: Dict) -> List[Dict[str, Any]]:
        """Create simulation configurations from input analysis"""
        simulations = []
        
        for sim_type in simulation_analysis["simulation_types"]:
            if sim_type == "hypothetical":
                # Extract hypothetical from input
                description = context.user_input
                brain_state = await self._extract_brain_state_from_context(context)
                
                sim_input = await self.original_system.setup_simulation(description, brain_state)
                if sim_input:
                    simulations.append({
                        "type": sim_type,
                        "input": sim_input,
                        "priority": simulation_analysis["urgency"],
                        "context": context
                    })
            
            elif sim_type == "outcome_prediction":
                # Create outcome prediction simulation
                simulations.append({
                    "type": sim_type,
                    "description": f"Predict outcome of: {context.user_input}",
                    "priority": simulation_analysis["urgency"],
                    "target_variables": self._identify_target_variables(context)
                })
            
            elif sim_type == "planning":
                # Extract planning objective
                objective = self._extract_planning_objective(context.user_input)
                simulations.append({
                    "type": sim_type,
                    "objective": objective,
                    "priority": "high",  # Planning is usually high priority
                    "constraints": self._identify_constraints(context, messages)
                })
        
        return simulations
    
    async def _run_context_aware_simulation(self, simulation: Dict[str, Any], 
                                          context: SharedContext, 
                                          messages: Dict) -> Dict[str, Any]:
        """Run a simulation with full context awareness"""
        sim_type = simulation.get("type")
        
        # Add context to simulation
        if "input" in simulation and hasattr(simulation["input"], "initial_state"):
            # Enhance initial state with context
            simulation["input"].initial_state.update({
                "emotional_context": context.emotional_state,
                "relationship_context": context.relationship_context,
                "goal_context": context.goal_context
            })
        
        # Run appropriate simulation type
        if sim_type == "hypothetical" and "input" in simulation:
            result = await self.original_system.run_simulation(simulation["input"])
        else:
            # Use generic simulation for other types
            brain_state = await self._extract_brain_state_from_context(context)
            result = await self.original_system.imagine_scenario(
                description=simulation.get("description", ""),
                current_brain_state=brain_state
            )
        
        # Enhance results with context
        result["context_factors"] = self._analyze_context_influence(result, context)
        
        # Store active simulation
        self.active_simulations[result.get("simulation_id", "unknown")] = {
            "result": result,
            "context": context,
            "timestamp": datetime.now()
        }
        
        return result
    
    async def _run_synthesis_simulations(self, context: SharedContext, 
                                       messages: Dict) -> List[Dict[str, Any]]:
        """Run queued simulations during synthesis phase"""
        synthesis_results = []
        
        # Run up to 3 queued simulations
        simulations_to_run = self.simulation_queue[:3]
        self.simulation_queue = self.simulation_queue[3:]
        
        for sim in simulations_to_run:
            result = await self._run_context_aware_simulation(sim, context, messages)
            synthesis_results.append(result)
        
        return synthesis_results

    async def _calculate_processing_confidence(self, processing_results: Dict[str, Any]) -> float:
        """Calculate overall confidence in processing results"""
        confidence_factors = []
        
        # Pattern detection confidence
        patterns = processing_results.get("pattern_detections", [])
        if patterns:
            # Calculate average pattern confidence
            pattern_confidences = []
            for pattern in patterns:
                if isinstance(pattern, dict) and "confidence" in pattern:
                    pattern_confidences.append(pattern["confidence"])
            
            if pattern_confidences:
                avg_pattern_confidence = sum(pattern_confidences) / len(pattern_confidences)
                confidence_factors.append(avg_pattern_confidence)
            else:
                confidence_factors.append(0.5)  # Default if no confidence values
        
        # Behavior evaluation confidence
        evaluations = processing_results.get("behavior_evaluations", [])
        if evaluations:
            # Extract confidence from evaluations
            eval_confidences = []
            
            for eval_data in evaluations:
                if isinstance(eval_data, dict):
                    # Check if it's a behavior evaluation result
                    if "confidence" in eval_data:
                        eval_confidences.append(eval_data["confidence"])
                    # Check if it's cached behavior data with baseline
                    elif "baseline_frequency" in eval_data:
                        # Use baseline frequency as a proxy for confidence
                        # Higher baseline = more confident in behavior patterns
                        baseline = eval_data.get("baseline_frequency", 0.5)
                        confidence = 0.5 + (baseline * 0.3)  # Map to 0.5-0.8 range
                        eval_confidences.append(confidence)
            
            if eval_confidences:
                avg_eval_confidence = sum(eval_confidences) / len(eval_confidences)
                confidence_factors.append(avg_eval_confidence)
            else:
                confidence_factors.append(0.7)  # Default confidence for evaluations
        
        # Mode processing stability
        mode_processing = processing_results.get("mode_processing", {})
        if mode_processing:
            # Check if mode processing is recent and stable
            if isinstance(mode_processing, dict):
                # If we have distribution info
                if "distribution" in mode_processing:
                    distribution = mode_processing["distribution"]
                    # Check distribution entropy (lower = more confident)
                    if distribution:
                        values = list(distribution.values())
                        max_value = max(values) if values else 0
                        # High dominance of one mode = high confidence
                        if max_value > 0.7:
                            confidence_factors.append(0.9)
                        elif max_value > 0.5:
                            confidence_factors.append(0.7)
                        else:
                            confidence_factors.append(0.5)
                
                # Check if we have dominant mode consistency
                if "dominant_mode" in mode_processing:
                    # Having a clear dominant mode increases confidence
                    confidence_factors.append(0.8)
        
        # Check mode blending history for stability
        if hasattr(self, 'mode_blending_history') and self.mode_blending_history:
            recent_history = self.mode_blending_history[-3:]
            if len(recent_history) >= 2:
                # Check consistency of dominant modes
                dominant_modes = []
                for h in recent_history:
                    if isinstance(h, dict) and "dominant_mode" in h:
                        dominant_modes.append(h["dominant_mode"])
                
                if dominant_modes:
                    # All same dominant mode = high confidence
                    if len(set(dominant_modes)) == 1:
                        confidence_factors.append(0.9)
                    # Some variation = medium confidence
                    elif len(set(dominant_modes)) == 2:
                        confidence_factors.append(0.7)
                    # High variation = lower confidence
                    else:
                        confidence_factors.append(0.5)
        
        # Conditioning results confidence
        conditioning_applied = processing_results.get("conditioning_applied", [])
        if conditioning_applied:
            # Having successful conditioning applications increases confidence
            confidence_factors.append(0.75)
        
        # Calculate overall confidence
        if confidence_factors:
            # Weighted average with slight bias toward lower values (conservative estimate)
            overall_confidence = sum(confidence_factors) / len(confidence_factors)
            # Apply slight reduction for conservative estimate
            overall_confidence *= 0.95
            return overall_confidence
        
        return 0.5  # Default confidence if no factors
    
    async def _synthesize_simulation_insights(self, simulations: List[Dict[str, Any]], 
                                            context: SharedContext, 
                                            messages: Dict) -> Dict[str, Any]:
        """Synthesize insights from multiple simulations"""
        insights = {
            "key_patterns": [],
            "convergent_outcomes": [],
            "divergent_possibilities": [],
            "confidence_levels": {},
            "actionable_insights": []
        }
        
        if not simulations:
            return insights
        
        # Extract patterns across simulations
        outcome_clusters = {}
        for sim in simulations:
            outcome = sim.get("predicted_outcome", "unknown")
            confidence = sim.get("confidence", 0.5)
            
            if outcome not in outcome_clusters:
                outcome_clusters[outcome] = []
            outcome_clusters[outcome].append(confidence)
        
        # Identify convergent outcomes
        for outcome, confidences in outcome_clusters.items():
            avg_confidence = sum(confidences) / len(confidences)
            if len(confidences) >= 2 and avg_confidence > 0.6:
                insights["convergent_outcomes"].append({
                    "outcome": outcome,
                    "confidence": avg_confidence,
                    "simulation_count": len(confidences)
                })
        
        # Extract key patterns
        all_patterns = []
        for sim in simulations:
            if "causal_analysis" in sim:
                patterns = sim["causal_analysis"].get("patterns", [])
                all_patterns.extend(patterns)
        
        # Deduplicate and rank patterns
        pattern_counts = {}
        for pattern in all_patterns:
            pattern_str = str(pattern)
            pattern_counts[pattern_str] = pattern_counts.get(pattern_str, 0) + 1
        
        # Get most common patterns
        top_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        insights["key_patterns"] = [{"pattern": p, "frequency": c} for p, c in top_patterns]
        
        # Identify divergent possibilities
        unique_outcomes = set()
        for sim in simulations:
            trajectory = sim.get("trajectory", [])
            for state in trajectory:
                if "state_variables" in state:
                    for var, value in state["state_variables"].items():
                        unique_outcomes.add(f"{var}={value:.2f}" if isinstance(value, float) else f"{var}={value}")
        
        if len(unique_outcomes) > 5:
            insights["divergent_possibilities"] = list(unique_outcomes)[:10]
        
        # Generate actionable insights
        for sim in simulations:
            if sim.get("confidence", 0) > 0.7:
                key_insights = sim.get("key_insights", [])
                for insight in key_insights[:2]:  # Top 2 insights per simulation
                    if insight not in insights["actionable_insights"]:
                        insights["actionable_insights"].append(insight)
        
        return insights
    
    async def _generate_predictive_guidance(self, synthesized_insights: Dict[str, Any], 
                                          context: SharedContext) -> Dict[str, Any]:
        """Generate predictive guidance for response generation"""
        guidance = {
            "likely_outcomes": [],
            "risk_factors": [],
            "opportunity_factors": [],
            "recommended_approach": ""
        }
        
        # Extract likely outcomes
        convergent_outcomes = synthesized_insights.get("convergent_outcomes", [])
        for outcome in convergent_outcomes:
            if outcome["confidence"] > 0.7:
                guidance["likely_outcomes"].append(outcome["outcome"])
        
        # Identify risk factors
        key_patterns = synthesized_insights.get("key_patterns", [])
        for pattern in key_patterns:
            if "risk" in str(pattern.get("pattern", "")).lower():
                guidance["risk_factors"].append(pattern["pattern"])
        
        # Identify opportunity factors
        actionable_insights = synthesized_insights.get("actionable_insights", [])
        for insight in actionable_insights:
            if any(word in insight.lower() for word in ["opportunity", "potential", "could"]):
                guidance["opportunity_factors"].append(insight)
        
        # Generate recommended approach based on context
        if context.emotional_state:
            emotion_valence = context.emotional_state.get("valence", 0)
            if emotion_valence < -0.5 and guidance["risk_factors"]:
                guidance["recommended_approach"] = "cautious_supportive"
            elif emotion_valence > 0.5 and guidance["opportunity_factors"]:
                guidance["recommended_approach"] = "optimistic_encouraging"
            else:
                guidance["recommended_approach"] = "balanced_analytical"
        else:
            guidance["recommended_approach"] = "neutral_informative"
        
        return guidance
    
    async def _generate_counterfactual_awareness(self, context: SharedContext, 
                                               messages: Dict) -> Dict[str, Any]:
        """Generate awareness of counterfactual possibilities"""
        awareness = {
            "alternative_paths": [],
            "missed_opportunities": [],
            "what_if_scenarios": []
        }
        
        # Check recent simulations for counterfactuals
        for sim_id, sim_data in self.active_simulations.items():
            result = sim_data["result"]
            if result.get("counterfactual"):
                awareness["alternative_paths"].append({
                    "condition": result["counterfactual"].get("description", ""),
                    "outcome": result.get("predicted_outcome", ""),
                    "probability": result.get("confidence", 0.5)
                })
        
        # Generate what-if scenarios based on context
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            for goal in active_goals[:2]:  # Top 2 goals
                awareness["what_if_scenarios"].append({
                    "scenario": f"What if we prioritized {goal.get('description', 'this goal')}?",
                    "potential_impact": "high" if goal.get("priority", 0) > 0.7 else "moderate"
                })
        
        # Identify missed opportunities from messages
        for module_name, module_messages in messages.items():
            if module_name == "goal_manager":
                for msg in module_messages:
                    if msg["type"] == "opportunity_missed":
                        awareness["missed_opportunities"].append(msg["data"])
        
        return awareness
    
    async def _suggest_imaginative_elements(self, context: SharedContext) -> List[str]:
        """Suggest imaginative elements for response"""
        suggestions = []
        
        # Based on emotional state
        if context.emotional_state:
            emotion_valence = context.emotional_state.get("valence", 0)
            if emotion_valence > 0.5:
                suggestions.append("optimistic_future_vision")
                suggestions.append("creative_possibilities")
            elif emotion_valence < -0.5:
                suggestions.append("transformative_reframing")
                suggestions.append("hope_despite_challenges")
        
        # Based on user input patterns
        if "imagine" in context.user_input.lower() or "dream" in context.user_input.lower():
            suggestions.append("vivid_scenario_painting")
            suggestions.append("sensory_details")
        
        # Based on relationship depth
        if context.relationship_context:
            intimacy = context.relationship_context.get("intimacy", 0.5)
            if intimacy > 0.7:
                suggestions.append("shared_future_vision")
                suggestions.append("collaborative_dreaming")
        
        return suggestions
    
    async def _project_future_states(self, context: SharedContext, 
                                   messages: Dict) -> List[Dict[str, Any]]:
        """Project possible future states"""
        projections = []
        
        # Short-term projection (next interaction)
        short_term = {
            "timeframe": "next_interaction",
            "emotional_state": self._project_emotional_state(context.emotional_state, "short"),
            "relationship_state": self._project_relationship_state(context.relationship_context, "short"),
            "goal_progress": self._project_goal_progress(context.goal_context, "short")
        }
        projections.append(short_term)
        
        # Medium-term projection (next few interactions)
        medium_term = {
            "timeframe": "next_few_interactions",
            "emotional_state": self._project_emotional_state(context.emotional_state, "medium"),
            "relationship_state": self._project_relationship_state(context.relationship_context, "medium"),
            "goal_progress": self._project_goal_progress(context.goal_context, "medium")
        }
        projections.append(medium_term)
        
        return projections
    
    # Specialized simulation runners
    
    async def _run_goal_simulation(self, simulation: Dict[str, Any], 
                                  goal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run goal-specific simulation"""
        brain_state = await self._get_current_brain_state()
        
        # Add goal-specific state
        brain_state["goal_activation"] = goal_data.get("priority", 0.5)
        brain_state["goal_type"] = goal_data.get("associated_need", "general")
        
        # Run simulation
        result = await self.original_system.imagine_scenario(
            description=simulation["description"],
            current_brain_state=brain_state
        )
        
        return result
    
    async def _run_emotional_simulation(self, simulation: Dict[str, Any], 
                                      emotional_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run emotion-specific simulation"""
        brain_state = await self._get_current_brain_state()
        
        # Set emotional state
        brain_state.update({
            "emotional_valence": emotional_data.get("valence", 0),
            "emotional_arousal": emotional_data.get("arousal", 0.5)
        })
        
        # Run simulation with emotional focus
        sim_input = await self.original_system.setup_simulation(
            simulation["description"],
            brain_state
        )
        
        if sim_input:
            sim_input.domain = "emotional"
            result = await self.original_system.run_simulation(sim_input)
            return result.model_dump() if hasattr(result, "model_dump") else result
        
        return {"success": False, "error": "Failed to setup emotional simulation"}
    
    async def _run_memory_simulation(self, simulation: Dict[str, Any], 
                                   memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run memory-based pattern simulation"""
        # Extract patterns from memories
        patterns = []
        for memory in memories:
            if "pattern" in memory.get("metadata", {}):
                patterns.append(memory["metadata"]["pattern"])
        
        # Create projection based on patterns
        brain_state = await self._get_current_brain_state()
        
        # Run pattern projection
        result = {
            "patterns": patterns,
            "projections": [],
            "confidence": 0.7 if patterns else 0.3
        }
        
        # Project each pattern forward
        for pattern in patterns[:3]:  # Limit to top 3 patterns
            projection = {
                "pattern": pattern,
                "likelihood": 0.6,  # Base likelihood
                "conditions": "similar_context"
            }
            result["projections"].append(projection)
        
        return result
    
    async def _run_prediction_simulation(self, simulation: Dict[str, Any], 
                                       prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run prediction-specific simulation"""
        brain_state = await self._get_current_brain_state()
        brain_state.update(simulation.get("initial_state", {}))
        
        # Run prediction
        result = await self.original_system.imagine_scenario(
            description=simulation["description"],
            current_brain_state=brain_state
        )
        
        # Extract prediction-specific results
        if "trajectory" in result and result["trajectory"]:
            final_state = result["trajectory"][-1]
            if simulation["target"] in final_state.get("state_variables", {}):
                result["predicted_outcome"] = final_state["state_variables"][simulation["target"]]
        
        return result
    
    async def _run_planning_simulation(self, simulation: Dict[str, Any], 
                                     planning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run planning-specific simulation"""
        brain_state = await self._get_current_brain_state()
        
        # Set goal condition
        sim_input = await self.original_system.setup_simulation(
            simulation["description"],
            brain_state
        )
        
        if sim_input:
            sim_input.goal_condition = simulation.get("goal_condition", {})
            sim_input.max_steps = simulation.get("max_steps", 20)
            
            result = await self.original_system.run_simulation(sim_input)
            return result.model_dump() if hasattr(result, "model_dump") else result
        
        return {"success": False, "error": "Failed to setup planning simulation"}
    
    async def _run_identity_simulation(self, simulation: Dict[str, Any], 
                                     identity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run identity evolution simulation"""
        brain_state = await self._get_current_brain_state()
        
        # Add identity-specific state
        brain_state["identity_coherence"] = identity_data.get("coherence_score", 0.8)
        brain_state["trait_dominance"] = max(identity_data.get("traits", {}).values()) if identity_data.get("traits") else 0.5
        
        # Run simulation
        result = await self.original_system.imagine_scenario(
            description=simulation["description"],
            current_brain_state=brain_state
        )
        
        return result
    
    async def _run_relationship_simulation(self, simulation: Dict[str, Any], 
                                         relationship_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run relationship projection simulation"""
        brain_state = await self._get_current_brain_state()
        
        # Add relationship state
        rel_state = relationship_data.get("relationship_context", {})
        brain_state.update({
            "relationship_depth": rel_state.get("depth", 0.5),
            "relationship_trust": rel_state.get("trust", 0.5),
            "relationship_intimacy": rel_state.get("intimacy", 0.5)
        })
        
        # Run simulation
        result = await self.original_system.imagine_scenario(
            description=simulation["description"],
            current_brain_state=brain_state
        )
        
        return result
    
    # Analysis helper methods
    
    async def _analyze_active_simulations(self) -> Dict[str, Any]:
        """Analyze currently active simulations"""
        active_count = len(self.active_simulations)
        
        # Categorize by type
        type_counts = {}
        avg_confidence = 0.0
        
        for sim_id, sim_data in self.active_simulations.items():
            result = sim_data["result"]
            sim_type = result.get("type", "unknown")
            type_counts[sim_type] = type_counts.get(sim_type, 0) + 1
            avg_confidence += result.get("confidence", 0.5)
        
        if active_count > 0:
            avg_confidence /= active_count
        
        return {
            "active_count": active_count,
            "type_distribution": type_counts,
            "average_confidence": avg_confidence,
            "oldest_simulation": min(
                (s["timestamp"] for s in self.active_simulations.values()),
                default=datetime.now()
            )
        }
    
    async def _analyze_simulation_patterns(self, messages: Dict) -> Dict[str, Any]:
        """Analyze patterns in simulation history"""
        patterns = {
            "common_scenarios": {},
            "outcome_trends": {},
            "confidence_trends": []
        }
        
        # Analyze recent simulations
        for sim_id, sim_data in list(self.active_simulations.items())[-20:]:  # Last 20
            result = sim_data["result"]
            
            # Track scenario types
            scenario = result.get("description", "")
            if scenario:
                scenario_type = self._classify_scenario(scenario)
                patterns["common_scenarios"][scenario_type] = patterns["common_scenarios"].get(scenario_type, 0) + 1
            
            # Track outcome trends
            outcome = result.get("predicted_outcome", "")
            if outcome:
                patterns["outcome_trends"][outcome] = patterns["outcome_trends"].get(outcome, 0) + 1
            
            # Track confidence
            patterns["confidence_trends"].append(result.get("confidence", 0.5))
        
        return patterns
    
    async def _generate_simulation_insights(self, context: SharedContext, 
                                          messages: Dict) -> List[str]:
        """Generate insights from simulation patterns"""
        insights = []
        
        # Analyze simulation success rate
        success_count = sum(1 for _, sim in self.active_simulations.items() 
                          if sim["result"].get("success", False))
        total_count = len(self.active_simulations)
        
        if total_count > 0:
            success_rate = success_count / total_count
            if success_rate > 0.8:
                insights.append("High simulation success rate indicates good predictive capability")
            elif success_rate < 0.5:
                insights.append("Low simulation success rate suggests high uncertainty")
        
        # Analyze convergence patterns
        if total_count >= 5:
            recent_outcomes = [sim["result"].get("predicted_outcome", "") 
                             for _, sim in list(self.active_simulations.items())[-5:]]
            unique_outcomes = len(set(recent_outcomes))
            
            if unique_outcomes == 1:
                insights.append("Recent simulations converge on a single outcome")
            elif unique_outcomes == len(recent_outcomes):
                insights.append("Recent simulations show high divergence in outcomes")
        
        return insights
    
    async def _identify_missed_opportunities(self, context: SharedContext, 
                                           messages: Dict) -> List[Dict[str, Any]]:
        """Identify missed simulation opportunities"""
        missed = []
        
        # Check if strong emotions weren't simulated
        if context.emotional_state:
            emotion_intensity = 0.0
            primary_emotion = context.emotional_state.get("primary_emotion", {})
            if isinstance(primary_emotion, dict):
                emotion_intensity = primary_emotion.get("intensity", 0.0)
            
            if emotion_intensity > 0.8:
                # Check if we ran emotional simulations
                emotional_sims = [s for s in self.active_simulations.values() 
                                if "emotional" in s["result"].get("type", "")]
                
                if not emotional_sims:
                    missed.append({
                        "type": "emotional_simulation",
                        "reason": "Strong emotion not explored",
                        "potential_value": "high"
                    })
        
        # Check if high-priority goals weren't simulated
        if context.goal_context:
            high_priority_goals = [g for g in context.goal_context.get("active_goals", [])
                                 if g.get("priority", 0) > 0.8]
            
            for goal in high_priority_goals:
                goal_id = goal.get("id")
                goal_sims = [s for s in self.active_simulations.values()
                           if goal_id in str(s["result"])]
                
                if not goal_sims:
                    missed.append({
                        "type": "goal_simulation",
                        "reason": f"High-priority goal '{goal.get('description', 'goal')}' not simulated",
                        "potential_value": "high"
                    })
        
        return missed
    
    # Helper utility methods
    
    async def _get_current_brain_state(self) -> Dict[str, Any]:
        """Get current brain state from context"""
        brain_state = {
            "emotional_valence": 0.0,
            "emotional_arousal": 0.5,
            "goal_activation": 0.5,
            "relationship_depth": 0.5,
            "uncertainty_level": 0.3,
            "exploration_drive": 0.5,
            "identity_coherence": 0.8,
            "cognitive_load": 0.3
        }
        
        # Extract from current context if available
        if hasattr(self, 'current_context') and self.current_context:
            context = self.current_context
            
            # Get emotional state
            if context.emotional_state:
                brain_state["emotional_valence"] = context.emotional_state.get("valence", 0.0)
                brain_state["emotional_arousal"] = context.emotional_state.get("arousal", 0.5)
                
                # Extract from primary emotion if available
                primary_emotion = context.emotional_state.get("primary_emotion", {})
                if isinstance(primary_emotion, dict):
                    emotion_intensity = primary_emotion.get("intensity", 0.5)
                    brain_state["emotional_arousal"] = emotion_intensity
            
            # Get goal activation
            if context.goal_context:
                active_goals = context.goal_context.get("active_goals", [])
                if active_goals:
                    # Average priority of active goals
                    avg_priority = sum(g.get("priority", 0.5) for g in active_goals) / len(active_goals)
                    brain_state["goal_activation"] = avg_priority
                    
                    # Increase cognitive load based on number of active goals
                    brain_state["cognitive_load"] = min(1.0, 0.3 + len(active_goals) * 0.1)
            
            # Get relationship depth
            if context.relationship_context:
                brain_state["relationship_depth"] = context.relationship_context.get("depth", 0.5)
                
                # Trust affects uncertainty
                trust = context.relationship_context.get("trust", 0.5)
                brain_state["uncertainty_level"] = 0.5 - (trust * 0.3)  # Higher trust = lower uncertainty
            
            # Get session context values
            if context.session_context:
                brain_state["uncertainty_level"] = context.session_context.get("uncertainty", brain_state["uncertainty_level"])
                brain_state["exploration_drive"] = context.session_context.get("curiosity", 0.5)
                
                # Check for specific task context
                if context.session_context.get("task_type") == "problem_solving":
                    brain_state["cognitive_load"] = min(1.0, brain_state["cognitive_load"] + 0.2)
                    brain_state["goal_activation"] = min(1.0, brain_state["goal_activation"] + 0.2)
        
        # Get from original system if available
        elif hasattr(self, 'original_system'):
            # Try to get emotional state from emotional core
            if hasattr(self.original_system, 'emotional_core') and self.original_system.emotional_core:
                try:
                    emotional_state = self.original_system.emotional_core.get_emotional_state()
                    brain_state["emotional_valence"] = self.original_system.emotional_core.get_emotional_valence()
                    brain_state["emotional_arousal"] = self.original_system.emotional_core.get_emotional_arousal()
                except:
                    pass
            
            # Try to get goal state from goal manager
            if hasattr(self.original_system, 'goal_manager') and self.original_system.goal_manager:
                try:
                    active_goals = await self.original_system.goal_manager.get_all_goals(status_filter=["active"])
                    if active_goals:
                        avg_priority = sum(g.get("priority", 0.5) for g in active_goals) / len(active_goals)
                        brain_state["goal_activation"] = avg_priority
                except:
                    pass
            
            # Try to get relationship state
            if hasattr(self.original_system, 'relationship_manager') and self.original_system.relationship_manager:
                try:
                    relationship_state = await self.original_system.relationship_manager.get_relationship_state()
                    brain_state["relationship_depth"] = relationship_state.get("depth", 0.5)
                except:
                    pass
        
        # Ensure all values are in valid range [0, 1] except valence [-1, 1]
        for key, value in brain_state.items():
            if key == "emotional_valence":
                brain_state[key] = max(-1.0, min(1.0, value))
            else:
                brain_state[key] = max(0.0, min(1.0, value))
        
        return brain_state
    
    def _extract_key_steps(self, result: Dict[str, Any]) -> List[str]:
        """Extract key steps from simulation result"""
        steps = []
        
        trajectory = result.get("trajectory", [])
        for i, state in enumerate(trajectory):
            if i == 0:
                continue  # Skip initial state
            
            # Look for significant changes
            if "last_action" in state:
                steps.append(f"Step {i}: {state['last_action']}")
        
        return steps[:5]  # Return top 5 steps
    
    def _generate_regulation_suggestions(self, result: Dict[str, Any]) -> List[str]:
        """Generate emotion regulation suggestions from simulation"""
        suggestions = []
        
        final_state = result.get("final_state", {})
        emotional_state = final_state.get("emotional_state", {})
        
        if emotional_state:
            valence = emotional_state.get("valence", 0)
            arousal = emotional_state.get("arousal", 0.5)
            
            if valence < -0.5 and arousal > 0.7:
                suggestions.append("Consider calming techniques to reduce arousal")
            elif valence < -0.5 and arousal < 0.3:
                suggestions.append("Consider activation strategies to increase engagement")
            elif valence > 0.5 and arousal > 0.8:
                suggestions.append("Channel high energy into productive activities")
        
        return suggestions
    
    def _extract_plan_steps(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract actionable plan steps from simulation"""
        steps = []
        
        trajectory = result.get("trajectory", [])
        for i, state in enumerate(trajectory):
            if "last_action" in state:
                step = {
                    "order": i,
                    "action": state["last_action"],
                    "expected_outcome": self._infer_outcome_from_state(state),
                    "confidence": result.get("confidence", 0.5) * (0.9 ** i)  # Decay confidence
                }
                steps.append(step)
        
        return steps
    
    def _analyze_resource_needs(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Analyze resource requirements from simulation"""
        resources = {
            "time": 0.0,
            "energy": 0.0,
            "attention": 0.0,
            "emotional_capacity": 0.0
        }
        
        trajectory = result.get("trajectory", [])
        
        # Estimate based on trajectory length
        resources["time"] = len(trajectory) * 0.1  # Each step requires time
        
        # Estimate based on state changes
        for state in trajectory:
            if "emotional_state" in state:
                arousal = state["emotional_state"].get("arousal", 0.5)
                resources["energy"] += arousal * 0.1
                resources["emotional_capacity"] += abs(state["emotional_state"].get("valence", 0)) * 0.1
            
            resources["attention"] += 0.05  # Each step requires attention
        
        return resources
    
    def _analyze_coherence_trajectory(self, result: Dict[str, Any]) -> List[float]:
        """Analyze identity coherence over simulation trajectory"""
        coherence_values = []
        
        trajectory = result.get("trajectory", [])
        for state in trajectory:
            # Simple coherence estimation based on state stability
            if "state_variables" in state:
                variance = self._calculate_state_variance(state["state_variables"])
                coherence = 1.0 - min(1.0, variance)  # Lower variance = higher coherence
                coherence_values.append(coherence)
        
        return coherence_values
    
    def _predict_milestones(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict relationship milestones from simulation"""
        milestones = []
        
        trajectory = result.get("trajectory", [])
        for i, state in enumerate(trajectory):
            if "state_variables" in state:
                trust = state["state_variables"].get("user_trust", 0.5)
                intimacy = state["state_variables"].get("intimacy", 0.5)
                
                # Check for milestone conditions
                if trust > 0.8 and intimacy > 0.7:
                    milestones.append({
                        "type": "deep_connection",
                        "estimated_step": i,
                        "probability": result.get("confidence", 0.5)
                    })
                elif trust > 0.9:
                    milestones.append({
                        "type": "complete_trust",
                        "estimated_step": i,
                        "probability": result.get("confidence", 0.5) * 0.8
                    })
        
        return milestones
    
    def _calculate_excitement_potential(self, result: Dict[str, Any]) -> float:
        """Calculate excitement potential from exploration simulation"""
        potential = 0.5  # Base potential
        
        # Check for novel outcomes
        unique_states = set()
        trajectory = result.get("trajectory", [])
        
        for state in trajectory:
            if "state_variables" in state:
                state_str = str(sorted(state["state_variables"].items()))
                unique_states.add(state_str)
        
        # More unique states = higher excitement potential
        novelty_factor = len(unique_states) / max(1, len(trajectory))
        potential += novelty_factor * 0.3
        
        # Check for positive emotional outcomes
        final_state = result.get("final_state", {})
        if "emotional_state" in final_state:
            valence = final_state["emotional_state"].get("valence", 0)
            if valence > 0:
                potential += valence * 0.2
        
        return min(1.0, potential)
    
    def _identify_learning_opportunities(self, result: Dict[str, Any]) -> List[str]:
        """Identify learning opportunities from exploration"""
        opportunities = []
        
        # Check for knowledge-related outcomes
        insights = result.get("key_insights", [])
        for insight in insights:
            if any(word in insight.lower() for word in ["learn", "discover", "understand", "realize"]):
                opportunities.append(insight)
        
        # Check for pattern discoveries
        if "causal_analysis" in result:
            patterns = result["causal_analysis"].get("patterns", [])
            if patterns:
                opportunities.append(f"Discover {len(patterns)} new patterns")
        
        return opportunities
    
    def _identify_target_variables(self, context: SharedContext) -> List[str]:
        """Identify target variables for prediction"""
        targets = []
        
        # Common prediction targets
        if context.relationship_context:
            targets.extend(["user_satisfaction", "relationship_depth", "trust"])
        
        if context.goal_context:
            targets.extend(["goal_progress", "goal_success"])
        
        if context.emotional_state:
            targets.extend(["emotional_valence", "emotional_stability"])
        
        return targets
    
    def _extract_planning_objective(self, user_input: str) -> str:
        """Extract planning objective from user input"""
        # Convert to lowercase for pattern matching
        input_lower = user_input.lower()
        
        # Define objective extraction patterns
        objective_patterns = [
            # Direct planning phrases
            (r"how can i\s+(.+?)(?:\?|$)", 1),
            (r"i want to\s+(.+?)(?:\.|$)", 1),
            (r"help me\s+(.+?)(?:\.|$)", 1),
            (r"i need to\s+(.+?)(?:\.|$)", 1),
            (r"plan to\s+(.+?)(?:\.|$)", 1),
            (r"trying to\s+(.+?)(?:\.|$)", 1),
            (r"goal is to\s+(.+?)(?:\.|$)", 1),
            # Question-based objectives
            (r"what's the best way to\s+(.+?)(?:\?|$)", 1),
            (r"how do i\s+(.+?)(?:\?|$)", 1),
            (r"can you help me\s+(.+?)(?:\?|$)", 1),
            # Indirect planning indicators
            (r"thinking about\s+(.+?)(?:\.|$)", 1),
            (r"considering\s+(.+?)(?:\.|$)", 1),
            (r"wondering how to\s+(.+?)(?:\.|$)", 1),
        ]
        
        # Try to extract objective using patterns
        import re
        for pattern, group_idx in objective_patterns:
            match = re.search(pattern, input_lower)
            if match:
                objective = match.group(group_idx).strip()
                # Clean up the objective
                objective = objective.rstrip('.,!?')
                # Capitalize first letter
                objective = objective[0].upper() + objective[1:] if objective else objective
                return objective
        
        # If no pattern matches, try to extract key action verbs
        action_verbs = ['achieve', 'accomplish', 'complete', 'finish', 'solve', 
                        'improve', 'learn', 'build', 'create', 'develop', 'reach']
        
        words = input_lower.split()
        for i, word in enumerate(words):
            if word in action_verbs and i < len(words) - 1:
                # Extract phrase starting from action verb
                objective_words = words[i:]
                objective = ' '.join(objective_words)
                objective = objective.rstrip('.,!?')
                return objective[0].upper() + objective[1:] if objective else objective
        
        # Final fallback: if input is short enough, use the whole thing
        if len(user_input) < 100:
            return user_input.strip()
        
        # Otherwise, extract first sentence/clause
        first_sentence = user_input.split('.')[0].strip()
        if len(first_sentence) < 100:
            return first_sentence
        
        # Last resort: first 80 characters
        return user_input[:80].strip() + "..."
    
    def _identify_constraints(self, context: SharedContext, messages: Dict) -> List[str]:
        """Identify constraints for planning"""
        constraints = []
        
        # Time constraints
        if context.session_context:
            if context.session_context.get("time_pressure", False):
                constraints.append("limited_time")
        
        # Emotional constraints
        if context.emotional_state:
            valence = context.emotional_state.get("valence", 0)
            if valence < -0.5:
                constraints.append("emotional_distress")
        
        # Resource constraints from messages
        for module_name, module_messages in messages.items():
            if module_name == "resource_manager":
                for msg in module_messages:
                    if msg["type"] == "low_resource":
                        constraints.append(f"low_{msg['data']['resource']}")
        
        return constraints
    
    def _analyze_context_influence(self, result: Dict[str, Any], 
                                  context: SharedContext) -> Dict[str, float]:
        """Analyze how context influenced simulation results"""
        influences = {}
        
        # Emotional influence
        if context.emotional_state and result.get("emotional_impact"):
            influences["emotional"] = 0.3
        
        # Goal influence
        if context.goal_context and "goal" in str(result.get("trajectory", [])):
            influences["goal"] = 0.4
        
        # Relationship influence
        if context.relationship_context and "relationship" in str(result.get("trajectory", [])):
            influences["relationship"] = 0.3
        
        return influences
    
    def _classify_scenario(self, scenario: str) -> str:
        """Classify scenario type from description"""
        scenario_lower = scenario.lower()
        
        if "goal" in scenario_lower or "achieve" in scenario_lower:
            return "goal_oriented"
        elif "emotion" in scenario_lower or "feel" in scenario_lower:
            return "emotional"
        elif "relationship" in scenario_lower or "together" in scenario_lower:
            return "relational"
        elif "what if" in scenario_lower:
            return "hypothetical"
        else:
            return "exploratory"
    
    def _infer_outcome_from_state(self, state: Dict[str, Any]) -> str:
        """Infer expected outcome from state"""
        if "state_variables" in state:
            # Find the highest value variable as likely outcome
            max_var = max(state["state_variables"].items(), 
                         key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
            return f"{max_var[0]} increases"
        
        return "state change"
    
    def _calculate_state_variance(self, state_variables: Dict[str, Any]) -> float:
        """Calculate variance in state variables"""
        numeric_values = [v for v in state_variables.values() if isinstance(v, (int, float))]
        
        if not numeric_values:
            return 0.0
        
        mean = sum(numeric_values) / len(numeric_values)
        variance = sum((x - mean) ** 2 for x in numeric_values) / len(numeric_values)
        
        return variance
    
    def _project_emotional_state(self, current_state: Optional[Dict[str, Any]], 
                               timeframe: str) -> Dict[str, Any]:
        """Project emotional state into future"""
        if not current_state:
            return {"valence": 0.0, "arousal": 0.5}
        
        valence = current_state.get("valence", 0.0)
        arousal = current_state.get("arousal", 0.5)
        
        # Apply decay based on timeframe
        decay = 0.9 if timeframe == "short" else 0.7 if timeframe == "medium" else 0.5
        
        return {
            "valence": valence * decay,
            "arousal": 0.5 + (arousal - 0.5) * decay  # Arousal trends toward baseline
        }
    
    def _project_relationship_state(self, current_state: Optional[Dict[str, Any]], 
                                  timeframe: str) -> Dict[str, Any]:
        """Project relationship state into future"""
        if not current_state:
            return {"depth": 0.5, "trust": 0.5}
        
        depth = current_state.get("depth", 0.5)
        trust = current_state.get("trust", 0.5)
        
        # Relationships tend to deepen slowly
        growth = 0.05 if timeframe == "short" else 0.1 if timeframe == "medium" else 0.15
        
        return {
            "depth": min(1.0, depth + growth * trust),  # Trust enables depth growth
            "trust": min(1.0, trust + growth * 0.5)  # Trust grows slower
        }
    
    def _project_goal_progress(self, current_state: Optional[Dict[str, Any]], 
                             timeframe: str) -> Dict[str, Any]:
        """Project goal progress into future"""
        if not current_state:
            return {"overall_progress": 0.5}
        
        active_goals = current_state.get("active_goals", [])
        if not active_goals:
            return {"overall_progress": 0.5}
        
        # Calculate average progress
        avg_progress = sum(g.get("progress", 0.5) for g in active_goals) / len(active_goals)
        
        # Project progress based on timeframe
        progress_rate = 0.1 if timeframe == "short" else 0.2 if timeframe == "medium" else 0.3
        
        return {
            "overall_progress": min(1.0, avg_progress + progress_rate),
            "goals_likely_completed": sum(1 for g in active_goals 
                                        if g.get("progress", 0) + progress_rate > 0.9)
        }
    
    # Delegate all other methods to the original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
