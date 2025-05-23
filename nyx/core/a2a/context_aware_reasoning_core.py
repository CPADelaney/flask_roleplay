# nyx/core/a2a/context_aware_reasoning_core.py

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import asyncio

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareReasoningCore(ContextAwareModule):
    """
    Context-aware wrapper for ReasoningCore with full A2A integration
    """
    
    def __init__(self, original_reasoning_core):
        super().__init__("reasoning_core")
        self.original_core = original_reasoning_core
        self.context_subscriptions = [
            "emotional_state_update", "memory_retrieval_complete", 
            "goal_context_available", "knowledge_update",
            "perception_input", "multimodal_integration",
            "causal_discovery_request", "conceptual_blend_request",
            "intervention_request", "counterfactual_query"
        ]
        
        # Track active reasoning processes
        self.active_models: Set[str] = set()
        self.active_spaces: Set[str] = set()
        self.active_interventions: Set[str] = set()
        self.reasoning_context: Dict[str, Any] = {}
    
    async def on_context_received(self, context: SharedContext):
        """Initialize reasoning processing for this context"""
        logger.debug(f"ReasoningCore received context for user: {context.user_id}")
        
        # Analyze input for reasoning-related content
        reasoning_implications = await self._analyze_input_for_reasoning(context.user_input)
        
        # Get relevant models and spaces for context
        relevant_models = await self._get_contextually_relevant_models(context)
        relevant_spaces = await self._get_contextually_relevant_spaces(context)
        
        # Send initial reasoning context to other modules
        await self.send_context_update(
            update_type="reasoning_context_available",
            data={
                "reasoning_implications": reasoning_implications,
                "available_models": relevant_models,
                "available_spaces": relevant_spaces,
                "active_reasoning_type": reasoning_implications.get("reasoning_type", "none"),
                "confidence": reasoning_implications.get("confidence", 0.0)
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect reasoning"""
        
        if update.update_type == "perception_input":
            # Update causal models with new perceptual data
            percept = update.data.get("percept")
            if percept and hasattr(self.original_core, 'update_with_perception'):
                await self.original_core.update_with_perception(percept)
                
                # Notify about model updates
                await self.send_context_update(
                    update_type="causal_models_updated",
                    data={
                        "updated_models": list(self.active_models),
                        "perception_modality": percept.modality if hasattr(percept, 'modality') else "unknown"
                    }
                )
        
        elif update.update_type == "causal_discovery_request":
            # Handle request for causal discovery
            model_id = update.data.get("model_id")
            if model_id:
                discovery_result = await self.original_core.discover_causal_relations(model_id)
                
                await self.send_context_update(
                    update_type="causal_discovery_complete",
                    data={
                        "model_id": model_id,
                        "discovery_result": discovery_result,
                        "new_relations": discovery_result.get("accepted_relations", 0)
                    },
                    priority=ContextPriority.HIGH
                )
        
        elif update.update_type == "conceptual_blend_request":
            # Handle request for conceptual blending
            space_ids = update.data.get("space_ids", [])
            blend_type = update.data.get("blend_type", "composition")
            
            if len(space_ids) >= 2:
                # Perform blending through the original core
                # This is a simplified version - the actual implementation would use the blending methods
                blend_result = await self._perform_contextual_blending(space_ids, blend_type)
                
                await self.send_context_update(
                    update_type="conceptual_blend_complete",
                    data={
                        "blend_result": blend_result,
                        "input_spaces": space_ids,
                        "blend_type": blend_type
                    }
                )
        
        elif update.update_type == "emotional_state_update":
            # Emotional state can influence causal reasoning
            emotional_data = update.data
            await self._adjust_reasoning_from_emotion(emotional_data)
        
        elif update.update_type == "goal_context_available":
            # Goals can guide reasoning direction
            goal_data = update.data
            await self._align_reasoning_with_goals(goal_data)
        
        elif update.update_type == "memory_retrieval_complete":
            # Use retrieved memories to inform causal models
            memory_data = update.data
            await self._inform_reasoning_from_memory(memory_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with full context awareness for reasoning"""
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Analyze input for reasoning needs
        reasoning_analysis = await self._analyze_input_for_reasoning(context.user_input)
        
        # Determine reasoning approach based on context
        reasoning_approach = await self._determine_contextual_reasoning_approach(
            context, reasoning_analysis, messages
        )
        
        # Execute reasoning based on approach
        reasoning_results = {}
        
        if reasoning_approach["type"] == "causal":
            reasoning_results = await self._execute_causal_reasoning(
                context, reasoning_approach
            )
        elif reasoning_approach["type"] == "conceptual":
            reasoning_results = await self._execute_conceptual_reasoning(
                context, reasoning_approach
            )
        elif reasoning_approach["type"] == "integrated":
            reasoning_results = await self._execute_integrated_reasoning(
                context, reasoning_approach
            )
        
        # Update context with reasoning results
        await self.send_context_update(
            update_type="reasoning_process_complete",
            data={
                "reasoning_type": reasoning_approach["type"],
                "reasoning_results": reasoning_results,
                "models_used": list(self.active_models),
                "spaces_used": list(self.active_spaces),
                "cross_module_integration": len(messages) > 0
            }
        )
        
        return {
            "reasoning_analysis": reasoning_analysis,
            "reasoning_approach": reasoning_approach,
            "reasoning_results": reasoning_results,
            "context_aware_reasoning": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze reasoning possibilities in current context"""
        # Get all relevant context
        messages = await self.get_cross_module_messages()
        
        # Analyze available reasoning resources
        available_models = list(self.original_core.causal_models.keys())
        available_spaces = list(self.original_core.concept_spaces.keys())
        available_blends = list(self.original_core.blends.keys())
        
        # Analyze reasoning potential based on context
        reasoning_potential = await self._analyze_reasoning_potential(
            context, messages, available_models, available_spaces
        )
        
        # Identify reasoning opportunities
        reasoning_opportunities = await self._identify_reasoning_opportunities(
            context, messages
        )
        
        # Assess reasoning coherence with other modules
        coherence_analysis = await self._analyze_reasoning_coherence(
            context, messages
        )
        
        return {
            "available_models": available_models,
            "available_spaces": available_spaces,
            "available_blends": available_blends,
            "reasoning_potential": reasoning_potential,
            "reasoning_opportunities": reasoning_opportunities,
            "coherence_analysis": coherence_analysis,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize reasoning insights for response generation"""
        messages = await self.get_cross_module_messages()
        
        # Generate reasoning-based insights
        reasoning_synthesis = {
            "causal_insights": await self._synthesize_causal_insights(context, messages),
            "conceptual_insights": await self._synthesize_conceptual_insights(context, messages),
            "counterfactual_insights": await self._synthesize_counterfactual_insights(context, messages),
            "intervention_suggestions": await self._synthesize_intervention_suggestions(context, messages),
            "reasoning_narrative": await self._generate_reasoning_narrative(context, messages),
            "confidence_assessment": await self._assess_reasoning_confidence(context)
        }
        
        # Check if we should announce any discoveries
        if reasoning_synthesis["causal_insights"].get("new_discoveries"):
            await self.send_context_update(
                update_type="causal_discovery_announcement",
                data={
                    "discoveries": reasoning_synthesis["causal_insights"]["new_discoveries"],
                    "impact": "high"
                },
                priority=ContextPriority.HIGH
            )
        
        return {
            "reasoning_synthesis": reasoning_synthesis,
            "synthesis_complete": True,
            "integrated_with_modules": list(messages.keys())
        }
    
    # ========================================================================================
    # REASONING ANALYSIS METHODS
    # ========================================================================================
    
    async def _analyze_input_for_reasoning(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input for reasoning-related implications"""
        input_lower = user_input.lower()
        
        analysis = {
            "suggests_causal_reasoning": any(kw in input_lower for kw in 
                ["cause", "effect", "why", "because", "leads to", "results in", "if then"]),
            "suggests_conceptual_reasoning": any(kw in input_lower for kw in 
                ["concept", "idea", "blend", "combine", "creative", "imagine"]),
            "suggests_counterfactual": any(kw in input_lower for kw in 
                ["what if", "would have", "could have", "alternative", "instead"]),
            "suggests_intervention": any(kw in input_lower for kw in 
                ["change", "intervene", "modify", "alter", "influence"]),
            "domain_keywords": self._extract_domain_keywords(input_lower),
            "confidence": 0.0
        }
        
        # Calculate confidence based on keyword matches
        confidence = 0.0
        if analysis["suggests_causal_reasoning"]:
            confidence += 0.3
        if analysis["suggests_conceptual_reasoning"]:
            confidence += 0.2
        if analysis["suggests_counterfactual"]:
            confidence += 0.3
        if analysis["suggests_intervention"]:
            confidence += 0.2
        
        analysis["confidence"] = min(1.0, confidence)
        
        # Determine primary reasoning type
        if analysis["suggests_counterfactual"]:
            analysis["reasoning_type"] = "counterfactual"
        elif analysis["suggests_causal_reasoning"] and analysis["suggests_conceptual_reasoning"]:
            analysis["reasoning_type"] = "integrated"
        elif analysis["suggests_causal_reasoning"]:
            analysis["reasoning_type"] = "causal"
        elif analysis["suggests_conceptual_reasoning"]:
            analysis["reasoning_type"] = "conceptual"
        else:
            analysis["reasoning_type"] = "none"
        
        return analysis
    
    async def _get_contextually_relevant_models(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Get causal models relevant to current context"""
        relevant_models = []
        
        # Extract domain from context
        domain_keywords = self._extract_domain_keywords(context.user_input.lower())
        
        for model_id, model in self.original_core.causal_models.items():
            relevance_score = 0.0
            
            # Check domain match
            if model.domain:
                for keyword in domain_keywords:
                    if keyword in model.domain.lower():
                        relevance_score += 0.3
            
            # Check if model relates to current goals
            if context.goal_context:
                active_goals = context.goal_context.get("active_goals", [])
                for goal in active_goals:
                    if goal.get("associated_need") in model.domain.lower():
                        relevance_score += 0.2
            
            # Check if model relates to emotional context
            if context.emotional_state:
                dominant_emotion = context.emotional_state.get("dominant_emotion")
                if dominant_emotion and dominant_emotion[0].lower() in model.metadata.get("emotional_relevance", []):
                    relevance_score += 0.1
            
            if relevance_score > 0.1:
                relevant_models.append({
                    "model_id": model_id,
                    "name": model.name,
                    "domain": model.domain,
                    "relevance_score": relevance_score,
                    "node_count": len(model.nodes),
                    "relation_count": len(model.relations)
                })
        
        # Sort by relevance
        relevant_models.sort(key=lambda m: m["relevance_score"], reverse=True)
        return relevant_models[:5]  # Top 5 models
    
    async def _get_contextually_relevant_spaces(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Get concept spaces relevant to current context"""
        relevant_spaces = []
        
        # Extract domain from context
        domain_keywords = self._extract_domain_keywords(context.user_input.lower())
        
        for space_id, space in self.original_core.concept_spaces.items():
            relevance_score = 0.0
            
            # Check domain match
            if space.domain:
                for keyword in domain_keywords:
                    if keyword in space.domain.lower():
                        relevance_score += 0.3
            
            # Check if space relates to memory context
            if context.memory_context:
                memory_types = context.memory_context.get("memory_types", [])
                if "experience" in memory_types and "experiential" in space.name.lower():
                    relevance_score += 0.2
            
            if relevance_score > 0.1:
                relevant_spaces.append({
                    "space_id": space_id,
                    "name": space.name,
                    "domain": space.domain,
                    "relevance_score": relevance_score,
                    "concept_count": len(space.concepts),
                    "relation_count": len(space.relations)
                })
        
        # Sort by relevance
        relevant_spaces.sort(key=lambda s: s["relevance_score"], reverse=True)
        return relevant_spaces[:5]  # Top 5 spaces
    
    async def _determine_contextual_reasoning_approach(self, context: SharedContext, 
                                                     reasoning_analysis: Dict[str, Any],
                                                     messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Determine the best reasoning approach based on full context"""
        approach = {
            "type": reasoning_analysis.get("reasoning_type", "none"),
            "confidence": reasoning_analysis.get("confidence", 0.0),
            "strategy": "default",
            "resources": []
        }
        
        # Adjust based on emotional context
        if context.emotional_state:
            dominant_emotion = context.emotional_state.get("dominant_emotion")
            if dominant_emotion:
                emotion_name, strength = dominant_emotion
                
                if emotion_name == "Curiosity" and strength > 0.5:
                    # Curiosity favors exploratory reasoning
                    approach["strategy"] = "exploratory"
                    if approach["type"] == "none":
                        approach["type"] = "conceptual"
                elif emotion_name == "Anxiety" and strength > 0.6:
                    # Anxiety favors understanding causes
                    approach["strategy"] = "explanatory"
                    if approach["type"] == "none":
                        approach["type"] = "causal"
        
        # Adjust based on goal context
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            for goal in active_goals:
                if "understand" in goal.get("description", "").lower():
                    approach["strategy"] = "analytical"
                elif "solve" in goal.get("description", "").lower():
                    approach["strategy"] = "problem_solving"
                    approach["type"] = "integrated"
        
        # Check messages from other modules
        if "knowledge_core" in messages:
            # Knowledge updates might require model updates
            approach["resources"].append("knowledge_integration")
        
        if "multimodal_integrator" in messages:
            # Multimodal input might provide evidence for causal discovery
            approach["resources"].append("perceptual_evidence")
        
        return approach
    
    # ========================================================================================
    # REASONING EXECUTION METHODS
    # ========================================================================================
    
    async def _execute_causal_reasoning(self, context: SharedContext, 
                                      approach: Dict[str, Any]) -> Dict[str, Any]:
        """Execute causal reasoning with context awareness"""
        results = {
            "models_analyzed": [],
            "causal_insights": [],
            "new_relations_discovered": 0
        }
        
        # Get relevant models
        relevant_models = await self._get_contextually_relevant_models(context)
        
        if not relevant_models:
            # Create a new model if needed
            domain = self._infer_domain_from_context(context)
            model_id = await self.original_core.create_causal_model(
                name=f"Context-driven model for {domain}",
                domain=domain,
                metadata={"created_from_context": True, "session_id": context.session_context.get("session_id")}
            )
            self.active_models.add(model_id)
            
            results["models_analyzed"].append({
                "model_id": model_id,
                "status": "newly_created"
            })
        else:
            # Analyze existing models
            for model_info in relevant_models[:2]:  # Analyze top 2 models
                model_id = model_info["model_id"]
                self.active_models.add(model_id)
                
                # Check if we should run causal discovery
                if approach["strategy"] == "exploratory":
                    discovery_result = await self.original_core.discover_causal_relations(model_id)
                    
                    results["models_analyzed"].append({
                        "model_id": model_id,
                        "discovery_result": discovery_result
                    })
                    
                    if discovery_result.get("accepted_relations", 0) > 0:
                        results["new_relations_discovered"] += discovery_result["accepted_relations"]
                else:
                    # Just analyze the model
                    model = self.original_core.causal_models[model_id]
                    
                    # Find relevant nodes based on context
                    relevant_nodes = []
                    for node_id, node in model.nodes.items():
                        if self._is_node_relevant_to_context(node, context):
                            relevant_nodes.append(node_id)
                    
                    results["models_analyzed"].append({
                        "model_id": model_id,
                        "relevant_nodes": relevant_nodes,
                        "total_nodes": len(model.nodes)
                    })
                
                # Extract causal insights
                insights = await self._extract_causal_insights(model_id, context)
                results["causal_insights"].extend(insights)
        
        return results
    
    async def _execute_conceptual_reasoning(self, context: SharedContext,
                                         approach: Dict[str, Any]) -> Dict[str, Any]:
        """Execute conceptual reasoning with context awareness"""
        results = {
            "spaces_analyzed": [],
            "blends_created": [],
            "conceptual_insights": []
        }
        
        # Get relevant spaces
        relevant_spaces = await self._get_contextually_relevant_spaces(context)
        
        if len(relevant_spaces) >= 2 and approach["strategy"] == "exploratory":
            # Create a blend
            space1_id = relevant_spaces[0]["space_id"]
            space2_id = relevant_spaces[1]["space_id"]
            
            self.active_spaces.add(space1_id)
            self.active_spaces.add(space2_id)
            
            # Find mappings between spaces
            mappings = await self._find_contextual_mappings(space1_id, space2_id, context)
            
            if mappings:
                # Create blend based on emotional context
                blend_type = self._determine_blend_type_from_context(context)
                
                blend_result = await self._create_contextual_blend(
                    space1_id, space2_id, mappings, blend_type
                )
                
                if blend_result:
                    results["blends_created"].append(blend_result)
        
        # Analyze spaces for insights
        for space_info in relevant_spaces[:3]:
            space_id = space_info["space_id"]
            self.active_spaces.add(space_id)
            
            insights = await self._extract_conceptual_insights(space_id, context)
            results["conceptual_insights"].extend(insights)
            
            results["spaces_analyzed"].append({
                "space_id": space_id,
                "insights_found": len(insights)
            })
        
        return results
    
    async def _execute_integrated_reasoning(self, context: SharedContext,
                                         approach: Dict[str, Any]) -> Dict[str, Any]:
        """Execute integrated causal and conceptual reasoning"""
        results = {
            "causal_results": {},
            "conceptual_results": {},
            "integration_insights": [],
            "creative_interventions": []
        }
        
        # First execute causal reasoning
        causal_results = await self._execute_causal_reasoning(context, approach)
        results["causal_results"] = causal_results
        
        # Then execute conceptual reasoning
        conceptual_results = await self._execute_conceptual_reasoning(context, approach)
        results["conceptual_results"] = conceptual_results
        
        # Integrate insights
        if causal_results["models_analyzed"] and conceptual_results["spaces_analyzed"]:
            # Find integration opportunities
            for model_info in causal_results["models_analyzed"]:
                model_id = model_info["model_id"]
                
                for space_info in conceptual_results["spaces_analyzed"]:
                    space_id = space_info["space_id"]
                    
                    # Check for integration potential
                    integration_score = await self._assess_integration_potential(
                        model_id, space_id, context
                    )
                    
                    if integration_score > 0.5:
                        # Create integrated model
                        integrated_result = await self.original_core.create_integrated_model(
                            domain=self._infer_domain_from_context(context),
                            base_on_causal=True
                        )
                        
                        results["integration_insights"].append({
                            "type": "integrated_model",
                            "result": integrated_result,
                            "integration_score": integration_score
                        })
                        
                        # If problem-solving strategy, suggest interventions
                        if approach["strategy"] == "problem_solving":
                            intervention = await self._suggest_creative_intervention(
                                model_id, context
                            )
                            if intervention:
                                results["creative_interventions"].append(intervention)
        
        return results
    
    # ========================================================================================
    # SYNTHESIS METHODS
    # ========================================================================================
    
    async def _synthesize_causal_insights(self, context: SharedContext, 
                                       messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Synthesize causal insights from active models"""
        insights = {
            "key_causal_factors": [],
            "causal_chains": [],
            "intervention_points": [],
            "new_discoveries": []
        }
        
        for model_id in self.active_models:
            if model_id not in self.original_core.causal_models:
                continue
                
            model = self.original_core.causal_models[model_id]
            
            # Find key causal factors (high centrality nodes)
            try:
                import networkx as nx
                centrality = nx.betweenness_centrality(model.graph)
                top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
                
                for node_id, centrality_score in top_nodes:
                    node = model.nodes.get(node_id)
                    if node:
                        insights["key_causal_factors"].append({
                            "node_name": node.name,
                            "centrality": centrality_score,
                            "model": model.name
                        })
            except:
                pass
            
            # Find causal chains relevant to context
            if context.user_input:
                relevant_chains = await self._find_relevant_causal_chains(model, context)
                insights["causal_chains"].extend(relevant_chains)
            
            # Identify intervention points
            intervention_points = await self._identify_intervention_points(model, context)
            insights["intervention_points"].extend(intervention_points)
        
        # Check for new discoveries from this session
        if hasattr(self, 'reasoning_context') and "discoveries" in self.reasoning_context:
            insights["new_discoveries"] = self.reasoning_context["discoveries"]
        
        return insights
    
    async def _synthesize_conceptual_insights(self, context: SharedContext,
                                           messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Synthesize conceptual insights from active spaces"""
        insights = {
            "key_concepts": [],
            "conceptual_patterns": [],
            "creative_possibilities": [],
            "blend_opportunities": []
        }
        
        for space_id in self.active_spaces:
            if space_id not in self.original_core.concept_spaces:
                continue
                
            space = self.original_core.concept_spaces[space_id]
            
            # Find key concepts
            key_concepts = await self._identify_key_concepts(space, context)
            insights["key_concepts"].extend(key_concepts)
            
            # Find conceptual patterns
            patterns = await self._identify_conceptual_patterns(space, context)
            insights["conceptual_patterns"].extend(patterns)
        
        # Find blend opportunities
        if len(self.active_spaces) >= 2:
            space_list = list(self.active_spaces)
            for i in range(len(space_list)):
                for j in range(i + 1, len(space_list)):
                    opportunity = await self._assess_blend_opportunity(
                        space_list[i], space_list[j], context
                    )
                    if opportunity["score"] > 0.5:
                        insights["blend_opportunities"].append(opportunity)
        
        return insights
    
    async def _synthesize_counterfactual_insights(self, context: SharedContext,
                                               messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Synthesize counterfactual insights"""
        insights = {
            "counterfactual_scenarios": [],
            "what_if_analyses": [],
            "alternative_paths": []
        }
        
        # Check if input suggests counterfactual reasoning
        if "what if" in context.user_input.lower() or "would have" in context.user_input.lower():
            for model_id in self.active_models:
                if model_id not in self.original_core.causal_models:
                    continue
                    
                # Generate counterfactual scenario
                scenario = await self._generate_counterfactual_scenario(model_id, context)
                if scenario:
                    insights["counterfactual_scenarios"].append(scenario)
                    
                    # Analyze alternative paths
                    alt_paths = await self._analyze_alternative_paths(model_id, scenario)
                    insights["alternative_paths"].extend(alt_paths)
        
        return insights
    
    async def _synthesize_intervention_suggestions(self, context: SharedContext,
                                               messages: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Synthesize intervention suggestions based on reasoning"""
        suggestions = []
        
        # Check goal context for intervention needs
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            
            for goal in active_goals:
                if goal.get("priority", 0) > 0.7:  # High priority goals
                    # Find causal model related to goal
                    for model_id in self.active_models:
                        model = self.original_core.causal_models.get(model_id)
                        if model and self._model_relates_to_goal(model, goal):
                            # Suggest intervention
                            suggestion = await self._generate_intervention_suggestion(
                                model_id, goal, context
                            )
                            if suggestion:
                                suggestions.append(suggestion)
        
        return suggestions
    
    async def _generate_reasoning_narrative(self, context: SharedContext,
                                         messages: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate a narrative explaining the reasoning process"""
        narrative_parts = []
        
        # Describe what type of reasoning was performed
        if self.active_models:
            narrative_parts.append(f"I analyzed {len(self.active_models)} causal models")
        
        if self.active_spaces:
            narrative_parts.append(f"explored {len(self.active_spaces)} conceptual spaces")
        
        # Describe key findings
        if hasattr(self, 'reasoning_context') and "key_findings" in self.reasoning_context:
            findings = self.reasoning_context["key_findings"]
            if findings:
                narrative_parts.append(f"and discovered {len(findings)} key insights")
        
        # Create coherent narrative
        if narrative_parts:
            narrative = "Through integrated reasoning, " + ", ".join(narrative_parts) + "."
        else:
            narrative = "I'm analyzing the situation from multiple perspectives."
        
        return narrative
    
    async def _assess_reasoning_confidence(self, context: SharedContext) -> float:
        """Assess confidence in reasoning results"""
        confidence = 0.5  # Base confidence
        
        # More models/spaces analyzed = higher confidence
        if len(self.active_models) > 0:
            confidence += 0.1 * min(3, len(self.active_models))
        
        if len(self.active_spaces) > 0:
            confidence += 0.1 * min(3, len(self.active_spaces))
        
        # Cross-module integration increases confidence
        messages = await self.get_cross_module_messages()
        if len(messages) > 0:
            confidence += 0.05 * min(4, len(messages))
        
        # New discoveries increase confidence
        if hasattr(self, 'reasoning_context') and self.reasoning_context.get("new_relations_discovered", 0) > 0:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    def _extract_domain_keywords(self, text: str) -> List[str]:
        """Extract domain-related keywords from text"""
        # Common domain indicators
        domain_patterns = [
            "climate", "health", "economics", "psychology", "technology",
            "relationships", "emotions", "learning", "behavior", "society",
            "environment", "politics", "art", "science", "philosophy"
        ]
        
        keywords = []
        for pattern in domain_patterns:
            if pattern in text:
                keywords.append(pattern)
        
        return keywords
    
    def _infer_domain_from_context(self, context: SharedContext) -> str:
        """Infer domain from context"""
        keywords = self._extract_domain_keywords(context.user_input.lower())
        
        if keywords:
            return keywords[0]  # Use first keyword as domain
        
        # Check emotional context
        if context.emotional_state:
            return "emotional_dynamics"
        
        # Check goal context
        if context.goal_context:
            return "goal_achievement"
        
        return "general"
    
    def _is_node_relevant_to_context(self, node, context: SharedContext) -> bool:
        """Check if a causal node is relevant to current context"""
        node_name_lower = node.name.lower()
        input_lower = context.user_input.lower()
        
        # Direct name match
        if any(word in node_name_lower for word in input_lower.split()):
            return True
        
        # Domain match
        if node.domain:
            domain_keywords = self._extract_domain_keywords(input_lower)
            if any(kw in node.domain.lower() for kw in domain_keywords):
                return True
        
        return False
    
    async def _extract_causal_insights(self, model_id: str, context: SharedContext) -> List[Dict[str, Any]]:
        """Extract causal insights from a model"""
        insights = []
        model = self.original_core.causal_models.get(model_id)
        
        if not model:
            return insights
        
        # Find causal paths relevant to input
        input_keywords = context.user_input.lower().split()
        
        for node_id, node in model.nodes.items():
            if any(kw in node.name.lower() for kw in input_keywords):
                # Found relevant node - trace causal paths
                ancestors = model.get_ancestors(node_id)[:3]  # Top 3 causes
                descendants = model.get_descendants(node_id)[:3]  # Top 3 effects
                
                insight = {
                    "type": "causal_path",
                    "central_factor": node.name,
                    "causes": [model.nodes[a].name for a in ancestors if a in model.nodes],
                    "effects": [model.nodes[d].name for d in descendants if d in model.nodes],
                    "model": model.name
                }
                
                insights.append(insight)
        
        return insights
    
    async def _find_contextual_mappings(self, space1_id: str, space2_id: str, 
                                     context: SharedContext) -> List[Dict[str, Any]]:
        """Find mappings between concept spaces based on context"""
        space1 = self.original_core.concept_spaces.get(space1_id)
        space2 = self.original_core.concept_spaces.get(space2_id)
        
        if not space1 or not space2:
            return []
        
        mappings = []
        
        # Weight mappings based on context relevance
        for concept1_id, concept1 in space1.concepts.items():
            for concept2_id, concept2 in space2.concepts.items():
                similarity = self.original_core._calculate_concept_similarity(
                    concept1, concept2, space1, space2
                )
                
                # Boost similarity if concepts relate to context
                if self._concept_relates_to_context(concept1, context):
                    similarity += 0.1
                if self._concept_relates_to_context(concept2, context):
                    similarity += 0.1
                
                if similarity >= 0.5:
                    mappings.append({
                        "concept1": concept1_id,
                        "concept2": concept2_id,
                        "similarity": min(1.0, similarity),
                        "context_boosted": True
                    })
        
        return mappings
    
    def _determine_blend_type_from_context(self, context: SharedContext) -> str:
        """Determine blend type based on context"""
        # Check emotional state
        if context.emotional_state:
            dominant_emotion = context.emotional_state.get("dominant_emotion")
            if dominant_emotion:
                emotion_name = dominant_emotion[0]
                
                if emotion_name == "Curiosity":
                    return "elaboration"  # Elaborate on ideas
                elif emotion_name in ["Frustration", "Conflict"]:
                    return "contrast"  # Find contrasts
                elif emotion_name in ["Joy", "Satisfaction"]:
                    return "fusion"  # Deep integration
        
        # Check goal context
        if context.goal_context:
            goal_types = [g.get("associated_need", "") for g in context.goal_context.get("active_goals", [])]
            
            if "novelty" in goal_types:
                return "elaboration"
            elif "coherence" in goal_types:
                return "completion"
        
        return "composition"  # Default
    
    def _concept_relates_to_context(self, concept: Dict[str, Any], context: SharedContext) -> bool:
        """Check if concept relates to context"""
        concept_name = concept.get("name", "").lower()
        input_lower = context.user_input.lower()
        
        # Check name match
        if any(word in concept_name for word in input_lower.split()):
            return True
        
        # Check property matches
        for prop_value in concept.get("properties", {}).values():
            if isinstance(prop_value, str) and any(word in prop_value.lower() for word in input_lower.split()):
                return True
        
        return False
    
    async def _create_contextual_blend(self, space1_id: str, space2_id: str,
                                    mappings: List[Dict[str, Any]], 
                                    blend_type: str) -> Optional[Dict[str, Any]]:
        """Create blend with context awareness"""
        space1 = self.original_core.concept_spaces.get(space1_id)
        space2 = self.original_core.concept_spaces.get(space2_id)
        
        if not space1 or not space2:
            return None
        
        # Use appropriate blend generation method
        blend_data = None
        
        if blend_type == "composition":
            blend_data = self.original_core._generate_composition_blend(space1, space2, mappings)
        elif blend_type == "fusion":
            blend_data = self.original_core._generate_fusion_blend(space1, space2, mappings)
        elif blend_type == "elaboration":
            blend_data = self.original_core._generate_elaboration_blend(space1, space2, mappings)
        elif blend_type == "contrast":
            blend_data = self.original_core._generate_contrast_blend(space1, space2, mappings)
        elif blend_type == "completion":
            blend_data = self.original_core._generate_completion_blend(space1, space2, mappings)
        
        if blend_data:
            # Update statistics
            self.original_core.stats["blends_created"] += 1
            
            # Store in reasoning context
            if "blends_created" not in self.reasoning_context:
                self.reasoning_context["blends_created"] = []
            self.reasoning_context["blends_created"].append(blend_data["id"])
        
        return blend_data
    
    async def _extract_conceptual_insights(self, space_id: str, context: SharedContext) -> List[Dict[str, Any]]:
        """Extract conceptual insights from a space"""
        insights = []
        space = self.original_core.concept_spaces.get(space_id)
        
        if not space:
            return insights
        
        # Find concepts relevant to input
        input_keywords = context.user_input.lower().split()
        
        for concept_id, concept in space.concepts.items():
            if any(kw in concept["name"].lower() for kw in input_keywords):
                # Found relevant concept
                related = space.get_related_concepts(concept_id)
                
                insight = {
                    "type": "conceptual_network",
                    "central_concept": concept["name"],
                    "related_concepts": [r["concept"]["name"] for r in related[:5]],
                    "space": space.name,
                    "properties": list(concept.get("properties", {}).keys())[:5]
                }
                
                insights.append(insight)
        
        return insights
    
    async def _perform_contextual_blending(self, space_ids: List[str], blend_type: str) -> Dict[str, Any]:
        """Perform complete contextual blending with all details"""
        if len(space_ids) < 2:
            return {"error": "Need at least 2 spaces for blending", "success": False}
        
        result = {
            "success": False,
            "blend_id": None,
            "blend_type": blend_type,
            "input_spaces": space_ids[:2],
            "concepts_blended": 0,
            "novel_concepts": [],
            "emergent_properties": [],
            "integration_insights": []
        }
        
        try:
            space1 = self.original_core.concept_spaces.get(space_ids[0])
            space2 = self.original_core.concept_spaces.get(space_ids[1])
            
            if not space1 or not space2:
                result["error"] = "Invalid space IDs"
                return result
            
            # Find mappings between spaces
            mappings = []
            mapping_scores = {}
            
            for c1_id, c1 in space1.concepts.items():
                for c2_id, c2 in space2.concepts.items():
                    similarity = self.original_core._calculate_concept_similarity(c1, c2, space1, space2)
                    
                    if similarity >= 0.5:
                        mappings.append({
                            "concept1": c1_id,
                            "concept2": c2_id,
                            "similarity": similarity
                        })
                        mapping_scores[(c1_id, c2_id)] = similarity
            
            if not mappings:
                result["error"] = "No suitable mappings found"
                return result
            
            # Create blend based on type
            blend_name = f"{space1.name}_{space2.name}_{blend_type}_blend"
            blend_id = f"blend_{len(self.original_core.blends)}"
            
            # Generate blended concepts based on blend type
            blended_concepts = {}
            
            if blend_type == "composition":
                # Standard composition - combine properties
                for mapping in mappings[:10]:  # Limit to top 10 mappings
                    c1 = space1.concepts[mapping["concept1"]]
                    c2 = space2.concepts[mapping["concept2"]]
                    
                    blended_concept = {
                        "name": f"{c1['name']}_{c2['name']}_composite",
                        "properties": {**c1.get("properties", {}), **c2.get("properties", {})},
                        "source_concepts": [c1["name"], c2["name"]],
                        "blend_strength": mapping["similarity"]
                    }
                    
                    blended_concepts[f"blend_{len(blended_concepts)}"] = blended_concept
                    result["concepts_blended"] += 1
            
            elif blend_type == "fusion":
                # Deep fusion - merge and transform properties
                for mapping in mappings[:7]:  # Fewer but deeper
                    c1 = space1.concepts[mapping["concept1"]]
                    c2 = space2.concepts[mapping["concept2"]]
                    
                    # Fuse properties
                    fused_props = {}
                    for key in set(c1.get("properties", {}).keys()) | set(c2.get("properties", {}).keys()):
                        val1 = c1.get("properties", {}).get(key)
                        val2 = c2.get("properties", {}).get(key)
                        
                        if val1 and val2:
                            fused_props[key] = f"fused({val1}, {val2})"
                        else:
                            fused_props[key] = val1 or val2
                    
                    blended_concept = {
                        "name": f"{c1['name']}~{c2['name']}",
                        "properties": fused_props,
                        "source_concepts": [c1["name"], c2["name"]],
                        "blend_strength": mapping["similarity"],
                        "fusion_type": "deep"
                    }
                    
                    blended_concepts[f"fusion_{len(blended_concepts)}"] = blended_concept
                    result["concepts_blended"] += 1
                    result["novel_concepts"].append(blended_concept["name"])
            
            elif blend_type == "elaboration":
                # Elaboration - expand and explore
                for mapping in mappings[:5]:
                    c1 = space1.concepts[mapping["concept1"]]
                    c2 = space2.concepts[mapping["concept2"]]
                    
                    # Generate elaborated variants
                    elaborations = [
                        {
                            "name": f"{c1['name']}_via_{c2['name']}",
                            "properties": {
                                **c1.get("properties", {}),
                                "elaborated_through": c2["name"],
                                "new_perspective": "cross_domain"
                            }
                        },
                        {
                            "name": f"{c2['name']}_as_{c1['name']}",
                            "properties": {
                                **c2.get("properties", {}),
                                "reframed_as": c1["name"],
                                "conceptual_shift": "metaphorical"
                            }
                        }
                    ]
                    
                    for i, elab in enumerate(elaborations):
                        blended_concepts[f"elab_{len(blended_concepts)}"] = elab
                        result["novel_concepts"].append(elab["name"])
                    
                    result["concepts_blended"] += 2
            
            elif blend_type == "contrast":
                # Contrast - highlight differences
                for mapping in mappings[:5]:
                    c1 = space1.concepts[mapping["concept1"]]
                    c2 = space2.concepts[mapping["concept2"]]
                    
                    # Find contrasting properties
                    contrasts = {}
                    all_props = set(c1.get("properties", {}).keys()) | set(c2.get("properties", {}).keys())
                    
                    for prop in all_props:
                        val1 = c1.get("properties", {}).get(prop)
                        val2 = c2.get("properties", {}).get(prop)
                        
                        if val1 and val2 and val1 != val2:
                            contrasts[f"contrast_{prop}"] = {"space1": val1, "space2": val2}
                    
                    if contrasts:
                        contrast_concept = {
                            "name": f"{c1['name']}_vs_{c2['name']}",
                            "properties": contrasts,
                            "type": "contrastive_analysis"
                        }
                        
                        blended_concepts[f"contrast_{len(blended_concepts)}"] = contrast_concept
                        result["concepts_blended"] += 1
            
            elif blend_type == "completion":
                # Completion - fill gaps
                # Find concepts in space1 without good matches in space2
                unmapped_space1 = set(space1.concepts.keys())
                unmapped_space2 = set(space2.concepts.keys())
                
                for mapping in mappings:
                    unmapped_space1.discard(mapping["concept1"])
                    unmapped_space2.discard(mapping["concept2"])
                
                # Create bridging concepts
                for unmapped_id in list(unmapped_space1)[:3]:
                    unmapped = space1.concepts[unmapped_id]
                    
                    bridge_concept = {
                        "name": f"{unmapped['name']}_bridge",
                        "properties": {
                            **unmapped.get("properties", {}),
                            "bridges_to": space2.name,
                            "completion_type": "gap_filler"
                        }
                    }
                    
                    blended_concepts[f"bridge_{len(blended_concepts)}"] = bridge_concept
                    result["novel_concepts"].append(bridge_concept["name"])
                    result["concepts_blended"] += 1
            
            # Identify emergent properties
            if len(blended_concepts) > 3:
                result["emergent_properties"].append("cross_domain_synthesis")
                
                if blend_type == "fusion":
                    result["emergent_properties"].append("unified_framework")
                elif blend_type == "elaboration":
                    result["emergent_properties"].append("expanded_possibility_space")
            
            # Generate integration insights
            result["integration_insights"] = [
                f"Successfully blended {result['concepts_blended']} concepts using {blend_type}",
                f"Discovered {len(result['novel_concepts'])} novel conceptual combinations",
                f"Blend reveals {len(result['emergent_properties'])} emergent properties"
            ]
            
            # Store the blend
            blend_data = {
                "id": blend_id,
                "name": blend_name,
                "type": blend_type,
                "input_spaces": [space1.name, space2.name],
                "concepts": blended_concepts,
                "mappings": mappings,
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "concept_count": len(blended_concepts),
                    "emergent_properties": result["emergent_properties"]
                }
            }
            
            self.original_core.blends[blend_id] = blend_data
            self.original_core.stats["blends_created"] += 1
            
            result["success"] = True
            result["blend_id"] = blend_id
            
        except Exception as e:
            logger.error(f"Error in contextual blending: {e}")
            result["error"] = str(e)
        
        return result
    
    async def _adjust_reasoning_from_emotion(self, emotional_data: Dict[str, Any]):
        """Adjust reasoning based on emotional state"""
        dominant_emotion = emotional_data.get("dominant_emotion")
        if not dominant_emotion:
            return
        
        emotion_name, strength = dominant_emotion
        
        # Store emotional influence in reasoning context
        self.reasoning_context["emotional_influence"] = {
            "emotion": emotion_name,
            "strength": strength,
            "timestamp": datetime.now().isoformat()
        }
        
        # Adjust reasoning parameters based on emotion
        if emotion_name == "Curiosity" and strength > 0.5:
            # Increase exploration in reasoning
            self.original_core.causal_config["discovery_threshold"] *= 0.8  # Lower threshold
            self.original_core.blending_config["default_mapping_threshold"] *= 0.9  # More mappings
        elif emotion_name == "Anxiety" and strength > 0.6:
            # Focus on understanding and control
            self.original_core.causal_config["min_relation_strength"] *= 1.2  # Higher standards
    
    async def _align_reasoning_with_goals(self, goal_data: Dict[str, Any]):
        """Align reasoning with active goals"""
        active_goals = goal_data.get("active_goals", [])
        
        # Store goal alignment in context
        self.reasoning_context["goal_alignment"] = {
            "aligned_goals": [],
            "timestamp": datetime.now().isoformat()
        }
        
        for goal in active_goals:
            if goal.get("priority", 0) > 0.6:
                # High priority goal - align reasoning
                goal_desc = goal.get("description", "").lower()
                
                if "understand" in goal_desc or "knowledge" in goal_desc:
                    # Focus on causal discovery
                    self.reasoning_context["goal_alignment"]["aligned_goals"].append({
                        "goal_id": goal.get("id"),
                        "alignment": "causal_discovery"
                    })
                elif "creative" in goal_desc or "novel" in goal_desc:
                    # Focus on conceptual blending
                    self.reasoning_context["goal_alignment"]["aligned_goals"].append({
                        "goal_id": goal.get("id"),
                        "alignment": "conceptual_blending"
                    })
    
    async def _inform_reasoning_from_memory(self, memory_data: Dict[str, Any]):
        """Use memory to inform reasoning"""
        retrieved_memories = memory_data.get("retrieved_memories", [])
        
        # Extract patterns from memories
        memory_patterns = []
        for memory in retrieved_memories:
            if memory.get("memory_type") == "experience":
                # Experience memories can inform causal models
                memory_patterns.append({
                    "type": "experiential",
                    "content": memory.get("content", ""),
                    "relevance": "causal"
                })
            elif memory.get("memory_type") == "reflection":
                # Reflection memories can inform conceptual understanding
                memory_patterns.append({
                    "type": "reflective",
                    "content": memory.get("content", ""),
                    "relevance": "conceptual"
                })
        
        # Store in reasoning context
        self.reasoning_context["memory_patterns"] = memory_patterns
    
    # ========================================================================================
    # ANALYSIS AND SYNTHESIS HELPER METHODS
    # ========================================================================================
    
    async def _analyze_reasoning_potential(self, context: SharedContext, messages: Dict,
                                        available_models: List[str], 
                                        available_spaces: List[str]) -> Dict[str, Any]:
        """Analyze the potential for different types of reasoning"""
        potential = {
            "causal_potential": 0.0,
            "conceptual_potential": 0.0,
            "integrated_potential": 0.0,
            "factors": []
        }
        
        # Causal potential
        if available_models:
            potential["causal_potential"] += 0.3
            potential["factors"].append("existing_causal_models")
        
        if "knowledge_core" in messages:
            potential["causal_potential"] += 0.2
            potential["factors"].append("knowledge_available")
        
        # Conceptual potential
        if available_spaces:
            potential["conceptual_potential"] += 0.3
            potential["factors"].append("existing_concept_spaces")
        
        if context.emotional_state and context.emotional_state.get("dominant_emotion"):
            emotion = context.emotional_state["dominant_emotion"][0]
            if emotion in ["Curiosity", "Wonder", "Excitement"]:
                potential["conceptual_potential"] += 0.2
                potential["factors"].append("creative_emotional_state")
        
        # Integrated potential
        if available_models and available_spaces:
            potential["integrated_potential"] = (
                potential["causal_potential"] + potential["conceptual_potential"]
            ) / 2
            potential["factors"].append("both_systems_available")
        
        return potential
    
    async def _identify_reasoning_opportunities(self, context: SharedContext,
                                             messages: Dict) -> List[Dict[str, Any]]:
        """Identify specific reasoning opportunities"""
        opportunities = []
        
        # Check for causal discovery opportunity
        if "perception_input" in [m.get("type") for msgs in messages.values() for m in msgs]:
            opportunities.append({
                "type": "causal_discovery",
                "trigger": "new_perceptual_data",
                "priority": 0.7
            })
        
        # Check for conceptual blending opportunity
        if len(self.original_core.concept_spaces) >= 2:
            opportunities.append({
                "type": "conceptual_blending",
                "trigger": "multiple_concept_spaces",
                "priority": 0.6
            })
        
        # Check for counterfactual reasoning opportunity
        if "what if" in context.user_input.lower():
            opportunities.append({
                "type": "counterfactual_reasoning",
                "trigger": "counterfactual_query",
                "priority": 0.9
            })
        
        return opportunities
    
    async def _analyze_reasoning_coherence(self, context: SharedContext,
                                        messages: Dict) -> Dict[str, Any]:
        """Analyze coherence between reasoning and other modules"""
        coherence = {
            "overall_score": 1.0,
            "issues": [],
            "alignments": []
        }
        
        # Check alignment with emotional state
        if context.emotional_state:
            emotion = context.emotional_state.get("dominant_emotion")
            if emotion and emotion[0] == "Confusion" and len(self.active_models) == 0:
                coherence["overall_score"] -= 0.2
                coherence["issues"].append("no_models_during_confusion")
            else:
                coherence["alignments"].append("emotion_reasoning_aligned")
        
        # Check alignment with goals
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            understanding_goals = [g for g in active_goals if "understand" in g.get("description", "").lower()]
            
            if understanding_goals and len(self.active_models) > 0:
                coherence["alignments"].append("goal_reasoning_aligned")
            elif understanding_goals and len(self.active_models) == 0:
                coherence["overall_score"] -= 0.15
                coherence["issues"].append("understanding_goal_without_models")
        
        return coherence
    
    # ========================================================================================
    # CAUSAL ANALYSIS HELPER METHODS
    # ========================================================================================
    
    async def _find_relevant_causal_chains(self, model, context: SharedContext) -> List[Dict[str, Any]]:
        """Find causal chains in model relevant to context"""
        chains = []
        
        # Extract key terms from input
        input_terms = set(context.user_input.lower().split())
        input_terms.update(self._extract_domain_keywords(context.user_input.lower()))
        
        # Find nodes matching input terms
        relevant_nodes = []
        for node_id, node in model.nodes.items():
            node_name_lower = node.name.lower()
            if any(term in node_name_lower for term in input_terms):
                relevant_nodes.append(node_id)
        
        # For each relevant node, trace causal chains
        for node_id in relevant_nodes[:3]:  # Limit to top 3 to avoid overwhelming
            # Get upstream chain (causes)
            upstream_chain = []
            current_nodes = [node_id]
            visited = set()
            
            for depth in range(3):  # Max depth of 3
                next_nodes = []
                for current in current_nodes:
                    if current in visited:
                        continue
                    visited.add(current)
                    
                    # Find parent nodes (causes)
                    for relation in model.relations:
                        if relation.target == current and relation.source not in visited:
                            next_nodes.append(relation.source)
                            upstream_chain.append({
                                "from": relation.source,
                                "to": current,
                                "strength": relation.strength,
                                "mechanism": relation.mechanism
                            })
                
                current_nodes = next_nodes
                if not current_nodes:
                    break
            
            # Get downstream chain (effects)
            downstream_chain = []
            current_nodes = [node_id]
            visited = {node_id}
            
            for depth in range(3):  # Max depth of 3
                next_nodes = []
                for current in current_nodes:
                    # Find child nodes (effects)
                    for relation in model.relations:
                        if relation.source == current and relation.target not in visited:
                            next_nodes.append(relation.target)
                            visited.add(relation.target)
                            downstream_chain.append({
                                "from": current,
                                "to": relation.target,
                                "strength": relation.strength,
                                "mechanism": relation.mechanism
                            })
                
                current_nodes = next_nodes
                if not current_nodes:
                    break
            
            # Create chain summary
            if upstream_chain or downstream_chain:
                chain = {
                    "central_node": model.nodes[node_id].name,
                    "central_node_id": node_id,
                    "upstream_depth": len(set(r["from"] for r in upstream_chain)),
                    "downstream_depth": len(set(r["to"] for r in downstream_chain)),
                    "total_relations": len(upstream_chain) + len(downstream_chain),
                    "upstream_chain": upstream_chain,
                    "downstream_chain": downstream_chain,
                    "chain_type": self._classify_chain_type(upstream_chain, downstream_chain),
                    "relevance_score": self._calculate_chain_relevance(
                        upstream_chain + downstream_chain, context
                    )
                }
                
                chains.append(chain)
        
        # Sort by relevance
        chains.sort(key=lambda c: c["relevance_score"], reverse=True)
        
        return chains
    
    async def _identify_intervention_points(self, model, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify points in causal model where interventions could be effective"""
        intervention_points = []
        
        # Strategy 1: Find high-centrality nodes (influential points)
        try:
            import networkx as nx
            
            # Calculate various centrality measures
            betweenness = nx.betweenness_centrality(model.graph)
            eigenvector = nx.eigenvector_centrality(model.graph, max_iter=1000)
            degree = nx.degree_centrality(model.graph)
            
            # Combine centrality scores
            combined_scores = {}
            for node_id in model.nodes:
                combined_scores[node_id] = (
                    betweenness.get(node_id, 0) * 0.4 +
                    eigenvector.get(node_id, 0) * 0.3 +
                    degree.get(node_id, 0) * 0.3
                )
            
            # Get top intervention candidates
            top_nodes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for node_id, score in top_nodes:
                node = model.nodes[node_id]
                
                # Analyze intervention potential
                intervention_point = {
                    "node_id": node_id,
                    "node_name": node.name,
                    "intervention_score": score,
                    "intervention_type": self._determine_intervention_type(node, model),
                    "expected_impact": self._estimate_intervention_impact(node_id, model),
                    "feasibility": self._assess_intervention_feasibility(node, context),
                    "downstream_effects": len(model.get_descendants(node_id)),
                    "upstream_causes": len(model.get_ancestors(node_id))
                }
                
                # Add context-specific reasoning
                if context.goal_context:
                    intervention_point["goal_alignment"] = self._assess_goal_alignment(
                        node, context.goal_context
                    )
                
                intervention_points.append(intervention_point)
                
        except Exception as e:
            logger.warning(f"NetworkX analysis failed: {e}")
            
            # Fallback: Simple analysis based on node connectivity
            for node_id, node in model.nodes.items():
                in_degree = sum(1 for r in model.relations if r.target == node_id)
                out_degree = sum(1 for r in model.relations if r.source == node_id)
                
                if out_degree > 2:  # Nodes with multiple effects
                    intervention_points.append({
                        "node_id": node_id,
                        "node_name": node.name,
                        "intervention_score": out_degree / 10.0,
                        "intervention_type": "high_impact",
                        "expected_impact": "multiple_downstream_effects",
                        "feasibility": 0.5,
                        "downstream_effects": out_degree,
                        "upstream_causes": in_degree
                    })
        
        # Sort by intervention score
        intervention_points.sort(key=lambda p: p["intervention_score"], reverse=True)
        
        return intervention_points
    
    def _classify_chain_type(self, upstream_chain: List[Dict], downstream_chain: List[Dict]) -> str:
        """Classify the type of causal chain"""
        if len(upstream_chain) > len(downstream_chain) * 2:
            return "convergent"  # Many causes lead to one effect
        elif len(downstream_chain) > len(upstream_chain) * 2:
            return "divergent"  # One cause leads to many effects
        elif len(upstream_chain) == 0:
            return "source"  # Starting point
        elif len(downstream_chain) == 0:
            return "sink"  # End point
        else:
            return "pathway"  # Middle of chain
    
    def _calculate_chain_relevance(self, chain_relations: List[Dict], context: SharedContext) -> float:
        """Calculate relevance of a causal chain to context"""
        relevance = 0.0
        
        # Base relevance on chain length and strength
        if chain_relations:
            avg_strength = sum(r["strength"] for r in chain_relations) / len(chain_relations)
            relevance += avg_strength * 0.3
            
            # Bonus for longer coherent chains
            if len(chain_relations) > 3:
                relevance += 0.2
        
        # Check goal alignment
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            if any("understand" in g.get("description", "").lower() for g in active_goals):
                relevance += 0.2  # Understanding goals favor causal chains
        
        # Check emotional context
        if context.emotional_state:
            dominant_emotion = context.emotional_state.get("dominant_emotion")
            if dominant_emotion and dominant_emotion[0] == "Curiosity":
                relevance += 0.1  # Curiosity increases relevance of all chains
        
        return min(1.0, relevance)
    
    def _determine_intervention_type(self, node, model) -> str:
        """Determine what type of intervention would be appropriate for a node"""
        # Check node properties
        if hasattr(node, 'node_type'):
            if node.node_type == 'observable':
                return "measurement"
            elif node.node_type == 'latent':
                return "indirect"
        
        # Check based on relationships
        in_relations = [r for r in model.relations if r.target == node.id]
        out_relations = [r for r in model.relations if r.source == node.id]
        
        if len(in_relations) == 0:
            return "root_cause_modification"
        elif len(out_relations) == 0:
            return "outcome_optimization"
        elif len(out_relations) > 3:
            return "leverage_point"
        else:
            return "pathway_modulation"
    
    def _estimate_intervention_impact(self, node_id: str, model) -> str:
        """Estimate the impact of intervening at a node"""
        descendants = model.get_descendants(node_id)
        
        if len(descendants) > 10:
            return "very_high"
        elif len(descendants) > 5:
            return "high"
        elif len(descendants) > 2:
            return "moderate"
        elif len(descendants) > 0:
            return "low"
        else:
            return "minimal"
    
    def _assess_intervention_feasibility(self, node, context: SharedContext) -> float:
        """Assess how feasible it would be to intervene at this node"""
        feasibility = 0.5  # Base feasibility
        
        # Check if node represents something controllable
        node_name_lower = node.name.lower()
        
        # Highly feasible interventions
        if any(term in node_name_lower for term in ["policy", "decision", "choice", "action", "behavior"]):
            feasibility += 0.3
        
        # Moderately feasible
        elif any(term in node_name_lower for term in ["process", "method", "approach", "strategy"]):
            feasibility += 0.2
        
        # Less feasible
        elif any(term in node_name_lower for term in ["weather", "natural", "inherent", "genetic"]):
            feasibility -= 0.2
        
        # Check context for constraints
        if context.constraints:
            if "limited_resources" in context.constraints:
                feasibility -= 0.1
            if "time_sensitive" in context.constraints:
                feasibility -= 0.1
        
        return max(0.0, min(1.0, feasibility))
    
    def _assess_goal_alignment(self, node, goal_context: Dict[str, Any]) -> float:
        """Assess how well an intervention aligns with active goals"""
        alignment = 0.0
        active_goals = goal_context.get("active_goals", [])
        
        node_name_lower = node.name.lower()
        
        for goal in active_goals:
            goal_desc = goal.get("description", "").lower()
            goal_priority = goal.get("priority", 0.5)
            
            # Check for keyword matches
            goal_keywords = set(goal_desc.split())
            node_keywords = set(node_name_lower.split())
            
            overlap = len(goal_keywords.intersection(node_keywords))
            if overlap > 0:
                alignment += overlap * 0.1 * goal_priority
            
            # Check for semantic alignment
            if goal.get("associated_need") in node_name_lower:
                alignment += 0.2 * goal_priority
        
        return min(1.0, alignment)
    
    # ========================================================================================
    # CONCEPTUAL ANALYSIS HELPER METHODS
    # ========================================================================================
    
    async def _identify_key_concepts(self, space, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify key concepts in a conceptual space"""
        key_concepts = []
        
        # Calculate concept importance scores
        for concept_id, concept in space.concepts.items():
            importance_score = 0.0
            
            # Factor 1: Connectivity (how many relations)
            relations_count = sum(1 for r in space.relations if 
                                r["source"] == concept_id or r["target"] == concept_id)
            importance_score += min(relations_count / 10.0, 0.3)
            
            # Factor 2: Property richness
            property_count = len(concept.get("properties", {}))
            importance_score += min(property_count / 20.0, 0.2)
            
            # Factor 3: Context relevance
            relevance = self._calculate_concept_relevance_to_context(concept, context)
            importance_score += relevance * 0.5
            
            if importance_score > 0.3:  # Threshold for key concepts
                # Get related concepts for context
                related = space.get_related_concepts(concept_id)
                
                key_concept = {
                    "concept_id": concept_id,
                    "name": concept["name"],
                    "importance_score": importance_score,
                    "properties": self._get_salient_properties(concept),
                    "relation_count": relations_count,
                    "related_concepts": [r["concept"]["name"] for r in related[:3]],
                    "semantic_role": self._determine_semantic_role(concept, space),
                    "context_alignment": relevance
                }
                
                key_concepts.append(key_concept)
        
        # Sort by importance
        key_concepts.sort(key=lambda c: c["importance_score"], reverse=True)
        
        return key_concepts[:10]  # Top 10 concepts
    
    async def _identify_conceptual_patterns(self, space, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify patterns in conceptual organization"""
        patterns = []
        
        # Pattern 1: Hierarchical structures
        hierarchies = self._find_hierarchical_patterns(space)
        for hierarchy in hierarchies:
            patterns.append({
                "type": "hierarchy",
                "root_concept": hierarchy["root"],
                "depth": hierarchy["depth"],
                "branch_count": hierarchy["branches"],
                "pattern_strength": hierarchy["consistency"],
                "example_path": hierarchy["example_path"]
            })
        
        # Pattern 2: Clusters (tightly connected groups)
        clusters = self._find_conceptual_clusters(space)
        for cluster in clusters:
            patterns.append({
                "type": "cluster",
                "central_concepts": cluster["centers"],
                "cluster_size": cluster["size"],
                "density": cluster["density"],
                "theme": self._infer_cluster_theme(cluster, space),
                "pattern_strength": cluster["cohesion"]
            })
        
        # Pattern 3: Bridge concepts (connect different areas)
        bridges = self._find_bridge_concepts(space)
        for bridge in bridges:
            patterns.append({
                "type": "bridge",
                "bridge_concept": bridge["concept"],
                "connected_regions": bridge["regions"],
                "bridge_strength": bridge["strength"],
                "integration_potential": bridge["potential"]
            })
        
        # Pattern 4: Conceptual gradients (smooth transitions)
        gradients = self._find_conceptual_gradients(space)
        for gradient in gradients:
            patterns.append({
                "type": "gradient",
                "dimension": gradient["dimension"],
                "start_concept": gradient["start"],
                "end_concept": gradient["end"],
                "intermediate_concepts": gradient["path"],
                "smoothness": gradient["smoothness"]
            })
        
        # Filter patterns by context relevance
        relevant_patterns = []
        for pattern in patterns:
            if self._pattern_relevant_to_context(pattern, context):
                relevant_patterns.append(pattern)
        
        return relevant_patterns
    
    def _calculate_concept_relevance_to_context(self, concept: Dict[str, Any], 
                                               context: SharedContext) -> float:
        """Calculate how relevant a concept is to the current context"""
        relevance = 0.0
        
        # Check name similarity
        concept_name = concept.get("name", "").lower()
        input_words = set(context.user_input.lower().split())
        
        # Direct word matches
        name_words = set(concept_name.split())
        word_overlap = len(name_words.intersection(input_words))
        relevance += word_overlap * 0.2
        
        # Property matches
        for prop_name, prop_value in concept.get("properties", {}).items():
            if isinstance(prop_value, str):
                prop_words = set(prop_value.lower().split())
                if prop_words.intersection(input_words):
                    relevance += 0.1
        
        # Domain alignment
        if hasattr(concept, "domain") and concept.domain:
            domain_keywords = self._extract_domain_keywords(context.user_input.lower())
            if any(kw in concept.domain.lower() for kw in domain_keywords):
                relevance += 0.2
        
        # Emotional context alignment
        if context.emotional_state:
            emotion = context.emotional_state.get("dominant_emotion")
            if emotion and self._concept_aligns_with_emotion(concept, emotion[0]):
                relevance += 0.15
        
        return min(1.0, relevance)
    
    def _get_salient_properties(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the most salient properties of a concept"""
        properties = concept.get("properties", {})
        
        if len(properties) <= 5:
            return properties
        
        # Rank properties by salience
        salient_props = {}
        
        # Prioritize certain property types
        priority_keys = ["definition", "purpose", "function", "type", "category"]
        
        for key in priority_keys:
            if key in properties:
                salient_props[key] = properties[key]
        
        # Add other properties up to limit
        for key, value in properties.items():
            if key not in salient_props and len(salient_props) < 5:
                salient_props[key] = value
        
        return salient_props
    
    def _determine_semantic_role(self, concept: Dict[str, Any], space) -> str:
        """Determine the semantic role of a concept in the space"""
        concept_id = concept.get("id", "")
        
        # Count different types of relations
        incoming_relations = [r for r in space.relations if r["target"] == concept_id]
        outgoing_relations = [r for r in space.relations if r["source"] == concept_id]
        
        # Analyze relation types
        if len(incoming_relations) > len(outgoing_relations) * 2:
            return "terminal"  # End point of many relations
        elif len(outgoing_relations) > len(incoming_relations) * 2:
            return "generative"  # Source of many relations
        elif len(incoming_relations) > 5 and len(outgoing_relations) > 5:
            return "hub"  # Central connector
        elif len(incoming_relations) == 0 and len(outgoing_relations) > 0:
            return "primitive"  # Basic building block
        elif len(outgoing_relations) == 0 and len(incoming_relations) > 0:
            return "composite"  # Built from other concepts
        else:
            return "intermediate"  # Regular concept
    
    def _find_hierarchical_patterns(self, space) -> List[Dict[str, Any]]:
        """Find hierarchical organization patterns in the concept space"""
        hierarchies = []
        
        # Look for IS-A or PART-OF relations
        hierarchical_relations = ["is_a", "type_of", "part_of", "subset_of", "instance_of"]
        
        # Build hierarchy trees
        roots = []
        for concept_id, concept in space.concepts.items():
            # Check if this could be a root (no hierarchical parents)
            has_parent = any(
                r["target"] == concept_id and r.get("relation_type") in hierarchical_relations
                for r in space.relations
            )
            
            if not has_parent:
                # Found potential root
                hierarchy = self._trace_hierarchy(concept_id, space, hierarchical_relations)
                if hierarchy["depth"] > 1:  # Only include actual hierarchies
                    hierarchies.append(hierarchy)
        
        return hierarchies
    
    def _trace_hierarchy(self, root_id: str, space, relation_types: List[str]) -> Dict[str, Any]:
        """Trace a hierarchical structure from a root concept"""
        hierarchy = {
            "root": space.concepts[root_id]["name"],
            "root_id": root_id,
            "depth": 0,
            "branches": 0,
            "consistency": 1.0,
            "example_path": []
        }
        
        # BFS to explore hierarchy
        queue = [(root_id, 0, [root_id])]
        visited = {root_id}
        max_depth = 0
        branch_count = 0
        
        while queue:
            current_id, depth, path = queue.pop(0)
            max_depth = max(max_depth, depth)
            
            # Find children
            children = [
                r["target"] for r in space.relations
                if r["source"] == current_id and 
                r.get("relation_type") in relation_types and
                r["target"] not in visited
            ]
            
            if children:
                branch_count += len(children) - 1  # -1 because first child continues branch
                
                for child_id in children:
                    visited.add(child_id)
                    new_path = path + [child_id]
                    queue.append((child_id, depth + 1, new_path))
                    
                    # Keep track of one example path
                    if depth + 1 > hierarchy["depth"]:
                        hierarchy["depth"] = depth + 1
                        hierarchy["example_path"] = [
                            space.concepts[cid]["name"] for cid in new_path
                        ]
        
        hierarchy["branches"] = branch_count
        
        return hierarchy
    
    def _find_conceptual_clusters(self, space) -> List[Dict[str, Any]]:
        """Find clusters of tightly connected concepts"""
        clusters = []
        
        # Simple clustering: find groups with high internal connectivity
        visited = set()
        
        for concept_id in space.concepts:
            if concept_id in visited:
                continue
            
            # Find connected component
            cluster = self._find_connected_component(concept_id, space, visited)
            
            if len(cluster) >= 3:  # Minimum cluster size
                # Calculate cluster metrics
                density = self._calculate_cluster_density(cluster, space)
                cohesion = self._calculate_cluster_cohesion(cluster, space)
                
                if density > 0.3:  # Minimum density threshold
                    clusters.append({
                        "centers": self._find_cluster_centers(cluster, space),
                        "members": cluster,
                        "size": len(cluster),
                        "density": density,
                        "cohesion": cohesion
                    })
        
        return clusters
    
    def _find_connected_component(self, start_id: str, space, visited: set) -> List[str]:
        """Find all concepts connected to start_id"""
        component = []
        queue = [start_id]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
                
            visited.add(current)
            component.append(current)
            
            # Find neighbors
            neighbors = set()
            for relation in space.relations:
                if relation["source"] == current:
                    neighbors.add(relation["target"])
                elif relation["target"] == current:
                    neighbors.add(relation["source"])
            
            queue.extend(neighbors - visited)
        
        return component
    
    def _calculate_cluster_density(self, cluster: List[str], space) -> float:
        """Calculate the density of connections within a cluster"""
        if len(cluster) < 2:
            return 0.0
        
        # Count internal edges
        internal_edges = 0
        for relation in space.relations:
            if relation["source"] in cluster and relation["target"] in cluster:
                internal_edges += 1
        
        # Calculate density (actual edges / possible edges)
        possible_edges = len(cluster) * (len(cluster) - 1)
        density = internal_edges / possible_edges if possible_edges > 0 else 0.0
        
        return density
    
    def _calculate_cluster_cohesion(self, cluster: List[str], space) -> float:
        """Calculate semantic cohesion of a cluster"""
        if len(cluster) < 2:
            return 0.0
        
        # Calculate average similarity between cluster members
        total_similarity = 0.0
        comparisons = 0
        
        for i, concept1_id in enumerate(cluster):
            for concept2_id in cluster[i+1:]:
                concept1 = space.concepts[concept1_id]
                concept2 = space.concepts[concept2_id]
                
                similarity = self._calculate_concept_similarity_simple(concept1, concept2)
                total_similarity += similarity
                comparisons += 1
        
        return total_similarity / comparisons if comparisons > 0 else 0.0
    
    def _calculate_concept_similarity_simple(self, concept1: Dict, concept2: Dict) -> float:
        """Simple concept similarity calculation"""
        similarity = 0.0
        
        # Name similarity
        name1_words = set(concept1["name"].lower().split())
        name2_words = set(concept2["name"].lower().split())
        name_overlap = len(name1_words.intersection(name2_words))
        similarity += name_overlap * 0.2
        
        # Property overlap
        props1 = set(concept1.get("properties", {}).keys())
        props2 = set(concept2.get("properties", {}).keys())
        prop_overlap = len(props1.intersection(props2))
        similarity += min(prop_overlap * 0.1, 0.3)
        
        return min(1.0, similarity)
    
    def _find_cluster_centers(self, cluster: List[str], space) -> List[str]:
        """Find the most central concepts in a cluster"""
        centrality_scores = {}
        
        for concept_id in cluster:
            # Count connections within cluster
            connections = 0
            for relation in space.relations:
                if relation["source"] == concept_id and relation["target"] in cluster:
                    connections += 1
                elif relation["target"] == concept_id and relation["source"] in cluster:
                    connections += 1
            
            centrality_scores[concept_id] = connections
        
        # Get top 3 centers
        sorted_centers = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)
        return [space.concepts[cid]["name"] for cid, _ in sorted_centers[:3]]
    
    def _infer_cluster_theme(self, cluster: Dict[str, Any], space) -> str:
        """Infer the thematic focus of a cluster"""
        # Get all concept names in cluster
        concept_names = [space.concepts[cid]["name"] for cid in cluster["members"]]
        
        # Extract common words
        word_freq = {}
        for name in concept_names:
            words = name.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Find most common meaningful word
        if word_freq:
            theme_word = max(word_freq.items(), key=lambda x: x[1])[0]
            return f"{theme_word}_related"
        
        return "mixed_concepts"
    
    def _find_bridge_concepts(self, space) -> List[Dict[str, Any]]:
        """Find concepts that bridge different regions of the space"""
        bridges = []
        
        # First, identify regions (using clusters)
        clusters = self._find_conceptual_clusters(space)
        
        if len(clusters) < 2:
            return bridges
        
        # Find concepts that connect different clusters
        for concept_id, concept in space.concepts.items():
            connected_clusters = set()
            
            # Check which clusters this concept connects to
            for relation in space.relations:
                if relation["source"] == concept_id:
                    # Find which cluster the target belongs to
                    for i, cluster in enumerate(clusters):
                        if relation["target"] in cluster["members"]:
                            connected_clusters.add(i)
                
                elif relation["target"] == concept_id:
                    # Find which cluster the source belongs to
                    for i, cluster in enumerate(clusters):
                        if relation["source"] in cluster["members"]:
                            connected_clusters.add(i)
            
            if len(connected_clusters) >= 2:
                # This concept bridges multiple clusters
                bridges.append({
                    "concept": concept["name"],
                    "concept_id": concept_id,
                    "regions": [clusters[i]["centers"][0] for i in connected_clusters],
                    "strength": len(connected_clusters) / len(clusters),
                    "potential": self._assess_bridge_potential(concept_id, connected_clusters, clusters, space)
                })
        
        return bridges
    
    def _assess_bridge_potential(self, concept_id: str, connected_clusters: set, 
                                clusters: List[Dict], space) -> float:
        """Assess the potential of a bridge concept for creating new connections"""
        potential = len(connected_clusters) / len(clusters)  # Base potential
        
        # Check if bridge has unique properties
        concept = space.concepts[concept_id]
        unique_props = len(concept.get("properties", {}))
        potential += min(unique_props / 10.0, 0.3)
        
        return min(1.0, potential)
    
    def _find_conceptual_gradients(self, space) -> List[Dict[str, Any]]:
        """Find smooth conceptual transitions (gradients) in the space"""
        gradients = []
        
        # Look for chains of concepts with gradual property changes
        for concept1_id, concept1 in space.concepts.items():
            for concept2_id, concept2 in space.concepts.items():
                if concept1_id == concept2_id:
                    continue
                
                # Find path between concepts
                path = self._find_conceptual_path(concept1_id, concept2_id, space)
                
                if path and len(path) >= 3:
                    # Assess gradient smoothness
                    smoothness = self._assess_path_smoothness(path, space)
                    
                    if smoothness > 0.6:  # Smooth gradient threshold
                        # Identify the dimension of change
                        dimension = self._identify_gradient_dimension(path, space)
                        
                        gradients.append({
                            "dimension": dimension,
                            "start": concept1["name"],
                            "end": concept2["name"],
                            "path": [space.concepts[cid]["name"] for cid in path],
                            "smoothness": smoothness,
                            "length": len(path)
                        })
        
        # Remove duplicate/reverse gradients
        unique_gradients = []
        seen_pairs = set()
        
        for gradient in gradients:
            pair = tuple(sorted([gradient["start"], gradient["end"]]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_gradients.append(gradient)
        
        return unique_gradients[:5]  # Limit to top 5 gradients
    
    def _find_conceptual_path(self, start_id: str, end_id: str, space) -> List[str]:
        """Find a path between two concepts"""
        # Simple BFS pathfinding
        queue = [(start_id, [start_id])]
        visited = {start_id}
        
        while queue:
            current_id, path = queue.pop(0)
            
            if current_id == end_id:
                return path
            
            # Find neighbors
            neighbors = set()
            for relation in space.relations:
                if relation["source"] == current_id:
                    neighbors.add(relation["target"])
                elif relation["target"] == current_id:
                    neighbors.add(relation["source"])
            
            for neighbor in neighbors - visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
        
        return []  # No path found
    
    def _assess_path_smoothness(self, path: List[str], space) -> float:
        """Assess how smooth the transitions are along a path"""
        if len(path) < 2:
            return 0.0
        
        total_similarity = 0.0
        
        for i in range(len(path) - 1):
            concept1 = space.concepts[path[i]]
            concept2 = space.concepts[path[i + 1]]
            
            similarity = self._calculate_concept_similarity_simple(concept1, concept2)
            total_similarity += similarity
        
        # Average similarity between adjacent concepts
        smoothness = total_similarity / (len(path) - 1)
        
        return smoothness
    
    def _identify_gradient_dimension(self, path: List[str], space) -> str:
        """Identify what dimension changes along a gradient"""
        # Look for systematic property changes
        changing_properties = {}
        
        for i in range(len(path) - 1):
            concept1 = space.concepts[path[i]]
            concept2 = space.concepts[path[i + 1]]
            
            props1 = set(concept1.get("properties", {}).keys())
            props2 = set(concept2.get("properties", {}).keys())
            
            # Track which properties change
            changed = props1.symmetric_difference(props2)
            for prop in changed:
                changing_properties[prop] = changing_properties.get(prop, 0) + 1
        
        if changing_properties:
            # Most frequently changing property
            dimension_prop = max(changing_properties.items(), key=lambda x: x[1])[0]
            return f"{dimension_prop}_dimension"
        
        return "abstract_dimension"
    
    def _pattern_relevant_to_context(self, pattern: Dict[str, Any], context: SharedContext) -> bool:
        """Check if a conceptual pattern is relevant to context"""
        pattern_type = pattern.get("type", "")
        
        # Check based on pattern type
        if pattern_type == "hierarchy":
            # Hierarchies relevant for understanding relationships
            if "relationship" in context.user_input.lower() or "structure" in context.user_input.lower():
                return True
        
        elif pattern_type == "cluster":
            # Clusters relevant for grouping/categorization
            if "group" in context.user_input.lower() or "category" in context.user_input.lower():
                return True
        
        elif pattern_type == "bridge":
            # Bridges relevant for integration/connection
            if "connect" in context.user_input.lower() or "integrate" in context.user_input.lower():
                return True
        
        elif pattern_type == "gradient":
            # Gradients relevant for transitions/progressions
            if "progress" in context.user_input.lower() or "transition" in context.user_input.lower():
                return True
        
        # General relevance check
        pattern_desc = str(pattern).lower()
        input_words = set(context.user_input.lower().split())
        
        return any(word in pattern_desc for word in input_words)
    
    # ========================================================================================
    # BLEND OPPORTUNITY ASSESSMENT
    # ========================================================================================
    
    async def _assess_blend_opportunity(self, space1_id: str, space2_id: str,
                                      context: SharedContext) -> Dict[str, Any]:
        """Assess the opportunity for blending two conceptual spaces"""
        space1 = self.original_core.concept_spaces.get(space1_id)
        space2 = self.original_core.concept_spaces.get(space2_id)
        
        if not space1 or not space2:
            return {"score": 0.0}
        
        opportunity = {
            "space1": space1.name,
            "space2": space2.name,
            "score": 0.0,
            "blend_type": "none",
            "potential_insights": [],
            "mapping_quality": 0.0,
            "context_alignment": 0.0
        }
        
        # Find potential mappings
        mappings = await self._find_contextual_mappings(space1_id, space2_id, context)
        
        if not mappings:
            return opportunity
        
        # Assess mapping quality
        mapping_quality = sum(m["similarity"] for m in mappings) / len(mappings)
        opportunity["mapping_quality"] = mapping_quality
        
        # Base score on mapping quality
        opportunity["score"] = mapping_quality * 0.4
        
        # Check for complementary structures
        space1_patterns = await self._identify_conceptual_patterns(space1, context)
        space2_patterns = await self._identify_conceptual_patterns(space2, context)
        
        complementarity = self._assess_pattern_complementarity(space1_patterns, space2_patterns)
        opportunity["score"] += complementarity * 0.3
        
        # Context alignment
        context_alignment = self._assess_blend_context_alignment(space1, space2, context)
        opportunity["context_alignment"] = context_alignment
        opportunity["score"] += context_alignment * 0.3
        
        # Determine best blend type
        if mapping_quality > 0.7:
            opportunity["blend_type"] = "fusion"  # Deep integration
        elif complementarity > 0.6:
            opportunity["blend_type"] = "completion"  # Fill gaps
        elif context.emotional_state and context.emotional_state.get("dominant_emotion", [""])[0] == "Curiosity":
            opportunity["blend_type"] = "elaboration"  # Explore possibilities
        else:
            opportunity["blend_type"] = "composition"  # Standard blend
        
        # Identify potential insights
        opportunity["potential_insights"] = self._identify_blend_insights(
            space1, space2, mappings, opportunity["blend_type"]
        )
        
        return opportunity
    
    def _assess_pattern_complementarity(self, patterns1: List[Dict], patterns2: List[Dict]) -> float:
        """Assess how well two sets of patterns complement each other"""
        complementarity = 0.0
        
        # Get pattern type distributions
        types1 = {p["type"] for p in patterns1}
        types2 = {p["type"] for p in patterns2}
        
        # Different pattern types suggest complementarity
        unique_to_1 = types1 - types2
        unique_to_2 = types2 - types1
        
        if unique_to_1 or unique_to_2:
            complementarity += 0.4
        
        # Check for hierarchies + clusters (good combination)
        if "hierarchy" in types1 and "cluster" in types2:
            complementarity += 0.3
        elif "cluster" in types1 and "hierarchy" in types2:
            complementarity += 0.3
        
        # Check for bridges in one and not the other (integration opportunity)
        if "bridge" in types1 and "bridge" not in types2:
            complementarity += 0.2
        
        return min(1.0, complementarity)
    
    def _assess_blend_context_alignment(self, space1, space2, context: SharedContext) -> float:
        """Assess how well a potential blend aligns with context"""
        alignment = 0.0
        
        # Check if both spaces relate to user query
        input_keywords = set(context.user_input.lower().split())
        
        space1_relevance = self._calculate_space_relevance(space1, input_keywords)
        space2_relevance = self._calculate_space_relevance(space2, input_keywords)
        
        # Both spaces should be somewhat relevant
        min_relevance = min(space1_relevance, space2_relevance)
        alignment += min_relevance * 0.5
        
        # Check goal alignment
        if context.goal_context:
            goals = context.goal_context.get("active_goals", [])
            
            for goal in goals:
                if "creative" in goal.get("description", "").lower():
                    alignment += 0.2  # Creativity goals favor blending
                elif "integrate" in goal.get("description", "").lower():
                    alignment += 0.3  # Integration goals strongly favor blending
        
        return min(1.0, alignment)
    
    def _calculate_space_relevance(self, space, keywords: set) -> float:
        """Calculate relevance of a conceptual space to keywords"""
        relevance = 0.0
        
        # Check space name and domain
        if any(kw in space.name.lower() for kw in keywords):
            relevance += 0.3
        
        if space.domain and any(kw in space.domain.lower() for kw in keywords):
            relevance += 0.2
        
        # Check concepts in space
        matching_concepts = 0
        for concept in space.concepts.values():
            if any(kw in concept["name"].lower() for kw in keywords):
                matching_concepts += 1
        
        relevance += min(matching_concepts / 10.0, 0.5)
        
        return min(1.0, relevance)
    
    def _identify_blend_insights(self, space1, space2, mappings: List[Dict], 
                               blend_type: str) -> List[str]:
        """Identify potential insights from blending"""
        insights = []
        
        if blend_type == "fusion":
            insights.append("Deep structural alignment could reveal hidden isomorphisms")
            insights.append("Unified framework might emerge from merged concepts")
        
        elif blend_type == "completion":
            insights.append("Gap-filling could reveal missing conceptual links")
            insights.append("Complementary structures might form complete picture")
        
        elif blend_type == "elaboration":
            insights.append("Novel concept combinations could generate creative solutions")
            insights.append("Exploratory blending might reveal unexpected connections")
        
        else:  # composition
            insights.append("Standard blending could create useful hybrid concepts")
            insights.append("Cross-domain mapping might enable knowledge transfer")
        
        # Add specific insights based on mappings
        if len(mappings) > 5:
            insights.append(f"Rich mapping structure ({len(mappings)} connections) suggests high integration potential")
        
        return insights
    
    # ========================================================================================
    # COUNTERFACTUAL AND INTERVENTION METHODS
    # ========================================================================================
    
    async def _generate_counterfactual_scenario(self, model_id: str, 
                                              context: SharedContext) -> Optional[Dict[str, Any]]:
        """Generate a counterfactual scenario based on context"""
        model = self.original_core.causal_models.get(model_id)
        if not model:
            return None
        
        # Parse counterfactual query
        query_lower = context.user_input.lower()
        
        # Extract the counterfactual condition
        condition = None
        if "what if" in query_lower:
            condition = query_lower.split("what if")[1].strip()
        elif "would have" in query_lower:
            parts = query_lower.split("would have")
            if len(parts) > 1:
                condition = parts[0].strip()
        
        if not condition:
            return None
        
        # Find relevant nodes for the condition
        relevant_nodes = []
        condition_words = set(condition.split())
        
        for node_id, node in model.nodes.items():
            node_words = set(node.name.lower().split())
            if node_words.intersection(condition_words):
                relevant_nodes.append(node_id)
        
        if not relevant_nodes:
            return None
        
        # Generate scenario
        scenario = {
            "condition": condition,
            "model": model.name,
            "intervention_nodes": relevant_nodes[:2],  # Limit to 2 nodes
            "original_state": {},
            "counterfactual_state": {},
            "changes": [],
            "confidence": 0.7
        }
        
        # For each intervention node, trace effects
        for node_id in scenario["intervention_nodes"]:
            node = model.nodes[node_id]
            
            # Record original state
            scenario["original_state"][node.name] = {
                "type": getattr(node, "node_type", "standard"),
                "properties": getattr(node, "properties", {})
            }
            
            # Create counterfactual state
            scenario["counterfactual_state"][node.name] = {
                "type": "intervened",
                "properties": {"counterfactual": True}
            }
            
            # Trace downstream effects
            effects = self._trace_intervention_effects(node_id, model)
            scenario["changes"].extend(effects)
        
        return scenario
    
    def _trace_intervention_effects(self, node_id: str, model) -> List[Dict[str, Any]]:
        """Trace the effects of intervening on a node"""
        effects = []
        
        # Get immediate effects
        immediate_effects = model.get_descendants(node_id, max_depth=1)
        
        for effect_id in immediate_effects:
            effect_node = model.nodes.get(effect_id)
            if effect_node:
                # Find the relation strength
                relation_strength = 0.5  # Default
                for relation in model.relations:
                    if relation.source == node_id and relation.target == effect_id:
                        relation_strength = relation.strength
                        break
                
                effects.append({
                    "affected_node": effect_node.name,
                    "effect_type": "direct",
                    "strength": relation_strength,
                    "mechanism": "causal_influence"
                })
        
        # Get secondary effects
        secondary_effects = model.get_descendants(node_id, max_depth=2)
        secondary_effects = set(secondary_effects) - set(immediate_effects) - {node_id}
        
        for effect_id in list(secondary_effects)[:3]:  # Limit secondary effects
            effect_node = model.nodes.get(effect_id)
            if effect_node:
                effects.append({
                    "affected_node": effect_node.name,
                    "effect_type": "indirect",
                    "strength": 0.3,  # Weaker for indirect
                    "mechanism": "propagated_influence"
                })
        
        return effects
    
    async def _analyze_alternative_paths(self, model_id: str, 
                                       scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze alternative causal paths given a counterfactual scenario"""
        model = self.original_core.causal_models.get(model_id)
        if not model:
            return []
        
        alternative_paths = []
        
        # For each intervention node, find alternative paths to key outcomes
        for node_id in scenario["intervention_nodes"]:
            # Find important outcome nodes (high centrality or user-mentioned)
            outcome_nodes = self._identify_outcome_nodes(model, scenario)
            
            for outcome_id in outcome_nodes:
                # Find paths that don't go through intervention node
                paths = self._find_alternative_causal_paths(
                    model, node_id, outcome_id
                )
                
                for path in paths:
                    alternative_paths.append({
                        "blocked_node": model.nodes[node_id].name,
                        "outcome": model.nodes[outcome_id].name,
                        "alternative_route": [model.nodes[n].name for n in path],
                        "path_strength": self._calculate_path_strength(path, model),
                        "viability": self._assess_path_viability(path, model)
                    })
        
        # Sort by viability
        alternative_paths.sort(key=lambda p: p["viability"], reverse=True)
        
        return alternative_paths[:5]  # Top 5 alternatives
    
    def _identify_outcome_nodes(self, model, scenario: Dict[str, Any]) -> List[str]:
        """Identify important outcome nodes in the model"""
        outcome_nodes = []
        
        # High out-degree nodes (consequences)
        for node_id, node in model.nodes.items():
            out_degree = sum(1 for r in model.relations if r.source == node_id)
            if out_degree == 0:  # Leaf nodes
                outcome_nodes.append(node_id)
        
        # Limit to top 3 by centrality
        if len(outcome_nodes) > 3:
            # Simple centrality: nodes with most incoming paths
            centrality = {}
            for node_id in outcome_nodes:
                ancestors = model.get_ancestors(node_id)
                centrality[node_id] = len(ancestors)
            
            sorted_outcomes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            outcome_nodes = [node_id for node_id, _ in sorted_outcomes[:3]]
        
        return outcome_nodes
    
    def _find_alternative_causal_paths(self, model, blocked_node: str, 
                                     target_node: str) -> List[List[str]]:
        """Find causal paths that don't go through blocked node"""
        alternative_paths = []
        
        # Modified BFS that avoids blocked node
        queue = [(target_node, [target_node])]
        visited = {target_node}
        
        while queue and len(alternative_paths) < 3:  # Find up to 3 paths
            current, path = queue.pop(0)
            
            # Find predecessors (going backwards from target)
            predecessors = []
            for relation in model.relations:
                if relation.target == current and relation.source != blocked_node:
                    predecessors.append(relation.source)
            
            for pred in predecessors:
                if pred in visited:
                    continue
                    
                visited.add(pred)
                new_path = [pred] + path
                
                # Check if we've found a complete path (no more predecessors)
                pred_predecessors = [r.source for r in model.relations if r.target == pred]
                if not pred_predecessors:
                    alternative_paths.append(new_path)
                else:
                    queue.append((pred, new_path))
        
        return alternative_paths
    
    def _calculate_path_strength(self, path: List[str], model) -> float:
        """Calculate the strength of a causal path"""
        if len(path) < 2:
            return 0.0
        
        total_strength = 1.0
        
        for i in range(len(path) - 1):
            # Find relation between consecutive nodes
            source = path[i]
            target = path[i + 1]
            
            relation_strength = 0.5  # Default
            for relation in model.relations:
                if relation.source == source and relation.target == target:
                    relation_strength = relation.strength
                    break
            
            # Multiply strengths (assuming independence)
            total_strength *= relation_strength
        
        return total_strength
    
    def _assess_path_viability(self, path: List[str], model) -> float:
        """Assess how viable an alternative path is"""
        # Base viability on path strength
        viability = self._calculate_path_strength(path, model)
        
        # Penalize very long paths
        if len(path) > 4:
            viability *= 0.7
        elif len(path) > 6:
            viability *= 0.5
        
        # Bonus for paths through controllable nodes
        for node_id in path[1:-1]:  # Intermediate nodes
            node = model.nodes[node_id]
            if any(term in node.name.lower() for term in ["policy", "decision", "action"]):
                viability *= 1.2
        
        return min(1.0, viability)
    
    # ========================================================================================
    # GOAL AND INTERVENTION METHODS
    # ========================================================================================
    
    def _model_relates_to_goal(self, model, goal: Dict[str, Any]) -> bool:
        """Check if a causal model relates to a goal"""
        goal_desc = goal.get("description", "").lower()
        goal_keywords = set(goal_desc.split())
        
        # Check model name and domain
        model_text = f"{model.name} {model.domain}".lower()
        if any(kw in model_text for kw in goal_keywords):
            return True
        
        # Check node names
        for node in model.nodes.values():
            if any(kw in node.name.lower() for kw in goal_keywords):
                return True
        
        # Check associated need
        associated_need = goal.get("associated_need", "").lower()
        if associated_need and associated_need in model_text:
            return True
        
        return False
    
    async def _generate_intervention_suggestion(self, model_id: str, goal: Dict[str, Any],
                                             context: SharedContext) -> Optional[Dict[str, Any]]:
        """Generate an intervention suggestion to achieve a goal"""
        model = self.original_core.causal_models.get(model_id)
        if not model:
            return None
        
        # Find nodes related to goal
        goal_nodes = []
        goal_keywords = set(goal.get("description", "").lower().split())
        
        for node_id, node in model.nodes.items():
            node_words = set(node.name.lower().split())
            if node_words.intersection(goal_keywords):
                goal_nodes.append(node_id)
        
        if not goal_nodes:
            return None
        
        # Find best intervention point for goal
        best_intervention = None
        best_score = 0.0
        
        for goal_node in goal_nodes:
            # Find nodes that influence this goal node
            influencers = model.get_ancestors(goal_node, max_depth=2)
            
            for influencer_id in influencers:
                influencer = model.nodes.get(influencer_id)
                if not influencer:
                    continue
                
                # Score this intervention point
                score = 0.0
                
                # Controllability
                if any(term in influencer.name.lower() for term in ["action", "decision", "behavior"]):
                    score += 0.4
                
                # Direct influence
                for relation in model.relations:
                    if relation.source == influencer_id and relation.target == goal_node:
                        score += relation.strength * 0.3
                        break
                
                # Feasibility
                feasibility = self._assess_intervention_feasibility(influencer, context)
                score += feasibility * 0.3
                
                if score > best_score:
                    best_score = score
                    best_intervention = {
                        "intervention_node": influencer.name,
                        "goal_node": model.nodes[goal_node].name,
                        "intervention_type": self._determine_intervention_type(influencer, model),
                        "expected_impact": self._estimate_intervention_impact(influencer_id, model),
                        "confidence": best_score,
                        "implementation_suggestions": self._generate_implementation_suggestions(
                            influencer, goal, context
                        )
                    }
        
        return best_intervention
    
    def _generate_implementation_suggestions(self, intervention_node, goal: Dict[str, Any],
                                           context: SharedContext) -> List[str]:
        """Generate specific suggestions for implementing an intervention"""
        suggestions = []
        node_name_lower = intervention_node.name.lower()
        
        # Generate suggestions based on intervention type
        if "behavior" in node_name_lower:
            suggestions.append("Establish clear behavioral targets and tracking metrics")
            suggestions.append("Use positive reinforcement to encourage desired behaviors")
        
        elif "decision" in node_name_lower:
            suggestions.append("Create decision frameworks or criteria")
            suggestions.append("Implement systematic decision review processes")
        
        elif "process" in node_name_lower:
            suggestions.append("Map current process and identify improvement points")
            suggestions.append("Implement incremental process changes with monitoring")
        
        elif "policy" in node_name_lower:
            suggestions.append("Draft clear policy guidelines with specific objectives")
            suggestions.append("Ensure stakeholder buy-in before implementation")
        
        # Add goal-specific suggestions
        if "understand" in goal.get("description", "").lower():
            suggestions.append("Document learnings and insights systematically")
        elif "improve" in goal.get("description", "").lower():
            suggestions.append("Set measurable improvement targets")
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    # ========================================================================================
    # INTEGRATION AND CREATIVE METHODS
    # ========================================================================================
    
    async def _assess_integration_potential(self, model_id: str, space_id: str,
                                          context: SharedContext) -> float:
        """Assess potential for integrating causal model with conceptual space"""
        model = self.original_core.causal_models.get(model_id)
        space = self.original_core.concept_spaces.get(space_id)
        
        if not model or not space:
            return 0.0
        
        potential = 0.0
        
        # Domain compatibility
        if model.domain and space.domain:
            if model.domain.lower() == space.domain.lower():
                potential += 0.3
            elif any(word in space.domain.lower() for word in model.domain.lower().split()):
                potential += 0.15
        
        # Concept-node overlap
        overlap_count = 0
        for node in model.nodes.values():
            for concept in space.concepts.values():
                if self._concepts_match(node.name, concept["name"]):
                    overlap_count += 1
        
        potential += min(overlap_count * 0.1, 0.4)
        
        # Structural compatibility
        if len(model.nodes) > 5 and len(space.concepts) > 5:
            potential += 0.1  # Both have rich structure
        
        # Context alignment
        if context.goal_context:
            goals = context.goal_context.get("active_goals", [])
            if any("integrate" in g.get("description", "").lower() for g in goals):
                potential += 0.2
        
        return min(1.0, potential)
    
    def _concepts_match(self, name1: str, name2: str) -> bool:
        """Check if two concept names match"""
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Exact match
        if name1_lower == name2_lower:
            return True
        
        # Word overlap
        words1 = set(name1_lower.split())
        words2 = set(name2_lower.split())
        
        overlap = len(words1.intersection(words2))
        if overlap >= min(len(words1), len(words2)) * 0.5:
            return True
        
        return False
    
    async def _suggest_creative_intervention(self, model_id: str, 
                                           context: SharedContext) -> Optional[Dict[str, Any]]:
        """Suggest a creative intervention based on causal understanding"""
        model = self.original_core.causal_models.get(model_id)
        if not model:
            return None
        
        # Find leverage points (nodes with high impact)
        leverage_points = []
        
        for node_id, node in model.nodes.items():
            descendants = model.get_descendants(node_id)
            if len(descendants) >= 3:  # Influences multiple outcomes
                leverage_points.append({
                    "node_id": node_id,
                    "node": node,
                    "impact_breadth": len(descendants)
                })
        
        if not leverage_points:
            return None
        
        # Select best leverage point
        best_point = max(leverage_points, key=lambda p: p["impact_breadth"])
        
        # Generate creative intervention
        intervention = {
            "target_node": best_point["node"].name,
            "intervention_class": "creative_leverage",
            "approach": self._generate_creative_approach(best_point["node"], context),
            "expected_outcomes": [
                model.nodes[desc].name 
                for desc in model.get_descendants(best_point["node_id"])[:3]
                if desc in model.nodes
            ],
            "creativity_factors": self._identify_creativity_factors(best_point["node"], model, context),
            "implementation_phases": self._design_intervention_phases(best_point["node"], context)
        }
        
        return intervention
    
    def _generate_creative_approach(self, node, context: SharedContext) -> str:
        """Generate a creative approach for intervention"""
        node_name_lower = node.name.lower()
        
        # Context-aware creative strategies
        if context.emotional_state:
            emotion = context.emotional_state.get("dominant_emotion", [""])[0]
            
            if emotion == "Curiosity":
                return f"Experimental exploration of {node.name} through systematic variation"
            elif emotion == "Frustration":
                return f"Radical reimagining of {node.name} to bypass current constraints"
        
        # Default creative strategies based on node type
        if "process" in node_name_lower:
            return "Process gamification to enhance engagement and outcomes"
        elif "behavior" in node_name_lower:
            return "Behavioral nudges using environmental design"
        elif "system" in node_name_lower:
            return "Systems thinking workshop to identify hidden connections"
        else:
            return f"Design thinking approach to reimagine {node.name}"
    
    def _identify_creativity_factors(self, node, model, context: SharedContext) -> List[str]:
        """Identify factors that enhance creative intervention potential"""
        factors = []
        
        # Node has many weak connections (opportunity for strengthening)
        weak_relations = [r for r in model.relations 
                         if (r.source == node.id or r.target == node.id) and r.strength < 0.3]
        if len(weak_relations) > 2:
            factors.append("multiple_weak_connections_to_strengthen")
        
        # Node bridges different domains
        connected_nodes = set()
        for relation in model.relations:
            if relation.source == node.id:
                connected_nodes.add(relation.target)
            elif relation.target == node.id:
                connected_nodes.add(relation.source)
        
        domains = set()
        for node_id in connected_nodes:
            if node_id in model.nodes and hasattr(model.nodes[node_id], 'domain'):
                domains.add(model.nodes[node_id].domain)
        
        if len(domains) > 1:
            factors.append("cross_domain_bridge_potential")
        
        # Context suggests innovation need
        if "new" in context.user_input.lower() or "innovative" in context.user_input.lower():
            factors.append("explicit_innovation_requirement")
        
        return factors
    
    def _design_intervention_phases(self, node, context: SharedContext) -> List[Dict[str, str]]:
        """Design phases for implementing creative intervention"""
        phases = [
            {
                "phase": "Discovery",
                "duration": "1-2 weeks",
                "activities": f"Map current state of {node.name} and identify constraints"
            },
            {
                "phase": "Ideation",
                "duration": "1 week",
                "activities": "Generate diverse intervention ideas using creative techniques"
            },
            {
                "phase": "Prototyping",
                "duration": "2-3 weeks",
                "activities": "Test small-scale versions of most promising interventions"
            },
            {
                "phase": "Implementation",
                "duration": "4-6 weeks",
                "activities": "Roll out refined intervention with continuous monitoring"
            },
            {
                "phase": "Evolution",
                "duration": "Ongoing",
                "activities": "Iterate based on outcomes and emergent opportunities"
            }
        ]
        
        # Adjust based on context
        if context.constraints and "time_sensitive" in context.constraints:
            # Compress timeline
            for phase in phases:
                phase["duration"] = "Accelerated"
        
        return phases
    
    # ========================================================================================
    # ADDITIONAL HELPER METHOD FOR BLENDING
    # ========================================================================================
    
    def _concept_aligns_with_emotion(self, concept: Dict[str, Any], emotion: str) -> bool:
        """Check if a concept aligns with an emotional state"""
        concept_name = concept.get("name", "").lower()
        properties = concept.get("properties", {})
        
        emotion_keywords = {
            "Curiosity": ["unknown", "explore", "discover", "mystery", "question"],
            "Joy": ["positive", "success", "achievement", "happiness", "reward"],
            "Frustration": ["obstacle", "challenge", "difficulty", "problem", "constraint"],
            "Anxiety": ["threat", "risk", "uncertainty", "danger", "concern"],
            "Satisfaction": ["complete", "fulfilled", "achieved", "resolved", "success"]
        }
        
        keywords = emotion_keywords.get(emotion, [])
        
        # Check concept name
        if any(kw in concept_name for kw in keywords):
            return True
        
        # Check properties
        for prop_value in properties.values():
            if isinstance(prop_value, str) and any(kw in prop_value.lower() for kw in keywords):
                return True
        
        return False
    
    # ========================================================================================
    # DELEGATE TO ORIGINAL CORE
    # ========================================================================================
    
    def __getattr__(self, name):
        """Delegate any missing methods to the original reasoning core"""
        return getattr(self.original_core, name)
