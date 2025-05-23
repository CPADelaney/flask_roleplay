# nyx/core/a2a/context_aware_reasoning_agents.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

from agents import Agent, Runner, RunContextWrapper
from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareReasoningAgents(ContextAwareModule):
    """
    Context-aware wrapper for reasoning agents with A2A integration
    """
    
    def __init__(self, triage_agent, integrated_agent, context_aware_reasoning_core):
        super().__init__("reasoning_agents")
        self.triage_agent = triage_agent
        self.integrated_agent = integrated_agent
        self.reasoning_core = context_aware_reasoning_core
        
        self.context_subscriptions = [
            "reasoning_request", "goal_context_available",
            "emotional_state_update", "knowledge_update",
            "urgent_reasoning_need"
        ]
        
        # Track active reasoning sessions
        self.active_sessions: Dict[str, Any] = {}
        self.agent_selection_history: List[Dict[str, Any]] = []
    
    async def on_context_received(self, context: SharedContext):
        """Initialize agent reasoning for this context"""
        logger.debug(f"ReasoningAgents received context for user: {context.user_id}")
        
        # Analyze input for agent routing
        routing_analysis = await self._analyze_input_for_routing(context.user_input)
        
        # Determine best agent based on context
        selected_agent = await self._select_agent_by_context(context, routing_analysis)
        
        # Send initial routing decision to other modules
        await self.send_context_update(
            update_type="reasoning_agent_selected",
            data={
                "selected_agent": selected_agent["name"],
                "routing_reason": selected_agent["reason"],
                "confidence": selected_agent["confidence"],
                "available_capabilities": self._get_agent_capabilities(selected_agent["agent"])
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect agent reasoning"""
        
        if update.update_type == "reasoning_request":
            # Direct reasoning request
            request_data = update.data
            query = request_data.get("query", "")
            request_type = request_data.get("type", "general")
            
            # Route to appropriate agent
            await self._handle_reasoning_request(query, request_type, update)
        
        elif update.update_type == "urgent_reasoning_need":
            # Urgent reasoning required
            urgency_data = update.data
            await self._handle_urgent_reasoning(urgency_data)
        
        elif update.update_type == "goal_context_available":
            # Goals might influence agent selection
            goal_data = update.data
            await self._update_agent_context_from_goals(goal_data)
        
        elif update.update_type == "emotional_state_update":
            # Emotions might affect reasoning approach
            emotional_data = update.data
            await self._update_agent_context_from_emotion(emotional_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input through reasoning agents with context awareness"""
        # Create session ID
        session_id = f"reasoning_{context.user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Initialize session
        self.active_sessions[session_id] = {
            "started": datetime.now(),
            "context": context,
            "agent_calls": 0,
            "handoffs": 0
        }
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Determine routing with full context
        routing_decision = await self._make_contextual_routing_decision(context, messages)
        
        # Create reasoning context for agents
        from nyx.core.reasoning_agents import ReasoningContext
        agent_context = ReasoningContext(
            knowledge_core=self.reasoning_core.knowledge_core,
            session_id=session_id,
            user_id=context.user_id
        )
        
        # Add context information
        agent_context.current_domain = self._extract_domain_from_context(context)
        
        # Run selected agent
        selected_agent = routing_decision["agent"]
        
        try:
            # Run agent with context wrapper
            result = await Runner.run(
                selected_agent,
                context.user_input,
                context=agent_context
            )
            
            # Track session stats
            self.active_sessions[session_id]["agent_calls"] = agent_context.total_calls
            self.active_sessions[session_id]["handoffs"] = agent_context.handoffs
            
            # Process agent output
            agent_output = result.final_output
            
            # Send completion update
            await self.send_context_update(
                update_type="reasoning_agent_complete",
                data={
                    "session_id": session_id,
                    "agent_output": agent_output,
                    "total_calls": agent_context.total_calls,
                    "handoffs": agent_context.handoffs,
                    "models_created": len(self.reasoning_core.active_models),
                    "spaces_created": len(self.reasoning_core.active_spaces)
                }
            )
            
            return {
                "agent_output": agent_output,
                "session_id": session_id,
                "routing_decision": routing_decision,
                "context_integrated": True
            }
            
        except Exception as e:
            logger.error(f"Error in reasoning agent execution: {e}")
            return {
                "error": str(e),
                "session_id": session_id,
                "fallback": "Reasoning processing encountered an issue"
            }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze agent capabilities in current context"""
        messages = await self.get_cross_module_messages()
        
        # Analyze each agent's relevance to current context
        agent_analysis = {
            "triage_agent": await self._analyze_agent_relevance(self.triage_agent, context, messages),
            "integrated_agent": await self._analyze_agent_relevance(self.integrated_agent, context, messages)
        }
        
        # Analyze current reasoning needs
        reasoning_needs = await self._analyze_reasoning_needs(context, messages)
        
        # Analyze agent selection patterns
        selection_patterns = self._analyze_selection_patterns()
        
        return {
            "agent_analysis": agent_analysis,
            "reasoning_needs": reasoning_needs,
            "selection_patterns": selection_patterns,
            "active_sessions": len(self.active_sessions),
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize agent reasoning for response generation"""
        messages = await self.get_cross_module_messages()
        
        # Get latest session if exists
        latest_session = None
        if self.active_sessions:
            session_id = max(self.active_sessions.keys())
            latest_session = self.active_sessions[session_id]
        
        # Synthesize reasoning narrative
        reasoning_synthesis = {
            "reasoning_approach": await self._synthesize_reasoning_approach(context, latest_session),
            "key_insights": await self._synthesize_key_insights(context, messages),
            "reasoning_confidence": await self._assess_reasoning_confidence(latest_session),
            "suggested_follow_ups": await self._suggest_follow_up_questions(context, latest_session),
            "integration_summary": await self._synthesize_integration_summary(messages)
        }
        
        return {
            "reasoning_synthesis": reasoning_synthesis,
            "synthesis_complete": True,
            "session_active": latest_session is not None
        }
    
    # ========================================================================================
    # ROUTING AND SELECTION METHODS
    # ========================================================================================
    
    async def _analyze_input_for_routing(self, user_input: str) -> Dict[str, Any]:
        """Analyze input to determine agent routing"""
        input_lower = user_input.lower()
        
        analysis = {
            "suggests_causal": any(kw in input_lower for kw in 
                ["cause", "effect", "why", "because", "leads to", "results in"]),
            "suggests_conceptual": any(kw in input_lower for kw in 
                ["concept", "idea", "blend", "combine", "creative"]),
            "suggests_integrated": any(kw in input_lower for kw in 
                ["analyze", "understand deeply", "complex", "integrate"]),
            "complexity_score": self._assess_query_complexity(user_input),
            "domain_specific": self._check_domain_specificity(user_input)
        }
        
        return analysis
    
    async def _select_agent_by_context(self, context: SharedContext, 
                                    routing_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Select best agent based on full context"""
        # Default to triage for routing
        selected_agent = self.triage_agent
        reason = "default_routing"
        confidence = 0.7
        
        # Check if we should go directly to integrated agent
        if routing_analysis["complexity_score"] > 0.7:
            selected_agent = self.integrated_agent
            reason = "high_complexity_query"
            confidence = 0.85
        elif routing_analysis["suggests_integrated"]:
            selected_agent = self.integrated_agent
            reason = "integrated_reasoning_needed"
            confidence = 0.8
        
        # Check emotional context
        if context.emotional_state:
            dominant_emotion = context.emotional_state.get("dominant_emotion")
            if dominant_emotion and dominant_emotion[0] == "Curiosity" and dominant_emotion[1] > 0.6:
                # High curiosity favors integrated reasoning
                selected_agent = self.integrated_agent
                reason = "curiosity_driven_exploration"
                confidence = 0.75
        
        # Check goal context
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            understanding_goals = [g for g in active_goals if "understand" in g.get("description", "").lower()]
            
            if understanding_goals:
                selected_agent = self.integrated_agent
                reason = "understanding_goal_active"
                confidence = 0.8
        
        # Record selection
        self.agent_selection_history.append({
            "timestamp": datetime.now().isoformat(),
            "selected": selected_agent.name,
            "reason": reason,
            "confidence": confidence,
            "context_factors": {
                "has_emotion": bool(context.emotional_state),
                "has_goals": bool(context.goal_context),
                "complexity": routing_analysis["complexity_score"]
            }
        })
        
        return {
            "agent": selected_agent,
            "name": selected_agent.name,
            "reason": reason,
            "confidence": confidence
        }
    
    async def _make_contextual_routing_decision(self, context: SharedContext,
                                             messages: Dict) -> Dict[str, Any]:
        """Make routing decision with full cross-module context"""
        # Start with basic routing analysis
        routing_analysis = await self._analyze_input_for_routing(context.user_input)
        
        # Enhance with cross-module information
        if "knowledge_core" in messages:
            # Knowledge availability might favor causal reasoning
            routing_analysis["knowledge_available"] = True
            routing_analysis["complexity_score"] += 0.1
        
        if "multimodal_integrator" in messages:
            # Multimodal input suggests integrated reasoning
            routing_analysis["multimodal_input"] = True
            routing_analysis["suggests_integrated"] = True
        
        # Make selection
        selected = await self._select_agent_by_context(context, routing_analysis)
        
        return {
            "agent": selected["agent"],
            "routing_analysis": routing_analysis,
            "confidence": selected["confidence"],
            "reason": selected["reason"]
        }
    
    # ========================================================================================
    # CONTEXT UPDATE HANDLERS
    # ========================================================================================
    
    async def _handle_reasoning_request(self, query: str, request_type: str, 
                                     update: ContextUpdate):
        """Handle direct reasoning request"""
        # Create minimal context for request
        from nyx.core.reasoning_agents import ReasoningContext
        
        session_id = f"direct_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        agent_context = ReasoningContext(
            knowledge_core=self.reasoning_core.knowledge_core,
            session_id=session_id,
            user_id=update.source_module  # Use source module as pseudo-user
        )
        
        # Route based on request type
        if request_type == "causal":
            # Direct to causal agent via integrated agent
            result = await Runner.run(
                self.integrated_agent,
                f"Analyze the causal relationships in: {query}",
                context=agent_context
            )
        elif request_type == "conceptual":
            # Direct to conceptual agent via integrated agent
            result = await Runner.run(
                self.integrated_agent,
                f"Explore the conceptual aspects of: {query}",
                context=agent_context
            )
        else:
            # Use triage
            result = await Runner.run(
                self.triage_agent,
                query,
                context=agent_context
            )
        
        # Send result back
        await self.send_context_update(
            update_type="reasoning_request_complete",
            data={
                "request_id": update.timestamp.isoformat(),
                "result": result.final_output,
                "request_type": request_type
            },
            target_modules=[update.source_module],
            scope=ContextScope.TARGETED
        )
    
    async def _handle_urgent_reasoning(self, urgency_data: Dict[str, Any]):
        """Handle urgent reasoning needs"""
        urgency_level = urgency_data.get("urgency_level", 0.5)
        reasoning_need = urgency_data.get("reasoning_need", "")
        
        if urgency_level > 0.8:
            # High urgency - use integrated agent directly
            from nyx.core.reasoning_agents import ReasoningContext
            
            session_id = f"urgent_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            agent_context = ReasoningContext(
                knowledge_core=self.reasoning_core.knowledge_core,
                session_id=session_id
            )
            
            # Run integrated agent
            result = await Runner.run(
                self.integrated_agent,
                reasoning_need,
                context=agent_context
            )
            
            # Send urgent result
            await self.send_context_update(
                update_type="urgent_reasoning_complete",
                data={
                    "urgency_level": urgency_level,
                    "result": result.final_output
                },
                priority=ContextPriority.CRITICAL
            )
    
    async def _update_agent_context_from_goals(self, goal_data: Dict[str, Any]):
        """Update agent context based on goals"""
        active_goals = goal_data.get("active_goals", [])
        
        # Find reasoning-related goals
        reasoning_goals = []
        for goal in active_goals:
            desc_lower = goal.get("description", "").lower()
            if any(kw in desc_lower for kw in ["understand", "analyze", "reason", "figure out"]):
                reasoning_goals.append(goal)
        
        # Store in agent context
        if reasoning_goals:
            logger.info(f"Found {len(reasoning_goals)} reasoning-related goals")
    
    async def _update_agent_context_from_emotion(self, emotional_data: Dict[str, Any]):
        """Update agent context based on emotional state"""
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if dominant_emotion:
            emotion_name, strength = dominant_emotion
            
            # Log emotional influence on reasoning
            logger.info(f"Emotional context: {emotion_name} (strength: {strength:.2f})")
    
    # ========================================================================================
    # ANALYSIS METHODS
    # ========================================================================================
    
    async def _analyze_agent_relevance(self, agent: Agent, context: SharedContext,
                                    messages: Dict) -> Dict[str, Any]:
        """Analyze agent relevance to current context"""
        relevance = {
            "agent_name": agent.name,
            "relevance_score": 0.5,
            "relevance_factors": []
        }
        
        # Check agent capabilities against context
        if hasattr(agent, 'tools'):
            tool_names = [tool.name for tool in agent.tools]
            
            # Check for causal tools
            if any("causal" in name for name in tool_names):
                if "cause" in context.user_input.lower() or "why" in context.user_input.lower():
                    relevance["relevance_score"] += 0.2
                    relevance["relevance_factors"].append("causal_tools_match_query")
            
            # Check for conceptual tools
            if any("concept" in name or "blend" in name for name in tool_names):
                if "idea" in context.user_input.lower() or "creative" in context.user_input.lower():
                    relevance["relevance_score"] += 0.2
                    relevance["relevance_factors"].append("conceptual_tools_match_query")
        
        # Check handoff capabilities
        if hasattr(agent, 'handoffs'):
            relevance["has_handoffs"] = len(agent.handoffs) > 0
            if relevance["has_handoffs"]:
                relevance["relevance_score"] += 0.1
                relevance["relevance_factors"].append("can_delegate")
        
        return relevance
    
    async def _analyze_reasoning_needs(self, context: SharedContext,
                                    messages: Dict) -> Dict[str, Any]:
        """Analyze what type of reasoning is needed"""
        needs = {
            "primary_need": "none",
            "secondary_needs": [],
            "complexity": "low",
            "suggested_approach": "triage"
        }
        
        # Analyze query complexity
        complexity_score = self._assess_query_complexity(context.user_input)
        
        if complexity_score > 0.7:
            needs["complexity"] = "high"
            needs["suggested_approach"] = "integrated"
        elif complexity_score > 0.4:
            needs["complexity"] = "medium"
            needs["suggested_approach"] = "specialized"
        
        # Determine primary need
        input_lower = context.user_input.lower()
        
        if "why" in input_lower or "cause" in input_lower:
            needs["primary_need"] = "causal_explanation"
        elif "what if" in input_lower:
            needs["primary_need"] = "counterfactual_analysis"
        elif "creative" in input_lower or "imagine" in input_lower:
            needs["primary_need"] = "conceptual_exploration"
        elif "understand" in input_lower:
            needs["primary_need"] = "deep_understanding"
        
        # Check for secondary needs based on context
        if context.emotional_state:
            needs["secondary_needs"].append("emotion_aware_reasoning")
        
        if context.goal_context:
            needs["secondary_needs"].append("goal_aligned_reasoning")
        
        return needs
    
    def _analyze_selection_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in agent selection history"""
        if not self.agent_selection_history:
            return {"pattern": "no_history"}
        
        # Count selections
        selection_counts = {}
        reason_counts = {}
        
        for selection in self.agent_selection_history[-20:]:  # Last 20 selections
            agent = selection["selected"]
            reason = selection["reason"]
            
            selection_counts[agent] = selection_counts.get(agent, 0) + 1
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        # Find most common
        most_selected = max(selection_counts.items(), key=lambda x: x[1])[0] if selection_counts else None
        most_common_reason = max(reason_counts.items(), key=lambda x: x[1])[0] if reason_counts else None
        
        return {
            "selection_counts": selection_counts,
            "reason_counts": reason_counts,
            "most_selected_agent": most_selected,
            "most_common_reason": most_common_reason,
            "total_selections": len(self.agent_selection_history)
        }
    
    # ========================================================================================
    # SYNTHESIS METHODS
    # ========================================================================================
    
    async def _synthesize_reasoning_approach(self, context: SharedContext,
                                          session: Optional[Dict[str, Any]]) -> str:
        """Synthesize description of reasoning approach"""
        if not session:
            return "Ready to apply integrated reasoning to analyze your query."
        
        # Describe based on session stats
        if session["handoffs"] > 0:
            return f"I engaged multiple specialized reasoning agents to thoroughly analyze your question."
        elif session["agent_calls"] > 5:
            return f"I performed an in-depth analysis using {session['agent_calls']} reasoning operations."
        else:
            return "I applied focused reasoning to address your query."
    
    async def _synthesize_key_insights(self, context: SharedContext,
                                    messages: Dict) -> List[str]:
        """Synthesize key insights from reasoning"""
        insights = []
        
        # Check for causal insights
        if self.reasoning_core.active_models:
            insights.append(f"Identified causal relationships across {len(self.reasoning_core.active_models)} models")
        
        # Check for conceptual insights  
        if self.reasoning_core.active_spaces:
            insights.append(f"Explored {len(self.reasoning_core.active_spaces)} conceptual spaces")
        
        # Check for discoveries
        if hasattr(self.reasoning_core, 'reasoning_context'):
            discoveries = self.reasoning_core.reasoning_context.get("new_relations_discovered", 0)
            if discoveries > 0:
                insights.append(f"Discovered {discoveries} new causal relationships")
        
        return insights
    
    async def _assess_reasoning_confidence(self, session: Optional[Dict[str, Any]]) -> float:
        """Assess confidence in reasoning results"""
        if not session:
            return 0.5
        
        confidence = 0.6  # Base confidence
        
        # More operations generally mean more thorough analysis
        if session["agent_calls"] > 5:
            confidence += 0.2
        elif session["agent_calls"] > 2:
            confidence += 0.1
        
        # Handoffs indicate comprehensive analysis
        if session["handoffs"] > 0:
            confidence += 0.1
        
        # Active models/spaces indicate grounded reasoning
        if self.reasoning_core.active_models or self.reasoning_core.active_spaces:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    async def _suggest_follow_up_questions(self, context: SharedContext,
                                        session: Optional[Dict[str, Any]]) -> List[str]:
        """Suggest follow-up questions based on reasoning"""
        suggestions = []
        
        if self.reasoning_core.active_models:
            suggestions.append("Would you like me to explore specific causal pathways in more detail?")
        
        if self.reasoning_core.active_spaces:
            suggestions.append("Should I create conceptual blends to generate novel insights?")
        
        # Check for counterfactual opportunities
        if "what if" not in context.user_input.lower() and self.reasoning_core.active_models:
            suggestions.append("Would you like to explore 'what if' scenarios with counterfactual reasoning?")
        
        return suggestions[:2]  # Limit to 2 suggestions
    
    async def _synthesize_integration_summary(self, messages: Dict) -> Dict[str, Any]:
        """Synthesize how reasoning integrated with other modules"""
        integration = {
            "modules_integrated": list(messages.keys()),
            "integration_depth": "none"
        }
        
        if not messages:
            return integration
        
        # Assess integration depth
        if len(messages) > 3:
            integration["integration_depth"] = "deep"
        elif len(messages) > 1:
            integration["integration_depth"] = "moderate"
        else:
            integration["integration_depth"] = "shallow"
        
        # Specific integrations
        if "emotional_core" in messages:
            integration["emotion_aware"] = True
        
        if "goal_manager" in messages:
            integration["goal_aligned"] = True
        
        if "memory_core" in messages:
            integration["memory_informed"] = True
        
        return integration
    
    # ========================================================================================
    # UTILITY METHODS
    # ========================================================================================
    
    def _assess_query_complexity(self, query: str) -> float:
        """Assess the complexity of a query"""
        complexity = 0.0
        
        # Length factor
        word_count = len(query.split())
        if word_count > 20:
            complexity += 0.3
        elif word_count > 10:
            complexity += 0.2
        elif word_count > 5:
            complexity += 0.1
        
        # Question word diversity
        question_words = ["why", "how", "what", "when", "where", "which"]
        question_count = sum(1 for qw in question_words if qw in query.lower())
        complexity += question_count * 0.15
        
        # Conceptual complexity indicators
        complex_indicators = ["relationship between", "integrate", "analyze", "compare", 
                            "contrast", "explain how", "multiple factors", "complex"]
        for indicator in complex_indicators:
            if indicator in query.lower():
                complexity += 0.1
        
        # Conditional complexity
        if "if" in query.lower() and "then" in query.lower():
            complexity += 0.2
        
        return min(1.0, complexity)
    
    def _check_domain_specificity(self, query: str) -> bool:
        """Check if query is domain-specific"""
        domain_keywords = [
            "climate", "economics", "psychology", "biology", "physics",
            "chemistry", "sociology", "technology", "medicine", "philosophy"
        ]
        
        query_lower = query.lower()
        return any(domain in query_lower for domain in domain_keywords)
    
    def _extract_domain_from_context(self, context: SharedContext) -> str:
        """Extract domain from context"""
        # Try to extract from user input
        domain_keywords = {
            "climate": "climate",
            "weather": "climate",
            "economic": "economics",
            "money": "economics",
            "market": "economics",
            "psychology": "psychology",
            "mind": "psychology",
            "behavior": "psychology",
            "emotion": "psychology"
        }
        
        input_lower = context.user_input.lower()
        for keyword, domain in domain_keywords.items():
            if keyword in input_lower:
                return domain
        
        return "general"
    
    def _get_agent_capabilities(self, agent: Agent) -> List[str]:
        """Get list of agent capabilities"""
        capabilities = []
        
        if hasattr(agent, 'tools'):
            for tool in agent.tools:
                capabilities.append(tool.name)
        
        if hasattr(agent, 'handoffs'):
            capabilities.append(f"can_handoff_to_{len(agent.handoffs)}_agents")
        
        return capabilities
    
    # ========================================================================================
    # DELEGATE TO COMPONENTS
    # ========================================================================================
    
    def __getattr__(self, name):
        """Delegate to appropriate component"""
        # First check triage agent
        if hasattr(self.triage_agent, name):
            return getattr(self.triage_agent, name)
        # Then check integrated agent
        elif hasattr(self.integrated_agent, name):
            return getattr(self.integrated_agent, name)
        # Finally check reasoning core
        else:
            return getattr(self.reasoning_core, name)
