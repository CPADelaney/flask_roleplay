# nyx/core/a2a/context_aware_capability_system.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareCapabilitySystem(ContextAwareModule):
    """
    Advanced Capability Assessment System with full context distribution capabilities
    """
    
    def __init__(self, original_capability_system):
        super().__init__("capability_system")
        self.original_system = original_capability_system
        self.context_subscriptions = [
            "goal_context_available", "task_request", "capability_query",
            "learning_opportunity", "skill_development_update", "performance_feedback",
            "capability_gap_identified", "system_limitation_encountered"
        ]
        self.recent_assessments = []
    
    async def on_context_received(self, context: SharedContext):
        """Initialize capability assessment for this context"""
        logger.debug(f"CapabilitySystem received context for user: {context.user_id}")
        
        # Assess if input requires capability evaluation
        needs_assessment = await self._check_capability_assessment_needed(context)
        
        if needs_assessment:
            # Perform initial capability assessment
            assessment = await self._assess_context_capabilities(context)
            
            await self.send_context_update(
                update_type="capability_assessment_available",
                data={
                    "assessment": assessment,
                    "current_capabilities": self._get_current_capabilities(),
                    "confidence_levels": self._get_capability_confidence()
                },
                priority=ContextPriority.HIGH
            )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        
        if update.update_type == "goal_context_available":
            # Assess capabilities needed for goals
            goal_data = update.data
            capability_requirements = await self._assess_goal_capability_requirements(goal_data)
            
            if capability_requirements["gaps"]:
                await self.send_context_update(
                    update_type="capability_gaps_for_goals",
                    data=capability_requirements,
                    priority=ContextPriority.HIGH
                )
        
        elif update.update_type == "task_request":
            # Assess if we can handle the task
            task_data = update.data
            feasibility = await self._assess_task_feasibility(task_data)
            
            await self.send_context_update(
                update_type="task_feasibility_assessment",
                data=feasibility,
                target_modules=[update.source_module],
                scope=ContextScope.TARGETED
            )
        
        elif update.update_type == "performance_feedback":
            # Update capability confidence based on performance
            feedback_data = update.data
            await self._update_capability_confidence(feedback_data)
        
        elif update.update_type == "system_limitation_encountered":
            # Record new limitation/gap
            limitation_data = update.data
            await self._record_capability_gap(limitation_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for capability-related queries"""
        user_input = context.user_input
        
        # Check if user is asking about capabilities
        if self._is_capability_query(user_input):
            query_type = self._determine_query_type(user_input)
            
            if query_type == "what_can_you_do":
                capabilities = self.original_system.capability_model.get_implemented_capabilities()
                response_data = {
                    "query_type": query_type,
                    "capabilities": [cap.to_dict() for cap in capabilities],
                    "total_capabilities": len(capabilities)
                }
            
            elif query_type == "can_you_do_x":
                # Assess specific capability
                goal = self._extract_capability_goal(user_input)
                assessment = await self.original_system.assess_required_capabilities(goal)
                response_data = {
                    "query_type": query_type,
                    "assessment": assessment,
                    "feasible": assessment["overall_feasibility"]["score"] > 0.6
                }
            
            elif query_type == "limitations":
                gaps = await self.original_system.identify_capability_gaps()
                response_data = {
                    "query_type": query_type,
                    "gaps": gaps,
                    "honest_assessment": True
                }
            
            # Send capability response
            await self.send_context_update(
                update_type="capability_query_response",
                data=response_data,
                priority=ContextPriority.HIGH
            )
        
        # Check if we should add a desired capability
        if self._suggests_new_capability(user_input):
            capability_suggestion = await self._extract_capability_suggestion(user_input, context)
            
            if capability_suggestion:
                result = await self.original_system.add_desired_capability(
                    name=capability_suggestion["name"],
                    description=capability_suggestion["description"],
                    category=capability_suggestion["category"],
                    examples=capability_suggestion.get("examples", [])
                )
                
                await self.send_context_update(
                    update_type="desired_capability_recorded",
                    data=result,
                    priority=ContextPriority.NORMAL
                )
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        return {
            "capability_query_detected": self._is_capability_query(user_input),
            "new_capability_suggested": self._suggests_new_capability(user_input),
            "processing_complete": True,
            "cross_module_informed": len(messages) > 0
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze capability requirements and gaps"""
        messages = await self.get_cross_module_messages()
        
        # Analyze capability usage patterns
        usage_patterns = await self._analyze_capability_usage(context, messages)
        
        # Identify emerging capability needs
        emerging_needs = await self._identify_emerging_needs(context, messages)
        
        # Assess capability development opportunities
        development_opportunities = await self._assess_development_opportunities(context)
        
        # Analyze capability dependencies
        dependency_analysis = self._analyze_capability_dependencies()
        
        return {
            "usage_patterns": usage_patterns,
            "emerging_needs": emerging_needs,
            "development_opportunities": development_opportunities,
            "dependency_analysis": dependency_analysis,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize capability insights for response"""
        messages = await self.get_cross_module_messages()
        
        # Generate capability-aware response elements
        synthesis = {
            "capability_disclosure": await self._generate_capability_disclosure(context),
            "confidence_modulation": self._calculate_response_confidence(context, messages),
            "suggested_alternatives": await self._suggest_capability_alternatives(context),
            "growth_acknowledgment": self._acknowledge_capability_growth(context),
            "limitation_transparency": self._express_limitations_appropriately(context)
        }
        
        # Check if we should proactively mention capabilities
        if self._should_mention_capabilities(context, messages):
            synthesis["proactive_capability_mention"] = True
            synthesis["relevant_capabilities"] = await self._get_relevant_capabilities(context)
        
        return synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _check_capability_assessment_needed(self, context: SharedContext) -> bool:
        """Check if capability assessment is needed"""
        assessment_triggers = [
            "can you", "are you able", "do you know how",
            "what can you do", "capabilities", "limitations",
            "possible", "feasible"
        ]
        
        user_input_lower = context.user_input.lower()
        return any(trigger in user_input_lower for trigger in assessment_triggers)
    
    async def _assess_context_capabilities(self, context: SharedContext) -> Dict[str, Any]:
        """Assess capabilities relevant to current context"""
        # Extract implied task from context
        implied_goal = context.user_input
        
        # Use original system's assessment
        assessment = await self.original_system.assess_required_capabilities(implied_goal)
        
        # Enhance with context awareness
        assessment["context_factors"] = {
            "emotional_readiness": self._assess_emotional_readiness(context.emotional_state),
            "goal_alignment": await self._assess_goal_alignment(context.goal_context),
            "memory_support": self._assess_memory_support(context.memory_context)
        }
        
        return assessment
    
    def _get_current_capabilities(self) -> List[Dict[str, Any]]:
        """Get list of current capabilities"""
        capabilities = self.original_system.capability_model.get_implemented_capabilities()
        return [
            {
                "name": cap.name,
                "category": cap.category,
                "confidence": cap.confidence,
                "status": cap.implementation_status
            }
            for cap in capabilities
        ]
    
    def _get_capability_confidence(self) -> Dict[str, float]:
        """Get confidence levels for each capability category"""
        capabilities = self.original_system.capability_model.get_all_capabilities()
        
        category_confidence = {}
        category_counts = {}
        
        for cap in capabilities:
            category = cap.category
            if category not in category_confidence:
                category_confidence[category] = 0.0
                category_counts[category] = 0
            
            category_confidence[category] += cap.confidence
            category_counts[category] += 1
        
        # Calculate averages
        for category in category_confidence:
            if category_counts[category] > 0:
                category_confidence[category] /= category_counts[category]
        
        return category_confidence
    
    async def _assess_goal_capability_requirements(self, goal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess what capabilities are needed for active goals"""
        active_goals = goal_data.get("active_goals", [])
        
        requirements = {
            "required_capabilities": [],
            "available_capabilities": [],
            "gaps": [],
            "confidence_by_goal": {}
        }
        
        for goal in active_goals:
            goal_text = goal.get("description", "")
            assessment = await self.original_system.assess_required_capabilities(goal_text)
            
            goal_id = goal.get("id", "unknown")
            requirements["confidence_by_goal"][goal_id] = assessment["overall_feasibility"]["score"]
            
            # Identify required capabilities
            for cap in assessment.get("relevant_capabilities", []):
                cap_info = cap["capability"]
                if cap_info not in requirements["required_capabilities"]:
                    requirements["required_capabilities"].append(cap_info)
                
                if cap_info["implementation_status"] == "implemented":
                    requirements["available_capabilities"].append(cap_info)
            
            # Identify gaps
            for gap in assessment.get("potential_gaps", []):
                requirements["gaps"].append({
                    "goal_id": goal_id,
                    "capability_gap": gap
                })
        
        return requirements
    
    async def _assess_task_feasibility(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess feasibility of a specific task"""
        task_description = task_data.get("description", "")
        
        # Use original system's assessment
        assessment = await self.original_system.assess_required_capabilities(task_description)
        
        feasibility = assessment["overall_feasibility"]
        
        return {
            "task": task_description,
            "feasible": feasibility["score"] > 0.6,
            "confidence": feasibility["confidence"],
            "assessment": feasibility["assessment"],
            "missing_capabilities": assessment.get("potential_gaps", []),
            "available_capabilities": [
                cap["capability"]["name"] 
                for cap in assessment.get("relevant_capabilities", [])
                if cap["capability"]["implementation_status"] == "implemented"
            ]
        }
    
    async def _update_capability_confidence(self, feedback_data: Dict[str, Any]):
        """Update capability confidence based on performance feedback"""
        capability_name = feedback_data.get("capability")
        performance_score = feedback_data.get("performance", 0.5)
        
        # Find and update capability
        capabilities = self.original_system.capability_model.get_all_capabilities()
        
        for cap in capabilities:
            if cap.name == capability_name:
                # Adjust confidence based on performance
                old_confidence = cap.confidence
                new_confidence = (old_confidence * 0.7) + (performance_score * 0.3)
                
                self.original_system.capability_model.update_capability(
                    cap.id,
                    {"confidence": new_confidence}
                )
                
                logger.info(f"Updated {capability_name} confidence: {old_confidence:.2f} -> {new_confidence:.2f}")
                break
    
    async def _record_capability_gap(self, limitation_data: Dict[str, Any]):
        """Record a newly discovered capability gap"""
        description = limitation_data.get("description", "Unknown limitation")
        context = limitation_data.get("context", "")
        
        # Add as desired capability
        await self.original_system.add_desired_capability(
            name=f"Handle_{limitation_data.get('type', 'unknown')}_limitation",
            description=description,
            category="system",
            examples=[context] if context else []
        )
    
    def _is_capability_query(self, text: str) -> bool:
        """Check if text is asking about capabilities"""
        query_patterns = [
            "can you", "are you able", "do you know",
            "what can you do", "your capabilities", "your limitations",
            "is it possible", "would you be able"
        ]
        
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in query_patterns)
    
    def _determine_query_type(self, text: str) -> str:
        """Determine the type of capability query"""
        text_lower = text.lower()
        
        if "what can you do" in text_lower or "capabilities" in text_lower:
            return "what_can_you_do"
        elif "limitations" in text_lower or "can't" in text_lower:
            return "limitations"
        elif "can you" in text_lower or "are you able" in text_lower:
            return "can_you_do_x"
        else:
            return "general"
    
    def _extract_capability_goal(self, text: str) -> str:
        """Extract the goal/task from a capability query"""
        # Remove the query part to get the actual task
        query_starters = [
            "can you", "are you able to", "do you know how to",
            "is it possible to", "would you be able to"
        ]
        
        text_lower = text.lower()
        for starter in query_starters:
            if starter in text_lower:
                # Extract everything after the query starter
                parts = text_lower.split(starter)
                if len(parts) > 1:
                    return parts[1].strip().rstrip("?")
        
        return text
    
    def _suggests_new_capability(self, text: str) -> bool:
        """Check if user is suggesting we need a new capability"""
        suggestion_patterns = [
            "you should be able",
            "it would be nice if you could",
            "i wish you could",
            "you need to learn",
            "why can't you"
        ]
        
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in suggestion_patterns)
    
    async def _extract_capability_suggestion(self, text: str, context: SharedContext) -> Optional[Dict[str, Any]]:
        """Extract a capability suggestion from user input"""
        # This is simplified - could use NLP for better extraction
        
        # Try to identify the suggested capability
        action_verbs = ["able to", "could", "learn to", "know how to"]
        
        text_lower = text.lower()
        for verb in action_verbs:
            if verb in text_lower:
                parts = text_lower.split(verb)
                if len(parts) > 1:
                    capability_description = parts[1].strip().rstrip(".")
                    
                    return {
                        "name": f"user_suggested_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        "description": f"Ability to {capability_description}",
                        "category": self._categorize_capability(capability_description),
                        "examples": [text]
                    }
        
        return None
    
    def _categorize_capability(self, description: str) -> str:
        """Categorize a capability based on its description"""
        category_keywords = {
            "cognitive": ["think", "reason", "analyze", "understand", "deduce"],
            "creative": ["create", "generate", "write", "compose", "design"],
            "technical": ["code", "program", "implement", "build", "debug"],
            "communication": ["talk", "explain", "describe", "communicate", "express"],
            "social": ["interact", "relate", "empathize", "connect", "collaborate"]
        }
        
        description_lower = description.lower()
        
        for category, keywords in category_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                return category
        
        return "general"
    
    def _assess_emotional_readiness(self, emotional_state: Dict[str, Any]) -> float:
        """Assess emotional readiness for capability execution"""
        if not emotional_state:
            return 0.7  # Neutral readiness
        
        # Positive emotions enhance capability confidence
        positive_emotions = ["Confidence", "Joy", "Curiosity", "Excitement"]
        negative_emotions = ["Anxiety", "Fear", "Frustration", "Confusion"]
        
        emotions = emotional_state.get("emotional_state", {})
        
        positive_score = sum(emotions.get(e, 0) for e in positive_emotions)
        negative_score = sum(emotions.get(e, 0) for e in negative_emotions)
        
        readiness = 0.5 + (positive_score * 0.1) - (negative_score * 0.1)
        return max(0.0, min(1.0, readiness))
    
    async def _assess_goal_alignment(self, goal_context: Dict[str, Any]) -> float:
        """Assess how well capabilities align with current goals"""
        if not goal_context:
            return 0.5
        
        active_goals = goal_context.get("active_goals", [])
        if not active_goals:
            return 0.5
        
        # Check if we have capabilities for active goals
        total_confidence = 0.0
        for goal in active_goals:
            goal_text = goal.get("description", "")
            assessment = await self.original_system.assess_required_capabilities(goal_text)
            total_confidence += assessment["overall_feasibility"]["score"]
        
        return total_confidence / len(active_goals) if active_goals else 0.5
    
    def _assess_memory_support(self, memory_context: Dict[str, Any]) -> float:
        """Assess memory support for capability execution"""
        if not memory_context:
            return 0.5
        
        # More memories generally mean better context for capability execution
        memory_count = memory_context.get("memory_count", 0)
        return min(1.0, 0.5 + (memory_count * 0.05))
    
    async def _analyze_capability_usage(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze how capabilities are being used"""
        usage_stats = {}
        
        # Count capability mentions across messages
        for module, module_messages in messages.items():
            for msg in module_messages:
                if "capability" in msg.get("type", "").lower():
                    cap_name = msg.get("data", {}).get("capability")
                    if cap_name:
                        usage_stats[cap_name] = usage_stats.get(cap_name, 0) + 1
        
        return {
            "most_used": max(usage_stats.items(), key=lambda x: x[1])[0] if usage_stats else None,
            "usage_distribution": usage_stats,
            "total_uses": sum(usage_stats.values())
        }
    
    async def _identify_emerging_needs(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Identify emerging capability needs from context"""
        emerging_needs = []
        
        # Look for system limitation messages
        for module, module_messages in messages.items():
            for msg in module_messages:
                if "limitation" in msg.get("type", "").lower() or "error" in msg.get("type", "").lower():
                    emerging_needs.append({
                        "source": module,
                        "need": msg.get("data", {}).get("description", "Unknown need"),
                        "priority": "high" if "critical" in str(msg).lower() else "medium"
                    })
        
        return emerging_needs[:5]  # Top 5 emerging needs
    
    async def _assess_development_opportunities(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify opportunities for capability development"""
        opportunities = []
        
        # Get low confidence capabilities
        capabilities = self.original_system.capability_model.get_all_capabilities()
        
        for cap in capabilities:
            if cap.confidence < 0.5 and cap.implementation_status == "implemented":
                opportunities.append({
                    "capability": cap.name,
                    "current_confidence": cap.confidence,
                    "improvement_potential": 1.0 - cap.confidence,
                    "category": cap.category
                })
        
        # Sort by improvement potential
        opportunities.sort(key=lambda x: x["improvement_potential"], reverse=True)
        
        return opportunities[:3]  # Top 3 opportunities
    
    def _analyze_capability_dependencies(self) -> Dict[str, Any]:
        """Analyze capability dependency structure"""
        dependency_graph = self.original_system.capability_model.get_capability_dependency_graph()
        
        # Find capabilities with most dependencies
        most_dependent = []
        most_required = []
        
        for cap_id, cap_data in dependency_graph.items():
            if len(cap_data["dependencies"]) > 2:
                most_dependent.append(cap_data["name"])
            if len(cap_data["required_by"]) > 2:
                most_required.append(cap_data["name"])
        
        return {
            "total_capabilities": len(dependency_graph),
            "most_dependent": most_dependent,
            "most_required": most_required,
            "isolated_capabilities": [
                cap_data["name"] for cap_id, cap_data in dependency_graph.items()
                if not cap_data["dependencies"] and not cap_data["required_by"]
            ]
        }
    
    async def _generate_capability_disclosure(self, context: SharedContext) -> str:
        """Generate appropriate capability disclosure for context"""
        if self._is_capability_query(context.user_input):
            return "transparent"  # Full disclosure when asked
        elif context.session_context.get("high_stakes"):
            return "cautious"  # Careful disclosure for important tasks
        else:
            return "natural"  # Natural, conversational disclosure
    
    def _calculate_response_confidence(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate confidence level for response"""
        base_confidence = 0.7
        
        # Adjust based on capability assessment if available
        for assessment in self.recent_assessments:
            if assessment.get("context_id") == id(context):
                feasibility = assessment.get("overall_feasibility", {})
                base_confidence = feasibility.get("confidence", base_confidence)
                break
        
        # Adjust based on emotional readiness
        emotional_readiness = self._assess_emotional_readiness(context.emotional_state)
        
        return (base_confidence * 0.7) + (emotional_readiness * 0.3)
    
    async def _suggest_capability_alternatives(self, context: SharedContext) -> List[str]:
        """Suggest alternative approaches when capabilities are limited"""
        alternatives = []
        
        # If we detected a capability query we can't fully satisfy
        if hasattr(self, '_last_assessment') and self._last_assessment:
            gaps = self._last_assessment.get("potential_gaps", [])
            
            for gap in gaps:
                if gap["suggested_capability"] == "image_generation":
                    alternatives.append("I can describe images in detail instead")
                elif gap["suggested_capability"] == "real_time_data":
                    alternatives.append("I can work with the information you provide")
        
        return alternatives
    
    def _acknowledge_capability_growth(self, context: SharedContext) -> Optional[str]:
        """Acknowledge when capabilities have grown"""
        # Check recent performance feedback
        growth_acknowledgments = []
        
        # This would check for recent capability improvements
        # For now, return None
        return None
    
    def _express_limitations_appropriately(self, context: SharedContext) -> Dict[str, Any]:
        """Determine how to express limitations"""
        return {
            "style": "constructive",  # Focus on what we CAN do
            "suggest_alternatives": True,
            "acknowledge_honestly": True
        }
    
    def _should_mention_capabilities(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Determine if we should proactively mention capabilities"""
        # Mention if user seems unsure what we can do
        uncertainty_indicators = ["what", "how", "can", "possible", "able"]
        uncertainty_score = sum(1 for ind in uncertainty_indicators if ind in context.user_input.lower())
        
        return uncertainty_score >= 2
    
    async def _get_relevant_capabilities(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Get capabilities most relevant to current context"""
        all_capabilities = self.original_system.capability_model.get_implemented_capabilities()
        
        # Simple relevance scoring based on context
        relevant = []
        
        for cap in all_capabilities:
            relevance = 0.0
            
            # Check if capability name/description matches input keywords
            if any(word in cap.name.lower() for word in context.user_input.lower().split()):
                relevance += 0.5
            
            if relevance > 0:
                relevant.append({
                    "name": cap.name,
                    "description": cap.description,
                    "relevance": relevance
                })
        
        # Sort by relevance
        relevant.sort(key=lambda x: x["relevance"], reverse=True)
        
        return relevant[:3]  # Top 3
    
    # Delegate all other methods to the original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
