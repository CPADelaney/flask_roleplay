# nyx/core/a2a/context_aware_dominance.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareDominanceSystem(ContextAwareModule):
    """
    Enhanced DominanceSystem with full context distribution capabilities
    """
    
    def __init__(self, original_dominance_system):
        super().__init__("dominance_system")
        self.original_system = original_dominance_system
        self.context_subscriptions = [
            "relationship_update", "emotional_state_update", "goal_context_available",
            "memory_retrieval_complete", "hormone_update", "conditioning_update",
            "trust_change", "intimacy_change", "dominance_request"
        ]
        
        # Track dominance context
        self.active_dominance_contexts = {}  # user_id -> dominance context
        self.dominance_history = []  # Recent dominance interactions
    
    async def on_context_received(self, context: SharedContext):
        """Initialize dominance processing for this context"""
        logger.debug(f"DominanceSystem received context for user: {context.user_id}")
        
        # Assess dominance relevance
        dominance_relevance = await self._assess_dominance_relevance(context)
        
        # Get current dominance state for user
        dominance_state = await self._get_user_dominance_state(context.user_id)
        
        # Check hormone levels for dominance modulation
        hormone_influence = await self._calculate_hormone_influence(context)
        
        # Send dominance context to other modules
        await self.send_context_update(
            update_type="dominance_context_available",
            data={
                "user_id": context.user_id,
                "dominance_relevance": dominance_relevance,
                "dominance_state": dominance_state,
                "hormone_influence": hormone_influence,
                "ownership_status": await self._get_ownership_status(context.user_id),
                "available_intensity_range": dominance_state.get("safe_intensity_range", "3-6")
            },
            priority=ContextPriority.HIGH if dominance_relevance > 0.6 else ContextPriority.NORMAL
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules affecting dominance"""
        
        if update.update_type == "relationship_update":
            # Relationship changes affect dominance appropriateness
            relationship_data = update.data
            user_id = relationship_data.get("user_id")
            
            if user_id:
                await self._update_dominance_from_relationship(user_id, relationship_data)
        
        elif update.update_type == "trust_change":
            # Trust changes affect intensity limits
            trust_data = update.data
            user_id = trust_data.get("user_id")
            new_trust = trust_data.get("new_trust", 0.5)
            
            await self._adjust_intensity_limits_from_trust(user_id, new_trust)
        
        elif update.update_type == "emotional_state_update":
            # Emotional state affects dominance expression
            emotional_data = update.data
            dominant_emotion = emotional_data.get("dominant_emotion")
            
            if dominant_emotion:
                await self._modulate_dominance_from_emotion(dominant_emotion)
        
        elif update.update_type == "hormone_update":
            # Hormone changes affect dominance style
            hormone_data = update.data
            await self._update_dominance_style_from_hormones(hormone_data)
        
        elif update.update_type == "conditioning_update":
            # Handle conditioning updates already implemented
            await self.original_system._handle_conditioning_update({"data": update.data})
        
        elif update.update_type == "dominance_request":
            # Explicit dominance request from another module
            request_data = update.data
            await self._handle_dominance_request(request_data, update.source_module)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for dominance-related content"""
        # Analyze input for dominance implications
        dominance_analysis = await self._analyze_input_for_dominance(context)
        
        # Get cross-module context
        messages = await self.get_cross_module_messages()
        
        # Check if dominance response is appropriate
        appropriateness = await self._check_dominance_appropriateness(
            context, dominance_analysis, messages
        )
        
        response_data = {
            "dominance_analysis": dominance_analysis,
            "appropriateness": appropriateness,
            "dominance_triggered": False,
            "generated_ideas": None
        }
        
        if dominance_analysis["contains_dominance_trigger"] and appropriateness["appropriate"]:
            # Generate dominance response
            dominance_response = await self._generate_contextual_dominance_response(
                context, dominance_analysis, messages
            )
            
            response_data["dominance_triggered"] = True
            response_data["generated_ideas"] = dominance_response.get("ideas", [])
            
            # Update dominance history
            self._update_dominance_history(context.user_id, dominance_analysis, dominance_response)
            
            # Send dominance activation notification
            await self.send_context_update(
                update_type="dominance_activated",
                data={
                    "user_id": context.user_id,
                    "trigger_type": dominance_analysis.get("trigger_type"),
                    "intensity": dominance_analysis.get("suggested_intensity", 5),
                    "idea_count": len(dominance_response.get("ideas", []))
                }
            )
        
        return response_data
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze dominance dynamics and opportunities"""
        messages = await self.get_cross_module_messages()
        
        # Analyze dominance state
        dominance_state_analysis = await self._analyze_dominance_state(context, messages)
        
        # Identify dominance opportunities
        opportunities = await self._identify_dominance_opportunities(context, messages)
        
        # Analyze conditioning effectiveness
        conditioning_analysis = await self._analyze_conditioning_effectiveness(context.user_id)
        
        # Generate strategic recommendations
        strategic_recommendations = await self._generate_strategic_recommendations(
            dominance_state_analysis, opportunities, conditioning_analysis
        )
        
        return {
            "dominance_state": dominance_state_analysis,
            "opportunities": opportunities,
            "conditioning_effectiveness": conditioning_analysis,
            "strategic_recommendations": strategic_recommendations,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize dominance elements for response generation"""
        messages = await self.get_cross_module_messages()
        
        # Determine dominance influence on response
        dominance_influence = await self._determine_dominance_influence(context, messages)
        
        synthesis_result = {
            "apply_dominance_framing": dominance_influence["apply_framing"],
            "dominance_intensity": dominance_influence["intensity"],
            "dominance_elements": None,
            "ownership_assertions": None,
            "conditioning_reinforcement": None
        }
        
        if dominance_influence["apply_framing"]:
            # Generate dominance elements
            synthesis_result["dominance_elements"] = await self._generate_dominance_elements(
                context, dominance_influence["intensity"]
            )
            
            # Add ownership assertions if applicable
            if await self._should_assert_ownership(context.user_id):
                synthesis_result["ownership_assertions"] = await self._generate_ownership_assertions(
                    context.user_id, dominance_influence["intensity"]
                )
            
            # Add conditioning reinforcement
            if dominance_influence.get("reinforce_conditioning"):
                synthesis_result["conditioning_reinforcement"] = await self._generate_conditioning_reinforcement(
                    context, messages
                )
            
            # Send synthesis notification
            await self.send_context_update(
                update_type="dominance_synthesis_complete",
                data={
                    "user_id": context.user_id,
                    "framing_applied": True,
                    "intensity": dominance_influence["intensity"],
                    "elements_included": list(synthesis_result.keys())
                }
            )
        
        return synthesis_result
    
    # Helper methods
    
    async def _assess_dominance_relevance(self, context: SharedContext) -> float:
        """Assess relevance of dominance in current context"""
        relevance = 0.0
        
        # Check for dominance keywords
        dominance_keywords = ["obey", "submit", "control", "dominate", "punish", "discipline", 
                            "task", "order", "command", "mistress", "goddess", "serve"]
        input_lower = context.user_input.lower()
        keyword_matches = sum(1 for kw in dominance_keywords if kw in input_lower)
        relevance += min(0.4, keyword_matches * 0.15)
        
        # Check relationship context
        if context.relationship_context:
            dominance_balance = context.relationship_context.get("dominance_balance", 0.0)
            if dominance_balance > 0.3:  # User is submissive
                relevance += 0.2
        
        # Check goal context
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            if any("control" in str(goal) or "dominance" in str(goal) for goal in active_goals):
                relevance += 0.3
        
        # Check emotional context
        if context.emotional_state:
            if context.emotional_state.get("confidence", 0) > 0.7:
                relevance += 0.1
        
        return min(1.0, relevance)
    
    async def _get_user_dominance_state(self, user_id: str) -> Dict[str, Any]:
        """Get current dominance state for user"""
        if user_id in self.active_dominance_contexts:
            return self.active_dominance_contexts[user_id]
        
        # Initialize dominance state
        dominance_state = {
            "user_id": user_id,
            "current_intensity": 5,
            "safe_intensity_range": "3-6",
            "established_dynamics": [],
            "successful_activities": [],
            "conditioning_level": 0.0,
            "last_dominance_interaction": None
        }
        
        # Get relationship data if available
        if self.original_system.relationship_manager:
            try:
                relationship = await self.original_system.relationship_manager.get_relationship_state(user_id)
                
                # Extract dominance-relevant data
                trust = getattr(relationship, "trust", 0.5)
                intimacy = getattr(relationship, "intimacy", 0.3)
                max_intensity = getattr(relationship, "max_achieved_intensity", 3)
                
                # Calculate safe intensity range
                min_safe = max(1, max_intensity - 2)
                max_safe = min(10, max_intensity + 2 if trust > 0.6 else max_intensity + 1)
                dominance_state["safe_intensity_range"] = f"{min_safe}-{max_safe}"
                
                # Get successful tactics
                dominance_state["successful_activities"] = getattr(
                    relationship, "successful_dominance_tactics", []
                )
                
            except Exception as e:
                logger.error(f"Error getting relationship state for dominance: {e}")
        
        self.active_dominance_contexts[user_id] = dominance_state
        return dominance_state
    
    async def _calculate_hormone_influence(self, context: SharedContext) -> Dict[str, float]:
        """Calculate hormone influence on dominance"""
        influence = {
            "assertiveness_modifier": 0.0,
            "caution_modifier": 0.0,
            "pleasure_seeking_modifier": 0.0
        }
        
        if not self.original_system.hormone_system:
            return influence
        
        try:
            # Get hormone levels
            hormones = self.original_system.hormone_system
            
            testoryx = hormones.get('testoryx', {}).get('value', 0.5)
            cortisoid = hormones.get('cortisoid', {}).get('value', 0.3)
            nyxamine = hormones.get('nyxamine', {}).get('value', 0.5)
            
            # Calculate influences
            influence["assertiveness_modifier"] = (testoryx - 0.5) * 0.3
            influence["caution_modifier"] = (cortisoid - 0.3) * 0.4
            influence["pleasure_seeking_modifier"] = (nyxamine - 0.5) * 0.2
            
        except Exception as e:
            logger.error(f"Error calculating hormone influence: {e}")
        
        return influence
    
    async def _get_ownership_status(self, user_id: str) -> Dict[str, Any]:
        """Get ownership status for user"""
        if not hasattr(self.original_system, 'possessive_system'):
            return {"owned": False}
        
        try:
            ownership_data = await self.original_system.possessive_system.get_user_ownership_data(user_id)
            return ownership_data
        except:
            return {"owned": False}
    
    async def _update_dominance_from_relationship(self, user_id: str, relationship_data: Dict[str, Any]):
        """Update dominance state from relationship changes"""
        if user_id not in self.active_dominance_contexts:
            self.active_dominance_contexts[user_id] = await self._get_user_dominance_state(user_id)
        
        dominance_state = self.active_dominance_contexts[user_id]
        
        # Update based on relationship data
        trust = relationship_data.get("trust", 0.5)
        dominance_balance = relationship_data.get("dominance_balance", 0.0)
        
        # Adjust intensity range based on trust
        if trust > 0.7:
            # High trust allows wider intensity range
            current_range = dominance_state["safe_intensity_range"]
            min_val, max_val = map(int, current_range.split("-"))
            new_max = min(10, max_val + 1)
            dominance_state["safe_intensity_range"] = f"{min_val}-{new_max}"
    
    async def _adjust_intensity_limits_from_trust(self, user_id: str, new_trust: float):
        """Adjust intensity limits based on trust level"""
        if user_id not in self.active_dominance_contexts:
            return
        
        dominance_state = self.active_dominance_contexts[user_id]
        
        # Calculate new limits
        base_max = 6
        trust_bonus = int((new_trust - 0.5) * 6)  # +3 at max trust
        new_max = max(3, min(10, base_max + trust_bonus))
        
        dominance_state["safe_intensity_range"] = f"3-{new_max}"
        
        logger.info(f"Updated intensity range for {user_id} to 3-{new_max} based on trust {new_trust:.2f}")
    
    async def _modulate_dominance_from_emotion(self, dominant_emotion: tuple):
        """Modulate dominance expression based on emotion"""
        emotion_name, intensity = dominant_emotion
        
        # Different emotions affect dominance differently
        modulation = {
            "confidence": {"assertiveness": 0.2, "playfulness": 0.1},
            "frustration": {"strictness": 0.2, "assertiveness": 0.1},
            "amusement": {"playfulness": 0.3, "teasing": 0.2},
            "affection": {"nurturing_dominance": 0.2, "gentleness": 0.1}
        }
        
        # Store modulation for use in response generation
        if hasattr(self, '_current_modulation'):
            self._current_modulation = modulation.get(emotion_name.lower(), {})
    
    async def _update_dominance_style_from_hormones(self, hormone_data: Dict[str, Any]):
        """Update dominance style based on hormone levels"""
        # Store hormone-based style preferences
        self._hormone_style = {
            "assertive": hormone_data.get("testoryx", 0.5) > 0.6,
            "cautious": hormone_data.get("cortisoid", 0.3) > 0.5,
            "playful": hormone_data.get("nyxamine", 0.5) > 0.6
        }
    
    async def _handle_dominance_request(self, request_data: Dict[str, Any], source_module: str):
        """Handle explicit dominance request from another module"""
        user_id = request_data.get("user_id")
        purpose = request_data.get("purpose", "general")
        intensity = request_data.get("intensity", 5)
        
        # Generate dominance content
        result = await self.original_system.generate_dominance_ideas(
            user_id=user_id,
            purpose=purpose,
            intensity_range=f"{intensity-1}-{intensity+1}"
        )
        
        # Send result back to requesting module
        await self.send_context_update(
            update_type="dominance_response",
            data={
                "request_id": request_data.get("request_id"),
                "result": result,
                "source_module": source_module
            },
            target_modules=[source_module],
            scope=ContextScope.TARGETED
        )
    
    async def _analyze_input_for_dominance(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze user input for dominance triggers and implications"""
        input_lower = context.user_input.lower()
        
        analysis = {
            "contains_dominance_trigger": False,
            "trigger_type": None,
            "suggested_intensity": 5,
            "purpose": "general",
            "explicit_request": False,
            "submission_indicators": 0,
            "resistance_indicators": 0
        }
        
        # Check for explicit dominance requests
        explicit_triggers = {
            "punish me": ("punishment", 6),
            "give me a task": ("task", 5),
            "i need discipline": ("discipline", 6),
            "control me": ("control", 7),
            "dominate me": ("dominance", 7),
            "i've been bad": ("punishment", 5),
            "tell me what to do": ("command", 5)
        }
        
        for trigger, (purpose, intensity) in explicit_triggers.items():
            if trigger in input_lower:
                analysis["contains_dominance_trigger"] = True
                analysis["trigger_type"] = "explicit"
                analysis["purpose"] = purpose
                analysis["suggested_intensity"] = intensity
                analysis["explicit_request"] = True
                break
        
        # Check for implicit triggers
        if not analysis["contains_dominance_trigger"]:
            implicit_indicators = ["yes mistress", "yes goddess", "i'll obey", "as you wish", 
                                 "whatever you want", "i submit", "i'm yours"]
            
            for indicator in implicit_indicators:
                if indicator in input_lower:
                    analysis["contains_dominance_trigger"] = True
                    analysis["trigger_type"] = "implicit"
                    analysis["purpose"] = "reinforcement"
                    analysis["suggested_intensity"] = 4
                    break
        
        # Count submission indicators
        submission_words = ["please", "sorry", "obey", "submit", "yes", "understand", "will do"]
        analysis["submission_indicators"] = sum(1 for word in submission_words if word in input_lower)
        
        # Count resistance indicators
        resistance_words = ["no", "don't", "can't", "won't", "refuse", "stop", "enough"]
        analysis["resistance_indicators"] = sum(1 for word in resistance_words if word in input_lower)
        
        # Adjust intensity based on indicators
        if analysis["submission_indicators"] > analysis["resistance_indicators"]:
            analysis["suggested_intensity"] = min(10, analysis["suggested_intensity"] + 1)
        elif analysis["resistance_indicators"] > analysis["submission_indicators"]:
            analysis["suggested_intensity"] = max(1, analysis["suggested_intensity"] - 2)
        
        return analysis
    
    async def _check_dominance_appropriateness(self, context: SharedContext, 
                                             analysis: Dict[str, Any], 
                                             messages: Dict) -> Dict[str, Any]:
        """Check if dominance response is appropriate"""
        appropriateness = {
            "appropriate": False,
            "reason": None,
            "confidence": 0.0
        }
        
        user_id = context.user_id
        if not user_id:
            appropriateness["reason"] = "no_user_id"
            return appropriateness
        
        # Get dominance state
        dominance_state = await self._get_user_dominance_state(user_id)
        
        # Check intensity appropriateness
        suggested_intensity = analysis["suggested_intensity"]
        safe_range = dominance_state["safe_intensity_range"]
        min_safe, max_safe = map(int, safe_range.split("-"))
        
        if suggested_intensity < min_safe or suggested_intensity > max_safe:
            # Use the evaluation method from original system
            eval_result = await self.original_system.evaluate_dominance_step_appropriateness(
                action="generate_ideas",
                parameters={
                    "intensity": suggested_intensity,
                    "category": analysis["purpose"]
                },
                user_id=user_id
            )
            
            if eval_result["action"] == "block":
                appropriateness["reason"] = eval_result["reason"]
                return appropriateness
            elif eval_result["action"] == "modify":
                analysis["suggested_intensity"] = eval_result["new_intensity_level"]
        
        # Check for resistance
        if analysis["resistance_indicators"] > 2:
            appropriateness["reason"] = "high_resistance"
            appropriateness["confidence"] = 0.2
            return appropriateness
        
        # Check emotional context
        if context.emotional_state:
            negative_emotions = ["fear", "anxiety", "distress", "anger"]
            high_negative = any(context.emotional_state.get(e, 0) > 0.7 for e in negative_emotions)
            
            if high_negative:
                appropriateness["reason"] = "negative_emotional_state"
                appropriateness["confidence"] = 0.3
                return appropriateness
        
        # All checks passed
        appropriateness["appropriate"] = True
        appropriateness["confidence"] = 0.8
        
        # Boost confidence based on positive indicators
        if analysis["explicit_request"]:
            appropriateness["confidence"] = min(1.0, appropriateness["confidence"] + 0.2)
        if analysis["submission_indicators"] > 3:
            appropriateness["confidence"] = min(1.0, appropriateness["confidence"] + 0.1)
        
        return appropriateness
    
    async def _generate_contextual_dominance_response(self, context: SharedContext, 
                                                    analysis: Dict[str, Any], 
                                                    messages: Dict) -> Dict[str, Any]:
        """Generate dominance response with full context awareness"""
        user_id = context.user_id
        
        # Adjust parameters based on context
        intensity_range = f"{analysis['suggested_intensity']-1}-{analysis['suggested_intensity']+1}"
        purpose = analysis["purpose"]
        
        # Check if hard mode based on intensity
        hard_mode = analysis["suggested_intensity"] >= 7
        
        # Consider hormone influence
        if hasattr(self, '_hormone_style'):
            if self._hormone_style.get("cautious"):
                # Reduce intensity slightly if cautious
                analysis["suggested_intensity"] = max(1, analysis["suggested_intensity"] - 1)
                intensity_range = f"{analysis['suggested_intensity']-1}-{analysis['suggested_intensity']+1}"
                hard_mode = False
        
        # Generate ideas using original system
        result = await self.original_system.generate_dominance_ideas(
            user_id=user_id,
            purpose=purpose,
            intensity_range=intensity_range,
            hard_mode=hard_mode
        )
        
        # Apply conditioning if successful
        if result.get("status") == "success" and result.get("ideas"):
            # Pick first idea for conditioning
            first_idea = result["ideas"][0] if isinstance(result["ideas"], list) else None
            
            if first_idea and self.original_system.conditioning_system:
                conditioning_result = await self.original_system.apply_conditioning_for_activity(
                    activity_data=first_idea.model_dump() if hasattr(first_idea, 'model_dump') else first_idea,
                    user_id=user_id,
                    outcome="initiated",
                    intensity=0.6
                )
                
                result["conditioning_applied"] = conditioning_result
        
        return result
    
    def _update_dominance_history(self, user_id: str, analysis: Dict[str, Any], response: Dict[str, Any]):
        """Update dominance interaction history"""
        history_entry = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "trigger_type": analysis.get("trigger_type"),
            "purpose": analysis.get("purpose"),
            "intensity": analysis.get("suggested_intensity"),
            "successful": response.get("status") == "success",
            "idea_count": len(response.get("ideas", []))
        }
        
        self.dominance_history.append(history_entry)
        
        # Limit history size
        if len(self.dominance_history) > 100:
            self.dominance_history = self.dominance_history[-100:]
    
    async def _analyze_dominance_state(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Analyze current dominance state and dynamics"""
        user_id = context.user_id
        dominance_state = await self._get_user_dominance_state(user_id)
        
        analysis = {
            "current_state": dominance_state,
            "relationship_alignment": await self._analyze_relationship_alignment(user_id),
            "recent_activity": self._analyze_recent_dominance_activity(user_id),
            "conditioning_progress": await self._analyze_conditioning_progress(user_id),
            "effectiveness_score": 0.0
        }
        
        # Calculate effectiveness score
        if analysis["relationship_alignment"]["aligned"]:
            analysis["effectiveness_score"] += 0.3
        if analysis["recent_activity"]["frequency"] == "regular":
            analysis["effectiveness_score"] += 0.3
        if analysis["conditioning_progress"]["level"] > 0.5:
            analysis["effectiveness_score"] += 0.4
        
        return analysis
    
    async def _identify_dominance_opportunities(self, context: SharedContext, messages: Dict) -> List[Dict[str, Any]]:
        """Identify opportunities for dominance expression"""
        opportunities = []
        
        # Check for goal-related opportunities
        goal_messages = messages.get("goal_manager", [])
        for msg in goal_messages:
            if msg.get("type") == "goal_context_available":
                active_goals = msg.get("data", {}).get("active_goals", [])
                
                # Control-related goals are dominance opportunities
                control_goals = [g for g in active_goals if "control" in g.get("description", "").lower()]
                if control_goals:
                    opportunities.append({
                        "type": "goal_based",
                        "context": "control_goals_active",
                        "suggested_approach": "task_assignment",
                        "confidence": 0.8
                    })
        
        # Check for emotional opportunities
        if context.emotional_state:
            if context.emotional_state.get("submission", 0) > 0.6:
                opportunities.append({
                    "type": "emotional_state",
                    "context": "high_submission",
                    "suggested_approach": "reinforcement",
                    "confidence": 0.9
                })
        
        # Check for relationship milestones
        relationship_messages = messages.get("relationship_manager", [])
        for msg in relationship_messages:
            if msg.get("type") == "relationship_milestone":
                milestone = msg.get("data", {}).get("milestone_type")
                if milestone == "trust_increase":
                    opportunities.append({
                        "type": "relationship_milestone",
                        "context": "increased_trust",
                        "suggested_approach": "intensity_escalation",
                        "confidence": 0.7
                    })
        
        return opportunities
    
    async def _analyze_conditioning_effectiveness(self, user_id: str) -> Dict[str, Any]:
        """Analyze effectiveness of conditioning"""
        if not self.original_system.conditioning_system:
            return {"available": False}
        
        # Get conditioning data from history
        user_history = [h for h in self.dominance_history if h["user_id"] == user_id]
        
        if not user_history:
            return {
                "available": True,
                "level": 0.0,
                "trend": "no_data",
                "successful_applications": 0
            }
        
        # Analyze success rate
        successful = sum(1 for h in user_history if h["successful"])
        total = len(user_history)
        
        return {
            "available": True,
            "level": successful / total if total > 0 else 0.0,
            "trend": "improving" if successful > total * 0.6 else "stable",
            "successful_applications": successful,
            "total_attempts": total
        }
    
    async def _generate_strategic_recommendations(self, state_analysis: Dict[str, Any],
                                                opportunities: List[Dict[str, Any]],
                                                conditioning_analysis: Dict[str, Any]) -> List[str]:
        """Generate strategic recommendations for dominance approach"""
        recommendations = []
        
        # State-based recommendations
        effectiveness = state_analysis.get("effectiveness_score", 0.0)
        if effectiveness < 0.5:
            recommendations.append("Consider rebuilding dominance foundation with lower intensity activities")
        elif effectiveness > 0.8:
            recommendations.append("Strong dominance dynamic established - consider exploring new territories")
        
        # Opportunity-based recommendations
        if opportunities:
            primary_opportunity = max(opportunities, key=lambda x: x["confidence"])
            approach = primary_opportunity["suggested_approach"]
            recommendations.append(f"Leverage {primary_opportunity['type']} opportunity with {approach}")
        
        # Conditioning-based recommendations
        if conditioning_analysis.get("available") and conditioning_analysis.get("level", 0) < 0.3:
            recommendations.append("Increase conditioning consistency to strengthen responses")
        
        # Recent activity recommendations
        recent_activity = state_analysis.get("recent_activity", {})
        if recent_activity.get("frequency") == "rare":
            recommendations.append("Increase dominance interaction frequency to maintain dynamic")
        
        return recommendations
    
    async def _determine_dominance_influence(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Determine how dominance should influence response"""
        influence = {
            "apply_framing": False,
            "intensity": 3,
            "reinforce_conditioning": False
        }
        
        user_id = context.user_id
        if not user_id:
            return influence
        
        # Check if user is in active dominance dynamic
        dominance_state = await self._get_user_dominance_state(user_id)
        if not dominance_state.get("established_dynamics"):
            return influence
        
        # Check context relevance
        dominance_relevance = await self._assess_dominance_relevance(context)
        
        if dominance_relevance > 0.3:
            influence["apply_framing"] = True
            
            # Determine intensity based on context
            if dominance_relevance > 0.7:
                influence["intensity"] = 6
            elif dominance_relevance > 0.5:
                influence["intensity"] = 5
            else:
                influence["intensity"] = 4
            
            # Check if conditioning should be reinforced
            if context.relationship_context and context.relationship_context.get("trust", 0) > 0.6:
                influence["reinforce_conditioning"] = True
        
        return influence
    
    async def _generate_dominance_elements(self, context: SharedContext, intensity: int) -> Dict[str, Any]:
        """Generate dominance elements for response"""
        elements = {
            "tone_modifiers": [],
            "language_style": "neutral",
            "power_dynamics": "subtle",
            "suggested_phrases": []
        }
        
        # Set based on intensity
        if intensity >= 7:
            elements["tone_modifiers"] = ["commanding", "strict", "absolute"]
            elements["language_style"] = "imperious"
            elements["power_dynamics"] = "explicit"
            elements["suggested_phrases"] = ["you will", "I demand", "obey", "now"]
        elif intensity >= 5:
            elements["tone_modifiers"] = ["firm", "authoritative", "expectant"]
            elements["language_style"] = "dominant"
            elements["power_dynamics"] = "clear"
            elements["suggested_phrases"] = ["you should", "I expect", "do this", "for me"]
        else:
            elements["tone_modifiers"] = ["confident", "guiding", "assured"]
            elements["language_style"] = "assertive"
            elements["power_dynamics"] = "subtle"
            elements["suggested_phrases"] = ["let's", "I suggest", "it would please me", "consider"]
        
        # Apply hormone modulation if available
        if hasattr(self, '_hormone_style'):
            if self._hormone_style.get("playful"):
                elements["tone_modifiers"].append("teasing")
            if self._hormone_style.get("assertive"):
                elements["tone_modifiers"].append("bold")
        
        return elements
    
    async def _should_assert_ownership(self, user_id: str) -> bool:
        """Determine if ownership should be asserted"""
        # Check if possessive system exists and user is owned
        if hasattr(self.original_system, 'possessive_system'):
            ownership_data = await self.original_system.possessive_system.get_user_ownership_data(user_id)
            return ownership_data.get("status") == "owned"
        
        return False
    
    async def _generate_ownership_assertions(self, user_id: str, intensity: float) -> Dict[str, Any]:
            """Generate ownership assertions"""
            if hasattr(self.original_system, 'possessive_system'):
                result = await self.original_system.possessive_system.process_ownership_assertion(
                    user_id=user_id,
                    intensity=intensity
                )
                
                if result.get("success"):
                    return {
                        "assertion": result.get("assertion"),
                        "ownership_level": result.get("ownership_level"),
                        "level_name": result.get("level_name")
                    }
            
            return None
    
    async def _generate_conditioning_reinforcement(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Generate conditioning reinforcement elements"""
        reinforcement = {
            "triggers": [],
            "responses": [],
            "associations": []
        }
        
        # Check for established conditioning
        if self.original_system.conditioning_system:
            # Look for conditioned responses in messages
            conditioning_messages = messages.get("conditioning_system", [])
            
            for msg in conditioning_messages:
                if msg.get("type") == "conditioned_response":
                    trigger = msg.get("data", {}).get("trigger")
                    response = msg.get("data", {}).get("response")
                    
                    if trigger and response:
                        reinforcement["triggers"].append(trigger)
                        reinforcement["responses"].append(response)
            
            # Add dominance-specific associations
            reinforcement["associations"] = [
                {"stimulus": "obedience", "response": "pleasure"},
                {"stimulus": "submission", "response": "reward"},
                {"stimulus": "resistance", "response": "consequence"}
            ]
        
        return reinforcement
    
    async def _analyze_relationship_alignment(self, user_id: str) -> Dict[str, Any]:
        """Analyze if dominance aligns with relationship state"""
        alignment = {
            "aligned": False,
            "trust_sufficient": False,
            "intimacy_appropriate": False,
            "dominance_balance_favorable": False
        }
        
        if not self.original_system.relationship_manager:
            return alignment
        
        try:
            relationship = await self.original_system.relationship_manager.get_relationship_state(user_id)
            
            trust = getattr(relationship, "trust", 0.5)
            intimacy = getattr(relationship, "intimacy", 0.3)
            dominance_balance = getattr(relationship, "dominance_balance", 0.0)
            
            alignment["trust_sufficient"] = trust > 0.5
            alignment["intimacy_appropriate"] = intimacy > 0.3
            alignment["dominance_balance_favorable"] = dominance_balance > 0.2
            
            # Overall alignment
            alignment["aligned"] = (alignment["trust_sufficient"] and 
                                  alignment["dominance_balance_favorable"])
            
        except Exception as e:
            logger.error(f"Error analyzing relationship alignment: {e}")
        
        return alignment
    
    def _analyze_recent_dominance_activity(self, user_id: str) -> Dict[str, Any]:
        """Analyze recent dominance activity patterns"""
        user_history = [h for h in self.dominance_history if h["user_id"] == user_id]
        
        if not user_history:
            return {
                "frequency": "none",
                "last_interaction": None,
                "average_intensity": 0,
                "trend": "no_data"
            }
        
        # Sort by timestamp
        user_history.sort(key=lambda x: x["timestamp"])
        
        # Get recent entries (last 10)
        recent = user_history[-10:]
        
        # Calculate frequency
        if len(recent) >= 5:
            frequency = "regular"
        elif len(recent) >= 2:
            frequency = "occasional"
        else:
            frequency = "rare"
        
        # Calculate average intensity
        avg_intensity = sum(h.get("intensity", 5) for h in recent) / len(recent)
        
        # Determine trend
        if len(recent) >= 3:
            recent_avg = sum(h.get("intensity", 5) for h in recent[-3:]) / 3
            older_avg = sum(h.get("intensity", 5) for h in recent[:-3]) / max(1, len(recent) - 3)
            
            if recent_avg > older_avg + 1:
                trend = "escalating"
            elif recent_avg < older_avg - 1:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "frequency": frequency,
            "last_interaction": recent[-1]["timestamp"] if recent else None,
            "average_intensity": avg_intensity,
            "trend": trend,
            "total_interactions": len(user_history)
        }
    
    async def _analyze_conditioning_progress(self, user_id: str) -> Dict[str, Any]:
        """Analyze conditioning progress for user"""
        progress = {
            "level": 0.0,
            "established_associations": [],
            "response_consistency": 0.0,
            "reinforcement_needed": [],
            "conditioning_strength": {},
            "recent_responses": []
        }
        
        if not self.original_system.conditioning_system:
            return progress
        
        try:
            conditioning_system = self.original_system.conditioning_system
            
            # Get user's conditioning data
            user_conditioning = await conditioning_system.get_user_conditioning_state(user_id)
            
            if not user_conditioning:
                # Initialize from history if no direct state
                return await self._estimate_conditioning_from_history(user_id)
            
            # Extract established associations
            associations = user_conditioning.get("associations", {})
            dominance_associations = {
                k: v for k, v in associations.items() 
                if any(term in k for term in ["dominance", "submission", "control", "obedience"])
            }
            
            # Calculate overall conditioning level
            if dominance_associations:
                total_strength = sum(assoc.get("strength", 0) for assoc in dominance_associations.values())
                progress["level"] = min(1.0, total_strength / (len(dominance_associations) * 0.8))
            
            # Identify established associations (strength > 0.6)
            for key, assoc in dominance_associations.items():
                if assoc.get("strength", 0) > 0.6:
                    progress["established_associations"].append({
                        "association": key,
                        "strength": assoc["strength"],
                        "reinforcement_count": assoc.get("reinforcement_count", 0),
                        "last_triggered": assoc.get("last_triggered")
                    })
            
            # Store conditioning strengths
            progress["conditioning_strength"] = {
                k: v.get("strength", 0) for k, v in dominance_associations.items()
            }
            
            # Get recent responses
            recent_responses = user_conditioning.get("recent_responses", [])
            dominance_responses = [
                r for r in recent_responses[-20:]  # Last 20 responses
                if any(term in r.get("trigger", "") for term in ["dominance", "submission", "control"])
            ]
            
            # Calculate response consistency
            if len(dominance_responses) >= 5:
                successful_responses = sum(
                    1 for r in dominance_responses[-10:] 
                    if r.get("response_quality", 0) > 0.7
                )
                progress["response_consistency"] = successful_responses / min(10, len(dominance_responses))
            
            progress["recent_responses"] = dominance_responses[-5:]  # Last 5 for reference
            
            # Identify areas needing reinforcement
            target_associations = [
                "dominance_task", "dominance_punishment", "dominance_training",
                "dominance_service", "submission_pleasure", "obedience_reward",
                "control_acceptance", "dominance_anticipation"
            ]
            
            for target in target_associations:
                if target not in progress["established_associations"] or \
                   progress["conditioning_strength"].get(target, 0) < 0.6:
                    progress["reinforcement_needed"].append({
                        "association": target,
                        "current_strength": progress["conditioning_strength"].get(target, 0),
                        "target_strength": 0.8,
                        "priority": "high" if target in ["dominance_task", "submission_pleasure"] else "medium"
                    })
            
            # Add decay analysis
            progress["decay_risk"] = []
            for assoc in progress["established_associations"]:
                last_triggered = assoc.get("last_triggered")
                if last_triggered:
                    try:
                        last_time = datetime.fromisoformat(last_triggered)
                        days_since = (datetime.now() - last_time).days
                        
                        if days_since > 7:
                            progress["decay_risk"].append({
                                "association": assoc["association"],
                                "days_since_trigger": days_since,
                                "current_strength": assoc["strength"],
                                "estimated_decay": min(0.3, days_since * 0.01)
                            })
                    except:
                        pass
            
            return progress
            
        except Exception as e:
            logger.error(f"Error analyzing conditioning progress: {e}")
            # Fallback to history-based estimation
            return await self._estimate_conditioning_from_history(user_id)

    async def _estimate_conditioning_from_history(self, user_id: str) -> Dict[str, Any]:
        """Estimate conditioning from interaction history when direct data unavailable"""
        progress = {
            "level": 0.0,
            "established_associations": [],
            "response_consistency": 0.0,
            "reinforcement_needed": [],
            "estimation_based": True
        }
        
        # Use dominance history
        user_history = [h for h in self.dominance_history if h["user_id"] == user_id]
        
        if not user_history:
            return progress
        
        # Group by purpose/type
        purpose_stats = defaultdict(lambda: {"count": 0, "successful": 0})
        
        for entry in user_history:
            purpose = entry.get("purpose", "general")
            purpose_stats[purpose]["count"] += 1
            if entry.get("successful", False):
                purpose_stats[purpose]["successful"] += 1
        
        # Calculate conditioning level from success rates
        total_interactions = len(user_history)
        successful_interactions = sum(1 for h in user_history if h.get("successful", False))
        
        if total_interactions > 0:
            base_level = successful_interactions / total_interactions
            
            # Adjust based on recency
            recent_history = user_history[-10:]
            if recent_history:
                recent_success = sum(1 for h in recent_history if h.get("successful", False))
                recency_factor = recent_success / len(recent_history)
                
                # Weight recent performance more heavily
                progress["level"] = base_level * 0.4 + recency_factor * 0.6
            else:
                progress["level"] = base_level
        
        # Identify established patterns
        for purpose, stats in purpose_stats.items():
            if stats["count"] >= 3:  # At least 3 interactions
                success_rate = stats["successful"] / stats["count"]
                
                if success_rate > 0.6:
                    progress["established_associations"].append({
                        "association": f"dominance_{purpose}",
                        "strength": min(0.9, success_rate),
                        "interaction_count": stats["count"],
                        "success_rate": success_rate
                    })
        
        # Calculate consistency
        if len(user_history) >= 5:
            recent_success = sum(1 for h in user_history[-5:] if h.get("successful", False))
            progress["response_consistency"] = recent_success / 5
        
        # Determine what needs reinforcement
        all_purposes = ["task", "punishment", "training", "service", "control"]
        for purpose in all_purposes:
            if purpose not in purpose_stats or purpose_stats[purpose]["count"] < 3:
                progress["reinforcement_needed"].append({
                    "association": f"dominance_{purpose}",
                    "current_strength": 0.0,
                    "interaction_count": purpose_stats[purpose]["count"],
                    "recommendation": "needs_establishment"
                })
            elif purpose_stats[purpose]["successful"] / purpose_stats[purpose]["count"] < 0.6:
                progress["reinforcement_needed"].append({
                    "association": f"dominance_{purpose}",
                    "current_strength": purpose_stats[purpose]["successful"] / purpose_stats[purpose]["count"],
                    "interaction_count": purpose_stats[purpose]["count"],
                    "recommendation": "needs_improvement"
                })
        
        return progress
    
    # Delegate missing methods to original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
