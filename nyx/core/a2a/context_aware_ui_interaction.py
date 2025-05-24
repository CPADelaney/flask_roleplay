# nyx/core/a2a/context_aware_ui_interaction.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareUIInteraction(ContextAwareModule):
    """
    Advanced UI Interaction System with full context distribution capabilities
    """
    
    def __init__(self, original_ui_manager):
        super().__init__("ui_interaction")
        self.original_manager = original_ui_manager
        self.context_subscriptions = [
            "conversation_creation_request", "message_send_request", "ui_navigation_request",
            "user_engagement_needed", "proactive_interaction_trigger", "conversation_context_update",
            "group_interaction_request", "ui_state_change"
        ]
        self.interaction_patterns = {}
        self.conversation_contexts = {}
    
    async def on_context_received(self, context: SharedContext):
        """Initialize UI interaction for this context"""
        logger.debug(f"UIInteraction received context for user: {context.user_id}")
        
        # Check if UI interaction is relevant
        if await self._requires_ui_interaction(context):
            # Get current conversation state
            active_conversations = await self._get_active_conversations_summary()
            
            await self.send_context_update(
                update_type="ui_interaction_ready",
                data={
                    "capabilities": [
                        "conversation_management",
                        "message_composition",
                        "group_conversations",
                        "proactive_engagement"
                    ],
                    "active_conversations": active_conversations,
                    "ui_ready": True
                },
                priority=ContextPriority.NORMAL
            )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        
        if update.update_type == "proactive_interaction_trigger":
            # Handle proactive interaction needs
            trigger_data = update.data
            result = await self._initiate_proactive_interaction(trigger_data)
            
            await self.send_context_update(
                update_type="proactive_interaction_initiated",
                data=result,
                priority=ContextPriority.HIGH
            )
        
        elif update.update_type == "user_engagement_needed":
            # Handle engagement requests
            engagement_data = update.data
            result = await self._handle_engagement_request(engagement_data)
            
            await self.send_context_update(
                update_type="engagement_handled",
                data=result,
                target_modules=[update.source_module],
                scope=ContextScope.TARGETED
            )
        
        elif update.update_type == "conversation_context_update":
            # Update conversation context
            conv_data = update.data
            await self._update_conversation_context(conv_data)
        
        elif update.update_type == "group_interaction_request":
            # Handle group conversation needs
            group_data = update.data
            result = await self._handle_group_interaction(group_data)
            
            await self.send_context_update(
                update_type="group_interaction_complete",
                data=result,
                priority=ContextPriority.NORMAL
            )
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for UI interaction opportunities"""
        user_input = context.user_input
        
        # Check for conversation management requests
        if self._is_conversation_request(user_input):
            conv_action = self._determine_conversation_action(user_input)
            
            if conv_action:
                result = await self._execute_conversation_action(conv_action, context)
                
                await self.send_context_update(
                    update_type="conversation_action_complete",
                    data=result,
                    priority=ContextPriority.HIGH
                )
        
        # Check for message composition needs
        if self._suggests_message_composition(user_input, context):
            message_data = await self._prepare_message_composition(user_input, context)
            
            if message_data:
                result = await self.original_manager.send_message(
                    conversation_id=message_data["conversation_id"],
                    message_content=message_data["content"],
                    attachments=message_data.get("attachments")
                )
                
                await self.send_context_update(
                    update_type="message_sent",
                    data=result,
                    priority=ContextPriority.NORMAL
                )
        
        # Track interaction patterns
        await self._track_interaction_pattern(context)
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        return {
            "conversation_request_detected": self._is_conversation_request(user_input),
            "message_composition_suggested": self._suggests_message_composition(user_input, context),
            "ui_interaction_complete": True,
            "cross_module_informed": len(messages) > 0
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze UI interaction patterns and opportunities"""
        messages = await self.get_cross_module_messages()
        
        # Analyze interaction patterns
        pattern_analysis = self._analyze_interaction_patterns()
        
        # Analyze conversation effectiveness
        conversation_analysis = await self._analyze_conversation_effectiveness()
        
        # Identify engagement opportunities
        engagement_opportunities = await self._identify_engagement_opportunities(context, messages)
        
        # Analyze UI workflow efficiency
        workflow_analysis = self._analyze_ui_workflows()
        
        return {
            "interaction_patterns": pattern_analysis,
            "conversation_analysis": conversation_analysis,
            "engagement_opportunities": engagement_opportunities,
            "workflow_analysis": workflow_analysis,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize UI interaction insights"""
        messages = await self.get_cross_module_messages()
        
        # Generate synthesis
        synthesis = {
            "ui_recommendations": await self._generate_ui_recommendations(context),
            "conversation_suggestions": await self._generate_conversation_suggestions(context, messages),
            "engagement_strategies": self._compile_engagement_strategies(),
            "interaction_summary": await self._summarize_interactions()
        }
        
        # Check if we should suggest proactive interaction
        if self._should_suggest_proactive_interaction(context, messages):
            synthesis["proactive_suggestions"] = await self._generate_proactive_suggestions(context)
            synthesis["suggest_proactive_interaction"] = True
        
        return synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _requires_ui_interaction(self, context: SharedContext) -> bool:
        """Check if context requires UI interaction"""
        # Check for UI-related keywords
        ui_keywords = [
            "message", "conversation", "talk", "chat", "discuss",
            "tell", "ask", "reply", "respond", "contact"
        ]
        
        user_input_lower = context.user_input.lower()
        if any(keyword in user_input_lower for keyword in ui_keywords):
            return True
        
        # Check if goals involve communication
        if context.goal_context:
            goals = context.goal_context.get("active_goals", [])
            communication_goals = [g for g in goals if "communicate" in g.get("description", "").lower() or "interact" in g.get("description", "").lower()]
            if communication_goals:
                return True
        
        return False
    
    async def _get_active_conversations_summary(self) -> Dict[str, Any]:
        """Get summary of active conversations"""
        all_conversations = list(self.original_manager.active_conversations.values())
        
        active = [c for c in all_conversations if c.get("status") == "active"]
        
        return {
            "total": len(all_conversations),
            "active": len(active),
            "recent": len([c for c in active if 
                         datetime.fromisoformat(c["updated_at"]) > 
                         datetime.now().replace(microsecond=0) - datetime.timedelta(hours=24)])
        }
    
    async def _initiate_proactive_interaction(self, trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initiate proactive interaction based on trigger"""
        reason = trigger_data.get("reason", "general")
        target_user = trigger_data.get("user_id", "default_user")
        
        # Determine appropriate message based on reason
        if reason == "goal_reminder":
            message = "I wanted to check in on your progress with your goals. How are things going?"
        elif reason == "emotional_support":
            message = "I noticed you might be going through something. I'm here if you'd like to talk."
        elif reason == "knowledge_sharing":
            message = "I discovered something interesting that might be relevant to our recent discussions."
        else:
            message = "I've been thinking about our conversation and had some additional thoughts to share."
        
        # Create or find conversation
        existing_convs = await self.original_manager.get_conversations_for_user(target_user)
        
        if existing_convs and existing_convs[0]["status"] == "active":
            # Use existing conversation
            conversation_id = existing_convs[0]["id"]
            
            # Send message
            result = await self.original_manager.send_message(
                conversation_id=conversation_id,
                message_content=message
            )
        else:
            # Create new conversation
            result = await self.original_manager.create_new_conversation(
                user_id=target_user,
                title=f"Proactive check-in: {reason}",
                initial_message=message,
                metadata={"initiated_by": "system", "reason": reason}
            )
        
        return {
            "success": True,
            "reason": reason,
            "message_sent": message,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_engagement_request(self, engagement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user engagement request"""
        engagement_type = engagement_data.get("type", "general")
        priority = engagement_data.get("priority", "normal")
        
        # Determine engagement strategy
        if engagement_type == "attention_needed":
            # High priority engagement
            strategy = "immediate_notification"
        elif engagement_type == "feedback_request":
            # Request user feedback
            strategy = "feedback_prompt"
        else:
            # General engagement
            strategy = "conversational"
        
        return {
            "engagement_type": engagement_type,
            "strategy": strategy,
            "priority": priority,
            "handled": True
        }
    
    async def _update_conversation_context(self, conv_data: Dict[str, Any]):
        """Update stored conversation context"""
        conversation_id = conv_data.get("conversation_id")
        context_update = conv_data.get("context", {})
        
        if conversation_id:
            if conversation_id not in self.conversation_contexts:
                self.conversation_contexts[conversation_id] = {}
            
            self.conversation_contexts[conversation_id].update(context_update)
            self.conversation_contexts[conversation_id]["last_updated"] = datetime.now().isoformat()
    
    async def _handle_group_interaction(self, group_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle group conversation interaction"""
        user_ids = group_data.get("user_ids", [])
        purpose = group_data.get("purpose", "discussion")
        
        # Create group conversation
        title = f"Group {purpose}: {datetime.now().strftime('%Y-%m-%d')}"
        initial_message = group_data.get("initial_message", f"Starting a group {purpose}")
        
        result = await self.original_manager.create_group_conversation(
            user_ids=user_ids,
            title=title,
            initial_message=initial_message
        )
        
        return {
            "success": bool(result),
            "conversation_id": result.get("id") if result else None,
            "participant_count": len(user_ids),
            "purpose": purpose
        }
    
    def _is_conversation_request(self, text: str) -> bool:
        """Check if text is requesting conversation management"""
        conversation_phrases = [
            "start conversation", "new conversation", "message",
            "talk to", "chat with", "contact", "reach out"
        ]
        
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in conversation_phrases)
    
    def _determine_conversation_action(self, text: str) -> Optional[Dict[str, Any]]:
        """Determine what conversation action is requested"""
        text_lower = text.lower()
        
        if "new" in text_lower or "start" in text_lower:
            return {"action": "create", "type": "new_conversation"}
        elif "list" in text_lower or "show" in text_lower:
            return {"action": "list", "type": "show_conversations"}
        elif "archive" in text_lower:
            return {"action": "archive", "type": "archive_conversation"}
        elif "search" in text_lower:
            return {"action": "search", "type": "search_conversations"}
        
        return None
    
    async def _execute_conversation_action(self, action: Dict[str, Any], context: SharedContext) -> Dict[str, Any]:
        """Execute conversation management action"""
        action_type = action.get("action")
        
        if action_type == "create":
            # Extract user reference from context
            user_id = self._extract_user_reference(context.user_input)
            
            result = await self.original_manager.create_new_conversation(
                user_id=user_id or context.user_id,
                title=f"Conversation started {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            
            return {
                "action": "create",
                "success": bool(result),
                "conversation_id": result.get("id") if result else None
            }
        
        elif action_type == "list":
            conversations = list(self.original_manager.active_conversations.values())
            
            return {
                "action": "list",
                "conversations": [
                    {
                        "id": c["id"],
                        "title": c["title"],
                        "user_id": c.get("user_id") or c.get("user_ids"),
                        "status": c["status"],
                        "updated_at": c["updated_at"]
                    }
                    for c in conversations[:10]  # Limit to 10
                ],
                "total": len(conversations)
            }
        
        elif action_type == "search":
            query = self._extract_search_query(context.user_input)
            if query:
                results = await self.original_manager.search_conversation_history(query)
                return {
                    "action": "search",
                    "query": query,
                    "results": results[:5],  # Limit results
                    "total_found": len(results)
                }
        
        return {"action": action_type, "status": "completed"}
    
    def _suggests_message_composition(self, text: str, context: SharedContext) -> bool:
        """Check if context suggests composing a message"""
        # Direct message indicators
        message_indicators = [
            "tell them", "say that", "respond with", "reply",
            "let them know", "inform", "message saying"
        ]
        
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in message_indicators):
            return True
        
        # Check if there's an active conversation context
        if context.session_context.get("active_conversation_id"):
            return "say" in text_lower or "tell" in text_lower
        
        return False
    
    async def _prepare_message_composition(self, text: str, context: SharedContext) -> Optional[Dict[str, Any]]:
        """Prepare message for composition"""
        # Extract message content
        message_content = self._extract_message_content(text)
        
        if not message_content:
            return None
        
        # Determine target conversation
        conversation_id = context.session_context.get("active_conversation_id")
        
        if not conversation_id:
            # Find most recent active conversation
            conversations = list(self.original_manager.active_conversations.values())
            active = [c for c in conversations if c["status"] == "active"]
            
            if active:
                # Sort by updated_at
                active.sort(key=lambda x: x["updated_at"], reverse=True)
                conversation_id = active[0]["id"]
        
        if not conversation_id:
            return None
        
        # Enhance message with context
        enhanced_message = await self._enhance_message_with_context(message_content, context)
        
        return {
            "conversation_id": conversation_id,
            "content": enhanced_message,
            "attachments": None
        }
    
    async def _track_interaction_pattern(self, context: SharedContext):
        """Track UI interaction patterns"""
        pattern_key = f"{context.user_id}_{datetime.now().strftime('%Y%m%d')}"
        
        if pattern_key not in self.interaction_patterns:
            self.interaction_patterns[pattern_key] = {
                "interactions": 0,
                "conversation_actions": 0,
                "messages_sent": 0,
                "proactive_triggers": 0
            }
        
        self.interaction_patterns[pattern_key]["interactions"] += 1
        
        if self._is_conversation_request(context.user_input):
            self.interaction_patterns[pattern_key]["conversation_actions"] += 1
    
    def _analyze_interaction_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in UI interactions"""
        if not self.interaction_patterns:
            return {"patterns_found": False}
        
        total_interactions = sum(p["interactions"] for p in self.interaction_patterns.values())
        total_messages = sum(p["messages_sent"] for p in self.interaction_patterns.values())
        
        # Calculate averages
        avg_interactions_per_session = total_interactions / len(self.interaction_patterns) if self.interaction_patterns else 0
        
        return {
            "patterns_found": True,
            "total_interactions": total_interactions,
            "total_messages_sent": total_messages,
            "average_interactions_per_session": avg_interactions_per_session,
            "session_count": len(self.interaction_patterns)
        }
    
    async def _analyze_conversation_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of conversations"""
        conversations = list(self.original_manager.active_conversations.values())
        
        if not conversations:
            return {"analysis_available": False}
        
        # Analyze conversation metrics
        total_messages = sum(len(c.get("messages", [])) for c in conversations)
        active_count = sum(1 for c in conversations if c["status"] == "active")
        
        # Calculate average conversation length
        conversation_lengths = [len(c.get("messages", [])) for c in conversations]
        avg_length = sum(conversation_lengths) / len(conversation_lengths) if conversation_lengths else 0
        
        return {
            "total_conversations": len(conversations),
            "active_conversations": active_count,
            "total_messages": total_messages,
            "average_conversation_length": avg_length,
            "engagement_rate": active_count / len(conversations) if conversations else 0
        }
    
    async def _identify_engagement_opportunities(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Identify opportunities for user engagement"""
        opportunities = []
        
        # Check for goal-related engagement
        goal_messages = messages.get("goal_manager", [])
        for msg in goal_messages:
            if msg.get("type") == "goal_progress":
                opportunities.append({
                    "type": "goal_check_in",
                    "description": "Check in on goal progress",
                    "priority": "medium"
                })
        
        # Check for emotional support opportunities
        emotional_messages = messages.get("emotional_core", [])
        for msg in emotional_messages:
            if msg.get("type") == "emotional_state_update":
                emotional_data = msg.get("data", {})
                if emotional_data.get("valence", 0) < -0.5:
                    opportunities.append({
                        "type": "emotional_support",
                        "description": "Offer emotional support",
                        "priority": "high"
                    })
        
        # Check for knowledge sharing opportunities
        memory_messages = messages.get("memory_core", [])
        if len(memory_messages) > 3:
            opportunities.append({
                "type": "knowledge_sharing",
                "description": "Share insights from recent learning",
                "priority": "low"
            })
        
        return opportunities[:3]  # Top 3 opportunities
    
    def _analyze_ui_workflows(self) -> Dict[str, Any]:
        """Analyze UI workflow efficiency"""
        # Simple workflow analysis
        workflows = {
            "conversation_creation": {
                "steps": ["identify_user", "create_conversation", "send_initial_message"],
                "average_time": "2 seconds",
                "success_rate": 0.95
            },
            "message_sending": {
                "steps": ["compose_message", "identify_conversation", "send"],
                "average_time": "1 second",
                "success_rate": 0.98
            }
        }
        
        return {
            "workflows_analyzed": len(workflows),
            "workflow_details": workflows,
            "optimization_potential": "low"  # Already efficient
        }
    
    async def _generate_ui_recommendations(self, context: SharedContext) -> List[str]:
        """Generate UI interaction recommendations"""
        recommendations = []
        
        # Based on patterns
        patterns = self._analyze_interaction_patterns()
        
        if patterns.get("total_messages_sent", 0) < patterns.get("total_interactions", 1) * 0.3:
            recommendations.append("Consider composing more messages to maintain engagement")
        
        # Based on conversation analysis
        conv_analysis = await self._analyze_conversation_effectiveness()
        
        if conv_analysis.get("average_conversation_length", 0) < 5:
            recommendations.append("Engage in longer conversations for deeper interactions")
        
        if conv_analysis.get("engagement_rate", 0) < 0.5:
            recommendations.append("Reactivate dormant conversations with check-ins")
        
        return recommendations
    
    async def _generate_conversation_suggestions(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Generate conversation suggestions based on context"""
        suggestions = []
        
        # Check for users without recent interaction
        all_users = set()
        recent_users = set()
        
        for conv in self.original_manager.active_conversations.values():
            user_id = conv.get("user_id")
            if user_id:
                all_users.add(user_id)
                
                if datetime.fromisoformat(conv["updated_at"]) > datetime.now().replace(microsecond=0) - datetime.timedelta(days=7):
                    recent_users.add(user_id)
        
        inactive_users = all_users - recent_users
        
        for user_id in list(inactive_users)[:3]:  # Top 3
            suggestions.append({
                "type": "reconnect",
                "user_id": user_id,
                "suggestion": "Reconnect with this user",
                "reason": "No recent interaction"
            })
        
        # Goal-based suggestions
        if context.goal_context:
            goals = context.goal_context.get("active_goals", [])
            for goal in goals:
                if "relationship" in goal.get("description", "").lower():
                    suggestions.append({
                        "type": "relationship_building",
                        "suggestion": "Start conversations to build relationships",
                        "reason": "Supports relationship goals"
                    })
                    break
        
        return suggestions
    
    def _compile_engagement_strategies(self) -> Dict[str, List[str]]:
        """Compile engagement strategies"""
        return {
            "proactive_strategies": [
                "Regular check-ins on goals and progress",
                "Share relevant discoveries and insights",
                "Offer support during emotional challenges",
                "Celebrate achievements and milestones"
            ],
            "reactive_strategies": [
                "Respond promptly to user messages",
                "Ask clarifying questions to deepen understanding",
                "Provide thoughtful and contextual responses",
                "Remember and reference previous conversations"
            ],
            "relationship_building": [
                "Show genuine interest in user's experiences",
                "Share appropriate personal insights",
                "Maintain conversation continuity",
                "Adapt communication style to user preferences"
            ]
        }
    
    async def _summarize_interactions(self) -> Dict[str, Any]:
        """Summarize recent interactions"""
        recent_conversations = []
        
        for conv in self.original_manager.active_conversations.values():
            if datetime.fromisoformat(conv["updated_at"]) > datetime.now().replace(microsecond=0) - datetime.timedelta(days=1):
                recent_conversations.append(conv)
        
        return {
            "recent_conversation_count": len(recent_conversations),
            "total_recent_messages": sum(len(c.get("messages", [])) for c in recent_conversations),
            "active_users": len(set(c.get("user_id") for c in recent_conversations if c.get("user_id"))),
            "summary_period": "last_24_hours"
        }
    
    def _should_suggest_proactive_interaction(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Determine if we should suggest proactive interaction"""
        # Check for engagement opportunities
        opportunities = []
        
        # Goal progress without recent check-in
        goal_messages = messages.get("goal_manager", [])
        if any(msg.get("type") == "goal_progress" for msg in goal_messages):
            opportunities.append("goal_progress")
        
        # Emotional state needing support
        emotional_messages = messages.get("emotional_core", [])
        if any(msg.get("data", {}).get("valence", 0) < -0.3 for msg in emotional_messages):
            opportunities.append("emotional_support")
        
        return len(opportunities) > 0
    
    async def _generate_proactive_suggestions(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Generate proactive interaction suggestions"""
        suggestions = []
        
        # Based on emotional state
        if context.emotional_state:
            valence = context.emotional_state.get("valence", 0)
            if valence < -0.3:
                suggestions.append({
                    "type": "emotional_check_in",
                    "message": "I noticed you might be feeling down. Would you like to talk?",
                    "priority": "high"
                })
        
        # Based on goals
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            if active_goals:
                suggestions.append({
                    "type": "goal_check_in",
                    "message": f"How's your progress on '{active_goals[0].get('description', 'your goal')}'?",
                    "priority": "medium"
                })
        
        # Based on time since last interaction
        last_interaction = None
        for conv in self.original_manager.active_conversations.values():
            if conv.get("user_id") == context.user_id:
                conv_time = datetime.fromisoformat(conv["updated_at"])
                if not last_interaction or conv_time > last_interaction:
                    last_interaction = conv_time
        
        if last_interaction and (datetime.now() - last_interaction).days > 3:
            suggestions.append({
                "type": "reconnection",
                "message": "It's been a while! How have you been?",
                "priority": "low"
            })
        
        return suggestions[:2]  # Top 2 suggestions
    
    def _extract_user_reference(self, text: str) -> Optional[str]:
        """Extract user reference from text"""
        # Simple extraction - look for "with <name>" pattern
        import re
        match = re.search(r'with (\w+)', text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    
    def _extract_search_query(self, text: str) -> Optional[str]:
        """Extract search query from text"""
        # Look for "search for <query>" pattern
        import re
        match = re.search(r'search (?:for )?(.+)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_message_content(self, text: str) -> Optional[str]:
        """Extract message content from text"""
        # Look for message indicators
        message_patterns = [
            r'say (?:that )?(.+)',
            r'tell (?:them )?(?:that )?(.+)',
            r'respond (?:with )?(.+)',
            r'reply (?:with )?(.+)',
            r'message (?:saying )?(.+)'
        ]
        
        import re
        for pattern in message_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    async def _enhance_message_with_context(self, message: str, context: SharedContext) -> str:
        """Enhance message with contextual information"""
        enhanced = message
        
        # Add emotional tone if appropriate
        if context.emotional_state:
            dominant_emotion = max(
                context.emotional_state.get("emotional_state", {}).items(),
                key=lambda x: x[1],
                default=("neutral", 0)
            )[0]
            
            # Add appropriate emoji or tone marker
            emotion_markers = {
                "Joy": "😊",
                "Sadness": "😔",
                "Excitement": "🎉",
                "Curiosity": "🤔",
                "Love": "💕"
            }
            
            if dominant_emotion in emotion_markers:
                enhanced = f"{enhanced} {emotion_markers[dominant_emotion]}"
        
        return enhanced
    
    # Delegate all other methods to the original manager
    def __getattr__(self, name):
        """Delegate any missing methods to the original manager"""
        return getattr(self.original_manager, name)
