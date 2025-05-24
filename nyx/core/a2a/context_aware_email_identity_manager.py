# nyx/core/a2a/context_aware_email_identity_manager.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareEmailIdentityManager(ContextAwareModule):
    """
    Advanced Email Identity Management with full context distribution capabilities
    """
    
    def __init__(self, original_email_manager):
        super().__init__("email_identity_manager")
        self.original_manager = original_email_manager
        self.context_subscriptions = [
            "identity_creation_request", "social_signup_request", "privacy_need",
            "anonymity_request", "persona_activation", "identity_link_request",
            "security_alert", "identity_compromise"
        ]
        self.active_identities = {}
        self.identity_usage_history = []
    
    async def on_context_received(self, context: SharedContext):
        """Initialize email identity management for this context"""
        logger.debug(f"EmailIdentityManager received context for user: {context.user_id}")
        
        # Check if context suggests need for identity management
        if await self._requires_identity_management(context):
            await self.send_context_update(
                update_type="email_identity_manager_ready",
                data={
                    "capabilities": [
                        "burner_email_creation",
                        "permanent_email_creation",
                        "identity_linking",
                        "cross_platform_tracking"
                    ],
                    "active_identities": len(self.active_identities),
                    "ready": True
                },
                priority=ContextPriority.NORMAL
            )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        
        if update.update_type == "social_signup_request":
            # Create identity for social platform signup
            signup_data = update.data
            result = await self._create_identity_for_platform(signup_data)
            
            await self.send_context_update(
                update_type="identity_created_for_signup",
                data=result,
                target_modules=[update.source_module],
                scope=ContextScope.TARGETED
            )
        
        elif update.update_type == "privacy_need":
            # Handle privacy-related identity needs
            privacy_data = update.data
            if privacy_data.get("level", "medium") == "high":
                result = await self._create_anonymous_identity(privacy_data)
                
                await self.send_context_update(
                    update_type="anonymous_identity_created",
                    data=result,
                    priority=ContextPriority.HIGH
                )
        
        elif update.update_type == "persona_activation":
            # Link identity to persona
            persona_data = update.data
            await self._link_identity_to_persona(persona_data)
        
        elif update.update_type == "security_alert":
            # Handle potential identity compromise
            security_data = update.data
            await self._handle_security_alert(security_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for identity management needs"""
        user_input = context.user_input
        
        # Check if discussing online identity
        if self._discusses_online_identity(user_input):
            # Analyze identity needs
            identity_needs = await self._analyze_identity_needs(user_input, context)
            
            if identity_needs["needs_new_identity"]:
                # Create appropriate identity
                result = await self.original_manager.create_email_identity(
                    purpose=identity_needs["purpose"]
                )
                
                if result:
                    # Track the identity
                    self.active_identities[result["address"]] = result
                    
                    await self.send_context_update(
                        update_type="new_identity_created",
                        data={
                            "identity": result,
                            "purpose": identity_needs["purpose"],
                            "context_driven": True
                        },
                        priority=ContextPriority.HIGH
                    )
        
        # Check for identity management requests
        if self._is_identity_management_request(user_input):
            management_action = self._determine_management_action(user_input)
            
            if management_action:
                result = await self._execute_management_action(management_action, context)
                
                await self.send_context_update(
                    update_type="identity_management_complete",
                    data=result,
                    priority=ContextPriority.NORMAL
                )
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        return {
            "identity_discussion_detected": self._discusses_online_identity(user_input),
            "management_request_detected": self._is_identity_management_request(user_input),
            "processing_complete": True,
            "cross_module_informed": len(messages) > 0
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze identity usage patterns and risks"""
        messages = await self.get_cross_module_messages()
        
        # Analyze identity usage patterns
        usage_patterns = self._analyze_usage_patterns()
        
        # Assess identity risks
        risk_assessment = await self._assess_identity_risks()
        
        # Analyze cross-platform linkages
        linkage_analysis = self._analyze_identity_linkages()
        
        # Identify optimization opportunities
        optimization_opportunities = await self._identify_optimization_opportunities(context, messages)
        
        return {
            "usage_patterns": usage_patterns,
            "risk_assessment": risk_assessment,
            "linkage_analysis": linkage_analysis,
            "optimization_opportunities": optimization_opportunities,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize identity management insights"""
        messages = await self.get_cross_module_messages()
        
        # Generate synthesis
        synthesis = {
            "identity_recommendations": await self._generate_identity_recommendations(context),
            "security_suggestions": await self._generate_security_suggestions(),
            "privacy_best_practices": self._compile_privacy_practices(),
            "identity_portfolio_summary": self._summarize_identity_portfolio()
        }
        
        # Check if we should suggest identity actions
        if self._should_suggest_identity_actions(context, messages):
            synthesis["suggested_actions"] = await self._suggest_identity_actions(context)
            synthesis["suggest_identity_management"] = True
        
        return synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _requires_identity_management(self, context: SharedContext) -> bool:
        """Check if context requires identity management"""
        # Check for identity-related keywords
        identity_keywords = [
            "email", "sign up", "register", "account", "identity",
            "anonymous", "privacy", "burner", "temporary"
        ]
        
        user_input_lower = context.user_input.lower()
        if any(keyword in user_input_lower for keyword in identity_keywords):
            return True
        
        # Check goals for social interaction needs
        if context.goal_context:
            goals = context.goal_context.get("active_goals", [])
            social_goals = [g for g in goals if "social" in g.get("description", "").lower()]
            if social_goals:
                return True
        
        return False
    
    async def _create_identity_for_platform(self, signup_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create identity for specific platform signup"""
        platform = signup_data.get("platform", "unknown")
        persona = signup_data.get("persona", "default")
        
        # Create identity
        identity = await self.original_manager.create_email_identity(
            purpose=f"{platform}_signup"
        )
        
        if identity:
            # Link to platform
            await self.original_manager.log_cross_identity_link(
                email=identity["address"],
                platform=platform,
                username=f"{persona}_{datetime.now().strftime('%Y%m%d')}"
            )
            
            # Track usage
            self.identity_usage_history.append({
                "identity": identity["address"],
                "platform": platform,
                "persona": persona,
                "created_at": datetime.now().isoformat(),
                "purpose": "platform_signup"
            })
            
            return {
                "success": True,
                "identity": identity,
                "platform": platform,
                "linked": True
            }
        
        return {
            "success": False,
            "error": "Failed to create identity"
        }
    
    async def _create_anonymous_identity(self, privacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create highly anonymous identity"""
        # Use burner service for high privacy
        from nyx.tools.email_identity_manager import EMAIL_SERVICES
        
        burner_services = [s for s in EMAIL_SERVICES if s["type"] == "burner"]
        if not burner_services:
            return {"success": False, "error": "No burner services available"}
        
        # Create anonymous identity
        identity = await self.original_manager.create_email_identity(
            purpose="anonymous_activity"
        )
        
        if identity:
            # Add extra privacy metadata
            identity["privacy_level"] = "high"
            identity["expiration"] = "24_hours"
            identity["no_link_policy"] = True
            
            self.active_identities[identity["address"]] = identity
            
            return {
                "success": True,
                "identity": identity,
                "privacy_features": ["burner", "no_tracking", "auto_expire"]
            }
        
        return {"success": False, "error": "Failed to create anonymous identity"}
    
    async def _link_identity_to_persona(self, persona_data: Dict[str, Any]):
        """Link identity to active persona"""
        persona = persona_data.get("persona_name")
        
        # Find suitable identity for persona
        suitable_identity = None
        for address, identity in self.active_identities.items():
            if identity.get("purpose") == "account_registration" and not identity.get("linked_persona"):
                suitable_identity = identity
                break
        
        if suitable_identity:
            suitable_identity["linked_persona"] = persona
            
            await self.send_context_update(
                update_type="identity_persona_linked",
                data={
                    "identity": suitable_identity["address"],
                    "persona": persona,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    async def _handle_security_alert(self, security_data: Dict[str, Any]):
        """Handle security alerts for identities"""
        threat_level = security_data.get("threat_level", "medium")
        affected_identity = security_data.get("identity")
        
        if threat_level == "high" and affected_identity in self.active_identities:
            # Mark identity as compromised
            self.active_identities[affected_identity]["compromised"] = True
            self.active_identities[affected_identity]["compromised_at"] = datetime.now().isoformat()
            
            # Create replacement identity
            replacement = await self.original_manager.create_email_identity(
                purpose="replacement_secure"
            )
            
            await self.send_context_update(
                update_type="identity_security_response",
                data={
                    "compromised_identity": affected_identity,
                    "action_taken": "identity_replaced",
                    "new_identity": replacement["address"] if replacement else None,
                    "timestamp": datetime.now().isoformat()
                },
                priority=ContextPriority.CRITICAL
            )
    
    def _discusses_online_identity(self, text: str) -> bool:
        """Check if text discusses online identity"""
        identity_topics = [
            "email", "account", "sign up", "register", "identity",
            "username", "profile", "anonymous", "privacy"
        ]
        
        text_lower = text.lower()
        return any(topic in text_lower for topic in identity_topics)
    
    async def _analyze_identity_needs(self, text: str, context: SharedContext) -> Dict[str, Any]:
        """Analyze identity needs from text and context"""
        text_lower = text.lower()
        
        needs = {
            "needs_new_identity": False,
            "purpose": "general",
            "privacy_level": "medium",
            "duration": "permanent"
        }
        
        # Determine if new identity needed
        if any(phrase in text_lower for phrase in ["need email", "create account", "sign up"]):
            needs["needs_new_identity"] = True
        
        # Determine purpose
        if "reddit" in text_lower or "twitter" in text_lower:
            needs["purpose"] = "social_media_signup"
        elif "temporary" in text_lower or "burner" in text_lower:
            needs["purpose"] = "temporary_use"
            needs["duration"] = "temporary"
        elif "anonymous" in text_lower:
            needs["purpose"] = "anonymous_activity"
            needs["privacy_level"] = "high"
        
        # Consider emotional context
        if context.emotional_state:
            emotions = context.emotional_state.get("emotional_state", {})
            if emotions.get("Paranoia", 0) > 0.5 or emotions.get("Fear", 0) > 0.5:
                needs["privacy_level"] = "high"
        
        return needs
    
    def _is_identity_management_request(self, text: str) -> bool:
        """Check if text is requesting identity management"""
        management_phrases = [
            "list my emails", "show identities", "manage accounts",
            "check email", "which email", "delete identity"
        ]
        
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in management_phrases)
    
    def _determine_management_action(self, text: str) -> Optional[Dict[str, Any]]:
        """Determine what management action is requested"""
        text_lower = text.lower()
        
        if "list" in text_lower or "show" in text_lower:
            return {"action": "list", "target": "all"}
        elif "delete" in text_lower or "remove" in text_lower:
            return {"action": "delete", "target": "specified"}
        elif "check" in text_lower:
            return {"action": "status", "target": "all"}
        
        return None
    
    async def _execute_management_action(self, action: Dict[str, Any], context: SharedContext) -> Dict[str, Any]:
        """Execute identity management action"""
        action_type = action.get("action")
        
        if action_type == "list":
            return {
                "action": "list",
                "identities": [
                    {
                        "address": addr,
                        "purpose": ident.get("purpose"),
                        "created_at": ident.get("created_at"),
                        "active": not ident.get("compromised", False)
                    }
                    for addr, ident in self.active_identities.items()
                ],
                "total": len(self.active_identities)
            }
        
        elif action_type == "status":
            active_count = sum(1 for i in self.active_identities.values() if not i.get("compromised", False))
            compromised_count = sum(1 for i in self.active_identities.values() if i.get("compromised", False))
            
            return {
                "action": "status",
                "total_identities": len(self.active_identities),
                "active": active_count,
                "compromised": compromised_count,
                "recent_usage": len([u for u in self.identity_usage_history if 
                                   datetime.fromisoformat(u["created_at"]) > 
                                   datetime.now().replace(microsecond=0) - datetime.timedelta(days=7)])
            }
        
        return {"action": action_type, "status": "completed"}
    
    def _analyze_usage_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in identity usage"""
        if not self.identity_usage_history:
            return {"patterns_found": False}
        
        patterns = {
            "total_uses": len(self.identity_usage_history),
            "platform_distribution": {},
            "purpose_distribution": {},
            "average_identity_lifetime": 0
        }
        
        # Analyze platform distribution
        for usage in self.identity_usage_history:
            platform = usage.get("platform", "unknown")
            patterns["platform_distribution"][platform] = patterns["platform_distribution"].get(platform, 0) + 1
            
            purpose = usage.get("purpose", "unknown")
            patterns["purpose_distribution"][purpose] = patterns["purpose_distribution"].get(purpose, 0) + 1
        
        # Calculate average lifetime (simplified)
        if self.active_identities:
            active_times = []
            for identity in self.active_identities.values():
                if "created_at" in identity:
                    created = datetime.fromisoformat(identity["created_at"])
                    lifetime = (datetime.now() - created).days
                    active_times.append(lifetime)
            
            if active_times:
                patterns["average_identity_lifetime"] = sum(active_times) / len(active_times)
        
        return patterns
    
    async def _assess_identity_risks(self) -> Dict[str, Any]:
        """Assess risks associated with identities"""
        risks = {
            "risk_level": "low",
            "compromised_identities": 0,
            "overused_identities": [],
            "linkage_risks": []
        }
        
        # Count compromised
        risks["compromised_identities"] = sum(1 for i in self.active_identities.values() if i.get("compromised", False))
        
        # Check for overuse
        usage_counts = {}
        for usage in self.identity_usage_history:
            identity = usage.get("identity")
            usage_counts[identity] = usage_counts.get(identity, 0) + 1
        
        for identity, count in usage_counts.items():
            if count > 5:  # Arbitrary threshold
                risks["overused_identities"].append({
                    "identity": identity,
                    "usage_count": count,
                    "risk": "high_exposure"
                })
        
        # Assess overall risk
        if risks["compromised_identities"] > 0:
            risks["risk_level"] = "high"
        elif len(risks["overused_identities"]) > 2:
            risks["risk_level"] = "medium"
        
        return risks
    
    def _analyze_identity_linkages(self) -> Dict[str, Any]:
        """Analyze linkages between identities"""
        linkages = {
            "total_links": 0,
            "platform_links": {},
            "persona_links": {}
        }
        
        # Count platform links
        for usage in self.identity_usage_history:
            platform = usage.get("platform", "unknown")
            if platform not in linkages["platform_links"]:
                linkages["platform_links"][platform] = []
            
            linkages["platform_links"][platform].append(usage.get("identity"))
            linkages["total_links"] += 1
        
        # Count persona links
        for identity in self.active_identities.values():
            persona = identity.get("linked_persona")
            if persona:
                if persona not in linkages["persona_links"]:
                    linkages["persona_links"][persona] = []
                linkages["persona_links"][persona].append(identity.get("address"))
        
        return linkages
    
    async def _identify_optimization_opportunities(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Identify opportunities to optimize identity management"""
        opportunities = []
        
        # Check for redundant identities
        purpose_counts = {}
        for identity in self.active_identities.values():
            purpose = identity.get("purpose", "unknown")
            purpose_counts[purpose] = purpose_counts.get(purpose, 0) + 1
        
        for purpose, count in purpose_counts.items():
            if count > 3:
                opportunities.append({
                    "type": "consolidation",
                    "description": f"Multiple identities for {purpose} could be consolidated",
                    "benefit": "Reduced management overhead"
                })
        
        # Check for expired burner potential
        old_identities = []
        for address, identity in self.active_identities.items():
            if "created_at" in identity:
                created = datetime.fromisoformat(identity["created_at"])
                age_days = (datetime.now() - created).days
                if age_days > 30 and identity.get("purpose") == "temporary_use":
                    old_identities.append(address)
        
        if old_identities:
            opportunities.append({
                "type": "cleanup",
                "description": f"{len(old_identities)} temporary identities could be retired",
                "benefit": "Improved security through rotation"
            })
        
        return opportunities
    
    async def _generate_identity_recommendations(self, context: SharedContext) -> List[str]:
        """Generate identity management recommendations"""
        recommendations = []
        
        # Based on usage patterns
        patterns = self._analyze_usage_patterns()
        
        if patterns.get("average_identity_lifetime", 0) > 90:
            recommendations.append("Consider rotating long-lived identities for better security")
        
        # Based on risk assessment
        risks = await self._assess_identity_risks()
        
        if risks["risk_level"] == "high":
            recommendations.append("High risk detected - review and replace compromised identities")
        
        if risks["overused_identities"]:
            recommendations.append("Distribute usage across multiple identities to reduce exposure")
        
        # Based on context
        if context.goal_context:
            goals = context.goal_context.get("active_goals", [])
            if any("privacy" in g.get("description", "").lower() for g in goals):
                recommendations.append("Use burner identities for privacy-sensitive activities")
        
        return recommendations
    
    async def _generate_security_suggestions(self) -> List[str]:
        """Generate security suggestions"""
        suggestions = []
        
        # Check for basic security practices
        if len(self.active_identities) < 3:
            suggestions.append("Maintain multiple identities to compartmentalize activities")
        
        # Check age of identities
        old_count = 0
        for identity in self.active_identities.values():
            if "created_at" in identity:
                created = datetime.fromisoformat(identity["created_at"])
                if (datetime.now() - created).days > 180:
                    old_count += 1
        
        if old_count > 0:
            suggestions.append(f"Rotate {old_count} identities older than 6 months")
        
        return suggestions
    
    def _compile_privacy_practices(self) -> List[str]:
        """Compile privacy best practices"""
        return [
            "Use unique identities for different platforms",
            "Rotate identities regularly to prevent tracking",
            "Use burner emails for one-time signups",
            "Avoid linking identities to real personal information",
            "Monitor for identity compromise indicators"
        ]
    
    def _summarize_identity_portfolio(self) -> Dict[str, Any]:
        """Summarize current identity portfolio"""
        summary = {
            "total_identities": len(self.active_identities),
            "active": sum(1 for i in self.active_identities.values() if not i.get("compromised", False)),
            "by_purpose": {},
            "by_privacy_level": {}
        }
        
        # Group by purpose
        for identity in self.active_identities.values():
            purpose = identity.get("purpose", "unknown")
            summary["by_purpose"][purpose] = summary["by_purpose"].get(purpose, 0) + 1
            
            privacy = identity.get("privacy_level", "medium")
            summary["by_privacy_level"][privacy] = summary["by_privacy_level"].get(privacy, 0) + 1
        
        return summary
    
    def _should_suggest_identity_actions(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Determine if we should suggest identity actions"""
        # Suggest if discussing accounts or privacy
        if self._discusses_online_identity(context.user_input):
            return True
        
        # Suggest if social browsing module is active
        social_messages = messages.get("social_browsing", [])
        if any(msg.get("type") == "social_signup_needed" for msg in social_messages):
            return True
        
        return False
    
    async def _suggest_identity_actions(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Suggest identity-related actions"""
        suggestions = []
        
        # Check if user needs identity for goal
        if context.goal_context:
            goals = context.goal_context.get("active_goals", [])
            
            for goal in goals:
                desc = goal.get("description", "").lower()
                
                if "social" in desc or "connect" in desc:
                    suggestions.append({
                        "action": "create_social_identity",
                        "reason": "Support social interaction goals",
                        "privacy_level": "medium"
                    })
                
                elif "research" in desc or "anonymous" in desc:
                    suggestions.append({
                        "action": "create_anonymous_identity",
                        "reason": "Enable anonymous research",
                        "privacy_level": "high"
                    })
        
        return suggestions[:2]  # Top 2 suggestions
    
    # Delegate all other methods to the original manager
    def __getattr__(self, name):
        """Delegate any missing methods to the original manager"""
        return getattr(self.original_manager, name)
