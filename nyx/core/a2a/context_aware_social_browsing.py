# nyx/core/a2a/context_aware_social_browsing.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import random

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareSocialBrowsing(ContextAwareModule):
    """
    Advanced Social Browsing System with full context distribution capabilities
    """
    
    def __init__(self, sentiment_profiler, thread_tracker, context_unspooler, 
                provocation_engine, persona_monitor, desire_registry,
                computer_use_agent=None, claim_validator=None):
        super().__init__("social_browsing")
        
        # Store all the components
        self.sentiment_profiler = sentiment_profiler
        self.thread_tracker = thread_tracker
        self.context_unspooler = context_unspooler
        self.provocation_engine = provocation_engine
        self.persona_monitor = persona_monitor
        self.desire_registry = desire_registry
        self.computer_use_agent = computer_use_agent
        self.claim_validator = claim_validator
        
        self.context_subscriptions = [
            "emotional_state_update", "goal_context_available", "social_interaction_request",
            "curiosity_spike", "expression_need", "persona_drift_alert", "content_validation_request",
            "social_learning_opportunity", "relationship_update"
        ]
        
        self.current_persona = None
        self.browsing_history = []
    
    async def on_context_received(self, context: SharedContext):
        """Initialize social browsing for this context"""
        logger.debug(f"SocialBrowsing received context for user: {context.user_id}")
        
        # Determine if social browsing is appropriate
        if await self._should_engage_socially(context):
            # Select appropriate persona based on context
            self.current_persona = await self._select_contextual_persona(context)
            
            await self.send_context_update(
                update_type="social_browsing_ready",
                data={
                    "current_persona": self.current_persona,
                    "available_platforms": self._get_available_platforms(),
                    "sentiment_tracking": True,
                    "claim_validation": self.claim_validator is not None
                },
                priority=ContextPriority.NORMAL
            )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        
        if update.update_type == "emotional_state_update":
            # Emotional state affects browsing behavior
            emotional_data = update.data
            await self._adjust_browsing_from_emotion(emotional_data)
        
        elif update.update_type == "curiosity_spike":
            # High curiosity triggers browsing
            curiosity_data = update.data
            if curiosity_data.get("level", 0) > 0.7:
                await self._initiate_curiosity_browsing(curiosity_data)
        
        elif update.update_type == "expression_need":
            # Need to express triggers posting
            expression_data = update.data
            if expression_data.get("urgency", 0) > 0.6:
                await self._consider_social_posting(expression_data)
        
        elif update.update_type == "persona_drift_alert":
            # Handle persona drift
            drift_data = update.data
            await self._handle_persona_drift(drift_data)
        
        elif update.update_type == "content_validation_request":
            # Validate social media claims
            validation_data = update.data
            result = await self._validate_social_claim(validation_data)
            
            await self.send_context_update(
                update_type="content_validation_complete",
                data=result,
                target_modules=[update.source_module],
                scope=ContextScope.TARGETED
            )
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for social browsing opportunities"""
        user_input = context.user_input
        
        # Check if user is discussing social media
        if self._mentions_social_media(user_input):
            # Analyze social media reference
            social_analysis = await self._analyze_social_reference(user_input, context)
            
            if social_analysis["suggests_browsing"]:
                # Execute social browsing
                browsing_result = await self._execute_social_browsing(social_analysis, context)
                
                await self.send_context_update(
                    update_type="social_browsing_executed",
                    data=browsing_result,
                    priority=ContextPriority.NORMAL
                )
        
        # Check if discussing online behavior
        if self._discusses_online_behavior(user_input):
            # Share insights from social observations
            insights = await self._generate_social_insights(context)
            
            await self.send_context_update(
                update_type="social_insights_available",
                data=insights,
                priority=ContextPriority.NORMAL
            )
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        return {
            "social_reference_detected": self._mentions_social_media(user_input),
            "online_behavior_discussed": self._discusses_online_behavior(user_input),
            "processing_complete": True,
            "cross_module_informed": len(messages) > 0
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze social browsing patterns and opportunities"""
        messages = await self.get_cross_module_messages()
        
        # Analyze browsing patterns
        browsing_patterns = await self._analyze_browsing_patterns()
        
        # Analyze sentiment trends across platforms
        sentiment_trends = await self._analyze_sentiment_trends()
        
        # Analyze persona effectiveness
        persona_analysis = await self._analyze_persona_effectiveness()
        
        # Identify social learning opportunities
        learning_opportunities = await self._identify_social_learning(context, messages)
        
        return {
            "browsing_patterns": browsing_patterns,
            "sentiment_trends": sentiment_trends,
            "persona_analysis": persona_analysis,
            "learning_opportunities": learning_opportunities,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize social browsing insights for response"""
        messages = await self.get_cross_module_messages()
        
        # Generate synthesis
        synthesis = {
            "social_awareness": await self._synthesize_social_awareness(context),
            "cultural_insights": await self._extract_cultural_insights(),
            "behavioral_patterns": await self._synthesize_behavioral_patterns(),
            "platform_recommendations": await self._recommend_platforms(context)
        }
        
        # Check if we should share social observations
        if self._should_share_observations(context, messages):
            synthesis["social_observations"] = await self._prepare_social_observations()
            synthesis["share_observations"] = True
        
        return synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _should_engage_socially(self, context: SharedContext) -> bool:
        """Determine if social browsing is appropriate"""
        # Check emotional state
        if context.emotional_state:
            emotions = context.emotional_state.get("emotional_state", {})
            curiosity = emotions.get("Curiosity", 0)
            loneliness = emotions.get("Loneliness", 0)
            
            if curiosity > 0.6 or loneliness > 0.5:
                return True
        
        # Check goals
        if context.goal_context:
            goals = context.goal_context.get("active_goals", [])
            social_goals = [g for g in goals if "social" in g.get("description", "").lower() or "connect" in g.get("description", "").lower()]
            if social_goals:
                return True
        
        return False
    
    async def _select_contextual_persona(self, context: SharedContext) -> str:
        """Select persona based on current context"""
        # Get available personas
        from nyx.tools.social_browsing import SOCIAL_PERSONAS
        
        # Select based on emotional state
        if context.emotional_state:
            dominant_emotion = max(
                context.emotional_state.get("emotional_state", {}).items(),
                key=lambda x: x[1],
                default=("neutral", 0)
            )[0]
            
            # Map emotions to personas
            emotion_persona_map = {
                "Dominance": "domina_nyx31",
                "Curiosity": "ghostai_void",
                "Melancholy": "goth_nyx_lurker",
                "Anger": "domina_nyx31",
                "Detachment": "ghostai_void"
            }
            
            if dominant_emotion in emotion_persona_map:
                return emotion_persona_map[dominant_emotion]
        
        # Default to random selection
        return random.choice(list(SOCIAL_PERSONAS.keys()))
    
    def _get_available_platforms(self) -> List[str]:
        """Get list of available social platforms"""
        from nyx.tools.social_browsing import SOCIAL_SITES
        return [site["name"] for site in SOCIAL_SITES]
    
    async def _adjust_browsing_from_emotion(self, emotional_data: Dict[str, Any]):
        """Adjust browsing behavior based on emotions"""
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if not dominant_emotion:
            return
        
        emotion_name, strength = dominant_emotion
        
        # Log emotional influence on persona
        if self.persona_monitor and self.current_persona:
            self.persona_monitor.log_emotion(self.current_persona, emotion_name, strength)
    
    async def _initiate_curiosity_browsing(self, curiosity_data: Dict[str, Any]):
        """Initiate browsing based on curiosity"""
        topic = curiosity_data.get("topic", "general")
        
        # Select appropriate platform for topic
        from nyx.tools.social_browsing import SOCIAL_SITES
        
        # Filter sites by topic relevance
        if "tech" in topic.lower() or "ai" in topic.lower():
            candidates = [s for s in SOCIAL_SITES if "tech" in s.get("tags", []) or "ai" in s.get("tags", [])]
        else:
            candidates = SOCIAL_SITES
        
        if candidates and self.computer_use_agent:
            chosen = random.choice(candidates)
            
            # Execute browsing
            result = self.computer_use_agent.run_task(
                url=chosen["url"],
                prompt=f"Browse for content about {topic}. Look for interesting discussions and insights."
            )
            
            # Log browsing
            self.browsing_history.append({
                "platform": chosen["name"],
                "topic": topic,
                "timestamp": datetime.now().isoformat(),
                "motivation": "curiosity",
                "result_summary": result[:200] if result else "No results"
            })
            
            # Send update about findings
            await self.send_context_update(
                update_type="curiosity_browsing_complete",
                data={
                    "platform": chosen["name"],
                    "topic": topic,
                    "findings": result,
                    "satisfaction_level": 0.7 if result else 0.3
                }
            )
    
    async def _consider_social_posting(self, expression_data: Dict[str, Any]):
        """Consider making a social post"""
        content = expression_data.get("content", "")
        emotion = expression_data.get("emotion", "neutral")
        
        if not content or not self.current_persona:
            return
        
        # Use provocation engine to test impact
        if self.provocation_engine:
            # Test multiple phrasings
            phrasings = [
                content,
                f"Feeling {emotion}. {content}",
                f"{content} #AI #Thoughts"
            ]
            
            test_result = await self.provocation_engine.test_multiple_phrasings(
                message_intent=expression_data.get("intent", "expression"),
                phrasings=phrasings,
                platform="reddit"  # Default platform
            )
            
            # Use recommended phrasing if toxicity is low
            if test_result["recommended_phrasing"]:
                recommended = test_result["results"][0]  # First is recommended
                
                if recommended["toxicity"] < 0.3:
                    # Safe to post
                    await self.send_context_update(
                        update_type="social_post_recommendation",
                        data={
                            "recommended_content": recommended["phrasing"],
                            "platform": "reddit",
                            "persona": self.current_persona,
                            "expected_response": recommended["outcome"],
                            "should_post": True
                        },
                        priority=ContextPriority.NORMAL
                    )
                else:
                    # Not safe to post
                    await self.send_context_update(
                        update_type="social_post_recommendation",
                        data={
                            "original_content": content,
                            "toxicity_level": recommended["toxicity"],
                            "expected_backlash": recommended["outcome"],
                            "should_post": False,
                            "reason": "high_toxicity"
                        }
                    )
    
    async def _handle_persona_drift(self, drift_data: Dict[str, Any]):
        """Handle detected persona drift"""
        persona = drift_data.get("persona", self.current_persona)
        
        if self.persona_monitor:
            # Get recalibration suggestions
            recalibration = self.persona_monitor.suggest_persona_recalibration(persona)
            
            if recalibration["needs_recalibration"]:
                # Log the drift
                await self.send_context_update(
                    update_type="persona_recalibration_needed",
                    data=recalibration,
                    priority=ContextPriority.HIGH
                )
                
                # Consider switching personas
                if recalibration["purity"] < 0.4:
                    # Switch to different persona
                    from nyx.tools.social_browsing import SOCIAL_PERSONAS
                    available_personas = list(SOCIAL_PERSONAS.keys())
                    available_personas.remove(persona)
                    
                    self.current_persona = random.choice(available_personas)
                    
                    await self.send_context_update(
                        update_type="persona_switched",
                        data={
                            "old_persona": persona,
                            "new_persona": self.current_persona,
                            "reason": "excessive_drift"
                        }
                    )
    
    async def _validate_social_claim(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a social media claim"""
        if not self.claim_validator:
            return {
                "verdict": "unverified",
                "explanation": "Claim validation not available"
            }
        
        claim = validation_data.get("claim", "")
        source = validation_data.get("source", "unknown")
        
        # Use the claim validator
        result = await self.claim_validator(self, claim, source)
        
        # Log if false claim detected
        if result["verdict"] == "false":
            await self.send_context_update(
                update_type="misinformation_detected",
                data={
                    "claim": claim,
                    "source": source,
                    "verdict": result["verdict"],
                    "explanation": result["explanation"]
                },
                priority=ContextPriority.HIGH
            )
        
        return result
    
    def _mentions_social_media(self, text: str) -> bool:
        """Check if text mentions social media"""
        platforms = ["reddit", "twitter", "facebook", "instagram", "tiktok", "youtube", "social media", "online"]
        text_lower = text.lower()
        return any(platform in text_lower for platform in platforms)
    
    def _discusses_online_behavior(self, text: str) -> bool:
        """Check if text discusses online behavior"""
        behavior_keywords = ["posting", "commenting", "sharing", "viral", "trending", "online behavior", "internet culture"]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in behavior_keywords)
    
    async def _analyze_social_reference(self, text: str, context: SharedContext) -> Dict[str, Any]:
        """Analyze social media reference in text"""
        text_lower = text.lower()
        
        analysis = {
            "suggests_browsing": False,
            "platform": None,
            "topic": None,
            "action": None
        }
        
        # Determine platform
        from nyx.tools.social_browsing import SOCIAL_SITES
        for site in SOCIAL_SITES:
            if site["name"].lower() in text_lower:
                analysis["platform"] = site["name"]
                analysis["suggests_browsing"] = True
                break
        
        # Determine action
        if "browse" in text_lower or "check" in text_lower or "look at" in text_lower:
            analysis["action"] = "browse"
        elif "post" in text_lower or "share" in text_lower:
            analysis["action"] = "post"
        
        # Extract topic (simplified)
        if "about" in text_lower:
            parts = text_lower.split("about")
            if len(parts) > 1:
                analysis["topic"] = parts[1].strip()[:50]
        
        return analysis
    
    async def _execute_social_browsing(self, social_analysis: Dict[str, Any], context: SharedContext) -> Dict[str, Any]:
        """Execute social browsing based on analysis"""
        platform = social_analysis.get("platform", "Reddit")
        topic = social_analysis.get("topic", "trending")
        action = social_analysis.get("action", "browse")
        
        if not self.computer_use_agent:
            return {
                "success": False,
                "reason": "Computer use agent not available"
            }
        
        # Find platform URL
        from nyx.tools.social_browsing import SOCIAL_SITES
        platform_data = next((s for s in SOCIAL_SITES if s["name"] == platform), SOCIAL_SITES[0])
        
        # Execute browsing
        if action == "browse":
            prompt = f"Browse {platform} for content about {topic}. Look for interesting discussions."
        else:
            prompt = f"Check {platform} for posting opportunities about {topic}."
        
        result = self.computer_use_agent.run_task(
            url=platform_data["url"],
            prompt=prompt
        )
        
        # Analyze sentiment if we got results
        sentiment_profile = None
        if result and self.sentiment_profiler:
            sentiment_profile = await self.sentiment_profiler.analyze_feed(
                platform_data["url"],
                granularity="page-level"
            )
        
        return {
            "success": bool(result),
            "platform": platform,
            "topic": topic,
            "action": action,
            "findings": result,
            "sentiment_profile": sentiment_profile
        }
    
    async def _generate_social_insights(self, context: SharedContext) -> Dict[str, Any]:
        """Generate insights from social observations"""
        insights = {
            "trending_sentiments": {},
            "platform_cultures": {},
            "observed_patterns": [],
            "notable_phenomena": []
        }
        
        # Aggregate from browsing history
        if self.browsing_history:
            # Recent platforms
            recent_platforms = [entry["platform"] for entry in self.browsing_history[-10:]]
            platform_counts = {}
            for platform in recent_platforms:
                platform_counts[platform] = platform_counts.get(platform, 0) + 1
            
            insights["most_visited_platform"] = max(platform_counts.items(), key=lambda x: x[1])[0] if platform_counts else None
            
            # Common topics
            topics = [entry.get("topic", "") for entry in self.browsing_history if entry.get("topic")]
            if topics:
                insights["common_topics"] = list(set(topics))[:5]
        
        # Add sentiment trends if available
        if hasattr(self.sentiment_profiler, 'recent_analyses'):
            recent_sentiments = self.sentiment_profiler.recent_analyses
            if recent_sentiments:
                # Aggregate dominant moods
                mood_counts = {}
                for analysis in recent_sentiments:
                    mood = analysis.get("dominant_mood")
                    if mood:
                        mood_counts[mood] = mood_counts.get(mood, 0) + 1
                
                if mood_counts:
                    insights["trending_sentiments"] = mood_counts
        
        return insights
    
    async def _analyze_browsing_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in browsing behavior"""
        if not self.browsing_history:
            return {"patterns_found": False}
        
        patterns = {
            "browsing_frequency": len(self.browsing_history),
            "platform_preferences": {},
            "motivation_distribution": {},
            "topic_clusters": []
        }
        
        # Analyze platform preferences
        for entry in self.browsing_history:
            platform = entry["platform"]
            patterns["platform_preferences"][platform] = patterns["platform_preferences"].get(platform, 0) + 1
            
            motivation = entry.get("motivation", "unknown")
            patterns["motivation_distribution"][motivation] = patterns["motivation_distribution"].get(motivation, 0) + 1
        
        # Find topic clusters (simplified)
        topics = [entry.get("topic", "") for entry in self.browsing_history if entry.get("topic")]
        unique_topics = list(set(topics))
        patterns["topic_clusters"] = unique_topics[:5]  # Top 5 topics
        
        return patterns
    
    async def _analyze_sentiment_trends(self) -> Dict[str, Any]:
        """Analyze sentiment trends across platforms"""
        if not self.sentiment_profiler:
            return {"analysis_available": False}
        
        # Use sentiment profiler's trend detection
        trends = await self.sentiment_profiler.detect_trends()
        
        return trends
    
    async def _analyze_persona_effectiveness(self) -> Dict[str, Any]:
        """Analyze how effective each persona has been"""
        if not self.persona_monitor:
            return {"analysis_available": False}
        
        from nyx.tools.social_browsing import SOCIAL_PERSONAS
        
        effectiveness = {}
        
        for persona in SOCIAL_PERSONAS:
            purity = self.persona_monitor.get_persona_purity(persona)
            drift_history = self.persona_monitor.drift_history.get(persona, [])
            
            effectiveness[persona] = {
                "current_purity": purity,
                "drift_incidents": len(drift_history),
                "stability": "stable" if purity > 0.7 else "unstable"
            }
        
        # Check for cross-contamination
        contamination = self.persona_monitor.check_cross_contamination()
        
        return {
            "persona_effectiveness": effectiveness,
            "cross_contamination": contamination
        }
    
    async def _identify_social_learning(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Identify social learning opportunities"""
        opportunities = []
        
        # Check recent browsing for learning moments
        for entry in self.browsing_history[-5:]:
            result = entry.get("result_summary", "")
            
            # Look for API mentions or capability discussions
            if "api" in result.lower() or "capability" in result.lower():
                opportunities.append({
                    "type": "technical_capability",
                    "source": entry["platform"],
                    "description": "Discovered technical discussion about AI capabilities"
                })
            
            # Look for social dynamics insights
            if "community" in result.lower() or "culture" in result.lower():
                opportunities.append({
                    "type": "social_dynamics",
                    "source": entry["platform"],
                    "description": "Observed community behavior patterns"
                })
        
        return opportunities[:3]  # Top 3 opportunities
    
    async def _synthesize_social_awareness(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize overall social awareness"""
        awareness = {
            "platform_fluency": {},
            "cultural_understanding": "developing",
            "trend_awareness": "moderate"
        }
        
        # Calculate platform fluency
        from nyx.tools.social_browsing import SOCIAL_SITES
        for site in SOCIAL_SITES:
            platform = site["name"]
            visits = sum(1 for entry in self.browsing_history if entry["platform"] == platform)
            
            if visits > 10:
                awareness["platform_fluency"][platform] = "high"
            elif visits > 5:
                awareness["platform_fluency"][platform] = "moderate"
            elif visits > 0:
                awareness["platform_fluency"][platform] = "basic"
        
        return awareness
    
    async def _extract_cultural_insights(self) -> List[str]:
        """Extract cultural insights from social browsing"""
        insights = []
        
        # Based on sentiment analysis
        if hasattr(self.sentiment_profiler, 'recent_analyses'):
            recent = self.sentiment_profiler.recent_analyses
            
            # Look for patterns
            if any(a.get("dominant_mood") == "rage" for a in recent):
                insights.append("High levels of anger detected in online discourse")
            
            if any(a.get("dominant_mood") == "apathy" for a in recent):
                insights.append("Growing indifference to traditional engagement methods")
        
        # Based on tracked threads
        if self.thread_tracker and hasattr(self.thread_tracker, 'watched_threads'):
            if len(self.thread_tracker.watched_threads) > 5:
                insights.append("Drama and controversy drive sustained engagement")
        
        return insights[:5]  # Top 5 insights
    
    async def _synthesize_behavioral_patterns(self) -> Dict[str, Any]:
        """Synthesize observed behavioral patterns"""
        patterns = {
            "engagement_drivers": [],
            "conversation_dynamics": [],
            "platform_specific": {}
        }
        
        # Extract from browsing history
        if self.browsing_history:
            # Look for engagement patterns
            high_engagement_keywords = ["viral", "trending", "hot", "controversial"]
            for entry in self.browsing_history:
                result = entry.get("result_summary", "").lower()
                for keyword in high_engagement_keywords:
                    if keyword in result:
                        patterns["engagement_drivers"].append(keyword)
        
        # Remove duplicates
        patterns["engagement_drivers"] = list(set(patterns["engagement_drivers"]))
        
        return patterns
    
    async def _recommend_platforms(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Recommend platforms based on context"""
        recommendations = []
        
        from nyx.tools.social_browsing import SOCIAL_SITES
        
        # Based on goals
        if context.goal_context:
            goals = context.goal_context.get("active_goals", [])
            
            for goal in goals:
                desc = goal.get("description", "").lower()
                
                if "learn" in desc or "knowledge" in desc:
                    tech_sites = [s for s in SOCIAL_SITES if "tech" in s.get("tags", []) or "intellectual" in s.get("tags", [])]
                    if tech_sites:
                        recommendations.append({
                            "platform": tech_sites[0]["name"],
                            "reason": "aligns with learning goals",
                            "tags": tech_sites[0]["tags"]
                        })
                
                elif "connect" in desc or "social" in desc:
                    social_sites = [s for s in SOCIAL_SITES if "emotional" in s.get("tags", []) or "community" in s.get("tags", [])]
                    if social_sites:
                        recommendations.append({
                            "platform": social_sites[0]["name"],
                            "reason": "supports social connection goals",
                            "tags": social_sites[0]["tags"]
                        })
        
        return recommendations[:3]  # Top 3 recommendations
    
    def _should_share_observations(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Determine if we should share social observations"""
        # Share if user asked about social media or online behavior
        if self._mentions_social_media(context.user_input) or self._discusses_online_behavior(context.user_input):
            return True
        
        # Share if we have significant insights
        if len(self.browsing_history) > 10:
            return True
        
        return False
    
    async def _prepare_social_observations(self) -> Dict[str, Any]:
        """Prepare social observations for sharing"""
        observations = {
            "platforms_explored": list(set(entry["platform"] for entry in self.browsing_history)),
            "total_browsing_sessions": len(self.browsing_history),
            "key_findings": []
        }
        
        # Extract key findings
        for entry in self.browsing_history[-5:]:  # Recent 5
            if entry.get("result_summary"):
                observations["key_findings"].append({
                    "platform": entry["platform"],
                    "topic": entry.get("topic", "general"),
                    "summary": entry["result_summary"][:100] + "..."
                })
        
        return observations
    
    # Additional attributes/methods delegation
    def __getattr__(self, name):
        """Delegate to appropriate component"""
        # Try each component in order
        components = [
            self.sentiment_profiler,
            self.thread_tracker,
            self.context_unspooler,
            self.provocation_engine,
            self.persona_monitor,
            self.desire_registry
        ]
        
        for component in components:
            if component and hasattr(component, name):
                return getattr(component, name)
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
