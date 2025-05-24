# nyx/core/a2a/context_aware_computer_use_agent.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareComputerUseAgent(ContextAwareModule):
    """
    Advanced Computer Use Agent with full context distribution capabilities
    """
    
    def __init__(self, original_computer_use_agent):
        super().__init__("computer_use_agent")
        self.original_agent = original_computer_use_agent
        self.context_subscriptions = [
            "web_task_request", "automation_request", "research_request",
            "visual_interaction_needed", "browser_navigation_request",
            "form_filling_request", "screenshot_analysis_request"
        ]
        self.task_history = []
        self.active_browser_context = None
    
    async def on_context_received(self, context: SharedContext):
        """Initialize computer use agent for this context"""
        logger.debug(f"ComputerUseAgent received context for user: {context.user_id}")
        
        # Check if input suggests computer use
        if await self._requires_computer_interaction(context):
            await self.send_context_update(
                update_type="computer_use_ready",
                data={
                    "capabilities": [
                        "web_browsing", "form_filling", "screenshot_analysis",
                        "click_interaction", "text_input", "navigation"
                    ],
                    "ready": True,
                    "context_aware": True
                },
                priority=ContextPriority.NORMAL
            )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        
        if update.update_type == "web_task_request":
            # Execute web task
            task_data = update.data
            result = await self._execute_web_task(task_data)
            
            await self.send_context_update(
                update_type="web_task_complete",
                data=result,
                target_modules=[update.source_module],
                scope=ContextScope.TARGETED
            )
        
        elif update.update_type == "research_request":
            # Conduct research using browser
            research_data = update.data
            result = await self._conduct_research(research_data)
            
            await self.send_context_update(
                update_type="research_complete",
                data=result,
                priority=ContextPriority.HIGH
            )
        
        elif update.update_type == "screenshot_analysis_request":
            # Analyze current screen
            analysis_data = update.data
            result = await self._analyze_screenshot(analysis_data)
            
            await self.send_context_update(
                update_type="screenshot_analysis_complete",
                data=result,
                target_modules=[update.source_module],
                scope=ContextScope.TARGETED
            )
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for computer use opportunities"""
        user_input = context.user_input
        
        # Detect web-related requests
        web_task = self._extract_web_task(user_input)
        
        if web_task:
            # Execute the task
            result = await self._execute_contextual_web_task(web_task, context)
            
            # Record task in history
            self.task_history.append({
                "task": web_task,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "context_id": id(context)
            })
            
            # Send task completion update
            await self.send_context_update(
                update_type="computer_task_executed",
                data={
                    "task": web_task,
                    "result": result,
                    "success": result.get("success", False)
                },
                priority=ContextPriority.HIGH
            )
        
        # Check if we need to maintain browser context
        if self._should_maintain_browser_context(user_input):
            await self._update_browser_context(context)
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        return {
            "web_task_detected": web_task is not None,
            "task_executed": web_task is not None,
            "browser_context_active": self.active_browser_context is not None,
            "cross_module_informed": len(messages) > 0
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze computer use patterns and opportunities"""
        messages = await self.get_cross_module_messages()
        
        # Analyze task patterns
        task_patterns = self._analyze_task_patterns()
        
        # Identify automation opportunities
        automation_opportunities = await self._identify_automation_opportunities(context, messages)
        
        # Analyze browser interaction efficiency
        efficiency_analysis = self._analyze_interaction_efficiency()
        
        # Assess task success rates
        success_analysis = self._analyze_task_success()
        
        return {
            "task_patterns": task_patterns,
            "automation_opportunities": automation_opportunities,
            "efficiency_analysis": efficiency_analysis,
            "success_analysis": success_analysis,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize computer use insights for response"""
        messages = await self.get_cross_module_messages()
        
        # Generate synthesis
        synthesis = {
            "task_recommendations": await self._generate_task_recommendations(context),
            "automation_suggestions": await self._suggest_automations(context, messages),
            "efficiency_tips": self._generate_efficiency_tips(),
            "browser_state_summary": self._summarize_browser_state()
        }
        
        # Check if we should suggest computer-assisted tasks
        if self._should_suggest_computer_tasks(context, messages):
            synthesis["suggested_tasks"] = await self._suggest_computer_tasks(context)
            synthesis["suggest_automation"] = True
        
        return synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _requires_computer_interaction(self, context: SharedContext) -> bool:
        """Check if context requires computer interaction"""
        interaction_keywords = [
            "browse", "search", "website", "click", "fill", "form",
            "navigate", "open", "check", "look at", "show me"
        ]
        
        user_input_lower = context.user_input.lower()
        return any(keyword in user_input_lower for keyword in interaction_keywords)
    
    async def _execute_web_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a web task with context awareness"""
        url = task_data.get("url", "https://www.google.com")
        prompt = task_data.get("prompt", "")
        
        try:
            # Use original agent to execute task
            result = self.original_agent.run_task(url=url, prompt=prompt)
            
            return {
                "success": True,
                "result": result,
                "task_type": "web_task",
                "url": url,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error executing web task: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_type": "web_task",
                "url": url
            }
    
    async def _conduct_research(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct research using web browsing"""
        topic = research_data.get("topic", "")
        depth = research_data.get("depth", "basic")
        
        # Determine research strategy
        if depth == "comprehensive":
            # Multiple searches and sources
            sources = [
                "https://www.google.com",
                "https://scholar.google.com",
                "https://en.wikipedia.org"
            ]
            prompt_template = "Research comprehensive information about {topic}. Look for authoritative sources."
        else:
            # Basic search
            sources = ["https://www.google.com"]
            prompt_template = "Find basic information about {topic}."
        
        results = []
        for source in sources:
            try:
                result = self.original_agent.run_task(
                    url=source,
                    prompt=prompt_template.format(topic=topic)
                )
                results.append({
                    "source": source,
                    "findings": result,
                    "success": True
                })
            except Exception as e:
                logger.error(f"Research error on {source}: {e}")
                results.append({
                    "source": source,
                    "error": str(e),
                    "success": False
                })
        
        return {
            "topic": topic,
            "depth": depth,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _analyze_screenshot(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze screenshot with context"""
        # This would integrate with the screenshot capability
        # For now, return a placeholder
        return {
            "analysis_type": "screenshot",
            "elements_detected": ["buttons", "text", "images"],
            "context": analysis_data,
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_web_task(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract web task from user input"""
        text_lower = text.lower()
        
        # Check for explicit URLs
        import re
        url_pattern = r'https?://[^\s]+'
        url_match = re.search(url_pattern, text)
        
        if url_match:
            return {
                "type": "navigate",
                "url": url_match.group(0),
                "action": "visit"
            }
        
        # Check for search requests
        if "search for" in text_lower or "google" in text_lower:
            search_terms = text_lower.split("search for")[-1].strip() if "search for" in text_lower else text_lower.split("google")[-1].strip()
            return {
                "type": "search",
                "query": search_terms,
                "engine": "google"
            }
        
        # Check for specific website mentions
        websites = {
            "wikipedia": "https://en.wikipedia.org",
            "youtube": "https://www.youtube.com",
            "reddit": "https://www.reddit.com",
            "twitter": "https://twitter.com"
        }
        
        for site_name, site_url in websites.items():
            if site_name in text_lower:
                return {
                    "type": "navigate",
                    "url": site_url,
                    "site": site_name
                }
        
        return None
    
    async def _execute_contextual_web_task(self, web_task: Dict[str, Any], context: SharedContext) -> Dict[str, Any]:
        """Execute web task with full context awareness"""
        task_type = web_task.get("type")
        
        if task_type == "search":
            query = web_task.get("query", "")
            
            # Enhance query with context if available
            if context.goal_context:
                active_goals = context.goal_context.get("active_goals", [])
                if active_goals:
                    # Add goal context to search
                    goal_keywords = " ".join([g.get("description", "")[:20] for g in active_goals[:2]])
                    enhanced_query = f"{query} {goal_keywords}"
                else:
                    enhanced_query = query
            else:
                enhanced_query = query
            
            return await self._execute_web_task({
                "url": "https://www.google.com",
                "prompt": f"Search for: {enhanced_query}"
            })
        
        elif task_type == "navigate":
            url = web_task.get("url", "")
            return await self._execute_web_task({
                "url": url,
                "prompt": f"Navigate to {url} and describe what you see"
            })
        
        else:
            return {"success": False, "error": f"Unknown task type: {task_type}"}
    
    def _should_maintain_browser_context(self, text: str) -> bool:
        """Check if we should maintain browser context"""
        continuity_phrases = [
            "then", "next", "after that", "continue", "keep going",
            "scroll down", "go back", "click on"
        ]
        
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in continuity_phrases)
    
    async def _update_browser_context(self, context: SharedContext):
        """Update browser context for continuity"""
        self.active_browser_context = {
            "last_url": self.task_history[-1]["task"].get("url") if self.task_history else None,
            "session_start": self.task_history[0]["timestamp"] if self.task_history else datetime.now().isoformat(),
            "task_count": len(self.task_history),
            "context_id": id(context)
        }
    
    def _analyze_task_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in task execution"""
        if not self.task_history:
            return {"patterns_found": False}
        
        patterns = {
            "total_tasks": len(self.task_history),
            "task_types": {},
            "common_sites": {},
            "success_rate": 0.0
        }
        
        success_count = 0
        
        for task_entry in self.task_history:
            task = task_entry["task"]
            result = task_entry["result"]
            
            # Count task types
            task_type = task.get("type", "unknown")
            patterns["task_types"][task_type] = patterns["task_types"].get(task_type, 0) + 1
            
            # Count sites
            if "url" in task:
                # Extract domain
                import re
                domain_match = re.search(r'https?://([^/]+)', task["url"])
                if domain_match:
                    domain = domain_match.group(1)
                    patterns["common_sites"][domain] = patterns["common_sites"].get(domain, 0) + 1
            
            # Count successes
            if result.get("success", False):
                success_count += 1
        
        patterns["success_rate"] = success_count / len(self.task_history) if self.task_history else 0.0
        
        return patterns
    
    async def _identify_automation_opportunities(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Identify opportunities for automation"""
        opportunities = []
        
        # Check for repetitive tasks
        if self.task_history:
            task_counts = {}
            for entry in self.task_history:
                task_key = f"{entry['task'].get('type')}_{entry['task'].get('url', '')}"
                task_counts[task_key] = task_counts.get(task_key, 0) + 1
            
            # Find repeated tasks
            for task_key, count in task_counts.items():
                if count >= 3:
                    opportunities.append({
                        "type": "repetitive_task",
                        "task": task_key,
                        "frequency": count,
                        "suggestion": "Consider automating this frequent task"
                    })
        
        # Check for multi-step processes
        if len(self.task_history) >= 5:
            # Look for sequences
            recent_tasks = self.task_history[-5:]
            if all(t["task"].get("type") == "navigate" for t in recent_tasks):
                opportunities.append({
                    "type": "navigation_sequence",
                    "suggestion": "Multiple navigation steps could be combined into a workflow"
                })
        
        return opportunities[:3]  # Top 3 opportunities
    
    def _analyze_interaction_efficiency(self) -> Dict[str, Any]:
        """Analyze efficiency of browser interactions"""
        if not self.task_history:
            return {"efficiency_score": 0.5}
        
        # Simple efficiency metrics
        total_tasks = len(self.task_history)
        successful_tasks = sum(1 for t in self.task_history if t["result"].get("success", False))
        
        efficiency_score = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        
        # Check for task completion time (would need timing data)
        avg_tasks_per_session = total_tasks  # Simplified
        
        return {
            "efficiency_score": efficiency_score,
            "success_rate": efficiency_score,
            "avg_tasks_per_session": avg_tasks_per_session,
            "optimization_potential": 1.0 - efficiency_score
        }
    
    def _analyze_task_success(self) -> Dict[str, Any]:
        """Analyze task success patterns"""
        if not self.task_history:
            return {"analysis_available": False}
        
        success_by_type = {}
        failure_reasons = []
        
        for entry in self.task_history:
            task_type = entry["task"].get("type", "unknown")
            success = entry["result"].get("success", False)
            
            if task_type not in success_by_type:
                success_by_type[task_type] = {"success": 0, "total": 0}
            
            success_by_type[task_type]["total"] += 1
            if success:
                success_by_type[task_type]["success"] += 1
            else:
                # Record failure reason
                error = entry["result"].get("error", "Unknown error")
                failure_reasons.append({
                    "task_type": task_type,
                    "error": error[:100]  # Truncate long errors
                })
        
        # Calculate success rates by type
        for task_type in success_by_type:
            data = success_by_type[task_type]
            data["success_rate"] = data["success"] / data["total"] if data["total"] > 0 else 0.0
        
        return {
            "success_by_type": success_by_type,
            "failure_reasons": failure_reasons[:5],  # Top 5 failures
            "overall_success_rate": sum(d["success"] for d in success_by_type.values()) / sum(d["total"] for d in success_by_type.values()) if success_by_type else 0.0
        }
    
    async def _generate_task_recommendations(self, context: SharedContext) -> List[str]:
        """Generate recommendations for computer use tasks"""
        recommendations = []
        
        # Based on task patterns
        patterns = self._analyze_task_patterns()
        
        if patterns.get("success_rate", 0) < 0.7:
            recommendations.append("Consider breaking complex tasks into smaller steps")
        
        if patterns.get("task_types", {}).get("search", 0) > 5:
            recommendations.append("Frequent searches detected - consider bookmarking common resources")
        
        # Based on context
        if context.goal_context:
            goals = context.goal_context.get("active_goals", [])
            if any("research" in g.get("description", "").lower() for g in goals):
                recommendations.append("Use advanced search operators for more precise research results")
        
        return recommendations
    
    async def _suggest_automations(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Suggest automation possibilities"""
        suggestions = []
        
        # Check for research patterns
        research_count = sum(1 for t in self.task_history if t["task"].get("type") == "search")
        if research_count > 3:
            suggestions.append({
                "type": "research_automation",
                "description": "Automate research workflows with saved search templates",
                "benefit": "Save time on repetitive searches"
            })
        
        # Check for form-filling patterns
        form_tasks = [t for t in self.task_history if "form" in str(t["task"]).lower()]
        if form_tasks:
            suggestions.append({
                "type": "form_automation",
                "description": "Create form-filling templates for common inputs",
                "benefit": "Reduce manual data entry"
            })
        
        return suggestions
    
    def _generate_efficiency_tips(self) -> List[str]:
        """Generate tips for more efficient computer use"""
        tips = []
        
        efficiency = self._analyze_interaction_efficiency()
        
        if efficiency["efficiency_score"] < 0.8:
            tips.append("Use keyboard shortcuts for faster navigation")
            tips.append("Batch similar tasks together for better efficiency")
        
        if self.task_history and len(self.task_history) > 10:
            tips.append("Consider using browser bookmarks for frequently visited sites")
        
        return tips
    
    def _summarize_browser_state(self) -> Dict[str, Any]:
        """Summarize current browser state"""
        if not self.active_browser_context:
            return {"active": False}
        
        return {
            "active": True,
            "current_session": {
                "start_time": self.active_browser_context.get("session_start"),
                "task_count": self.active_browser_context.get("task_count", 0),
                "last_url": self.active_browser_context.get("last_url")
            }
        }
    
    def _should_suggest_computer_tasks(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Determine if we should suggest computer-assisted tasks"""
        # Suggest if user seems to need information
        info_keywords = ["find", "search", "look up", "check", "what is", "how to"]
        user_input_lower = context.user_input.lower()
        
        return any(keyword in user_input_lower for keyword in info_keywords)
    
    async def _suggest_computer_tasks(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Suggest relevant computer-assisted tasks"""
        suggestions = []
        
        user_input_lower = context.user_input.lower()
        
        # Suggest searches based on keywords
        if "learn" in user_input_lower or "understand" in user_input_lower:
            suggestions.append({
                "task": "research",
                "description": "Research comprehensive information on the topic",
                "url": "https://scholar.google.com"
            })
        
        if "news" in user_input_lower or "latest" in user_input_lower:
            suggestions.append({
                "task": "news_search",
                "description": "Check latest news and updates",
                "url": "https://news.google.com"
            })
        
        if "how to" in user_input_lower:
            suggestions.append({
                "task": "tutorial_search",
                "description": "Find tutorials and guides",
                "url": "https://www.youtube.com"
            })
        
        return suggestions[:2]  # Top 2 suggestions
    
    # Delegate all other methods to the original agent
    def __getattr__(self, name):
        """Delegate any missing methods to the original agent"""
        return getattr(self.original_agent, name)
