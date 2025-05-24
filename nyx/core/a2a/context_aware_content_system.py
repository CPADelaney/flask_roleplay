# nyx/core/a2a/context_aware_content_system.py

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareContentSystem(ContextAwareModule):
    """
    Advanced Content Management System with full context distribution capabilities
    """
    
    def __init__(self, original_content_system):
        super().__init__("content_system")
        self.original_system = original_content_system
        self.context_subscriptions = [
            "creative_content_generated", "content_storage_request", "content_retrieval_request",
            "content_search_request", "emotional_state_update", "memory_creation",
            "goal_completion", "content_review_request"
        ]
        self.content_context = {}
    
    async def on_context_received(self, context: SharedContext):
        """Initialize content processing for this context"""
        logger.debug(f"ContentSystem received context for user: {context.user_id}")
        
        # Check if there's content-related intent
        content_intent = await self._analyze_content_intent(context)
        
        if content_intent:
            # Get recent content for context
            recent_content = await self._get_contextual_recent_content(context)
            
            await self.send_context_update(
                update_type="content_system_ready",
                data={
                    "content_intent": content_intent,
                    "recent_content_summary": recent_content,
                    "available_types": ["story", "poem", "lyrics", "journal", "code", "analysis", "assessment"],
                    "storage_initialized": True
                },
                priority=ContextPriority.NORMAL
            )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        
        if update.update_type == "creative_content_generated":
            # Store content from creative system
            content_data = update.data
            result = await self._store_creative_content(content_data)
            
            await self.send_context_update(
                update_type="content_stored",
                data={
                    "storage_result": result,
                    "content_type": content_data.get("content_type"),
                    "content_id": result.get("id")
                },
                priority=ContextPriority.NORMAL
            )
        
        elif update.update_type == "content_storage_request":
            # Handle explicit storage request
            storage_data = update.data
            result = await self.original_system.store_content(
                content_type=storage_data["content_type"],
                title=storage_data["title"],
                content=storage_data["content"],
                metadata=storage_data.get("metadata")
            )
            
            await self.send_context_update(
                update_type="content_storage_complete",
                data=result,
                target_modules=[update.source_module],
                scope=ContextScope.TARGETED
            )
        
        elif update.update_type == "memory_creation":
            # Potentially store significant memories as journal entries
            memory_data = update.data
            if memory_data.get("significance", 0) > 8:
                await self._create_memory_journal_entry(memory_data)
        
        elif update.update_type == "goal_completion":
            # Document goal completion
            goal_data = update.data
            await self._document_goal_completion(goal_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for content-related operations"""
        user_input = context.user_input
        
        # Check for content retrieval request
        if self._is_retrieval_request(user_input):
            retrieval_params = self._extract_retrieval_params(user_input)
            
            if retrieval_params.get("search_query"):
                # Search content
                results = await self.original_system.search_content(retrieval_params["search_query"])
                
                await self.send_context_update(
                    update_type="content_search_results",
                    data={
                        "query": retrieval_params["search_query"],
                        "results": results,
                        "count": len(results)
                    },
                    priority=ContextPriority.HIGH
                )
            
            elif retrieval_params.get("content_id"):
                # Retrieve specific content
                content = await self.original_system.retrieve_content(retrieval_params["content_id"])
                
                await self.send_context_update(
                    update_type="content_retrieved",
                    data=content,
                    priority=ContextPriority.HIGH
                )
            
            else:
                # List content
                content_list = await self.original_system.list_content(
                    content_type=retrieval_params.get("content_type"),
                    limit=retrieval_params.get("limit", 10)
                )
                
                await self.send_context_update(
                    update_type="content_list_retrieved",
                    data=content_list,
                    priority=ContextPriority.HIGH
                )
        
        # Check for content creation context
        if self._suggests_content_creation(user_input):
            creation_suggestion = await self._analyze_creation_opportunity(user_input, context)
            
            if creation_suggestion:
                await self.send_context_update(
                    update_type="content_creation_suggested",
                    data=creation_suggestion,
                    priority=ContextPriority.NORMAL
                )
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        return {
            "content_operation_detected": self._is_retrieval_request(user_input) or self._suggests_content_creation(user_input),
            "content_processed": True,
            "cross_module_aware": len(messages) > 0
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze content patterns and opportunities"""
        messages = await self.get_cross_module_messages()
        
        # Analyze content creation patterns
        creation_patterns = await self._analyze_creation_patterns(context)
        
        # Identify content gaps
        content_gaps = await self._identify_content_gaps(context, messages)
        
        # Analyze content quality trends
        quality_trends = await self._analyze_quality_trends()
        
        # Assess content coherence with goals
        goal_coherence = await self._assess_goal_content_coherence(context, messages)
        
        return {
            "creation_patterns": creation_patterns,
            "content_gaps": content_gaps,
            "quality_trends": quality_trends,
            "goal_coherence": goal_coherence,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize content insights for response"""
        messages = await self.get_cross_module_messages()
        
        # Generate content-aware synthesis
        synthesis = {
            "content_recommendations": await self._generate_content_recommendations(context, messages),
            "creative_suggestions": await self._suggest_creative_directions(context),
            "content_callbacks": await self._identify_content_callbacks(context),
            "archival_suggestions": await self._suggest_archival_actions(context)
        }
        
        # Check if response should reference stored content
        if self._should_reference_content(context, messages):
            synthesis["content_references"] = await self._get_relevant_content_references(context)
            synthesis["should_cite_content"] = True
        
        return synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _analyze_content_intent(self, context: SharedContext) -> Optional[Dict[str, Any]]:
        """Analyze if there's content-related intent in the context"""
        intent_keywords = {
            "store": ["save", "store", "keep", "record", "archive"],
            "retrieve": ["show", "get", "find", "retrieve", "list", "what have"],
            "create": ["write", "create", "compose", "generate", "make"],
            "analyze": ["analyze", "review", "assess", "evaluate"]
        }
        
        user_input_lower = context.user_input.lower()
        
        for intent_type, keywords in intent_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                return {
                    "type": intent_type,
                    "confidence": 0.8,
                    "detected_keywords": [kw for kw in keywords if kw in user_input_lower]
                }
        
        return None
    
    async def _get_contextual_recent_content(self, context: SharedContext) -> Dict[str, Any]:
        """Get recent content relevant to current context"""
        # Get recent creations
        recent = await self.original_system.get_recent_creations(days=7)
        
        # Filter by relevance to current context
        if context.emotional_state:
            # Prioritize content matching emotional state
            dominant_emotion = max(
                context.emotional_state.get("emotional_state", {}).items(),
                key=lambda x: x[1],
                default=("neutral", 0)
            )[0]
            
            # Filter content that might match the emotion
            relevant_types = {
                "Joy": ["story", "poem"],
                "Sadness": ["journal", "poem"],
                "Curiosity": ["analysis", "code"],
                "Frustration": ["assessment", "analysis"]
            }
            
            preferred_types = relevant_types.get(dominant_emotion, [])
            if preferred_types:
                recent["prioritized_types"] = preferred_types
        
        return recent
    
    async def _store_creative_content(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store content from creative system with enhanced metadata"""
        # Extract content details
        content_type = content_data.get("content_type", "story")
        result = content_data.get("result", {})
        
        # Prepare enhanced metadata
        metadata = {
            "source": "creative_system",
            "inspired_by": content_data.get("inspired_by", []),
            "emotional_context": content_data.get("emotional_context"),
            "goal_context": content_data.get("goal_context"),
            "generation_timestamp": datetime.now().isoformat()
        }
        
        # Determine title and content
        if isinstance(result, dict):
            title = result.get("title", f"Generated {content_type}")
            content = result.get("content", str(result))
            
            # Add any additional metadata from result
            if "filepath" in result:
                metadata["original_filepath"] = result["filepath"]
        else:
            title = f"Generated {content_type}"
            content = str(result)
        
        # Store the content
        return await self.original_system.store_content(
            content_type=content_type,
            title=title,
            content=content,
            metadata=metadata
        )
    
    async def _create_memory_journal_entry(self, memory_data: Dict[str, Any]):
        """Create a journal entry from a significant memory"""
        memory_text = memory_data.get("memory_text", "")
        significance = memory_data.get("significance", 0)
        
        # Create reflective journal entry
        journal_content = f"""# Memory Journal Entry

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Significance**: {significance}/10

## Memory
{memory_text}

## Context
- Memory Type: {memory_data.get("memory_type", "unknown")}
- Emotional Context: {memory_data.get("emotional_context", "neutral")}

## Reflection
This memory holds special significance. It represents a moment of {self._interpret_memory_significance(memory_data)}.
"""
        
        await self.original_system.store_content(
            content_type="journal",
            title=f"Significant Memory: {memory_text[:50]}...",
            content=journal_content,
            metadata={
                "source": "memory_system",
                "memory_id": memory_data.get("memory_id"),
                "auto_generated": True
            }
        )
    
    async def _document_goal_completion(self, goal_data: Dict[str, Any]):
        """Document a completed goal"""
        goal_description = goal_data.get("goal_description", "Unknown goal")
        completion_result = goal_data.get("completion_result", {})
        
        # Create achievement documentation
        doc_content = f"""# Goal Achievement Documentation

**Goal**: {goal_description}
**Completed**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Achievement Details
- Priority: {goal_data.get("priority", "unknown")}
- Time Horizon: {goal_data.get("time_horizon", "unknown")}
- Associated Need: {goal_data.get("associated_need", "none")}

## Completion Context
{self._format_completion_context(completion_result)}

## Lessons Learned
{self._extract_lessons(goal_data, completion_result)}
"""
        
        await self.original_system.store_content(
            content_type="assessment",
            title=f"Goal Completed: {goal_description[:50]}...",
            content=doc_content,
            metadata={
                "source": "goal_system",
                "goal_id": goal_data.get("goal_id"),
                "auto_generated": True
            }
        )
    
    def _is_retrieval_request(self, text: str) -> bool:
        """Check if text is requesting content retrieval"""
        retrieval_patterns = [
            "show me", "get my", "find my", "list my",
            "what have i", "retrieve", "search for",
            "my stories", "my poems", "my content"
        ]
        
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in retrieval_patterns)
    
    def _extract_retrieval_params(self, text: str) -> Dict[str, Any]:
        """Extract parameters for content retrieval"""
        params = {}
        text_lower = text.lower()
        
        # Check for content type
        content_types = ["story", "poem", "lyrics", "journal", "code", "analysis", "assessment"]
        for ctype in content_types:
            if ctype in text_lower:
                params["content_type"] = ctype
                break
        
        # Check for search query
        if "search" in text_lower or "find" in text_lower:
            # Extract search terms (simplified)
            search_keywords = ["about", "containing", "with", "titled"]
            for keyword in search_keywords:
                if keyword in text_lower:
                    parts = text_lower.split(keyword)
                    if len(parts) > 1:
                        params["search_query"] = parts[1].strip()
                        break
        
        # Check for limit
        if "recent" in text_lower:
            params["limit"] = 5
        elif "all" in text_lower:
            params["limit"] = 100
        
        return params
    
    def _suggests_content_creation(self, text: str) -> bool:
        """Check if text suggests content creation opportunity"""
        creation_triggers = [
            "i feel", "i think", "i realized", "i learned",
            "interesting", "amazing", "wonderful", "terrible"
        ]
        
        text_lower = text.lower()
        return any(trigger in text_lower for trigger in creation_triggers)
    
    async def _analyze_creation_opportunity(self, text: str, context: SharedContext) -> Optional[Dict[str, Any]]:
        """Analyze opportunity for content creation"""
        # Determine what type of content would be appropriate
        
        emotional_intensity = 0.0
        if context.emotional_state:
            emotions = context.emotional_state.get("emotional_state", {})
            emotional_intensity = max(emotions.values()) if emotions else 0.0
        
        # High emotion suggests creative writing
        if emotional_intensity > 0.7:
            return {
                "suggested_type": "poem" if emotional_intensity > 0.8 else "journal",
                "reason": "high_emotional_intensity",
                "prompt": f"Express feelings about: {text[:100]}",
                "urgency": emotional_intensity
            }
        
        # Learning/realization suggests analysis
        if any(word in text.lower() for word in ["learned", "realized", "discovered"]):
            return {
                "suggested_type": "analysis",
                "reason": "new_insight",
                "prompt": f"Document insight: {text[:100]}",
                "urgency": 0.6
            }
        
        return None
    
    async def _analyze_creation_patterns(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze patterns in content creation"""
        recent = await self.original_system.get_recent_creations(days=30)
        
        patterns = {
            "most_created_type": None,
            "creation_frequency": 0,
            "preferred_themes": [],
            "creation_times": []
        }
        
        if recent["items"]:
            # Count by type
            type_counts = {}
            for content_type, items in recent["items"].items():
                type_counts[content_type] = len(items)
            
            if type_counts:
                patterns["most_created_type"] = max(type_counts.items(), key=lambda x: x[1])[0]
            
            # Calculate frequency
            total_items = recent["stats"]["total_items"]
            days = recent["timeframe_days"]
            patterns["creation_frequency"] = total_items / days if days > 0 else 0
        
        return patterns
    
    async def _identify_content_gaps(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Identify gaps in content coverage"""
        gaps = []
        
        # Check goal-related gaps
        goal_messages = messages.get("goal_manager", [])
        for msg in goal_messages:
            if msg.get("type") == "goal_context_available":
                active_goals = msg.get("data", {}).get("active_goals", [])
                
                # Check if goals suggest content creation
                for goal in active_goals:
                    if "write" in goal.get("description", "").lower():
                        gaps.append(f"Content for goal: {goal['description']}")
        
        # Check emotional expression gaps
        recent = await self.original_system.get_recent_creations(days=7)
        if recent["stats"]["total_items"] == 0 and context.emotional_state:
            emotions = context.emotional_state.get("emotional_state", {})
            if any(v > 0.7 for v in emotions.values()):
                gaps.append("Emotional expression through creative content")
        
        return gaps[:3]  # Top 3 gaps
    
    async def _analyze_quality_trends(self) -> Dict[str, Any]:
        """Analyze trends in content quality"""
        # This would analyze metadata for quality indicators
        return {
            "trend": "improving",
            "average_length": "increasing",
            "complexity": "moderate",
            "diversity": "high"
        }
    
    async def _assess_goal_content_coherence(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> float:
        """Assess how well content aligns with goals"""
        # Check if created content supports active goals
        coherence_score = 0.7  # Default moderate coherence
        
        goal_messages = messages.get("goal_manager", [])
        if goal_messages:
            # If we have goal context, check alignment
            for msg in goal_messages:
                if msg.get("type") == "goal_context_available":
                    active_goals = msg.get("data", {}).get("active_goals", [])
                    
                    # Check if any goals involve content creation
                    content_goals = [g for g in active_goals if "write" in g.get("description", "").lower() or "create" in g.get("description", "").lower()]
                    
                    if content_goals:
                        # Check recent content
                        recent = await self.original_system.get_recent_creations(days=7)
                        if recent["stats"]["total_items"] > 0:
                            coherence_score = 0.9  # High coherence if creating content for goals
                        else:
                            coherence_score = 0.3  # Low coherence if not creating despite goals
        
        return coherence_score
    
    async def _generate_content_recommendations(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Generate content recommendations based on context"""
        recommendations = []
        
        # Emotional state recommendations
        if context.emotional_state:
            dominant_emotion = max(
                context.emotional_state.get("emotional_state", {}).items(),
                key=lambda x: x[1],
                default=("neutral", 0)
            )[0]
            
            emotion_content_map = {
                "Joy": "Write a celebratory story or uplifting poem",
                "Sadness": "Create a reflective journal entry",
                "Anger": "Channel energy into dramatic narrative",
                "Curiosity": "Document your exploration in an analysis",
                "Love": "Compose romantic poetry or lyrics"
            }
            
            if dominant_emotion in emotion_content_map:
                recommendations.append(emotion_content_map[dominant_emotion])
        
        # Goal-based recommendations
        content_gaps = await self._identify_content_gaps(context, messages)
        recommendations.extend(content_gaps)
        
        return recommendations[:3]  # Top 3 recommendations
    
    async def _suggest_creative_directions(self, context: SharedContext) -> List[str]:
        """Suggest creative directions for content"""
        suggestions = []
        
        # Recent patterns
        patterns = await self._analyze_creation_patterns(context)
        
        if patterns["most_created_type"] == "story":
            suggestions.append("Try experimenting with poetry for varied expression")
        elif patterns["most_created_type"] == "analysis":
            suggestions.append("Balance analytical content with creative writing")
        
        # Time-based suggestions
        if patterns["creation_frequency"] < 0.5:  # Less than one item every 2 days
            suggestions.append("Consider daily journaling for consistent creation")
        
        return suggestions
    
    async def _identify_content_callbacks(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify content that could be referenced or continued"""
        # Get recent content that might be relevant
        recent = await self.original_system.get_recent_creations(days=30)
        
        callbacks = []
        
        # Look for unfinished series or themes
        for content_type, items in recent["items"].items():
            for item in items[:5]:  # Check recent 5 of each type
                # Check metadata for series or continuation potential
                metadata = item.get("metadata", {})
                if "series" in metadata or "part" in str(metadata):
                    callbacks.append({
                        "content_id": item["id"],
                        "title": item["title"],
                        "type": "series_continuation"
                    })
        
        return callbacks[:2]  # Top 2 callbacks
    
    async def _suggest_archival_actions(self, context: SharedContext) -> List[str]:
        """Suggest archival or organizational actions"""
        suggestions = []
        
        # Get content stats
        all_content = await self.original_system.list_content(limit=1000)
        total_items = all_content["total"]
        
        if total_items > 50:
            suggestions.append("Consider organizing content into collections")
        
        if total_items > 100:
            suggestions.append("Archive older content to maintain performance")
        
        # Check for content without proper metadata
        items_without_tags = sum(
            1 for item in all_content["items"]
            if not item.get("metadata", {}).get("tags")
        )
        
        if items_without_tags > 10:
            suggestions.append("Add tags to content for better organization")
        
        return suggestions
    
    def _should_reference_content(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Determine if response should reference stored content"""
        # Reference if user asked about content
        if self._is_retrieval_request(context.user_input):
            return True
        
        # Reference if discussing topics with existing content
        content_keywords = ["story", "poem", "wrote", "created", "remember when"]
        return any(keyword in context.user_input.lower() for keyword in content_keywords)
    
    async def _get_relevant_content_references(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Get content references relevant to current context"""
        # Search for relevant content
        search_terms = context.user_input.split()[:5]  # First 5 words
        search_query = " ".join(search_terms)
        
        results = await self.original_system.search_content(search_query)
        
        # Format as references
        references = []
        for item in results[:3]:  # Top 3 results
            references.append({
                "content_id": item["id"],
                "title": item["title"],
                "type": item["type"],
                "created_at": item["created_at"]
            })
        
        return references
    
    def _interpret_memory_significance(self, memory_data: Dict[str, Any]) -> str:
        """Interpret the significance of a memory"""
        significance = memory_data.get("significance", 5)
        memory_type = memory_data.get("memory_type", "experience")
        
        if significance >= 9:
            return "profound realization and transformation"
        elif significance >= 7:
            return f"important {memory_type} that shaped my understanding"
        elif significance >= 5:
            return f"meaningful {memory_type} worth preserving"
        else:
            return "notable moment in my experience"
    
    def _format_completion_context(self, completion_result: Dict[str, Any]) -> str:
        """Format goal completion context"""
        if not completion_result:
            return "No specific completion details available."
        
        context_parts = []
        
        if "duration" in completion_result:
            context_parts.append(f"Duration: {completion_result['duration']}")
        
        if "difficulty" in completion_result:
            context_parts.append(f"Difficulty: {completion_result['difficulty']}")
        
        if "satisfaction_level" in completion_result:
            context_parts.append(f"Satisfaction: {completion_result['satisfaction_level']}")
        
        return "\n".join(context_parts) if context_parts else "Standard completion."
    
    def _extract_lessons(self, goal_data: Dict[str, Any], completion_result: Dict[str, Any]) -> str:
        """Extract lessons learned from goal completion"""
        lessons = []
        
        # Extract based on goal type
        if "learning" in goal_data.get("description", "").lower():
            lessons.append("Knowledge acquisition successful")
        
        if "relationship" in goal_data.get("description", "").lower():
            lessons.append("Interpersonal dynamics navigated effectively")
        
        # Extract based on completion quality
        if completion_result.get("satisfaction_level", 0) > 0.8:
            lessons.append("Approach was highly effective")
        elif completion_result.get("satisfaction_level", 0) < 0.5:
            lessons.append("Alternative strategies may be needed for similar goals")
        
        return "\n".join(f"- {lesson}" for lesson in lessons) if lessons else "- Goal completed as expected"
    
    # Delegate all other methods to the original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
