# nyx/core/a2a/context_aware_cross_user_experience.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareCrossUserExperience(ContextAwareModule):
    """
    Enhanced CrossUserExperienceManager with context distribution capabilities
    """
    
    def __init__(self, original_cross_user_manager):
        super().__init__("cross_user_experience")
        self.original_manager = original_cross_user_manager
        self.context_subscriptions = [
            "user_preference_update", "experience_shared", "permission_change",
            "relationship_update", "memory_retrieval_complete", "experience_request"
        ]
        
        # Track active sharing sessions
        self.active_sharing_sessions = {}
    
    async def on_context_received(self, context: SharedContext):
        """Initialize cross-user experience processing for this context"""
        logger.debug(f"CrossUserExperience received context for user: {context.user_id}")
        
        # Check if this context involves cross-user elements
        cross_user_relevance = await self._assess_cross_user_relevance(context)
        
        if cross_user_relevance > 0.3:
            # Get user's sharing preferences
            user_preferences = await self._get_user_sharing_preferences(context.user_id)
            
            # Check for potential experience matches
            potential_matches = await self._find_potential_experience_matches(context)
            
            # Send cross-user context to other modules
            await self.send_context_update(
                update_type="cross_user_context_available",
                data={
                    "user_id": context.user_id,
                    "sharing_preferences": user_preferences,
                    "potential_matches": len(potential_matches),
                    "cross_user_relevance": cross_user_relevance,
                    "sharing_enabled": user_preferences.get("cross_user_sharing_preference", 0.3) > 0.2
                },
                priority=ContextPriority.NORMAL
            )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules affecting cross-user experience"""
        
        if update.update_type == "memory_retrieval_complete":
            # Check if retrieved memories could be shared
            memory_data = update.data
            memories = memory_data.get("retrieved_memories", [])
            
            shareable = await self._identify_shareable_memories(memories)
            if shareable:
                await self.send_context_update(
                    update_type="shareable_experiences_identified",
                    data={
                        "shareable_count": len(shareable),
                        "experience_types": list(set(m.get("memory_type") for m in shareable)),
                        "average_significance": sum(m.get("significance", 0) for m in shareable) / len(shareable)
                    }
                )
        
        elif update.update_type == "relationship_update":
            # Relationship changes affect sharing permissions
            relationship_data = update.data
            user_id = relationship_data.get("user_id")
            
            if user_id:
                await self._update_sharing_permissions_from_relationship(user_id, relationship_data)
        
        elif update.update_type == "experience_request":
            # Handle explicit requests for shared experiences
            request_data = update.data
            query = request_data.get("query", "")
            requesting_user = request_data.get("user_id")
            
            if query and requesting_user:
                await self._process_experience_request(requesting_user, query, update.source_module)
        
        elif update.update_type == "user_preference_update":
            # Update sharing preferences
            pref_data = update.data
            user_id = pref_data.get("user_id")
            preferences = pref_data.get("preferences", {})
            
            if user_id and "sharing_preferences" in preferences:
                await self.original_manager.set_user_preference(
                    user_id, 
                    preferences["sharing_preferences"]
                )
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for cross-user experience opportunities"""
        # Analyze input for cross-user relevance
        cross_user_analysis = await self._analyze_input_for_cross_user(context.user_input)
        
        # Get messages from other modules
        messages = await self.get_cross_module_messages()
        
        if cross_user_analysis["requests_shared_experience"]:
            # Process request for shared experiences
            search_results = await self._search_cross_user_experiences(
                context.user_id,
                cross_user_analysis["query"],
                cross_user_analysis["filters"]
            )
            
            # Send notification about search
            await self.send_context_update(
                update_type="cross_user_search_performed",
                data={
                    "user_id": context.user_id,
                    "query": cross_user_analysis["query"],
                    "results_found": len(search_results.get("experiences", [])),
                    "source_users": search_results.get("source_users", [])
                }
            )
            
            return {
                "cross_user_search": search_results,
                "search_performed": True,
                "analysis": cross_user_analysis
            }
        
        # Check if current experience is shareable
        if cross_user_analysis["creates_shareable_experience"]:
            await self._mark_experience_as_shareable(context, cross_user_analysis)
        
        return {
            "cross_user_analysis": cross_user_analysis,
            "cross_user_relevant": cross_user_analysis["relevance"] > 0.3,
            "processing_complete": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze cross-user sharing opportunities and patterns"""
        messages = await self.get_cross_module_messages()
        
        # Analyze sharing patterns
        sharing_analysis = await self._analyze_sharing_patterns(context.user_id)
        
        # Identify collaboration opportunities
        collaboration_opportunities = await self._identify_collaboration_opportunities(
            context, messages
        )
        
        # Analyze permission compatibility
        permission_analysis = await self._analyze_permission_compatibility(context.user_id)
        
        return {
            "sharing_patterns": sharing_analysis,
            "collaboration_opportunities": collaboration_opportunities,
            "permission_compatibility": permission_analysis,
            "active_sharing_sessions": len(self.active_sharing_sessions),
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize cross-user elements for response generation"""
        messages = await self.get_cross_module_messages()
        
        # Determine if response should include shared experiences
        should_include_shared = await self._should_include_shared_experiences(context, messages)
        
        synthesis_result = {
            "include_shared_experiences": should_include_shared["include"],
            "shared_experience_context": None,
            "cross_user_insights": None,
            "personalization_notes": None
        }
        
        if should_include_shared["include"]:
            # Get relevant shared experiences
            shared_experiences = await self._get_relevant_shared_experiences(
                context.user_id,
                context.user_input,
                should_include_shared["reason"]
            )
            
            if shared_experiences:
                synthesis_result["shared_experience_context"] = shared_experiences
                synthesis_result["cross_user_insights"] = await self._generate_cross_user_insights(
                    shared_experiences
                )
                synthesis_result["personalization_notes"] = await self._generate_personalization_notes(
                    context.user_id,
                    shared_experiences
                )
                
                # Notify about shared experience use
                await self.send_context_update(
                    update_type="shared_experiences_in_response",
                    data={
                        "count": len(shared_experiences),
                        "source_users": list(set(exp.get("source_user_id") for exp in shared_experiences)),
                        "purpose": should_include_shared["reason"]
                    }
                )
        
        return synthesis_result
    
    # Helper methods
    
    async def _assess_cross_user_relevance(self, context: SharedContext) -> float:
        """Assess relevance of current context for cross-user features"""
        relevance = 0.0
        
        # Keywords that suggest cross-user interest
        cross_user_keywords = ["others", "someone else", "other people", "shared", "common",
                              "similar experience", "anyone else", "community"]
        
        input_lower = context.user_input.lower()
        keyword_matches = sum(1 for kw in cross_user_keywords if kw in input_lower)
        relevance += min(0.4, keyword_matches * 0.15)
        
        # Check for learning/curiosity context
        if context.goal_context and "knowledge" in str(context.goal_context):
            relevance += 0.2
        
        # Check for social context
        if context.relationship_context:
            relevance += 0.1
        
        return min(1.0, relevance)
    
    async def _get_user_sharing_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user's sharing preferences"""
        try:
            return await self.original_manager.get_user_preference(None, user_id)
        except:
            return {
                "cross_user_sharing_preference": 0.3,
                "experience_sharing_preference": 0.5,
                "privacy_level": 0.5
            }
    
    async def _find_potential_experience_matches(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Find potential matching experiences from other users"""
        if not self.original_manager.memory_core:
            return []
        
        try:
            # Extract key concepts from context
            key_concepts = await self._extract_key_concepts(context)
            
            if not key_concepts:
                return []
            
            # Get user's preferences
            user_prefs = await self._get_user_sharing_preferences(context.user_id)
            min_compatibility = user_prefs.get("min_compatibility_threshold", 0.6)
            
            # Find compatible users
            compatible_users = []
            for user_id, prefs in self.original_manager.user_preference_profiles.items():
                if user_id != context.user_id:
                    compatibility = self._calculate_user_compatibility(
                        user_prefs, prefs
                    )
                    if compatibility >= min_compatibility:
                        compatible_users.append((user_id, compatibility))
            
            # Sort by compatibility
            compatible_users.sort(key=lambda x: x[1], reverse=True)
            
            # Search for experiences from compatible users
            potential_matches = []
            for user_id, compatibility in compatible_users[:10]:  # Top 10 compatible users
                # Check permissions
                permission_key = f"{user_id}:{context.user_id}"
                if permission_key not in self.original_manager.permission_matrix:
                    # Calculate permission if not cached
                    permission = await self.original_manager.calculate_sharing_permission(
                        None, user_id, context.user_id
                    )
                    if permission.get("permission_level", 0) < 0.3:
                        continue
                
                # Search user's experiences
                search_query = " ".join(key_concepts[:3])  # Use top 3 concepts
                try:
                    experiences = await self.original_manager.memory_core.search_memories(
                        user_id=user_id,
                        query=search_query,
                        memory_type="experience",
                        limit=5
                    )
                    
                    for exp in experiences:
                        # Calculate match score
                        match_score = self._calculate_experience_match_score(
                            exp, key_concepts, context
                        )
                        
                        if match_score > 0.5:
                            potential_matches.append({
                                "experience": exp,
                                "source_user_id": user_id,
                                "compatibility": compatibility,
                                "match_score": match_score,
                                "combined_score": (compatibility * 0.3 + match_score * 0.7)
                            })
                            
                except Exception as e:
                    logger.error(f"Error searching experiences for user {user_id}: {e}")
                    continue
            
            # Sort by combined score
            potential_matches.sort(key=lambda x: x["combined_score"], reverse=True)
            
            return potential_matches[:20]  # Return top 20 matches
            
        except Exception as e:
            logger.error(f"Error finding potential experience matches: {e}")
            return []

    def _calculate_experience_match_score(self, experience: Dict[str, Any], 
                                        key_concepts: List[str], 
                                        context: SharedContext) -> float:
        """Calculate how well an experience matches the current context"""
        score = 0.0
        
        # Check concept overlap
        exp_text = experience.get("memory_text", "").lower()
        exp_tags = [tag.lower() for tag in experience.get("tags", [])]
        
        concept_matches = 0
        for concept in key_concepts:
            if concept.lower() in exp_text or concept.lower() in exp_tags:
                concept_matches += 1
        
        if key_concepts:
            score += (concept_matches / len(key_concepts)) * 0.4
        
        # Check emotional alignment
        if context.emotional_state and experience.get("emotional_context"):
            exp_emotions = experience["emotional_context"]
            emotion_alignment = 0.0
            
            for emotion, intensity in context.emotional_state.items():
                if emotion in exp_emotions:
                    # Similar emotions increase match
                    diff = abs(intensity - exp_emotions[emotion])
                    emotion_alignment += (1.0 - diff) * intensity
            
            if context.emotional_state:
                score += (emotion_alignment / len(context.emotional_state)) * 0.3
        
        # Check temporal relevance (prefer recent experiences)
        if experience.get("timestamp"):
            try:
                exp_time = datetime.fromisoformat(experience["timestamp"])
                days_old = (datetime.now() - exp_time).days
                
                if days_old < 7:
                    score += 0.2
                elif days_old < 30:
                    score += 0.1
                elif days_old < 90:
                    score += 0.05
            except:
                pass
        
        # Check significance
        significance = experience.get("significance", 5) / 10.0
        score += significance * 0.1
        
        return min(1.0, score)

    def _calculate_user_compatibility(self, prefs1: Dict[str, Any], prefs2: Dict[str, Any]) -> float:
        """Calculate compatibility between user preferences"""
        compatibility_score = 0.0
        weight_total = 0.0
        
        # Scenario preference compatibility (40% weight)
        scenario_weight = 0.4
        scenario_sim = 0.0
        scenario_count = 0
        
        scenarios1 = prefs1.get("scenario_preferences", {})
        scenarios2 = prefs2.get("scenario_preferences", {})
        
        for scenario in set(scenarios1.keys()) | set(scenarios2.keys()):
            pref1 = scenarios1.get(scenario, 0.5)
            pref2 = scenarios2.get(scenario, 0.5)
            
            # Calculate similarity (1 - normalized difference)
            similarity = 1.0 - abs(pref1 - pref2)
            scenario_sim += similarity
            scenario_count += 1
        
        if scenario_count > 0:
            compatibility_score += (scenario_sim / scenario_count) * scenario_weight
            weight_total += scenario_weight
        
        # Emotional preference compatibility (30% weight)
        emotion_weight = 0.3
        emotion_sim = 0.0
        emotion_count = 0
        
        emotions1 = prefs1.get("emotional_preferences", {})
        emotions2 = prefs2.get("emotional_preferences", {})
        
        for emotion in set(emotions1.keys()) | set(emotions2.keys()):
            pref1 = emotions1.get(emotion, 0.5)
            pref2 = emotions2.get(emotion, 0.5)
            
            similarity = 1.0 - abs(pref1 - pref2)
            emotion_sim += similarity
            emotion_count += 1
        
        if emotion_count > 0:
            compatibility_score += (emotion_sim / emotion_count) * emotion_weight
            weight_total += emotion_weight
        
        # Privacy level compatibility (20% weight)
        privacy_weight = 0.2
        privacy1 = prefs1.get("privacy_level", 0.5)
        privacy2 = prefs2.get("privacy_level", 0.5)
        
        privacy_diff = abs(privacy1 - privacy2)
        if privacy_diff < 0.2:  # Very compatible
            compatibility_score += privacy_weight
        elif privacy_diff < 0.4:  # Somewhat compatible
            compatibility_score += privacy_weight * 0.5
        
        weight_total += privacy_weight
        
        # Sharing preference alignment (10% weight)
        sharing_weight = 0.1
        sharing1 = prefs1.get("cross_user_sharing_preference", 0.3)
        sharing2 = prefs2.get("cross_user_sharing_preference", 0.3)
        
        # Both should be willing to share
        min_sharing = min(sharing1, sharing2)
        compatibility_score += min_sharing * sharing_weight
        weight_total += sharing_weight
        
        # Normalize by total weight
        if weight_total > 0:
            return compatibility_score / weight_total
        
        return 0.5  # Default compatibility
    
    async def _identify_shareable_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify which memories could be shared"""
        shareable = []
        
        for memory in memories:
            # Check shareability criteria
            if (memory.get("significance", 0) > 6 and
                memory.get("memory_type") in ["experience", "reflection", "abstraction"] and
                not memory.get("private", False)):
                
                shareable.append(memory)
        
        return shareable
    
    async def _update_sharing_permissions_from_relationship(self, user_id: str, relationship_data: Dict[str, Any]):
        """Update sharing permissions based on relationship changes"""
        trust = relationship_data.get("trust", 0.5)
        intimacy = relationship_data.get("intimacy", 0.3)
        
        # Higher trust and intimacy = more sharing
        new_sharing_preference = min(0.9, (trust + intimacy) / 2)
        
        await self.original_manager.set_user_preference(
            user_id,
            {"cross_user_sharing_preference": new_sharing_preference}
        )
        
        logger.info(f"Updated sharing preference for {user_id} to {new_sharing_preference:.2f}")
    
    async def _process_experience_request(self, user_id: str, query: str, source_module: str):
        """Process a request for shared experiences"""
        # Create a sharing session
        session_id = f"share_{user_id}_{datetime.now().timestamp()}"
        self.active_sharing_sessions[session_id] = {
            "user_id": user_id,
            "query": query,
            "source_module": source_module,
            "started_at": datetime.now()
        }
        
        # Search for experiences
        results = await self.original_manager.find_cross_user_experiences(
            target_user_id=user_id,
            query=query,
            limit=5
        )
        
        # Send results back to requesting module
        await self.send_context_update(
            update_type="cross_user_experiences_found",
            data={
                "session_id": session_id,
                "results": results,
                "query": query
            },
            target_modules=[source_module],
            scope=ContextScope.TARGETED
        )
    
    async def _analyze_input_for_cross_user(self, user_input: str) -> Dict[str, Any]:
        """Analyze input for cross-user relevance"""
        input_lower = user_input.lower()
        
        analysis = {
            "requests_shared_experience": False,
            "creates_shareable_experience": False,
            "query": "",
            "filters": {},
            "relevance": 0.0
        }
        
        # Check for experience requests
        request_patterns = [
            "what do others think about",
            "has anyone else",
            "show me experiences",
            "other people's",
            "similar experiences",
            "shared experiences"
        ]
        
        for pattern in request_patterns:
            if pattern in input_lower:
                analysis["requests_shared_experience"] = True
                analysis["query"] = user_input  # Use full input as query
                analysis["relevance"] = 0.8
                break
        
        # Check for shareable content creation
        shareable_indicators = [
            "i learned that",
            "i discovered",
            "interesting experience",
            "want to share",
            "others might benefit"
        ]
        
        for indicator in shareable_indicators:
            if indicator in input_lower:
                analysis["creates_shareable_experience"] = True
                analysis["relevance"] = max(analysis["relevance"], 0.6)
                break
        
        return analysis
    
    async def _search_cross_user_experiences(self, user_id: str, query: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Search for cross-user experiences"""
        # Use the original manager's search functionality
        results = await self.original_manager.find_cross_user_experiences(
            target_user_id=user_id,
            query=query,
            scenario_type=filters.get("scenario_type"),
            limit=filters.get("limit", 5)
        )
        
        return results
    
    async def _mark_experience_as_shareable(self, context: SharedContext, analysis: Dict[str, Any]):
        """Mark current experience as shareable"""
        # Send update to memory system
        await self.send_context_update(
            update_type="mark_experience_shareable",
            data={
                "user_id": context.user_id,
                "content": context.user_input,
                "shareable_type": analysis.get("content_type", "general"),
                "sharing_preference": 0.7  # Default sharing preference
            },
            target_modules=["memory_core"],
            scope=ContextScope.TARGETED
        )
    
    async def _analyze_sharing_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's sharing patterns"""
        stats = await self.original_manager.get_sharing_statistics(user_id)
        
        patterns = {
            "total_shared": stats.get("total_shares", 0),
            "sharing_frequency": "low",  # Would calculate based on time
            "preferred_categories": self._extract_preferred_categories(stats),
            "reciprocity_score": self._calculate_reciprocity(stats)
        }
        
        # Determine sharing frequency
        if patterns["total_shared"] > 20:
            patterns["sharing_frequency"] = "high"
        elif patterns["total_shared"] > 10:
            patterns["sharing_frequency"] = "medium"
        
        return patterns
    
    async def _identify_collaboration_opportunities(self, context: SharedContext, messages: Dict) -> List[Dict[str, Any]]:
        """Identify opportunities for cross-user collaboration"""
        opportunities = []
        
        # Check for shared goals
        goal_messages = messages.get("goal_manager", [])
        for msg in goal_messages:
            if msg.get("type") == "goal_context_available":
                active_goals = msg.get("data", {}).get("active_goals", [])
                
                # Knowledge goals are good for sharing
                knowledge_goals = [g for g in active_goals if "knowledge" in g.get("description", "").lower()]
                if knowledge_goals:
                    opportunities.append({
                        "type": "knowledge_sharing",
                        "context": "active_learning_goals",
                        "relevance": 0.7
                    })
        
        # Check for emotional experiences
        if context.emotional_state:
            high_emotions = [e for e, v in context.emotional_state.items() if v > 0.7]
            if high_emotions:
                opportunities.append({
                    "type": "emotional_support",
                    "context": f"high_{high_emotions[0]}",
                    "relevance": 0.6
                })
        
        return opportunities
    
    async def _analyze_permission_compatibility(self, user_id: str) -> Dict[str, Any]:
        """Analyze permission compatibility with other users"""
        # Get user's preferences
        user_prefs = await self._get_user_sharing_preferences(user_id)
        
        # Find compatible users (simplified)
        compatible_count = 0
        total_users = len(self.original_manager.user_preference_profiles)
        
        if total_users > 1:
            # Count users with compatible sharing preferences
            for other_id, other_prefs in self.original_manager.user_preference_profiles.items():
                if other_id != user_id:
                    if abs(user_prefs.get("privacy_level", 0.5) - 
                          other_prefs.get("privacy_level", 0.5)) < 0.3:
                        compatible_count += 1
        
        return {
            "compatibility_ratio": compatible_count / max(1, total_users - 1),
            "compatible_users": compatible_count,
            "total_other_users": total_users - 1
        }
    
    async def _should_include_shared_experiences(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Determine if response should include shared experiences"""
        should_include = {
            "include": False,
            "reason": None,
            "confidence": 0.0
        }
        
        # Check for explicit requests
        for module_msgs in messages.values():
            for msg in module_msgs:
                if msg.get("type") == "cross_user_search_performed":
                    should_include["include"] = True
                    should_include["reason"] = "explicit_request"
                    should_include["confidence"] = 1.0
                    return should_include
        
        # Check for learning context
        if context.goal_context and "knowledge" in str(context.goal_context):
            should_include["include"] = True
            should_include["reason"] = "learning_enhancement"
            should_include["confidence"] = 0.7
        
        # Check for emotional support context
        elif context.emotional_state:
            negative_emotions = ["sadness", "anxiety", "frustration", "loneliness"]
            if any(context.emotional_state.get(e, 0) > 0.6 for e in negative_emotions):
                should_include["include"] = True
                should_include["reason"] = "emotional_support"
                should_include["confidence"] = 0.8
        
        return should_include
    
    async def _get_relevant_shared_experiences(self, user_id: str, user_input: str, reason: str) -> List[Dict[str, Any]]:
        """Get shared experiences relevant to current context"""
        # Determine search parameters based on reason
        search_params = {
            "target_user_id": user_id,
            "query": user_input,
            "limit": 3
        }
        
        if reason == "emotional_support":
            search_params["scenario_type"] = "emotional"
        elif reason == "learning_enhancement":
            search_params["scenario_type"] = "educational"
        
        # Search for experiences
        results = await self.original_manager.find_cross_user_experiences(**search_params)
        
        return results.get("experiences", [])
    
    async def _generate_cross_user_insights(self, shared_experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights from shared experiences"""
        if not shared_experiences:
            return {}
        
        insights = {
            "common_themes": [],
            "unique_perspectives": [],
            "pattern_observations": []
        }
        
        # Extract themes (simplified)
        all_tags = []
        for exp in shared_experiences:
            all_tags.extend(exp.get("tags", []))
        
        # Count tag frequencies
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Common themes are tags that appear multiple times
        insights["common_themes"] = [tag for tag, count in tag_counts.items() if count > 1]
        
        # Unique perspectives are single occurrences
        insights["unique_perspectives"] = [tag for tag, count in tag_counts.items() if count == 1][:3]
        
        # Pattern observations
        if len(shared_experiences) >= 3:
            insights["pattern_observations"].append("Multiple users have similar experiences")
        
        return insights
    
    async def _generate_personalization_notes(self, user_id: str, shared_experiences: List[Dict[str, Any]]) -> List[str]:
        """Generate notes on how to personalize shared experiences"""
        notes = []
        
        # Get user preferences
        user_prefs = await self._get_user_sharing_preferences(user_id)
        
        if user_prefs.get("privacy_level", 0.5) > 0.7:
            notes.append("Present shared experiences with anonymity emphasized")
        
        if user_prefs.get("emotional_preferences", {}).get("empathy", 0.5) > 0.7:
            notes.append("Frame shared experiences with emotional connection")
        
        return notes
    
    async def _extract_key_concepts(self, context: SharedContext) -> List[str]:
        """Extract key concepts from context for matching"""
        concepts = []
        
        # Extract from user input (simplified - would use NLP in real implementation)
        words = context.user_input.lower().split()
        
        # Filter for meaningful words (simplified)
        meaningful_words = [w for w in words if len(w) > 4 and w not in 
                          ["about", "think", "would", "could", "should", "there"]]
        
        concepts.extend(meaningful_words[:5])
        
        # Add emotional concepts if present
        if context.emotional_state:
            high_emotions = [e for e, v in context.emotional_state.items() if v > 0.6]
            concepts.extend(high_emotions)
        
        return list(set(concepts))
    
    def _extract_preferred_categories(self, stats: Dict[str, Any]) -> List[str]:
        """Extract preferred sharing categories from statistics"""
        scenario_shares = stats.get("scenario_shares", {})
        if not scenario_shares:
            return []
        
        # Sort by frequency
        sorted_categories = sorted(scenario_shares.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 3
        return [cat for cat, _ in sorted_categories[:3]]
    
    def _calculate_reciprocity(self, stats: Dict[str, Any]) -> float:
        """Calculate reciprocity score from sharing statistics"""
        user_shares = stats.get("user_shares", {})
        if not user_shares:
            return 0.5
        
        total_shared = 0
        total_received = 0
        
        for user_data in user_shares.values():
            total_shared += user_data.get("shared", 0)
            total_received += user_data.get("received", 0)
        
        if total_shared + total_received == 0:
            return 0.5
        
        # Calculate balance
        return min(total_shared, total_received) / max(total_shared, total_received)
    
    # Delegate missing methods to original manager
    def __getattr__(self, name):
        """Delegate any missing methods to the original manager"""
        return getattr(self.original_manager, name)
