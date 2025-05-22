# nyx/core/a2a/context_aware_knowledge_core.py

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import json

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareKnowledgeCore(ContextAwareModule):
    """
    Enhanced KnowledgeCore with full context distribution capabilities
    """
    
    def __init__(self, original_knowledge_core):
        super().__init__("knowledge_core")
        self.original_core = original_knowledge_core
        self.context_subscriptions = [
            "memory_retrieval_complete", "goal_context_available", "emotional_state_update",
            "relationship_updates", "curiosity_trigger", "learning_opportunity",
            "knowledge_request", "pattern_detected", "abstraction_created"
        ]
        
    async def on_context_received(self, context: SharedContext):
        """Initialize knowledge processing for this context"""
        logger.debug(f"KnowledgeCore received context for user: {context.user_id}")
        
        # Analyze input for knowledge-related content
        knowledge_implications = await self._analyze_input_for_knowledge(context.user_input)
        
        # Check for relevant knowledge gaps
        knowledge_gaps = await self._identify_relevant_gaps(context)
        
        # Get relevant exploration targets
        exploration_targets = await self._get_contextual_exploration_targets(context)
        
        # Send initial knowledge context to other modules
        await self.send_context_update(
            update_type="knowledge_context_available",
            data={
                "knowledge_implications": knowledge_implications,
                "knowledge_gaps": knowledge_gaps,
                "exploration_targets": exploration_targets,
                "current_domains": await self._get_active_domains(context),
                "curiosity_level": await self._calculate_curiosity_level(context)
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        
        if update.update_type == "memory_retrieval_complete":
            # Extract knowledge from retrieved memories
            memory_data = update.data
            await self._extract_knowledge_from_memories(memory_data)
            
        elif update.update_type == "goal_context_available":
            # Align knowledge exploration with active goals
            goal_data = update.data
            await self._align_exploration_with_goals(goal_data)
            
        elif update.update_type == "emotional_state_update":
            # Emotional state affects curiosity and learning
            emotional_data = update.data
            await self._adjust_curiosity_from_emotion(emotional_data)
            
        elif update.update_type == "pattern_detected":
            # Another module detected a pattern
            pattern_data = update.data
            await self._process_external_pattern(pattern_data)
            
        elif update.update_type == "learning_opportunity":
            # Learning opportunity identified
            opportunity_data = update.data
            await self._process_learning_opportunity(opportunity_data)
            
        elif update.update_type == "knowledge_request":
            # Direct request for knowledge
            request_data = update.data
            await self._handle_knowledge_request(request_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with knowledge awareness"""
        # Analyze input for knowledge content
        knowledge_analysis = await self._analyze_input_for_knowledge(context.user_input)
        
        # Query relevant knowledge
        query_results = []
        if knowledge_analysis["contains_question"] or knowledge_analysis["seeks_information"]:
            query_results = await self._query_contextual_knowledge(context)
        
        # Check if we should create new knowledge
        if knowledge_analysis["contains_assertion"] or knowledge_analysis["provides_information"]:
            new_knowledge = await self._create_knowledge_from_input(context)
        
        # Identify learning opportunities
        learning_opportunities = await self._identify_learning_opportunities(context)
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Send knowledge processing update
        await self.send_context_update(
            update_type="knowledge_processing_complete",
            data={
                "knowledge_analysis": knowledge_analysis,
                "query_results": query_results,
                "learning_opportunities": learning_opportunities,
                "knowledge_added": knowledge_analysis.get("new_knowledge_id"),
                "cross_module_insights": len(messages)
            }
        )
        
        return {
            "knowledge_processed": True,
            "analysis": knowledge_analysis,
            "query_results": query_results,
            "learning_opportunities": learning_opportunities
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze knowledge in context of current situation"""
        # Get current knowledge state
        knowledge_stats = await self.original_core.get_knowledge_statistics()
        
        # Analyze knowledge coherence
        coherence_analysis = await self._analyze_knowledge_coherence(context)
        
        # Identify knowledge patterns
        patterns = await self._identify_knowledge_patterns(context)
        
        # Assess knowledge quality
        quality_assessment = await self._assess_knowledge_quality(context)
        
        # Generate curiosity targets
        curiosity_targets = await self._generate_curiosity_targets(context)
        
        # Cross-module knowledge integration
        messages = await self.get_cross_module_messages()
        integration_opportunities = await self._identify_integration_opportunities(messages)
        
        return {
            "knowledge_stats": knowledge_stats,
            "coherence_analysis": coherence_analysis,
            "patterns": patterns,
            "quality_assessment": quality_assessment,
            "curiosity_targets": curiosity_targets,
            "integration_opportunities": integration_opportunities
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize knowledge components for response"""
        messages = await self.get_cross_module_messages()
        
        # Create knowledge-informed synthesis
        knowledge_synthesis = {
            "relevant_facts": await self._extract_relevant_facts(context),
            "knowledge_connections": await self._identify_knowledge_connections(context),
            "learning_insights": await self._generate_learning_insights(context, messages),
            "curiosity_prompts": await self._generate_curiosity_prompts(context),
            "knowledge_coherence": await self._check_response_knowledge_coherence(context)
        }
        
        # Check if we should express curiosity
        curiosity_expression = await self._should_express_curiosity(context, messages)
        if curiosity_expression["should_express"]:
            await self.send_context_update(
                update_type="curiosity_expression_needed",
                data=curiosity_expression,
                priority=ContextPriority.HIGH
            )
        
        return knowledge_synthesis
    
    # ========================================================================================
    # KNOWLEDGE-SPECIFIC HELPER METHODS
    # ========================================================================================
    
    async def _analyze_input_for_knowledge(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input for knowledge-related content"""
        input_lower = user_input.lower()
        
        analysis = {
            "contains_question": "?" in user_input or any(q in input_lower for q in ["what", "how", "why", "when", "where", "who"]),
            "seeks_information": any(kw in input_lower for kw in ["tell me", "explain", "describe", "what is", "how does"]),
            "contains_assertion": any(kw in input_lower for kw in ["is", "are", "was", "were"]) and "?" not in user_input,
            "provides_information": len(user_input.split()) > 10 and not analysis.get("contains_question", False),
            "references_learning": any(kw in input_lower for kw in ["learn", "understand", "know", "knowledge", "curious"]),
            "complexity_level": self._assess_input_complexity(user_input)
        }
        
        # Extract potential topics
        analysis["potential_topics"] = self._extract_topics(user_input)
        
        # Assess information quality if providing information
        if analysis["provides_information"]:
            analysis["information_quality"] = self._assess_information_quality(user_input)
        
        return analysis
    
    async def _identify_relevant_gaps(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify knowledge gaps relevant to current context"""
        # Get all knowledge gaps
        all_gaps = await self.original_core.identify_knowledge_gaps()
        
        # Filter by relevance to current context
        relevant_gaps = []
        for gap in all_gaps:
            relevance = await self._calculate_gap_relevance(gap, context)
            if relevance > 0.3:
                gap["context_relevance"] = relevance
                relevant_gaps.append(gap)
        
        # Sort by relevance * importance
        relevant_gaps.sort(key=lambda g: g["context_relevance"] * g["importance"], reverse=True)
        
        return relevant_gaps[:5]  # Top 5 gaps
    
    async def _get_contextual_exploration_targets(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Get exploration targets relevant to current context"""
        # Get active exploration targets
        targets = await self.original_core.get_exploration_targets(limit=20)
        
        # Score by context relevance
        scored_targets = []
        for target in targets:
            relevance = await self._calculate_target_relevance(target, context)
            if relevance > 0.2:
                target["context_relevance"] = relevance
                scored_targets.append(target)
        
        # Sort by combined score
        scored_targets.sort(key=lambda t: t["priority_score"] * t["context_relevance"], reverse=True)
        
        return scored_targets[:5]
    
    async def _extract_knowledge_from_memories(self, memory_data: Dict[str, Any]):
        """Extract and store knowledge from retrieved memories"""
        memories = memory_data.get("retrieved_memories", [])
        
        for memory in memories:
            # Check if this memory contains extractable knowledge
            if await self._contains_extractable_knowledge(memory):
                # Extract knowledge components
                knowledge_components = await self._extract_knowledge_components(memory)
                
                for component in knowledge_components:
                    # Add to knowledge graph
                    try:
                        knowledge_id = await self.original_core.add_knowledge(
                            type=component["type"],
                            content=component["content"],
                            source=f"memory_{memory.get('id', 'unknown')}",
                            confidence=component["confidence"]
                        )
                        
                        # Link to source memory
                        if knowledge_id and memory.get("id"):
                            await self.original_core.add_relation(
                                source_id=knowledge_id,
                                target_id=f"memory_{memory['id']}",
                                type="extracted_from",
                                weight=0.8
                            )
                    except Exception as e:
                        logger.error(f"Error extracting knowledge from memory: {e}")
    
    async def _align_exploration_with_goals(self, goal_data: Dict[str, Any]):
        """Align knowledge exploration with active goals"""
        active_goals = goal_data.get("active_goals", [])
        
        for goal in active_goals:
            # Identify knowledge needed for goal
            needed_knowledge = await self._identify_goal_knowledge_needs(goal)
            
            for knowledge_need in needed_knowledge:
                # Create or boost exploration target
                existing_targets = await self.original_core.get_exploration_targets()
                
                matching_target = None
                for target in existing_targets:
                    if (target["domain"] == knowledge_need["domain"] and 
                        target["topic"] == knowledge_need["topic"]):
                        matching_target = target
                        break
                
                if matching_target:
                    # Boost urgency for existing target
                    result = await self.original_core.record_exploration(
                        target_id=matching_target["id"],
                        result={"urgency_boost": 0.2, "goal_alignment": goal["id"]}
                    )
                else:
                    # Create new exploration target
                    target_id = await self.original_core.create_exploration_target(
                        domain=knowledge_need["domain"],
                        topic=knowledge_need["topic"],
                        importance=knowledge_need["importance"],
                        urgency=0.7  # High urgency for goal-aligned knowledge
                    )
    
    async def _adjust_curiosity_from_emotion(self, emotional_data: Dict[str, Any]):
        """Adjust curiosity based on emotional state"""
        emotional_state = emotional_data.get("emotional_state", {})
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if not dominant_emotion:
            return
        
        emotion_name, intensity = dominant_emotion
        
        # Map emotions to curiosity adjustments
        emotion_curiosity_map = {
            "Curiosity": 0.3,
            "Interest": 0.25,
            "Excitement": 0.2,
            "Wonder": 0.35,
            "Confusion": 0.15,  # Confusion can drive learning
            "Boredom": 0.1,     # Mild boost to seek stimulation
            "Fear": -0.2,       # Reduces exploration
            "Anxiety": -0.15,
            "Contentment": -0.1  # Reduces drive to explore
        }
        
        adjustment = emotion_curiosity_map.get(emotion_name, 0.0) * intensity
        
        # Apply adjustment to exploration targets
        if adjustment != 0:
            targets = await self.original_core.get_exploration_targets()
            for target in targets[:5]:  # Adjust top 5 targets
                await self.original_core.record_exploration(
                    target_id=target["id"],
                    result={"emotional_adjustment": adjustment}
                )
    
    async def _query_contextual_knowledge(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Query knowledge relevant to current context"""
        # Build query from context
        query_components = []
        
        # Add user input topics
        topics = self._extract_topics(context.user_input)
        query_components.extend(topics)
        
        # Add active goal topics
        if context.goal_context:
            goal_topics = self._extract_goal_topics(context.goal_context)
            query_components.extend(goal_topics)
        
        # Query knowledge for each component
        all_results = []
        for component in query_components[:3]:  # Limit to top 3
            results = await self.original_core.query_knowledge({
                "content_filter": {"topic": component},
                "limit": 5
            })
            all_results.extend(results)
        
        # Deduplicate and sort by relevance
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                result["context_relevance"] = await self._calculate_knowledge_relevance(result, context)
                unique_results.append(result)
        
        unique_results.sort(key=lambda r: r["context_relevance"] * r["confidence"], reverse=True)
        
        return unique_results[:10]
    
    async def _create_knowledge_from_input(self, context: SharedContext) -> Optional[Dict[str, Any]]:
        """Create new knowledge from user input"""
        # Extract knowledge components
        components = await self._extract_input_knowledge_components(context.user_input)
        
        if not components:
            return None
        
        created_knowledge = []
        for component in components:
            try:
                # Determine knowledge type
                k_type = self._determine_knowledge_type(component)
                
                # Create knowledge node
                knowledge_id = await self.original_core.add_knowledge(
                    type=k_type,
                    content=component["content"],
                    source=f"user_input_{context.user_id}",
                    confidence=component.get("confidence", 0.7)
                )
                
                if knowledge_id:
                    created_knowledge.append({
                        "id": knowledge_id,
                        "type": k_type,
                        "content": component["content"]
                    })
                    
                    # Create exploration target for new knowledge
                    domain = component["content"].get("domain", "general")
                    topic = component["content"].get("topic", "unknown")
                    
                    await self.original_core.create_exploration_target(
                        domain=domain,
                        topic=topic,
                        importance=0.6,
                        urgency=0.5,
                        knowledge_gap=0.3  # We have some knowledge but could learn more
                    )
            except Exception as e:
                logger.error(f"Error creating knowledge from input: {e}")
        
        return {
            "created_count": len(created_knowledge),
            "knowledge_nodes": created_knowledge
        }
    
    async def _identify_learning_opportunities(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify opportunities for learning from current context"""
        opportunities = []
        
        # Check for questions we can't fully answer
        if "?" in context.user_input:
            confidence = await self._assess_answer_confidence(context.user_input)
            if confidence < 0.7:
                opportunities.append({
                    "type": "knowledge_gap",
                    "description": "Question reveals knowledge gap",
                    "priority": 0.8,
                    "suggested_action": "explore_topic"
                })
        
        # Check for conceptual connections
        concepts = self._extract_concepts(context.user_input)
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                if not await self._has_connection(concept1, concept2):
                    opportunities.append({
                        "type": "missing_connection",
                        "description": f"Potential connection between {concept1} and {concept2}",
                        "priority": 0.5,
                        "suggested_action": "explore_relationship"
                    })
        
        # Check for pattern opportunities
        if len(context.context_updates) > 5:
            patterns = await self._detect_context_patterns(context.context_updates)
            for pattern in patterns:
                opportunities.append({
                    "type": "pattern_learning",
                    "description": f"Pattern detected: {pattern['description']}",
                    "priority": pattern["confidence"],
                    "suggested_action": "create_abstraction"
                })
        
        return opportunities
    
    async def _should_express_curiosity(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Determine if we should express curiosity in response"""
        # Calculate curiosity factors
        factors = {
            "knowledge_gaps": len(await self._identify_relevant_gaps(context)),
            "learning_opportunities": len(await self._identify_learning_opportunities(context)),
            "unanswered_questions": self._count_unanswered_questions(context),
            "exploration_targets": len(await self._get_contextual_exploration_targets(context)),
            "emotional_curiosity": self._get_emotional_curiosity(messages)
        }
        
        # Calculate overall curiosity score
        curiosity_score = (
            factors["knowledge_gaps"] * 0.2 +
            factors["learning_opportunities"] * 0.25 +
            factors["unanswered_questions"] * 0.3 +
            factors["exploration_targets"] * 0.15 +
            factors["emotional_curiosity"] * 0.1
        )
        
        should_express = curiosity_score > 0.6
        
        if should_express:
            # Generate curiosity expression
            expression = await self._generate_curiosity_expression(context, factors)
            
            return {
                "should_express": True,
                "curiosity_score": curiosity_score,
                "factors": factors,
                "expression": expression
            }
        
        return {"should_express": False, "curiosity_score": curiosity_score}
    
    # ========================================================================================
    # UTILITY METHODS
    # ========================================================================================
    
    def _assess_input_complexity(self, text: str) -> float:
        """Assess the complexity of input text"""
        # Simple heuristic based on length and vocabulary
        words = text.split()
        avg_word_length = sum(len(w) for w in words) / max(1, len(words))
        
        # Factors: sentence length, word length, punctuation
        complexity = min(1.0, (
            len(words) / 50.0 * 0.3 +  # Longer = more complex
            avg_word_length / 10.0 * 0.3 +  # Longer words = more complex
            text.count(",") / 5.0 * 0.2 +  # More clauses = more complex
            text.count(";") / 2.0 * 0.2  # Semicolons indicate complexity
        ))
        
        return complexity
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract potential topics from text"""
        # Simple extraction - in production use NLP
        words = text.lower().split()
        
        # Filter out common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        topics = []
        for i, word in enumerate(words):
            if len(word) > 3 and word not in stopwords:
                # Check for compound topics
                if i < len(words) - 1:
                    next_word = words[i + 1]
                    if len(next_word) > 3 and next_word not in stopwords:
                        topics.append(f"{word} {next_word}")
                topics.append(word)
        
        return list(set(topics))[:5]  # Unique topics, max 5
    
    def _assess_information_quality(self, text: str) -> float:
        """Assess the quality of information in text"""
        # Factors: specificity, clarity, completeness
        quality = 0.5  # Base quality
        
        # Specificity - numbers, names, dates increase quality
        import re
        if re.search(r'\d+', text):
            quality += 0.1
        if re.search(r'[A-Z][a-z]+', text):  # Proper nouns
            quality += 0.1
        
        # Clarity - complete sentences increase quality
        if text.endswith('.') and len(text.split()) > 5:
            quality += 0.1
        
        # Hedging words decrease quality
        hedging_words = ["maybe", "perhaps", "might", "could", "possibly", "probably"]
        for word in hedging_words:
            if word in text.lower():
                quality -= 0.05
        
        return max(0.1, min(1.0, quality))
    
    async def _calculate_gap_relevance(self, gap: Dict[str, Any], context: SharedContext) -> float:
        """Calculate relevance of a knowledge gap to current context"""
        relevance = 0.0
        
        # Check topic overlap
        gap_topic = gap.get("topic", "").lower()
        context_topics = self._extract_topics(context.user_input)
        
        for topic in context_topics:
            if topic in gap_topic or gap_topic in topic:
                relevance += 0.4
                break
        
        # Check domain relevance
        if context.session_context.get("domain") == gap.get("domain"):
            relevance += 0.2
        
        # Check goal alignment
        if context.goal_context:
            for goal in context.goal_context.get("active_goals", []):
                if gap.get("domain") in goal.get("description", "").lower():
                    relevance += 0.2
                    break
        
        # Boost for high importance gaps
        relevance += gap.get("importance", 0.5) * 0.2
        
        return min(1.0, relevance)
    
    def _determine_knowledge_type(self, component: Dict[str, Any]) -> str:
        """Determine the type of knowledge from a component"""
        content = component.get("content", {})
        
        # Check for different knowledge types
        if "definition" in content or "is defined as" in str(content):
            return "concept"
        elif "rule" in content or "always" in str(content) or "never" in str(content):
            return "rule"
        elif "because" in str(content) or "causes" in str(content):
            return "causal"
        elif any(word in str(content).lower() for word in ["believe", "think", "opinion"]):
            return "belief"
        else:
            return "fact"
    
    async def _generate_curiosity_expression(self, context: SharedContext, factors: Dict[str, Any]) -> str:
        """Generate an appropriate curiosity expression"""
        # Prioritize based on factors
        if factors["unanswered_questions"] > 2:
            return "I find myself wondering about several aspects of this topic..."
        elif factors["knowledge_gaps"] > 3:
            return "This touches on some areas I'd like to understand better..."
        elif factors["learning_opportunities"] > 2:
            return "There are some interesting connections here I'd like to explore..."
        else:
            return "I'm curious to learn more about this..."
    
    # Delegate unknown methods to original core
    def __getattr__(self, name):
        """Delegate any missing methods to the original core"""
        return getattr(self.original_core, name)
