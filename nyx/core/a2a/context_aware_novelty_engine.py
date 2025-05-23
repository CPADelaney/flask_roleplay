# nyx/core/a2a/context_aware_novelty_engine.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareNoveltyEngine(ContextAwareModule):
    """
    Advanced NoveltyEngine with full context distribution capabilities
    """
    
    def __init__(self, original_novelty_engine):
        super().__init__("novelty_engine")
        self.original_engine = original_novelty_engine
        self.context_subscriptions = [
            "memory_retrieval_complete", "emotional_state_update", 
            "knowledge_update", "pattern_detected", "goal_context_available",
            "creative_insight", "conceptual_blend_created", "causal_model_update",
            "imagination_simulation_complete", "reflection_generated"
        ]
        
        # Track context-inspired ideas
        self.context_inspired_ideas = {}
        self.cross_module_patterns = []
    
    async def on_context_received(self, context: SharedContext):
        """Process incoming context for creative inspiration"""
        logger.debug(f"NoveltyEngine received context for user: {context.user_id}")
        
        # Extract creative seeds from context
        creative_seeds = await self._extract_creative_seeds(context)
        
        # Check for cross-module pattern opportunities
        pattern_opportunities = await self._identify_pattern_opportunities(context)
        
        # Generate context-inspired ideas if seeds are promising
        if creative_seeds.get("high_potential_seeds"):
            idea_generation_task = self._generate_context_inspired_ideas(creative_seeds, context)
            # Don't await - let it run asynchronously
            import asyncio
            asyncio.create_task(idea_generation_task)
        
        # Send initial creative context
        await self.send_context_update(
            update_type="creative_context_available",
            data={
                "creative_seeds": creative_seeds,
                "pattern_opportunities": pattern_opportunities,
                "active_ideas": len(self.original_engine.context.generated_ideas),
                "creativity_potential": creative_seeds.get("creativity_potential", 0.5)
            },
            priority=ContextPriority.MEDIUM
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules for creative inspiration"""
        
        if update.update_type == "memory_retrieval_complete":
            # Use retrieved memories as creative inspiration
            memory_data = update.data
            memories = memory_data.get("memories", [])
            
            if memories:
                # Extract concepts from memories for creative blending
                memory_concepts = []
                for memory in memories[:5]:  # Top 5 memories
                    if memory.get("memory_text"):
                        memory_concepts.append({
                            "text": memory["memory_text"],
                            "type": memory.get("memory_type", "unknown"),
                            "significance": memory.get("significance", 0.5)
                        })
                
                if memory_concepts:
                    # Generate memory-inspired ideas
                    await self._generate_memory_inspired_ideas(memory_concepts)
        
        elif update.update_type == "emotional_state_update":
            # Adjust creative approach based on emotional state
            emotional_data = update.data
            dominant_emotion = emotional_data.get("dominant_emotion")
            
            if dominant_emotion:
                emotion_name, intensity = dominant_emotion
                await self._adjust_creative_approach(emotion_name, intensity)
        
        elif update.update_type == "pattern_detected":
            # Use detected patterns for pattern-based creativity
            pattern_data = update.data
            pattern_type = pattern_data.get("pattern_type")
            pattern_details = pattern_data.get("details", {})
            
            if pattern_type and pattern_details:
                await self._generate_pattern_based_ideas(pattern_type, pattern_details)
        
        elif update.update_type == "goal_context_available":
            # Align creative generation with active goals
            goal_data = update.data
            active_goals = goal_data.get("active_goals", [])
            
            if active_goals:
                # Focus creativity on goal-relevant domains
                goal_domains = set()
                for goal in active_goals:
                    if goal.get("associated_need"):
                        domain = self._map_need_to_creative_domain(goal["associated_need"])
                        if domain:
                            goal_domains.add(domain)
                
                if goal_domains:
                    await self._focus_creative_domains(list(goal_domains))
        
        elif update.update_type == "knowledge_update":
            # Incorporate new knowledge into creative processes
            knowledge_data = update.data
            new_concepts = knowledge_data.get("new_concepts", [])
            updated_models = knowledge_data.get("updated_models", [])
            
            if new_concepts or updated_models:
                await self._integrate_new_knowledge(new_concepts, updated_models)
        
        elif update.update_type == "reflection_generated":
            # Use reflections for meta-creative insights
            reflection_data = update.data
            reflection_text = reflection_data.get("reflection_text")
            reflection_type = reflection_data.get("reflection_type")
            
            if reflection_text and reflection_type == "creative":
                await self._process_creative_reflection(reflection_text)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with context-aware creative generation"""
        # Check if input suggests creative needs
        creative_analysis = await self._analyze_creative_potential(context.user_input)
        
        # Get cross-module messages for integrated creativity
        messages = await self.get_cross_module_messages()
        
        # Determine optimal creative approach
        creative_approach = await self._determine_creative_approach(context, messages, creative_analysis)
        
        # Generate ideas if high creative potential
        generated_ideas = []
        if creative_analysis.get("creative_potential", 0) > 0.6:
            # Use the original engine with context enhancement
            if creative_approach.get("technique"):
                idea = await self.original_engine.generate_novel_idea(
                    technique=creative_approach["technique"],
                    domain=creative_approach.get("domain"),
                    concepts=creative_approach.get("concepts", [])
                )
                if idea:
                    generated_ideas.append(idea)
                    # Track context-inspired idea
                    self.context_inspired_ideas[idea.id] = {
                        "context_summary": self._summarize_context(context),
                        "approach": creative_approach,
                        "timestamp": datetime.now().isoformat()
                    }
        
        # Send update about creative processing
        if generated_ideas:
            await self.send_context_update(
                update_type="novel_ideas_generated",
                data={
                    "ideas": [{"id": i.id, "title": i.title, "novelty_score": i.novelty_score} 
                             for i in generated_ideas],
                    "creative_approach": creative_approach,
                    "context_inspired": True
                },
                priority=ContextPriority.MEDIUM
            )
        
        return {
            "creative_processing": True,
            "ideas_generated": len(generated_ideas),
            "creative_analysis": creative_analysis,
            "approach_used": creative_approach
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze creative opportunities in current context"""
        # Get all active ideas
        recent_ideas = await self.original_engine.get_generated_ideas(limit=20)
        
        # Analyze idea patterns
        idea_patterns = await self._analyze_idea_patterns(recent_ideas)
        
        # Evaluate creative momentum
        creative_momentum = await self._evaluate_creative_momentum(context, recent_ideas)
        
        # Identify unexplored creative spaces
        unexplored_spaces = await self._identify_unexplored_spaces(context)
        
        # Cross-module creative opportunities
        messages = await self.get_cross_module_messages()
        cross_module_opportunities = await self._analyze_cross_module_creativity(messages)
        
        return {
            "recent_ideas_count": len(recent_ideas),
            "idea_patterns": idea_patterns,
            "creative_momentum": creative_momentum,
            "unexplored_spaces": unexplored_spaces,
            "cross_module_opportunities": cross_module_opportunities,
            "context_inspired_ratio": len([i for i in recent_ideas if i.id in self.context_inspired_ideas]) / max(1, len(recent_ideas))
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize creative insights for response generation"""
        # Get relevant ideas for synthesis
        filter_criteria = {"min_relevance": 0.6, "max_ideas": 3}
        relevant_ideas = await self._get_contextually_relevant_ideas(context, filter_criteria)
        
        # Generate creative synthesis
        creative_synthesis = {
            "featured_ideas": [],
            "creative_insights": [],
            "suggested_explorations": [],
            "creative_momentum_summary": ""
        }
        
        # Process featured ideas
        for idea in relevant_ideas:
            creative_synthesis["featured_ideas"].append({
                "title": idea.title,
                "description": idea.description,
                "relevance": idea.novelty_score * 0.5 + idea.usefulness_score * 0.5
            })
        
        # Generate insights from cross-module patterns
        if self.cross_module_patterns:
            for pattern in self.cross_module_patterns[-3:]:  # Latest 3 patterns
                creative_synthesis["creative_insights"].append(
                    f"Pattern discovered: {pattern['description']}"
                )
        
        # Suggest future explorations
        messages = await self.get_cross_module_messages()
        suggestions = await self._generate_exploration_suggestions(context, messages)
        creative_synthesis["suggested_explorations"] = suggestions
        
        # Summarize creative momentum
        momentum_data = await self._evaluate_creative_momentum(context, relevant_ideas)
        creative_synthesis["creative_momentum_summary"] = self._summarize_momentum(momentum_data)
        
        # Send synthesis update
        await self.send_context_update(
            update_type="creative_synthesis_complete",
            data={
                "synthesis": creative_synthesis,
                "ideas_synthesized": len(relevant_ideas),
                "insights_generated": len(creative_synthesis["creative_insights"])
            },
            priority=ContextPriority.MEDIUM
        )
        
        return creative_synthesis
    
    # Advanced helper methods
    
    async def _extract_creative_seeds(self, context: SharedContext) -> Dict[str, Any]:
        """Extract potential creative seeds from context"""
        seeds = {
            "concepts": [],
            "patterns": [],
            "tensions": [],
            "high_potential_seeds": [],
            "creativity_potential": 0.0
        }
        
        # Extract from user input
        if context.user_input:
            input_concepts = self._extract_concepts_from_text(context.user_input)
            seeds["concepts"].extend(input_concepts)
        
        # Extract from emotional state
        if context.emotional_state:
            emotional_tensions = self._identify_emotional_tensions(context.emotional_state)
            seeds["tensions"].extend(emotional_tensions)
        
        # Extract from memory context
        if context.memory_context:
            memory_patterns = self._extract_memory_patterns(context.memory_context)
            seeds["patterns"].extend(memory_patterns)
        
        # Identify high-potential seeds
        all_seeds = seeds["concepts"] + seeds["patterns"] + seeds["tensions"]
        if all_seeds:
            # Score each seed
            for seed in all_seeds:
                score = self._score_creative_seed(seed)
                if score > 0.7:
                    seeds["high_potential_seeds"].append(seed)
        
        # Calculate overall creativity potential
        if seeds["high_potential_seeds"]:
            seeds["creativity_potential"] = min(1.0, len(seeds["high_potential_seeds"]) * 0.3)
        elif all_seeds:
            seeds["creativity_potential"] = 0.3
        
        return seeds
    
    async def _identify_pattern_opportunities(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify opportunities for pattern-based creativity"""
        opportunities = []
        
        # Check for conceptual patterns
        if context.reasoning_context:
            concepts = context.reasoning_context.get("active_concepts", [])
            if len(concepts) >= 2:
                opportunities.append({
                    "type": "conceptual_blending",
                    "elements": concepts[:3],
                    "potential": 0.8
                })
        
        # Check for emotional patterns
        if context.emotional_state and context.session_context:
            past_emotions = context.session_context.get("emotional_history", [])
            if past_emotions:
                opportunities.append({
                    "type": "emotional_trajectory",
                    "pattern": "dynamic_emotional_flow",
                    "potential": 0.6
                })
        
        # Check for behavioral patterns
        if context.action_context:
            recent_actions = context.action_context.get("recent_actions", [])
            if len(recent_actions) >= 3:
                opportunities.append({
                    "type": "behavioral_pattern",
                    "pattern": "action_sequence",
                    "potential": 0.7
                })
        
        return opportunities
    
    async def _generate_context_inspired_ideas(self, creative_seeds: Dict[str, Any], context: SharedContext):
        """Generate ideas inspired by context asynchronously"""
        try:
            # Select best technique based on seeds
            technique = self._select_technique_for_seeds(creative_seeds)
            
            # Extract concepts for generation
            concepts = []
            for seed in creative_seeds.get("high_potential_seeds", []):
                if isinstance(seed, dict) and "text" in seed:
                    concepts.append(seed["text"])
                elif isinstance(seed, str):
                    concepts.append(seed)
            
            if not concepts:
                return
            
            # Generate idea using original engine
            idea = await self.original_engine.generate_novel_idea(
                technique=technique,
                concepts=concepts[:5]  # Limit concepts
            )
            
            if idea:
                # Track as context-inspired
                self.context_inspired_ideas[idea.id] = {
                    "seeds": creative_seeds,
                    "context_summary": self._summarize_context(context),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Send notification
                await self.send_context_update(
                    update_type="context_inspired_idea_generated",
                    data={
                        "idea_id": idea.id,
                        "idea_title": idea.title,
                        "novelty_score": idea.novelty_score,
                        "technique": technique,
                        "seed_count": len(creative_seeds.get("high_potential_seeds", []))
                    },
                    priority=ContextPriority.MEDIUM
                )
                
        except Exception as e:
            logger.error(f"Error generating context-inspired ideas: {e}")
    
    async def _generate_memory_inspired_ideas(self, memory_concepts: List[Dict[str, Any]]):
        """Generate ideas inspired by memories"""
        try:
            # Extract texts from memory concepts
            concepts = [m["text"] for m in memory_concepts if m.get("text")]
            
            if not concepts:
                return
            
            # Use memory-appropriate technique
            technique = "analogical_reasoning" if len(concepts) > 2 else "conceptual_blending"
            
            # Generate idea
            idea = await self.original_engine.generate_novel_idea(
                technique=technique,
                concepts=concepts
            )
            
            if idea:
                # Add memory references
                idea.related_memories = [m.get("id", "unknown") for m in memory_concepts]
                
                # Send update
                await self.send_context_update(
                    update_type="memory_inspired_idea_generated",
                    data={
                        "idea_id": idea.id,
                        "idea_title": idea.title,
                        "memory_count": len(memory_concepts),
                        "technique": technique
                    },
                    priority=ContextPriority.LOW
                )
                
        except Exception as e:
            logger.error(f"Error generating memory-inspired ideas: {e}")
    
    async def _adjust_creative_approach(self, emotion_name: str, intensity: float):
        """Adjust creative approach based on emotional state"""
        # Map emotions to creative techniques
        emotion_technique_map = {
            "Joy": ["random_stimulus", "conceptual_blending"],
            "Curiosity": ["bisociation", "perspective_shifting"],
            "Frustration": ["constraint_relaxation", "pattern_breaking"],
            "Excitement": ["random_stimulus", "provocation"],
            "Contemplation": ["analogical_reasoning", "conceptual_reasoning"],
            "Anxiety": ["perspective_shifting", "pattern_breaking"]
        }
        
        preferred_techniques = emotion_technique_map.get(emotion_name, ["conceptual_blending"])
        
        # Adjust source probabilities based on emotion
        if hasattr(self.original_engine, "source_probabilities"):
            if emotion_name in ["Joy", "Excitement"]:
                self.original_engine.source_probabilities[self.original_engine.ObservationSource.SENSORY] = 0.2
                self.original_engine.source_probabilities[self.original_engine.ObservationSource.PATTERN] = 0.1
            elif emotion_name in ["Contemplation", "Curiosity"]:
                self.original_engine.source_probabilities[self.original_engine.ObservationSource.MEMORY] = 0.2
                self.original_engine.source_probabilities[self.original_engine.ObservationSource.PATTERN] = 0.15
    
    async def _generate_pattern_based_ideas(self, pattern_type: str, pattern_details: Dict[str, Any]):
        """Generate ideas based on detected patterns"""
        try:
            # Map pattern types to techniques
            pattern_technique_map = {
                "behavioral": "pattern_breaking",
                "conceptual": "conceptual_blending",
                "temporal": "perspective_shifting",
                "causal": "causal_reasoning",
                "emotional": "random_stimulus"
            }
            
            technique = pattern_technique_map.get(pattern_type, "bisociation")
            
            # Extract concepts from pattern
            concepts = []
            if "elements" in pattern_details:
                concepts.extend(pattern_details["elements"][:3])
            if "description" in pattern_details:
                concepts.append(pattern_details["description"])
            
            if concepts:
                idea = await self.original_engine.generate_novel_idea(
                    technique=technique,
                    concepts=concepts
                )
                
                if idea:
                    # Track pattern
                    self.cross_module_patterns.append({
                        "pattern_type": pattern_type,
                        "idea_id": idea.id,
                        "description": f"Pattern-inspired idea from {pattern_type} pattern",
                        "timestamp": datetime.now().isoformat()
                    })
                    
        except Exception as e:
            logger.error(f"Error generating pattern-based ideas: {e}")
    
    def _map_need_to_creative_domain(self, need: str) -> Optional[str]:
        """Map needs to creative domains"""
        need_domain_map = {
            "knowledge": "science",
            "connection": "social_systems",
            "control_expression": "business",
            "novelty": "art",
            "agency": "philosophy",
            "coherence": "psychology",
            "safety": "environment",
            "intimacy": "psychology",
            "pleasure_indulgence": "entertainment"
        }
        return need_domain_map.get(need)
    
    async def _focus_creative_domains(self, domains: List[str]):
        """Focus creative generation on specific domains"""
        # This could modify the engine's domain selection probabilities
        # For now, just track the focus
        logger.info(f"Focusing creative generation on domains: {domains}")
    
    async def _integrate_new_knowledge(self, new_concepts: List[Any], updated_models: List[Any]):
        """Integrate new knowledge into creative processes"""
        # Could trigger knowledge-based idea generation
        if new_concepts:
            # Use new concepts for idea generation
            concept_texts = [str(c) for c in new_concepts[:3]]
            if concept_texts:
                await self.original_engine.generate_novel_idea(
                    technique="conceptual_reasoning",
                    concepts=concept_texts
                )
    
    async def _process_creative_reflection(self, reflection_text: str):
        """Process creative reflections for meta-insights"""
        # Extract insights from reflection
        if "pattern" in reflection_text.lower() or "insight" in reflection_text.lower():
            self.cross_module_patterns.append({
                "pattern_type": "meta_creative",
                "description": reflection_text[:100],
                "timestamp": datetime.now().isoformat()
            })
    
    async def _analyze_creative_potential(self, user_input: str) -> Dict[str, float]:
        """Analyze creative potential in user input"""
        creative_indicators = {
            "question_marks": user_input.count("?") * 0.1,
            "creative_words": sum(0.15 for word in ["imagine", "create", "idea", "what if", "design", "invent", "dream"]
                                if word in user_input.lower()),
            "complexity": min(1.0, len(user_input.split()) / 50) * 0.3,
            "openness": 0.3 if any(word in user_input.lower() for word in ["could", "might", "maybe", "perhaps"]) else 0
        }
        
        creative_potential = min(1.0, sum(creative_indicators.values()))
        
        return {
            "creative_potential": creative_potential,
            "indicators": creative_indicators,
            "suggests_creativity": creative_potential > 0.5
        }
    
    async def _determine_creative_approach(self, context: SharedContext, messages: List[Any], 
                                         creative_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal creative approach based on full context"""
        approach = {
            "technique": None,
            "domain": None,
            "concepts": [],
            "reasoning": ""
        }
        
        # High creative potential - use advanced techniques
        if creative_analysis.get("creative_potential", 0) > 0.7:
            # Check for reasoning availability
            if context.reasoning_context:
                approach["technique"] = "conceptual_reasoning"
                approach["reasoning"] = "High creative potential with reasoning available"
            else:
                approach["technique"] = "conceptual_blending"
                approach["reasoning"] = "High creative potential, using blending"
        
        # Medium creative potential - use contextual techniques
        elif creative_analysis.get("creative_potential", 0) > 0.4:
            # Check emotional state
            if context.emotional_state:
                primary_emotion = context.emotional_state.get("primary_emotion", {}).get("name")
                if primary_emotion in ["Curiosity", "Excitement"]:
                    approach["technique"] = "bisociation"
                elif primary_emotion in ["Frustration", "Confusion"]:
                    approach["technique"] = "constraint_relaxation"
                else:
                    approach["technique"] = "perspective_shifting"
                approach["reasoning"] = f"Medium potential with {primary_emotion} emotion"
        
        # Extract concepts from input
        if context.user_input:
            approach["concepts"] = self._extract_concepts_from_text(context.user_input)[:5]
        
        # Determine domain from context
        if context.session_context and "current_topic" in context.session_context:
            topic = context.session_context["current_topic"]
            approach["domain"] = self._map_topic_to_domain(topic)
        
        return approach
    
    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple extraction - in production would use NLP
        words = text.split()
        # Filter out common words
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "was", "are", "were"}
        concepts = [w for w in words if w.lower() not in stopwords and len(w) > 3]
        return concepts[:5]  # Top 5 concepts
    
    def _identify_emotional_tensions(self, emotional_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify creative tensions in emotional state"""
        tensions = []
        
        # Check for conflicting emotions
        if "emotion_blend" in emotional_state:
            blend = emotional_state["emotion_blend"]
            if len(blend) >= 2:
                # Find opposing emotions
                opposing_pairs = [
                    ("Joy", "Sadness"),
                    ("Trust", "Fear"),
                    ("Excitement", "Anxiety")
                ]
                for pair in opposing_pairs:
                    if any(e["name"] == pair[0] for e in blend) and any(e["name"] == pair[1] for e in blend):
                        tensions.append({
                            "type": "emotional_opposition",
                            "elements": list(pair),
                            "text": f"Tension between {pair[0]} and {pair[1]}"
                        })
        
        return tensions
    
    def _extract_memory_patterns(self, memory_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract patterns from memory context"""
        patterns = []
        
        if "retrieved_memories" in memory_context:
            memories = memory_context["retrieved_memories"]
            # Look for recurring themes
            themes = {}
            for memory in memories:
                if "tags" in memory:
                    for tag in memory["tags"]:
                        themes[tag] = themes.get(tag, 0) + 1
            
            # Patterns from recurring themes
            for theme, count in themes.items():
                if count >= 2:
                    patterns.append({
                        "type": "memory_theme",
                        "theme": theme,
                        "text": f"Recurring theme: {theme}",
                        "strength": count / len(memories)
                    })
        
        return patterns
    
    def _score_creative_seed(self, seed: Any) -> float:
        """Score a creative seed's potential"""
        if isinstance(seed, dict):
            # Score based on seed properties
            score = 0.5  # Base score
            if "type" in seed:
                if seed["type"] in ["emotional_opposition", "memory_theme"]:
                    score += 0.2
            if "strength" in seed:
                score += seed["strength"] * 0.3
            return min(1.0, score)
        return 0.5  # Default score
    
    def _summarize_context(self, context: SharedContext) -> str:
        """Create a summary of context for tracking"""
        elements = []
        
        if context.emotional_state:
            emotion = context.emotional_state.get("primary_emotion", {}).get("name")
            if emotion:
                elements.append(f"emotion:{emotion}")
        
        if context.session_context:
            topic = context.session_context.get("current_topic")
            if topic:
                elements.append(f"topic:{topic}")
        
        if context.user_id:
            elements.append(f"user:{context.user_id}")
        
        return "_".join(elements) if elements else "general_context"
    
    async def _analyze_idea_patterns(self, ideas: List[Any]) -> Dict[str, Any]:
        """Analyze patterns in generated ideas"""
        patterns = {
            "technique_distribution": {},
            "domain_distribution": {},
            "average_novelty": 0.0,
            "trending_techniques": []
        }
        
        if not ideas:
            return patterns
        
        # Analyze techniques
        for idea in ideas:
            technique = idea.technique_used
            patterns["technique_distribution"][technique] = patterns["technique_distribution"].get(technique, 0) + 1
        
        # Analyze domains
        for idea in ideas:
            domain = idea.domain
            patterns["domain_distribution"][domain] = patterns["domain_distribution"].get(domain, 0) + 1
        
        # Calculate average novelty
        novelty_scores = [idea.novelty_score for idea in ideas if hasattr(idea, "novelty_score")]
        if novelty_scores:
            patterns["average_novelty"] = sum(novelty_scores) / len(novelty_scores)
        
        # Find trending techniques (most used)
        if patterns["technique_distribution"]:
            sorted_techniques = sorted(patterns["technique_distribution"].items(), key=lambda x: x[1], reverse=True)
            patterns["trending_techniques"] = [t[0] for t in sorted_techniques[:3]]
        
        return patterns
    
    async def _evaluate_creative_momentum(self, context: SharedContext, recent_ideas: List[Any]) -> Dict[str, Any]:
        """Evaluate current creative momentum"""
        momentum = {
            "level": 0.0,
            "trend": "stable",
            "factors": {}
        }
        
        # Factor 1: Idea generation rate
        if recent_ideas:
            # Check timestamps to calculate rate
            idea_count = len(recent_ideas)
            if idea_count > 5:
                momentum["factors"]["high_generation_rate"] = 0.3
            elif idea_count > 2:
                momentum["factors"]["moderate_generation_rate"] = 0.2
        
        # Factor 2: Novelty scores
        if recent_ideas:
            avg_novelty = sum(i.novelty_score for i in recent_ideas if hasattr(i, "novelty_score")) / len(recent_ideas)
            if avg_novelty > 0.7:
                momentum["factors"]["high_novelty"] = 0.3
            elif avg_novelty > 0.5:
                momentum["factors"]["moderate_novelty"] = 0.2
        
        # Factor 3: Context alignment
        if self.context_inspired_ideas:
            context_ratio = len([i for i in recent_ideas if i.id in self.context_inspired_ideas]) / max(1, len(recent_ideas))
            if context_ratio > 0.5:
                momentum["factors"]["strong_context_alignment"] = 0.2
        
        # Calculate overall momentum
        momentum["level"] = sum(momentum["factors"].values())
        
        # Determine trend
        if momentum["level"] > 0.6:
            momentum["trend"] = "increasing"
        elif momentum["level"] < 0.3:
            momentum["trend"] = "decreasing"
        
        return momentum
    
    async def _identify_unexplored_spaces(self, context: SharedContext) -> List[Dict[str, str]]:
        """Identify unexplored creative spaces"""
        unexplored = []
        
        # Get used techniques and domains
        recent_ideas = await self.original_engine.get_generated_ideas(limit=20)
        used_techniques = set(i.technique_used for i in recent_ideas if hasattr(i, "technique_used"))
        used_domains = set(i.domain for i in recent_ideas if hasattr(i, "domain"))
        
        # Find unused techniques
        all_techniques = set(self.original_engine.context.techniques.keys())
        unused_techniques = all_techniques - used_techniques
        
        for technique in unused_techniques:
            unexplored.append({
                "type": "technique",
                "name": technique,
                "description": self.original_engine.context.techniques.get(technique, "")
            })
        
        # Find unused domains
        all_domains = set(self.original_engine.context.domains)
        unused_domains = all_domains - used_domains
        
        for domain in unused_domains:
            unexplored.append({
                "type": "domain",
                "name": domain,
                "description": f"Unexplored domain: {domain}"
            })
        
        return unexplored[:5]  # Top 5 unexplored spaces
    
    async def _analyze_cross_module_creativity(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Analyze cross-module creative opportunities"""
        opportunities = []
        
        # Look for memory + emotion combinations
        has_memory = any(m.get("module_name") == "memory_core" for m in messages)
        has_emotion = any(m.get("module_name") == "emotional_core" for m in messages)
        
        if has_memory and has_emotion:
            opportunities.append({
                "type": "memory_emotion_blend",
                "description": "Blend emotional memories for creative insight",
                "potential": 0.8
            })
        
        # Look for goal + reasoning combinations
        has_goal = any(m.get("module_name") == "goal_manager" for m in messages)
        has_reasoning = any(m.get("module_name") == "reasoning_core" for m in messages)
        
        if has_goal and has_reasoning:
            opportunities.append({
                "type": "goal_reasoning_synthesis",
                "description": "Use reasoning to creatively achieve goals",
                "potential": 0.7
            })
        
        return opportunities
    
    async def _get_contextually_relevant_ideas(self, context: SharedContext, 
                                             filter_criteria: Dict[str, Any]) -> List[Any]:
        """Get ideas relevant to current context"""
        all_ideas = await self.original_engine.get_generated_ideas(limit=20)
        
        # Score ideas by relevance
        scored_ideas = []
        for idea in all_ideas:
            score = 0.0
            
            # Check if context-inspired
            if idea.id in self.context_inspired_ideas:
                score += 0.3
            
            # Check domain match
            if hasattr(idea, "domain") and context.session_context:
                topic = context.session_context.get("current_topic", "")
                if topic and self._map_topic_to_domain(topic) == idea.domain:
                    score += 0.2
            
            # Check novelty threshold
            if hasattr(idea, "novelty_score") and idea.novelty_score > filter_criteria.get("min_relevance", 0.5):
                score += idea.novelty_score * 0.5
            
            if score > 0:
                scored_ideas.append((score, idea))
        
        # Sort by score and return top ideas
        scored_ideas.sort(key=lambda x: x[0], reverse=True)
        max_ideas = filter_criteria.get("max_ideas", 3)
        
        return [idea for score, idea in scored_ideas[:max_ideas]]
    
    async def _generate_exploration_suggestions(self, context: SharedContext, 
                                              messages: List[Any]) -> List[str]:
        """Generate suggestions for creative exploration"""
        suggestions = []
        
        # Based on unexplored spaces
        unexplored = await self._identify_unexplored_spaces(context)
        for space in unexplored[:2]:
            if space["type"] == "technique":
                suggestions.append(f"Try {space['name']} technique for fresh perspectives")
            elif space["type"] == "domain":
                suggestions.append(f"Explore ideas in the {space['name']} domain")
        
        # Based on cross-module opportunities
        opportunities = await self._analyze_cross_module_creativity(messages)
        for opp in opportunities[:1]:
            suggestions.append(opp["description"])
        
        # Based on emotional state
        if context.emotional_state:
            emotion = context.emotional_state.get("primary_emotion", {}).get("name")
            if emotion == "Curiosity":
                suggestions.append("Your curiosity could lead to breakthrough bisociations")
            elif emotion == "Frustration":
                suggestions.append("Channel frustration into constraint-breaking innovations")
        
        return suggestions[:3]  # Max 3 suggestions
    
    def _summarize_momentum(self, momentum_data: Dict[str, Any]) -> str:
        """Create a text summary of creative momentum"""
        level = momentum_data.get("level", 0)
        trend = momentum_data.get("trend", "stable")
        
        if level > 0.7:
            return f"Creative momentum is high and {trend} - ride this wave of innovation!"
        elif level > 0.4:
            return f"Moderate creative momentum, {trend} trend - good foundation for exploration"
        else:
            return f"Creative momentum is building - consider new techniques or domains"
    
    def _map_topic_to_domain(self, topic: str) -> str:
        """Map conversation topic to creative domain"""
        topic_lower = topic.lower()
        
        if any(word in topic_lower for word in ["tech", "computer", "ai", "software"]):
            return "technology"
        elif any(word in topic_lower for word in ["art", "music", "creative", "design"]):
            return "art"
        elif any(word in topic_lower for word in ["science", "research", "study"]):
            return "science"
        elif any(word in topic_lower for word in ["business", "work", "career"]):
            return "business"
        elif any(word in topic_lower for word in ["mind", "think", "feel", "emotion"]):
            return "psychology"
        elif any(word in topic_lower for word in ["life", "existence", "meaning"]):
            return "philosophy"
        else:
            return "general"
    
    def _select_technique_for_seeds(self, creative_seeds: Dict[str, Any]) -> str:
        """Select best technique based on creative seeds"""
        # Analyze seed types
        has_concepts = bool(creative_seeds.get("concepts"))
        has_patterns = bool(creative_seeds.get("patterns"))
        has_tensions = bool(creative_seeds.get("tensions"))
        
        # Select technique based on seed types
        if has_tensions:
            return "constraint_relaxation"
        elif has_patterns:
            return "pattern_breaking"
        elif has_concepts and len(creative_seeds["concepts"]) > 2:
            return "conceptual_blending"
        else:
            return "bisociation"
    
    # Delegate all other methods to the original engine
    def __getattr__(self, name):
        """Delegate any missing methods to the original engine"""
        return getattr(self.original_engine, name)
