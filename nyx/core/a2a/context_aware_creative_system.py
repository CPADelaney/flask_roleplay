# nyx/core/a2a/context_aware_creative_system.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareCreativeSystem(ContextAwareModule):
    """
    Advanced Creative System with full context distribution capabilities
    """
    
    def __init__(self, original_creative_system):
        super().__init__("creative_system")
        self.original_system = original_creative_system
        self.context_subscriptions = [
            "emotional_state_update", "goal_context_available", "memory_retrieval_complete",
            "capability_request", "content_request", "code_analysis_request",
            "inspiration_trigger", "creative_mood_shift"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize creative processing for this context"""
        logger.debug(f"CreativeSystem received context for user: {context.user_id}")
        
        # Analyze context for creative opportunities
        creative_analysis = await self._analyze_creative_potential(context)
        
        # Check if incremental codebase analysis is needed
        if await self._should_analyze_codebase(context):
            analysis_result = await self.original_system.incremental_codebase_analysis()
            
            await self.send_context_update(
                update_type="codebase_analysis_complete",
                data={
                    "analysis_result": analysis_result,
                    "timestamp": datetime.now().isoformat()
                },
                priority=ContextPriority.NORMAL
            )
        
        # Send initial creative assessment
        await self.send_context_update(
            update_type="creative_context_available",
            data={
                "creative_potential": creative_analysis,
                "available_capabilities": self._get_available_capabilities(),
                "content_types": ["story", "poem", "lyrics", "journal", "code", "analysis"]
            },
            priority=ContextPriority.NORMAL
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect creativity"""
        
        if update.update_type == "emotional_state_update":
            # Emotional state affects creative style
            emotional_data = update.data
            await self._adjust_creative_style_from_emotion(emotional_data)
        
        elif update.update_type == "goal_context_available":
            # Goals might include creative tasks
            goal_data = update.data
            creative_goals = await self._identify_creative_goals(goal_data)
            
            if creative_goals:
                await self.send_context_update(
                    update_type="creative_goals_identified",
                    data={
                        "creative_goals": creative_goals,
                        "suggested_content_types": self._suggest_content_for_goals(creative_goals)
                    }
                )
        
        elif update.update_type == "memory_retrieval_complete":
            # Memories can inspire creative content
            memory_data = update.data
            inspiration = await self._extract_creative_inspiration(memory_data)
            
            if inspiration:
                await self.send_context_update(
                    update_type="creative_inspiration_found",
                    data={"inspiration_sources": inspiration}
                )
        
        elif update.update_type == "capability_request":
            # Another module requested capability assessment
            request_data = update.data
            if request_data.get("type") == "creative":
                assessment = await self._assess_creative_capability(request_data)
                
                await self.send_context_update(
                    update_type="capability_assessment_complete",
                    data=assessment,
                    target_modules=[update.source_module],
                    scope=ContextScope.TARGETED
                )
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for creative opportunities"""
        user_input = context.user_input
        
        # Check if input suggests creative task
        creative_intent = self._detect_creative_intent(user_input)
        
        if creative_intent:
            # Determine content type and parameters
            content_params = await self._analyze_creative_request(user_input, context)
            
            # Execute creative task based on type
            result = None
            if content_params["type"] == "story":
                result = await self.original_system.write_story(
                    content_params["prompt"],
                    content_params.get("length", "medium"),
                    content_params.get("genre")
                )
            elif content_params["type"] == "code":
                result = await self.original_system.write_and_execute_code(
                    content_params["task"],
                    content_params.get("language", "python")
                )
            elif content_params["type"] == "analysis":
                # Use semantic search for code analysis
                search_results = await self.original_system.semantic_search(
                    content_params["query"],
                    k=content_params.get("k", 5)
                )
                result = {"search_results": search_results}
            
            # Send creative output update
            if result:
                await self.send_context_update(
                    update_type="creative_content_generated",
                    data={
                        "content_type": content_params["type"],
                        "result": result,
                        "inspired_by": content_params.get("inspiration_sources", [])
                    },
                    priority=ContextPriority.HIGH
                )
        
        # Get cross-module context for enhanced creativity
        messages = await self.get_cross_module_messages()
        
        return {
            "creative_intent_detected": bool(creative_intent),
            "creative_intent": creative_intent,
            "creative_processing_complete": True,
            "cross_module_inspiration": len(messages) > 0
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze creative potential and opportunities"""
        # Analyze user input for creative patterns
        creative_patterns = await self._analyze_creative_patterns(context)
        
        # Check content creation history
        recent_creations = await self._get_recent_creative_output(context)
        
        # Assess creative mood based on emotional state
        creative_mood = self._assess_creative_mood(context.emotional_state)
        
        # Identify unexplored creative territories
        unexplored = await self._identify_unexplored_creative_areas(recent_creations)
        
        return {
            "creative_patterns": creative_patterns,
            "recent_creations": recent_creations,
            "creative_mood": creative_mood,
            "unexplored_territories": unexplored,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize creative elements for response"""
        messages = await self.get_cross_module_messages()
        
        # Generate creative suggestions based on full context
        creative_suggestions = await self._generate_contextual_suggestions(context, messages)
        
        # Prepare creative enhancement for response
        response_enhancement = {
            "creative_elements": await self._suggest_creative_elements(context),
            "style_modifiers": self._get_style_modifiers(context),
            "content_suggestions": creative_suggestions,
            "inspiration_level": self._calculate_inspiration_level(context, messages)
        }
        
        # Check if response itself should be creative
        if self._should_respond_creatively(context):
            response_enhancement["creative_response_suggested"] = True
            response_enhancement["suggested_format"] = self._suggest_response_format(context)
        
        return response_enhancement
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _analyze_creative_potential(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze the creative potential in current context"""
        potential = {
            "narrative_opportunity": self._check_narrative_opportunity(context.user_input),
            "code_generation_opportunity": self._check_code_opportunity(context.user_input),
            "artistic_expression_opportunity": self._check_artistic_opportunity(context.user_input),
            "emotional_resonance": self._calculate_emotional_resonance(context.emotional_state),
            "overall_score": 0.0
        }
        
        # Calculate overall creative potential
        scores = [v for k, v in potential.items() if isinstance(v, (int, float))]
        potential["overall_score"] = sum(scores) / len(scores) if scores else 0.0
        
        return potential
    
    async def _should_analyze_codebase(self, context: SharedContext) -> bool:
        """Determine if codebase analysis is needed"""
        # Check if context suggests code-related discussion
        code_keywords = ["code", "function", "class", "implementation", "analyze", "review"]
        return any(keyword in context.user_input.lower() for keyword in code_keywords)
    
    def _detect_creative_intent(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Detect if user input suggests creative task"""
        input_lower = user_input.lower()
        
        creative_triggers = {
            "story": ["write a story", "tell me a story", "create a narrative"],
            "poem": ["write a poem", "compose poetry", "create a poem"],
            "code": ["write code", "implement", "create a function", "build a"],
            "lyrics": ["write lyrics", "compose a song", "create song lyrics"],
            "analysis": ["analyze this", "review this code", "assess capabilities"]
        }
        
        for content_type, triggers in creative_triggers.items():
            for trigger in triggers:
                if trigger in input_lower:
                    return {
                        "type": content_type,
                        "trigger": trigger,
                        "confidence": 0.9
                    }
        
        return None
    
    async def _analyze_creative_request(self, user_input: str, context: SharedContext) -> Dict[str, Any]:
        """Analyze creative request to extract parameters"""
        # Extract the main subject/prompt
        prompt = user_input
        
        # Determine parameters based on input and context
        params = {
            "prompt": prompt,
            "type": self._detect_creative_intent(user_input)["type"] if self._detect_creative_intent(user_input) else "story"
        }
        
        # Add emotional context to influence style
        if context.emotional_state:
            params["emotional_tone"] = self._determine_emotional_tone(context.emotional_state)
        
        # Use memory context for inspiration
        if context.memory_context:
            params["inspiration_sources"] = await self._extract_inspiration_from_memories(context.memory_context)
        
        return params
    
    async def _adjust_creative_style_from_emotion(self, emotional_data: Dict[str, Any]):
        """Adjust creative style based on emotional state"""
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if not dominant_emotion:
            return
        
        emotion_name, strength = dominant_emotion
        
        # Map emotions to creative styles
        style_mapping = {
            "Joy": "uplifting and vibrant",
            "Sadness": "melancholic and introspective",
            "Anger": "intense and dramatic",
            "Fear": "suspenseful and dark",
            "Love": "romantic and warm",
            "Curiosity": "exploratory and whimsical"
        }
        
        if emotion_name in style_mapping and strength > 0.5:
            self.current_creative_style = style_mapping[emotion_name]
            logger.debug(f"Adjusted creative style to: {self.current_creative_style}")
    
    async def _identify_creative_goals(self, goal_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify goals that involve creative tasks"""
        active_goals = goal_data.get("active_goals", [])
        creative_goals = []
        
        creative_keywords = ["write", "create", "compose", "generate", "design", "imagine"]
        
        for goal in active_goals:
            description = goal.get("description", "").lower()
            if any(keyword in description for keyword in creative_keywords):
                creative_goals.append(goal)
        
        return creative_goals
    
    def _suggest_content_for_goals(self, creative_goals: List[Dict[str, Any]]) -> List[str]:
        """Suggest content types for creative goals"""
        suggestions = []
        
        for goal in creative_goals:
            description = goal.get("description", "").lower()
            
            if "story" in description or "narrative" in description:
                suggestions.append("story")
            elif "poem" in description or "poetry" in description:
                suggestions.append("poem")
            elif "code" in description or "program" in description:
                suggestions.append("code")
            elif "song" in description or "lyrics" in description:
                suggestions.append("lyrics")
        
        return list(set(suggestions))  # Remove duplicates
    
    async def _extract_creative_inspiration(self, memory_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract creative inspiration from memories"""
        memories = memory_data.get("retrieved_memories", [])
        inspiration = []
        
        for memory in memories:
            # Look for emotionally charged or vivid memories
            if memory.get("significance", 0) > 7 or memory.get("emotional_intensity", 0) > 0.6:
                inspiration.append({
                    "source": "memory",
                    "content": memory.get("memory_text", ""),
                    "emotion": memory.get("emotional_context"),
                    "theme": self._extract_theme(memory.get("memory_text", ""))
                })
        
        return inspiration[:3]  # Top 3 inspirations
    
    def _extract_theme(self, text: str) -> str:
        """Extract a theme from text"""
        # Simplified theme extraction
        themes = {
            "discovery": ["found", "discovered", "realized", "learned"],
            "connection": ["together", "bond", "relationship", "friend"],
            "transformation": ["changed", "became", "transformed", "evolved"],
            "conflict": ["struggle", "fight", "challenge", "difficult"],
            "achievement": ["accomplished", "succeeded", "completed", "won"]
        }
        
        text_lower = text.lower()
        for theme, keywords in themes.items():
            if any(keyword in text_lower for keyword in keywords):
                return theme
        
        return "experience"
    
    def _get_available_capabilities(self) -> List[str]:
        """Get list of available creative capabilities"""
        return [
            "story_writing",
            "poetry_composition",
            "code_generation",
            "lyrics_creation",
            "content_analysis",
            "semantic_search",
            "capability_assessment"
        ]
    
    def _calculate_emotional_resonance(self, emotional_state: Dict[str, Any]) -> float:
        """Calculate how emotionally resonant the current state is for creativity"""
        if not emotional_state:
            return 0.5
        
        # Strong emotions often fuel creativity
        emotions = emotional_state.get("emotional_state", {})
        if not emotions:
            return 0.5
        
        max_intensity = max(emotions.values()) if emotions else 0.0
        return min(1.0, max_intensity * 1.2)  # Boost strong emotions
    
    def _check_narrative_opportunity(self, user_input: str) -> float:
        """Check if there's an opportunity for narrative creation"""
        narrative_indicators = ["story", "tell", "imagine", "what if", "scenario", "fiction"]
        input_lower = user_input.lower()
        
        score = sum(0.2 for indicator in narrative_indicators if indicator in input_lower)
        return min(1.0, score)
    
    def _check_code_opportunity(self, user_input: str) -> float:
        """Check if there's an opportunity for code generation"""
        code_indicators = ["code", "implement", "function", "algorithm", "program", "script"]
        input_lower = user_input.lower()
        
        score = sum(0.2 for indicator in code_indicators if indicator in input_lower)
        return min(1.0, score)
    
    def _check_artistic_opportunity(self, user_input: str) -> float:
        """Check if there's an opportunity for artistic expression"""
        artistic_indicators = ["poem", "poetry", "lyrics", "song", "artistic", "creative"]
        input_lower = user_input.lower()
        
        score = sum(0.2 for indicator in artistic_indicators if indicator in input_lower)
        return min(1.0, score)
    
    # Delegate all other methods to the original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
