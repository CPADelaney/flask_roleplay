# nyx/nyx_decision_engine.py

import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from nyx.memory_system import NyxMemorySystem
from nyx.user_model_manager import UserModelManager
from nyx.llm_integration import generate_text_completion

logger = logging.getLogger(__name__)

class NyxDecisionEngine:
    """
    Advanced decision-making engine for Nyx that combines:
    - User model information
    - Memory-based context
    - Narrative structure awareness
    - Adaptive personality
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.memory_system = NyxMemorySystem(user_id, conversation_id)
        self.user_model_manager = UserModelManager(user_id, conversation_id)
        
        # Decision history for the current session
        self.decision_history = []
        
    async def get_response(
        self, 
        user_message: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a contextually appropriate response from Nyx.
        
        Args:
            user_message: The user's message
            context: Additional context (environment, NPCs, etc.)
            
        Returns:
            Response object with text and metadata
        """
        # 1. Get relevant memories
        memories = await self._get_relevant_memories(user_message, context)
        
        # 2. Get user model guidance
        user_guidance = await self.user_model_manager.get_response_guidance()
        
        # 3. Get or generate current emotional state
        emotional_state = await self._get_emotional_state(user_message, context)
        
        # 4. Get narrative context
        narrative_context = await self._get_narrative_context(context)
        
        # 5. Detect potential user revelations
        user_revelations = await self._detect_user_revelations(user_message, context)
        
        # 6. Construct decision parameters
        decision_params = {
            "user_message": user_message,
            "memories": [m["memory_text"] for m in memories],
            "user_guidance": user_guidance,
            "emotional_state": emotional_state,
            "narrative_context": narrative_context,
            "user_revelations": user_revelations,
            "context": context,
            "decision_history": self._format_decision_history()
        }
        
        # 7. Generate adaptive response
        response = await self._generate_response(decision_params)
        
        # 8. Process and analyze response for learning
        processed_response = await self._process_response(
            user_message, 
            response, 
            decision_params
        )
        
        # 9. Update decision history
        self._update_decision_history(processed_response)
        
        return processed_response
    
    async def _get_relevant_memories(
        self, 
        user_message: str, 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get memories relevant to current interaction."""
        # Query current and user-level memories
        memories = await self.memory_system.retrieve_memories(
            query=user_message,
            scopes=["game", "user", "global"],
            memory_types=["observation", "reflection", "abstraction"],
            limit=10,
            min_significance=3,
            context=context
        )
        
        # Also specifically get player model reflections
        reflections = await self.memory_system.retrieve_memories(
            query="player personality preferences behavior",
            scopes=["user", "game"],
            memory_types=["reflection"],
            limit=3,
            min_significance=4
        )
        
        # Add reflections to memories if not already present
        for reflection in reflections:
            if not any(m["id"] == reflection["id"] for m in memories):
                memories.append(reflection)
        
        return memories
    
    async def _get_emotional_state(
        self,
        user_message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get or generate Nyx's current emotional state."""
        # Check if context already has emotional state
        if "nyx_emotional_state" in context:
            return context["nyx_emotional_state"]
        
        # Check for recent emotional state in database
        async with asyncio.Pool.create_pool(dsn=get_db_connection()) as pool:
            async with pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT emotional_state FROM NyxAgentState
                    WHERE user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
                
                if row and row["emotional_state"]:
                    return json.loads(row["emotional_state"])
        
        # Default emotional state
        return {
            "primary_emotion": "neutral",
            "intensity": 0.3,
            "secondary_emotions": {
                "curiosity": 0.4,
                "amusement": 0.2
            },
            "confidence": 0.7
        }
    
    async def _get_narrative_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get current narrative context and state."""
        narrative_context = {}
        
        # Extract from provided context
        if "narrative_context" in context:
            return context["narrative_context"]
        
        # Retrieve from database
        async with asyncio.Pool.create_pool(dsn=get_db_connection()) as pool:
            async with pool.acquire() as conn:
                # Get narrative arcs
                arc_row = await conn.fetchrow("""
                    SELECT value FROM CurrentRoleplay 
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'NyxNarrativeArcs'
                """, self.user_id, self.conversation_id)
                
                if arc_row:
                    narrative_context["arcs"] = json.loads(arc_row["value"])
                
                # Get current plot stage
                plot_row = await conn.fetchrow("""
                    SELECT value FROM CurrentRoleplay 
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentPlotStage'
                """, self.user_id, self.conversation_id)
                
                if plot_row:
                    narrative_context["plot_stage"] = plot_row["value"]
                
                # Get tension level
                tension_row = await conn.fetchrow("""
                    SELECT value FROM CurrentRoleplay 
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentTension'
                """, self.user_id, self.conversation_id)
                
                if tension_row:
                    try:
                        narrative_context["tension"] = int(tension_row["value"])
                    except (ValueError, TypeError):
                        narrative_context["tension"] = 0
        
        return narrative_context
    
    async def _detect_user_revelations(
        self,
        user_message: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect if user is revealing new information that should be tracked.
        Returns list of detected revelations.
        """
        # Simple keyword-based detection for demonstration
        revelations = []
        
        # Look for potential preference revelations
        lower_message = user_message.lower()
        
        # Check for explicit kink mentions
        kink_keywords = {
            "ass": ["ass", "booty", "behind", "rear"],
            "feet": ["feet", "foot", "toes"],
            "goth": ["goth", "gothic", "dark", "black clothes"],
            "tattoos": ["tattoo", "ink", "inked"],
            "piercings": ["piercing", "pierced", "stud", "ring"],
            "latex": ["latex", "rubber", "shiny"],
            "leather": ["leather", "leathery"],
            "humiliation": ["humiliate", "embarrassed", "ashamed", "pathetic"],
            "submission": ["submit", "obey", "serve", "kneel"]
        }
        
        for kink, keywords in kink_keywords.items():
            if any(keyword in lower_message for keyword in keywords):
                # Check if this seems to be a positive statement
                sentiment = self._analyze_simple_sentiment(lower_message)
                if sentiment != "negative":
                    revelations.append({
                        "type": "kink_preference",
                        "kink": kink,
                        "intensity": 0.7 if sentiment == "positive" else 0.4,
                        "source": "explicit_mention"
                    })
        
        # Check for behavior pattern indicators
        if "don't tell me what to do" in lower_message or "i won't" in lower_message:
            revelations.append({
                "type": "behavior_pattern",
                "pattern": "resistance",
                "intensity": 0.6,
                "source": "explicit_statement"
            })
        
        if "yes mistress" in lower_message or "i'll obey" in lower_message:
            revelations.append({
                "type": "behavior_pattern",
                "pattern": "submission",
                "intensity": 0.8,
                "source": "explicit_statement"
            })
        
        # TODO: Add more sophisticated detection using NLP
        
        return revelations
    
    def _analyze_simple_sentiment(self, text: str) -> str:
        """Very simple sentiment analysis (positive/neutral/negative)."""
        positive_words = ["like", "love", "enjoy", "good", "great", "nice", "yes", "please"]
        negative_words = ["don't", "hate", "dislike", "bad", "worse", "no", "never"]
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"
    
    async def _generate_response(self, params: Dict[str, Any]) -> str:
        """Generate Nyx's response based on decision parameters."""
        # Build context for LLM
        # Get system prompt (Nyx's persona)
        system_prompt = await self._get_system_prompt(params)
        
        # Construct user context
        context_info = self._build_context_info(params)
        
        # Format prompt
        user_prompt = f"""
{context_info}

User: {params['user_message']}

Respond as Nyx, with your response reflecting your understanding of the user's preferences and the current context.
"""
        
        # Call LLM with appropriate framing
        response = await generate_text_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        return response
    
    async def _get_system_prompt(self, params: Dict[str, Any]) -> str:
        """Get system prompt for LLM, potentially customized for user."""
        from nyx.prompts import SYSTEM_PROMPT, PRIVATE_REFLECTION_INSTRUCTIONS
        
        user_guidance = params.get("user_guidance", {})
        
        # Get base personality traits from guidance
        personality_traits = user_guidance.get("personality_traits", {})
        
        # Customize intensity and style based on user model
        customized_prompt = SYSTEM_PROMPT
        
        # Add private reflection instructions
        return customized_prompt + "\n\n" + PRIVATE_REFLECTION_INSTRUCTIONS
    
    def _build_context_info(self, params: Dict[str, Any]) -> str:
        """
        Build context information to include in the prompt.
        This explains the current state, relevant memories, etc.
        """
        context = params.get("context", {})
        memories = params.get("memories", [])
        user_guidance = params.get("user_guidance", {})
        narrative_context = params.get("narrative_context", {})
        user_revelations = params.get("user_revelations", [])
        
        # Format relevant memories
        memory_text = ""
        if memories:
            memory_text = "### Relevant Memories ###\n"
            memory_text += "\n".join([f"- {memory}" for memory in memories[:5]])
            
            # Add reflections if available
            reflections = user_guidance.get("reflections", [])
            if reflections:
                memory_text += "\n\n### Player Reflections ###\n"
                memory_text += "\n".join([f"- {reflection}" for reflection in reflections])
        
        # Format narrative context
        narrative_text = ""
        if narrative_context:
            narrative_text = "### Narrative Context ###\n"
            
            if "plot_stage" in narrative_context:
                narrative_text += f"- Current plot stage: {narrative_context['plot_stage']}\n"
                
            if "tension" in narrative_context:
                narrative_text += f"- Current tension level: {narrative_context['tension']}/10\n"
                
            if "arcs" in narrative_context:
                arcs = narrative_context["arcs"]
                if "active_arcs" in arcs and arcs["active_arcs"]:
                    active_arc = arcs["active_arcs"][0]
                    narrative_text += f"- Active arc: {active_arc.get('name', 'Unknown')}"
                    if "progress" in active_arc:
                        narrative_text += f" (Progress: {active_arc['progress']}%)\n"
        
        # Format user model guidance
        guidance_text = ""
        if user_guidance:
            guidance_text = "### User Guidance ###\n"
            
            top_kinks = user_guidance.get("top_kinks", [])
            if top_kinks:
                kink_str = ", ".join([f"{k} (level {l})" for k, l in top_kinks])
                guidance_text += f"- Top interests: {kink_str}\n"
                
            suggested_intensity = user_guidance.get("suggested_intensity", 0.5)
            guidance_text += f"- Suggested intensity: {suggested_intensity:.1f}/1.0\n"
            
            behavior_patterns = user_guidance.get("behavior_patterns", {})
            if behavior_patterns:
                pattern_str = ", ".join([f"{k}: {v}" for k, v in behavior_patterns.items()])
                guidance_text += f"- Behavior patterns: {pattern_str}"
        
        # Format user revelations
        revelations_text = ""
        if user_revelations:
            revelations_text = "### User Revelations ###\n"
            for revelation in user_revelations:
                if revelation["type"] == "kink_preference":
                    revelations_text += f"- User mentioned interest in: {revelation['kink']} (intensity: {revelation['intensity']:.1f})\n"
                elif revelation["type"] == "behavior_pattern":
                    revelations_text += f"- User showed {revelation['pattern']} behavior (intensity: {revelation['intensity']:.1f})\n"
        
        # Game state
        game_state = ""
        for key, value in context.items():
            if key in ["location", "time_of_day", "npc_present"]:
                game_state += f"- {key}: {value}\n"
        
        if game_state:
            game_state = "### Current Game State ###\n" + game_state
        
        # Combine all sections
        sections = []
        if memory_text:
            sections.append(memory_text)
        if narrative_text:
            sections.append(narrative_text)
        if guidance_text:
            sections.append(guidance_text)
        if revelations_text:
            sections.append(revelations_text)
        if game_state:
            sections.append(game_state)
        
        return "\n\n".join(sections)
    
    async def _process_response(
        self,
        user_message: str,
        response: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process generated response for learning and improvement.
        
        Args:
            user_message: The user's message
            response: Generated response
            params: Decision parameters
            
        Returns:
            Processed response with metadata
        """
        # Extract revelations from params
        user_revelations = params.get("user_revelations", [])
        
        # Process kink revelations
        for revelation in user_revelations:
            if revelation["type"] == "kink_preference":
                await self.user_model_manager.track_kink_preference(
                    kink_name=revelation["kink"],
                    intensity=revelation["intensity"],
                    detected_from=revelation["source"]
                )
            elif revelation["type"] == "behavior_pattern":
                await self.user_model_manager.track_behavior_pattern(
                    pattern_type="response_style",
                    pattern_value=revelation["pattern"],
                    intensity=revelation["intensity"]
                )
        
        # Create memory of this interaction
        memory_text = f"Player said: '{user_message[:100]}...' I responded: '{response[:100]}...'"
        await self.memory_system.add_memory(
            memory_text=memory_text,
            memory_type="observation",
            memory_scope="game",
            significance=4,
            tags=["conversation", "player_interaction"],
            metadata={
                "user_message": user_message,
                "nyx_response": response,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Track conversation pattern
        await self.user_model_manager.track_conversation_response(
            user_message=user_message,
            nyx_response=response,
            conversation_context=params.get("context")
        )
        
        # Analyze if we should generate image
        should_generate_image = self._should_generate_image(response, params)
        
        # Return processed response
        return {
            "text": response,
            "generate_image": should_generate_image,
            "revelations_processed": len(user_revelations),
            "timestamp": datetime.now().isoformat()
        }
    
    def _should_generate_image(self, response: str, params: Dict[str, Any]) -> bool:
        """Determine if an image should be generated."""
        # For now, simple detection of image-worthy content
        # This could be expanded with more sophisticated rules
        
        # Check for keywords indicating action or state changes
        action_keywords = ["now you see", "suddenly", "appears", "reveals", "wearing", "dressed in"]
        has_action = any(keyword in response.lower() for keyword in action_keywords)
        
        # Only trigger occasionally to avoid too many images
        import random
        if has_action and random.random() < 0.3:
            return True
        
        return False
    
    def _format_decision_history(self) -> List[Dict[str, Any]]:
        """Format decision history for context."""
        # Only include last few decisions to avoid context bloat
        return self.decision_history[-5:] if self.decision_history else []
    
    def _update_decision_history(self, response: Dict[str, Any]):
        """Update decision history with new response."""
        # Add to decision history
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "response_text": response["text"][:100],  # Truncate for storage
            "revelations_processed": response.get("revelations_processed", 0)
        }
        
        self.decision_history.append(history_entry)
        
        # Keep only last 20 decisions
        if len(self.decision_history) > 20:
            self.decision_history = self.decision_history[-20:]
