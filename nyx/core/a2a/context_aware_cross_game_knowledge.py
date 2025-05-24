# nyx/core/a2a/context_aware_cross_game_knowledge.py

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import time

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareCrossGameKnowledge(ContextAwareModule):
    """
    Context-aware cross-game knowledge system that learns and transfers insights between games
    """
    
    def __init__(self, original_knowledge_system):
        super().__init__("cross_game_knowledge")
        self.original_system = original_knowledge_system
        self.context_subscriptions = [
            "game_identified", "memory_retrieval_complete", "goal_updates",
            "action_detected", "location_change", "quest_update", "emotional_state_update",
            "learning_milestone", "gameplay_event"
        ]
        self.current_game_context = {}
        self.pending_insights = []
        
    async def on_context_received(self, context: SharedContext):
        """Initialize cross-game processing for this context"""
        logger.debug(f"CrossGameKnowledge received context for user: {context.user_id}")
        
        # Check if we have game identification in context
        game_id = context.session_context.get("game_id")
        game_name = context.session_context.get("game_name")
        
        if game_id or game_name:
            # Load game knowledge
            self.current_game_context = {
                "game_id": game_id,
                "game_name": game_name,
                "start_time": time.time()
            }
            
            # Find similar games
            if game_name and game_name in self.original_system.games:
                similar_games = self.original_system.get_similar_games(game_name)
                
                # Get applicable insights
                insights = self.original_system.get_applicable_insights(game_name)
                
                # Send initial knowledge context
                await self.send_context_update(
                    update_type="cross_game_knowledge_available",
                    data={
                        "current_game": game_name,
                        "similar_games": similar_games,
                        "applicable_insights": insights,
                        "total_knowledge_base": len(self.original_system.games)
                    },
                    priority=ContextPriority.HIGH
                )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        
        if update.update_type == "game_identified":
            # New game identified
            game_data = update.data
            game_name = game_data.get("game_name")
            
            if game_name:
                # Update current game context
                self.current_game_context["game_name"] = game_name
                
                # Check if game exists in knowledge base
                if game_name not in self.original_system.games:
                    # Add new game
                    await self._add_new_game(game_data)
                
                # Get and send relevant knowledge
                await self._process_game_knowledge(game_name)
        
        elif update.update_type == "action_detected":
            # Action detected - check for mechanics
            action_data = update.data
            action_type = action_data.get("action_type")
            
            if action_type and self.current_game_context.get("game_name"):
                # Record mechanic usage
                await self._record_mechanic_usage(action_type, action_data)
        
        elif update.update_type == "memory_retrieval_complete":
            # Memory retrieved - check for cross-game patterns
            memory_data = update.data
            memories = memory_data.get("retrieved_memories", [])
            
            # Analyze memories for cross-game patterns
            patterns = await self._analyze_memory_patterns(memories)
            
            if patterns:
                await self.send_context_update(
                    update_type="cross_game_patterns_found",
                    data={
                        "patterns": patterns,
                        "source": "memory_analysis"
                    }
                )
        
        elif update.update_type == "quest_update":
            # Quest update - check for similar quest patterns
            quest_data = update.data
            await self._analyze_quest_patterns(quest_data)
        
        elif update.update_type == "learning_milestone":
            # Learning milestone - consolidate knowledge
            learning_data = update.data
            await self._consolidate_learning(learning_data)
        
        elif update.update_type == "gameplay_event":
            # Significant gameplay event
            event_data = update.data
            await self._process_gameplay_event(event_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with cross-game knowledge awareness"""
        user_input = context.user_input.lower()
        
        # Check for cross-game queries
        cross_game_analysis = {
            "requests_comparison": any(kw in user_input for kw in ["similar to", "like", "compare", "reminds me of"]),
            "requests_tips": any(kw in user_input for kw in ["tips", "advice", "help", "strategy"]),
            "mentions_other_game": self._detect_game_mention(user_input),
            "asks_about_mechanic": self._detect_mechanic_query(user_input)
        }
        
        result = {
            "cross_game_relevance": any(cross_game_analysis.values()),
            "analysis": cross_game_analysis
        }
        
        # If relevant, get cross-game insights
        if cross_game_analysis["requests_comparison"] or cross_game_analysis["mentions_other_game"]:
            insights = await self._get_relevant_insights(context, cross_game_analysis)
            result["insights"] = insights
            
            # Send insights to other modules
            if insights:
                await self.send_context_update(
                    update_type="cross_game_insights_ready",
                    data={
                        "insights": insights,
                        "triggered_by": "user_query"
                    }
                )
        
        return result
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze current game situation with cross-game knowledge"""
        messages = await self.get_cross_module_messages()
        
        # Get current game state from other modules
        game_state = self._extract_game_state(context, messages)
        
        # Analyze for patterns
        pattern_analysis = await self._analyze_current_patterns(game_state)
        
        # Discover new patterns
        new_patterns = self.original_system.discover_patterns()
        
        # Generate insights based on current situation
        situational_insights = await self._generate_situational_insights(game_state, pattern_analysis)
        
        return {
            "game_state_analysis": game_state,
            "pattern_analysis": pattern_analysis,
            "new_patterns": new_patterns,
            "situational_insights": situational_insights,
            "knowledge_stats": self.original_system.get_stats()
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize cross-game knowledge for response"""
        messages = await self.get_cross_module_messages()
        
        # Prepare knowledge contributions
        synthesis = {
            "relevant_insights": [],
            "mechanic_tips": [],
            "similar_game_references": [],
            "learning_suggestions": []
        }
        
        # Get current game
        game_name = self.current_game_context.get("game_name")
        
        if game_name:
            # Get insights to include
            insights = self.original_system.get_applicable_insights(game_name)[:2]
            synthesis["relevant_insights"] = [
                f"From {i['source_game']}: {i['insight']}" for i in insights
            ]
            
            # Get mechanic tips
            current_mechanic = self._get_current_mechanic(messages)
            if current_mechanic:
                mechanic_data = self.original_system.mechanics.get(current_mechanic)
                if mechanic_data:
                    synthesis["mechanic_tips"] = mechanic_data.examples[:2]
            
            # Get similar game references
            similar = self.original_system.get_similar_games(game_name, max_games=2)
            synthesis["similar_game_references"] = [
                f"{g['name']} ({int(g['similarity']*100)}% similar)" for g in similar
            ]
            
            # Get learning opportunities
            opportunities = self.original_system.get_learning_opportunities(game_name)
            synthesis["learning_suggestions"] = [
                opp["description"] for opp in opportunities[:2]
            ]
        
        # Check if consolidation needed
        if time.time() - self.original_system.last_auto_save_time > 300:
            consolidation_result = self.original_system.consolidate_knowledge()
            synthesis["knowledge_consolidated"] = True
            synthesis["consolidation_stats"] = consolidation_result
        
        return synthesis
    
    # Helper methods
    
    async def _add_new_game(self, game_data: Dict[str, Any]):
        """Add a newly identified game to the knowledge base"""
        game_name = game_data.get("game_name", "Unknown Game")
        genre = game_data.get("genre", [])
        
        # Extract mechanics from game data
        mechanics = []
        if "mechanics" in game_data:
            mechanics = game_data["mechanics"]
        
        # Add to knowledge base
        self.original_system.add_game(
            game_id=game_data.get("game_id", f"game_{int(time.time())}"),
            game_name=game_name,
            genre=genre,
            mechanics=mechanics,
            description=game_data.get("description", "")
        )
        
        logger.info(f"Added new game to knowledge base: {game_name}")
    
    async def _process_game_knowledge(self, game_name: str):
        """Process and send knowledge for a game"""
        # Get similar games
        similar_games = self.original_system.get_similar_games(game_name)
        
        # Get applicable insights
        insights = self.original_system.get_applicable_insights(game_name)
        
        # Get learning opportunities
        opportunities = self.original_system.get_learning_opportunities(game_name)
        
        # Send comprehensive knowledge update
        await self.send_context_update(
            update_type="game_knowledge_processed",
            data={
                "game_name": game_name,
                "similar_games": similar_games,
                "insights": insights,
                "learning_opportunities": opportunities,
                "mechanics": self.original_system.games.get(game_name, {}).get("mechanics", [])
            },
            priority=ContextPriority.HIGH
        )
    
    async def _record_mechanic_usage(self, action_type: str, action_data: Dict[str, Any]):
        """Record usage of a game mechanic"""
        game_name = self.current_game_context.get("game_name")
        
        if not game_name:
            return
        
        # Map action to mechanic
        mechanic_id = self._action_to_mechanic(action_type)
        
        if mechanic_id:
            # Record usage
            success = action_data.get("success", True)
            self.original_system.record_mechanic_usage(mechanic_id, success)
            
            # Check for pattern detection
            if self.original_system.action_recognition:
                pattern = self.original_system.action_recognition.detect_pattern()
                
                if pattern:
                    await self.send_context_update(
                        update_type="action_pattern_detected",
                        data={
                            "pattern": pattern,
                            "mechanic": mechanic_id,
                            "game": game_name
                        }
                    )
    
    async def _analyze_memory_patterns(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze memories for cross-game patterns"""
        patterns = []
        
        # Look for game-related memories
        game_memories = [m for m in memories if "game" in m.get("tags", [])]
        
        # Group by game
        games_mentioned = {}
        for memory in game_memories:
            # Extract game name from memory
            game_name = self._extract_game_from_memory(memory)
            if game_name:
                if game_name not in games_mentioned:
                    games_mentioned[game_name] = []
                games_mentioned[game_name].append(memory)
        
        # Look for patterns across games
        if len(games_mentioned) > 1:
            # Found memories from multiple games
            pattern = {
                "type": "multi_game_experience",
                "games": list(games_mentioned.keys()),
                "pattern": "Experience across multiple games suggests transferable skills"
            }
            patterns.append(pattern)
        
        return patterns
    
    async def _analyze_quest_patterns(self, quest_data: Dict[str, Any]):
        """Analyze quest patterns across games"""
        current_game = self.current_game_context.get("game_name")
        quest_type = quest_data.get("quest_type", "generic")
        
        if current_game:
            # Look for similar quest patterns in other games
            similar_patterns = []
            
            for insight in self.original_system.insights:
                if "quest" in insight.context or "objective" in insight.context:
                    if insight.target_game != current_game:
                        similar_patterns.append({
                            "source_game": insight.source_game,
                            "pattern": insight.content,
                            "relevance": insight.relevance
                        })
            
            if similar_patterns:
                await self.send_context_update(
                    update_type="similar_quest_patterns",
                    data={
                        "current_quest": quest_data,
                        "similar_patterns": similar_patterns[:3]
                    }
                )
    
    async def _consolidate_learning(self, learning_data: Dict[str, Any]):
        """Consolidate learning into cross-game knowledge"""
        game_name = self.current_game_context.get("game_name")
        
        if not game_name:
            return
        
        # Extract learning type
        learning_type = learning_data.get("type", "general")
        content = learning_data.get("content", "")
        
        # Create insight if significant
        if learning_type in ["mechanic_mastery", "strategy_discovered", "pattern_learned"]:
            # Find most similar game
            similar_games = self.original_system.get_similar_games(game_name, max_games=1)
            
            if similar_games:
                target_game = similar_games[0]["name"]
                mechanic = learning_data.get("mechanic", "general")
                
                # Generate insight
                insight = self.original_system.generate_insight(
                    source_game=game_name,
                    target_game=target_game,
                    mechanic=mechanic
                )
                
                if "error" not in insight:
                    await self.send_context_update(
                        update_type="new_insight_generated",
                        data={
                            "insight": insight,
                            "source": "learning_consolidation"
                        }
                    )
    
    async def _process_gameplay_event(self, event_data: Dict[str, Any]):
        """Process significant gameplay events"""
        event_type = event_data.get("type")
        significance = event_data.get("significance", 5)
        
        # Only process highly significant events
        if significance >= 7:
            game_name = self.current_game_context.get("game_name")
            
            if game_name:
                # Check if this event type creates a pattern
                self.pending_insights.append({
                    "game": game_name,
                    "event": event_type,
                    "data": event_data,
                    "timestamp": time.time()
                })
                
                # If we have enough similar events, create a pattern
                similar_events = [
                    e for e in self.pending_insights 
                    if e["event"] == event_type and e["game"] == game_name
                ]
                
                if len(similar_events) >= 3:
                    # Create learning pattern
                    pattern = self.original_system.add_learning_pattern(
                        pattern_type="gameplay_pattern",
                        description=f"Pattern observed in {game_name}: {event_type} events",
                        games=[game_name],
                        mechanics=[],
                        confidence=0.7
                    )
                    
                    # Clear processed events
                    self.pending_insights = [
                        e for e in self.pending_insights 
                        if e["event"] != event_type or e["game"] != game_name
                    ]
    
    def _detect_game_mention(self, text: str) -> Optional[str]:
        """Detect if text mentions a game"""
        for game_name in self.original_system.games.keys():
            if game_name.lower() in text:
                return game_name
        return None
    
    def _detect_mechanic_query(self, text: str) -> Optional[str]:
        """Detect if text asks about a mechanic"""
        mechanic_keywords = ["mechanic", "system", "how does", "how do I", "tips for"]
        
        if any(kw in text for kw in mechanic_keywords):
            # Check for specific mechanics
            for mechanic_name in self.original_system.mechanics.keys():
                if mechanic_name.replace("_", " ") in text:
                    return mechanic_name
        
        return None
    
    async def _get_relevant_insights(self, context: SharedContext, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get insights relevant to the current context"""
        game_name = self.current_game_context.get("game_name")
        
        if not game_name:
            return []
        
        # Get base insights
        insights = self.original_system.get_applicable_insights(game_name)
        
        # If another game mentioned, get comparative insights
        other_game = analysis.get("mentions_other_game")
        if other_game:
            # Generate comparative insight
            comparative = self.original_system.generate_insight(
                source_game=other_game,
                target_game=game_name
            )
            
            if "error" not in comparative:
                insights.append(comparative)
        
        return insights[:3]  # Limit to top 3
    
    def _extract_game_state(self, context: SharedContext, messages: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Extract current game state from context and messages"""
        game_state = {
            "game_name": self.current_game_context.get("game_name"),
            "current_action": None,
            "current_location": None,
            "active_quests": [],
            "emotional_state": context.emotional_state
        }
        
        # Extract from messages
        for module_name, module_messages in messages.items():
            if module_name == "action_recognition":
                for msg in module_messages:
                    if msg["type"] == "action_detected":
                        game_state["current_action"] = msg["data"].get("action_type")
                        
            elif module_name == "spatial_memory":
                for msg in module_messages:
                    if msg["type"] == "location_update":
                        game_state["current_location"] = msg["data"].get("location_name")
                        
            elif module_name == "quest_manager":
                for msg in module_messages:
                    if msg["type"] == "active_quests":
                        game_state["active_quests"] = msg["data"].get("quests", [])
        
        return game_state
    
    async def _analyze_current_patterns(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current game state for patterns"""
        patterns = {
            "detected_patterns": [],
            "potential_patterns": []
        }
        
        # Check for action patterns
        if game_state.get("current_action"):
            # Look for similar actions in pattern database
            for pattern in self.original_system.learning_patterns.values():
                if pattern.pattern_type == "action_pattern":
                    patterns["detected_patterns"].append({
                        "pattern": pattern.description,
                        "confidence": pattern.confidence
                    })
        
        return patterns
    
    async def _generate_situational_insights(self, game_state: Dict[str, Any], patterns: Dict[str, Any]) -> List[str]:
        """Generate insights based on current situation"""
        insights = []
        
        game_name = game_state.get("game_name")
        if not game_name:
            return insights
        
        # Generate location-based insights
        if game_state.get("current_location"):
            location = game_state["current_location"]
            insights.append(f"In {location}, consider checking for hidden items common in similar games")
        
        # Generate action-based insights
        if game_state.get("current_action"):
            action = game_state["current_action"]
            # Look for tips about this action
            for mechanic in self.original_system.mechanics.values():
                if action in mechanic.name:
                    if mechanic.examples:
                        insights.append(f"Tip: {mechanic.examples[0]}")
                    break
        
        # Generate emotional state insights
        if game_state.get("emotional_state"):
            dominant_emotion = max(game_state["emotional_state"].items(), key=lambda x: x[1])[0]
            if dominant_emotion == "Frustration":
                insights.append("Similar games suggest taking a different approach when stuck")
        
        return insights[:3]  # Limit insights
    
    def _get_current_mechanic(self, messages: Dict[str, List[Dict]]) -> Optional[str]:
        """Extract current mechanic from messages"""
        for module_name, module_messages in messages.items():
            if module_name == "action_recognition":
                for msg in module_messages:
                    if msg["type"] == "action_detected":
                        action = msg["data"].get("action_type", "")
                        # Map action to mechanic
                        return self._action_to_mechanic(action)
        return None
    
    def _action_to_mechanic(self, action_type: str) -> Optional[str]:
        """Map action type to mechanic ID"""
        # Simple mapping - in real system would be more sophisticated
        action_mechanic_map = {
            "combat": "combat",
            "craft": "crafting",
            "explore": "open_world",
            "travel": "fast_travel",
            "skill": "skill_tree"
        }
        
        for action_key, mechanic in action_mechanic_map.items():
            if action_key in action_type.lower():
                return mechanic
        
        return None
    
    def _extract_game_from_memory(self, memory: Dict[str, Any]) -> Optional[str]:
        """Extract game name from a memory"""
        # Check memory text
        memory_text = memory.get("text", "").lower()
        
        # Look for game names
        for game_name in self.original_system.games.keys():
            if game_name.lower() in memory_text:
                return game_name
        
        # Check metadata
        metadata = memory.get("metadata", {})
        if "game" in metadata:
            return metadata["game"]
        
        return None
    
    # Delegate other methods to original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
