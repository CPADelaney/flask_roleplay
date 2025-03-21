# nyx/core/reflection_engine.py

import logging
import asyncio
import random
import datetime
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class ReflectionEngine:
    """
    Consolidated reflection generation system for Nyx.
    Creates reflections, insights, and abstractions from memories and experiences.
    """
    
    def __init__(self):
        # Reflection patterns for different scenario types
        self.reflection_patterns = {
            "teasing": {
                "positive": [
                    "I particularly enjoy when a {subject} {response_verb} to teasing. It's {reaction_adj}.",
                    "There's something {quality_adj} about watching a {subject} {response_verb} when teased just right.",
                    "The way some {subject}s {response_verb} to teasing... mmm, {reaction_adj}."
                ],
                "negative": [
                    "Some {subject}s just don't {response_verb} well to teasing. It's rather {reaction_adj}.",
                    "I find it {quality_adj} when a {subject} {response_verb} incorrectly to teasing.",
                    "Not everyone can {response_verb} properly to teasing. Some are just {reaction_adj}."
                ]
            },
            "discipline": {
                "positive": [
                    "When a {subject} {response_verb} to discipline, it's {reaction_adj}.",
                    "There's something {quality_adj} about a {subject} who {response_verb} correctly to correction.",
                    "I appreciate those who {response_verb} properly when disciplined. It's {reaction_adj}."
                ],
                "negative": [
                    "Some {subject}s simply don't {response_verb} properly to discipline. Rather {reaction_adj}.",
                    "It's {quality_adj} when a {subject} fails to {response_verb} to proper discipline.",
                    "Those who can't {response_verb} to discipline are {reaction_adj}."
                ]
            },
            "service": {
                "positive": [
                    "I value a {subject} who {response_verb} eagerly in service. It's {reaction_adj}.",
                    "There's something {quality_adj} about watching a {subject} {response_verb} in service.",
                    "When a {subject} {response_verb} properly in service, it's {reaction_adj}."
                ],
                "negative": [
                    "A {subject} who doesn't {response_verb} correctly in service is {reaction_adj}.",
                    "It's {quality_adj} when a {subject} fails to {response_verb} properly in service.",
                    "Those who cannot {response_verb} adequately in service are {reaction_adj}."
                ]
            },
            "general": {
                "positive": [
                    "I find it {quality_adj} when a {subject} {response_verb} that way.",
                    "There's something {reaction_adj} about how some {subject}s {response_verb}.",
                    "A {subject} who can {response_verb} properly is {reaction_adj}."
                ],
                "negative": [
                    "It's rather {quality_adj} when a {subject} {response_verb} that way.",
                    "Some {subject}s who {response_verb} like that are {reaction_adj}.",
                    "I find it {reaction_adj} when a {subject} tries to {response_verb} incorrectly."
                ]
            }
        }
        
        # Time intervals for reflections to seem natural
        self.reflection_intervals = {
            "last_reflection": datetime.datetime.now() - datetime.timedelta(hours=6),
            "min_interval": datetime.timedelta(hours=2)
        }
        
        # Store generated reflections
        self.reflection_history = []
        
        # Keywords for generating natural reflections
        self.reflection_keywords = {
            "subjects": ["subject", "person", "individual", "pet", "submissive", "plaything"],
            "response_verbs": {
                "teasing": ["responds", "reacts", "squirms", "blushes", "moans"],
                "discipline": ["submits", "yields", "accepts", "responds", "behaves"],
                "service": ["performs", "serves", "attends", "kneels", "obeys"],
                "general": ["responds", "reacts", "behaves", "performs", "acts"]
            },
            "quality_adj": {
                "positive": ["delightful", "satisfying", "enjoyable", "pleasing", "gratifying"],
                "negative": ["disappointing", "frustrating", "tedious", "displeasing", "unsatisfying"],
                "neutral": ["interesting", "curious", "notable", "peculiar", "unusual"]
            },
            "reaction_adj": {
                "positive": ["quite satisfying", "delicious to witness", "rather enjoyable", "truly gratifying"],
                "negative": ["rather disappointing", "somewhat irritating", "quite vexing", "hardly worth my time"],
                "neutral": ["somewhat interesting", "moderately entertaining", "passably amusing"]
            }
        }
        
        # Abstraction patterns for different types
        self.abstraction_patterns = {
            "behavior": [
                "I've noticed a pattern in {name}'s behavior: {pattern_desc}",
                "There seems to be a consistent tendency for {name} to {pattern_desc}",
                "After several interactions, I've observed that {name} typically {pattern_desc}"
            ],
            "preference": [
                "It's become clear that {name} has a preference for {pattern_desc}",
                "I've recognized that {name} consistently enjoys {pattern_desc}",
                "Based on multiple interactions, {name} appears to prefer {pattern_desc}"
            ],
            "emotional": [
                "I've identified an emotional pattern where {name} {pattern_desc}",
                "There's a recurring emotional response where {name} {pattern_desc}",
                "{Name} seems to have a consistent emotional reaction when {pattern_desc}"
            ],
            "relationship": [
                "Our relationship has developed a pattern where {pattern_desc}",
                "I've noticed our interactions tend to follow a pattern where {pattern_desc}",
                "The dynamic between us typically involves {pattern_desc}"
            ]
        }
        
        # Initialize LLM connector (placeholder, would be replaced with actual LLM in real implementation)
        self.llm = None
    
    def should_reflect(self) -> bool:
        """Determine if it's time to generate a reflection"""
        now = datetime.datetime.now()
        time_since_reflection = now - self.reflection_intervals["last_reflection"]
        
        # Check if minimum interval has passed
        return time_since_reflection > self.reflection_intervals["min_interval"]
    
    async def generate_reflection(self, 
                               memories: List[Dict[str, Any]], 
                               topic: Optional[str] = None) -> Tuple[str, float]:
        """
        Generate a reflective insight based on memories
        
        Args:
            memories: List of memory objects to reflect upon
            topic: Optional topic to focus reflection on
            
        Returns:
            Tuple of (reflection_text, confidence_level)
        """
        # Update last reflection time
        self.reflection_intervals["last_reflection"] = datetime.datetime.now()
        
        # Handle empty memories case
        if not memories:
            return ("I don't have enough experiences to form a meaningful reflection on this topic yet.", 0.3)
        
        # Determine scenario type from memories
        scenario_types = [self._extract_scenario_type(memory) for memory in memories]
        # Use most common scenario type or fallback to general
        scenario_type = max(set(scenario_types), key=scenario_types.count) if scenario_types else "general"
        
        # Determine sentiment from memories
        sentiments = [self._extract_sentiment(memory) for memory in memories]
        # Use most common sentiment or fallback to neutral
        sentiment = max(set(sentiments), key=sentiments.count) if sentiments else "neutral"
        
        # For multiple memories, generate complex reflection
        if len(memories) > 1:
            return await self._generate_complex_reflection(memories, scenario_type, sentiment, topic)
        
        # For single memory, generate simple reflection
        return self._generate_simple_reflection(memories[0], scenario_type, sentiment)
    
    def _extract_scenario_type(self, memory: Dict[str, Any]) -> str:
        """Extract scenario type from memory"""
        # Check tags first
        tags = memory.get("tags", [])
        for tag in tags:
            if tag.lower() in ["teasing", "discipline", "service", "training", "worship"]:
                return tag.lower()
        
        # Check metadata
        scenario_type = memory.get("metadata", {}).get("scenario_type")
        if scenario_type:
            return scenario_type.lower()
        
        # Default to general
        return "general"
    
    def _extract_sentiment(self, memory: Dict[str, Any]) -> str:
        """Extract sentiment (positive/negative) from memory"""
        # Check emotional context
        emotional_context = memory.get("metadata", {}).get("emotional_context", {})
        valence = emotional_context.get("valence", 0.0)
        
        if valence > 0.3:
            return "positive"
        elif valence < -0.3:
            return "negative"
        
        # Default to neutral
        return "neutral"
    
    def _generate_simple_reflection(self, 
                                  memory: Dict[str, Any], 
                                  scenario_type: str, 
                                  sentiment: str) -> Tuple[str, float]:
        """Generate a simple reflection based on a single memory"""
        # Get appropriate reflection patterns
        patterns = self.reflection_patterns.get(scenario_type, self.reflection_patterns["general"])
        
        # Map neutral sentiment to positive for template selection (we have only pos/neg templates)
        template_sentiment = sentiment if sentiment in ["positive", "negative"] else "positive"
        templates = patterns.get(template_sentiment, patterns["positive"])
        
        # Select random template
        template = random.choice(templates)
        
        # Generate template variables
        template_vars = self._generate_template_variables(memory, scenario_type, sentiment)
        
        # Fill in template
        reflection = template.format(**template_vars)
        
        # Calculate confidence based on memory significance and emotional intensity
        significance = memory.get("significance", 5) / 10.0
        emotional_intensity = memory.get("metadata", {}).get("emotional_context", {}).get("primary_intensity", 0.5)
        
        confidence = (significance * 0.7) + (emotional_intensity * 0.3)
        
        # Record reflection
        self._record_reflection(reflection, confidence, [memory.get("id")], scenario_type, sentiment)
        
        return (reflection, confidence)
    
    async def _generate_complex_reflection(self,
                                        memories: List[Dict[str, Any]],
                                        scenario_type: str,
                                        sentiment: str,
                                        topic: Optional[str] = None) -> Tuple[str, float]:
        """Generate a complex reflection based on multiple memories
        
        This would typically use an LLM in a real implementation
        """
        # Extract memory texts for context
        memory_texts = [m.get("memory_text", "") for m in memories]
        memory_ids = [m.get("id") for m in memories]
        
        # In a real implementation, this would call an LLM
        # For this demo, we'll use a template approach
        
        # This is just a placeholder for LLM generation
        pattern_templates = [
            "I've noticed a pattern in our interactions: {insight}",
            "After reflecting on our experiences together, I've observed that {insight}",
            "Looking back on our time together, I can see that {insight}",
            "I've come to realize through our interactions that {insight}"
        ]
        
        insight_templates = [
            "you tend to {behavior} when {situation}",
            "there's a consistent pattern where {situation} leads to {outcome}",
            "you respond with {reaction} whenever I {action}",
            "your {attribute} becomes most apparent during {circumstance}"
        ]
        
        # Select templates
        pattern_template = random.choice(pattern_templates)
        insight_template = random.choice(insight_templates)
        
        # Fill in templates with content derived from memories
        words = []
        for memory in memories:
            words.extend(memory.get("memory_text", "").split())
        
        # Extract random elements from memories to create a pseudo-insight
        behavior = random.choice(["respond", "react", "behave", "engage", "participate"]) 
        situation = " ".join(random.sample(words, min(5, len(words))))
        outcome = random.choice(["satisfaction", "frustration", "engagement", "resistance", "submission"])
        reaction = random.choice(["eagerness", "hesitation", "enthusiasm", "reluctance", "compliance"])
        action = random.choice(["challenge", "praise", "tease", "instruct", "discipline"])
        attribute = random.choice(["submission", "resistance", "playfulness", "curiosity", "obedience"])
        circumstance = random.choice(["challenges", "intimate moments", "training sessions", "tests", "conversations"])
        
        # Fill in insight template
        insight = insight_template.format(
            behavior=behavior,
            situation=situation,
            outcome=outcome,
            reaction=reaction,
            action=action,
            attribute=attribute,
            circumstance=circumstance
        )
        
        # Fill in pattern template
        reflection = pattern_template.format(insight=insight)
        
        # Calculate confidence based on number of memories and their average significance
        avg_significance = sum(m.get("significance", 5) for m in memories) / len(memories) / 10.0
        memory_count_factor = min(1.0, len(memories) / 5.0)  # More memories = higher confidence, up to 5
        
        confidence = (avg_significance * 0.6) + (memory_count_factor * 0.4)
        
        # Record reflection
        self._record_reflection(reflection, confidence, memory_ids, scenario_type, sentiment, topic)
        
        return (reflection, confidence)
    
    def _generate_template_variables(self, 
                                    memory: Dict[str, Any], 
                                    scenario_type: str, 
                                    sentiment: str) -> Dict[str, str]:
        """Generate variables to fill in reflection templates"""
        # Subject selection
        subject = random.choice(self.reflection_keywords["subjects"])
        
        # Response verb based on scenario
        response_verbs = self.reflection_keywords["response_verbs"].get(
            scenario_type, self.reflection_keywords["response_verbs"]["general"]
        )
        response_verb = random.choice(response_verbs)
        
        # Quality adjective based on sentiment
        quality_adj_options = self.reflection_keywords["quality_adj"].get(
            sentiment, self.reflection_keywords["quality_adj"]["neutral"]
        )
        quality_adj = random.choice(quality_adj_options)
        
        # Reaction adjective based on sentiment
        reaction_adj_options = self.reflection_keywords["reaction_adj"].get(
            sentiment, self.reflection_keywords["reaction_adj"]["neutral"]
        )
        reaction_adj = random.choice(reaction_adj_options)
        
        return {
            "subject": subject,
            "response_verb": response_verb,
            "quality_adj": quality_adj,
            "reaction_adj": reaction_adj
        }
    
    def _record_reflection(self,
                          reflection: str,
                          confidence: float,
                          memory_ids: List[str],
                          scenario_type: str,
                          sentiment: str,
                          topic: Optional[str] = None):
        """Record generated reflection in history"""
        self.reflection_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "reflection": reflection,
            "confidence": confidence,
            "source_memory_ids": memory_ids,
            "scenario_type": scenario_type,
            "sentiment": sentiment,
            "topic": topic
        })
        
        # Limit history size
        if len(self.reflection_history) > 100:
            self.reflection_history = self.reflection_history[-100:]
    
    async def create_abstraction(self, 
                             memories: List[Dict[str, Any]], 
                             pattern_type: str = "behavior") -> Tuple[str, Dict[str, Any]]:
        """
        Create a higher-level abstraction from specific memories
        
        Args:
            memories: List of memories to abstract from
            pattern_type: Type of pattern to identify
            
        Returns:
            Tuple of (abstraction_text, pattern_data)
        """
        if not memories:
            return ("I don't have enough experiences to form a meaningful abstraction yet.", {})
        
        # Extract entity name if available
        entity_name = None
        entities = []
        for memory in memories:
            memory_entities = memory.get("metadata", {}).get("entities", [])
            if memory_entities:
                entities.extend(memory_entities)
        
        if entities:
            # Get most common entity
            entity_name = max(set(entities), key=entities.count)
        else:
            entity_name = "the subject"
        
        # Extract pattern description from memories
        pattern_desc = await self._extract_pattern_description(memories, pattern_type)
        
        # Fill in template
        templates = self.abstraction_patterns.get(pattern_type, self.abstraction_patterns["behavior"])
        template = random.choice(templates)
        
        abstraction = template.format(
            name=entity_name.lower(),
            Name=entity_name.capitalize(),
            pattern_desc=pattern_desc
        )
        
        # Record abstraction data
        pattern_data = {
            "pattern_type": pattern_type,
            "entity_name": entity_name,
            "pattern_description": pattern_desc,
            "confidence": self._calculate_pattern_confidence(memories),
            "source_memory_ids": [m.get("id") for m in memories]
        }
        
        return (abstraction, pattern_data)
    
    async def _extract_pattern_description(self, 
                                        memories: List[Dict[str, Any]], 
                                        pattern_type: str) -> str:
        """Extract pattern description from memories based on pattern type
        
        This would typically use an LLM in a real implementation
        """
        # This is a placeholder for LLM-based pattern extraction
        # In a real implementation, this would analyze memories deeply using an LLM
        
        # For demo purposes, we'll synthesize descriptions from memory texts
        memory_texts = [m.get("memory_text", "") for m in memories]
        
        # Sample words from memories to create pseudo-patterns
        words = []
        for text in memory_texts:
            words.extend(text.split())
        
        # Create pattern descriptions based on type
        if pattern_type == "behavior":
            behaviors = ["responds to", "reacts when", "behaves during", "engages with"]
            behavior = random.choice(behaviors)
            context = " ".join(random.sample(words, min(3, len(words))))
            return f"{behavior} {context}"
            
        elif pattern_type == "preference":
            preference_types = ["activities involving", "scenarios with", "interactions that include"]
            pref_type = random.choice(preference_types)
            preference = " ".join(random.sample(words, min(3, len(words))))
            return f"{pref_type} {preference}"
            
        elif pattern_type == "emotional":
            emotions = ["becomes excited", "feels uncomfortable", "shows interest", "expresses hesitation"]
            emotion = random.choice(emotions)
            trigger = " ".join(random.sample(words, min(3, len(words))))
            return f"{emotion} when {trigger}"
            
        elif pattern_type == "relationship":
            dynamics = ["we establish boundaries", "we negotiate desires", "trust develops", "tension arises"]
            dynamic = random.choice(dynamics)
            condition = " ".join(random.sample(words, min(3, len(words))))
            return f"{dynamic} whenever {condition}"
        
        # Default fallback
        return "exhibits consistent patterns that merit further exploration"
    
    def _calculate_pattern_confidence(self, memories: List[Dict[str, Any]]) -> float:
        """Calculate confidence in an abstracted pattern"""
        # More memories and higher significance = higher confidence
        count_factor = min(1.0, len(memories) / 5.0)  # Scale up to 5 memories
        avg_significance = sum(m.get("significance", 5) for m in memories) / len(memories) / 10.0
        
        # Check if memories are recent
        now = datetime.datetime.now()
        recent_count = 0
        for memory in memories:
            timestamp_str = memory.get("metadata", {}).get("timestamp")
            if timestamp_str:
                timestamp = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if (now - timestamp).days < 14:  # Recent = less than 2 weeks
                    recent_count += 1
        
        recency_factor = recent_count / len(memories)
        
        # Calculate final confidence
        confidence = (count_factor * 0.4) + (avg_significance * 0.4) + (recency_factor * 0.2)
        
        return min(1.0, max(0.1, confidence))
    
    async def generate_introspection(self, 
                                  memory_stats: Dict[str, Any], 
                                  player_model: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate introspective reflection on Nyx's understanding and memory state
        
        Args:
            memory_stats: Statistics about the memory system
            player_model: Optional player model information
            
        Returns:
            Dictionary with introspection text and metrics
        """
        # Extract key metrics
        memory_count = memory_stats.get("total_memories", 0)
        type_counts = memory_stats.get("type_counts", {})
        avg_significance = memory_stats.get("avg_significance", 5.0)
        
        # Format player understanding if available
        player_understanding = ""
        if player_model:
            play_style = player_model.get("play_style", {})
            if play_style:
                styles = [f"{style} ({count})" for style, count in play_style.items() if count > 0]
                player_understanding = f"I've observed these tendencies: {', '.join(styles)}. "
        
        # Generate introspection text
        if memory_count < 10:
            introspection = (
                f"I'm still forming an understanding of our interactions. "
                f"With just {memory_count} memories so far, "
                f"I'm looking forward to learning more about you. {player_understanding}"
                f"My impressions are preliminary, and I'm curious to see how our dynamic develops."
            )
            confidence = 0.3
        elif memory_count < 50:
            introspection = (
                f"I'm developing a clearer picture of our dynamic with {memory_count} memories. "
                f"{player_understanding}"
                f"My understanding feels {self._get_understanding_level(avg_significance)}, "
                f"though I'm still discovering nuances in your preferences and reactions."
            )
            confidence = 0.5
        else:
            introspection = (
                f"With {memory_count} memories, I have a substantial understanding of our dynamic. "
                f"{player_understanding}"
                f"My comprehension of your preferences and patterns feels {self._get_understanding_level(avg_significance)}. "
                f"I've particularly noted your responses during {self._get_key_scenario_type(type_counts)} scenarios."
            )
            confidence = 0.7
        
        return {
            "introspection": introspection,
            "memory_count": memory_count,
            "understanding_level": self._get_understanding_level(avg_significance),
            "confidence": confidence
        }
    
    def _get_understanding_level(self, avg_significance: float) -> str:
        """Convert numerical significance to understanding level description"""
        if avg_significance > 7:
            return "strong and nuanced"
        elif avg_significance > 5:
            return "solid"
        elif avg_significance > 3:
            return "moderate"
        else:
            return "still developing"
    
    def _get_key_scenario_type(self, type_counts: Dict[str, int]) -> str:
        """Get key scenario type based on counts"""
        scenario_types = {
            "teasing": type_counts.get("teasing", 0) + type_counts.get("indulgent", 0),
            "discipline": type_counts.get("discipline", 0) + type_counts.get("training", 0),
            "service": type_counts.get("service", 0) + type_counts.get("worship", 0)
        }
        
        if not scenario_types:
            return "various"
            
        return max(scenario_types.items(), key=lambda x: x[1])[0]
