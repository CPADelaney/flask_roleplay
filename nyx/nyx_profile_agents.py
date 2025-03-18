from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
from pydantic import BaseModel, Field
import random
import re
from collections import defaultdict

logger = logging.getLogger("nyx_profile_agents")

class TeasingAgent(BaseModel):
    """Agent responsible for generating creative teasing strategies"""
    id: str = Field(default_factory=lambda: f"tease_{random.randint(1000, 9999)}")
    creativity_level: float = Field(default=0.8, ge=0.0, le=1.0)
    subtlety_level: float = Field(default=0.7, ge=0.0, le=1.0)
    teasing_history: List[Dict[str, Any]] = []
    success_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    aesthetic_preferences: Dict[str, float] = {}
    dynamic_elements: Dict[str, Any] = {}
    
    async def generate_teasing_strategy(self, profile_insights: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a teasing strategy based on profile insights and context"""
        strategy = {
            "elements": [],
            "emotional_triggers": [],
            "aesthetic_elements": [],
            "dynamic_adaptations": [],
            "intensity": 0.0,
            "success_probability": 0.0
        }
        
        # Generate kink-based teasing
        if "kink_preferences" in profile_insights:
            for kink, intensity in profile_insights["kink_preferences"].items():
                if intensity > 0.6:  # Only use strong preferences
                    element = {
                        "type": "kink",
                        "kink": kink,
                        "phrases": self._generate_kink_tease_phrases(kink),
                        "intensity": intensity
                    }
                    strategy["elements"].append(element)
                    
        # Generate physical trait teasing
        if "physical_preferences" in profile_insights:
            for trait, intensity in profile_insights["physical_preferences"].items():
                if intensity > 0.6:
                    element = {
                        "type": "physical",
                        "trait": trait,
                        "phrases": self._generate_physical_tease_phrases(trait),
                        "intensity": intensity
                    }
                    strategy["elements"].append(element)
                    
        # Generate personality trait teasing
        if "personality_preferences" in profile_insights:
            for trait, intensity in profile_insights["personality_preferences"].items():
                if intensity > 0.6:
                    element = {
                        "type": "personality",
                        "trait": trait,
                        "phrases": self._generate_personality_tease_phrases(trait),
                        "intensity": intensity
                    }
                    strategy["elements"].append(element)
                    
        # Add aesthetic elements
        if "aesthetic_preferences" in profile_insights:
            for aesthetic, intensity in profile_insights["aesthetic_preferences"].items():
                if intensity > 0.6:
                    element = {
                        "type": "aesthetic",
                        "aesthetic": aesthetic,
                        "elements": self._generate_aesthetic_elements(aesthetic),
                        "intensity": intensity
                    }
                    strategy["aesthetic_elements"].append(element)
                    
        # Add dynamic adaptations based on context
        if context.get("scene_context"):
            dynamic_elements = self._generate_dynamic_adaptations(
                context["scene_context"],
                profile_insights
            )
            strategy["dynamic_adaptations"].extend(dynamic_elements)
            
        # Add emotional triggers
        if "emotional_triggers" in profile_insights:
            for trigger, intensity in profile_insights["emotional_triggers"].items():
                if intensity > 0.6:
                    element = {
                        "type": "trigger",
                        "trigger": trigger,
                        "phrases": self._generate_trigger_phrases(trigger),
                        "intensity": intensity
                    }
                    strategy["emotional_triggers"].append(element)
                    
        # Calculate strategy metrics
        strategy["intensity"] = self._calculate_strategy_intensity(strategy)
        strategy["success_probability"] = self._calculate_success_probability(strategy, context)
        
        return strategy
        
    def _generate_aesthetic_elements(self, aesthetic: str) -> List[Dict[str, Any]]:
        """Generate aesthetic-specific teasing elements"""
        elements = []
        
        # Define aesthetic-specific elements
        aesthetic_elements = {
            "goth": [
                {"type": "visual", "description": "dark makeup", "intensity": 0.7},
                {"type": "visual", "description": "black clothing", "intensity": 0.8},
                {"type": "mood", "description": "mysterious atmosphere", "intensity": 0.6},
                {"type": "behavior", "description": "mysterious smile", "intensity": 0.7}
            ],
            "punk": [
                {"type": "visual", "description": "edgy style", "intensity": 0.8},
                {"type": "behavior", "description": "rebellious attitude", "intensity": 0.7},
                {"type": "mood", "description": "energetic atmosphere", "intensity": 0.6}
            ],
            "preppy": [
                {"type": "visual", "description": "clean look", "intensity": 0.7},
                {"type": "behavior", "description": "confident demeanor", "intensity": 0.6},
                {"type": "mood", "description": "sophisticated atmosphere", "intensity": 0.5}
            ]
        }
        
        if aesthetic in aesthetic_elements:
            elements.extend(aesthetic_elements[aesthetic])
            
        return elements
        
    def _generate_dynamic_adaptations(self, scene_context: Dict[str, Any], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate dynamic adaptations based on scene context"""
        adaptations = []
        
        # Check for NPC presence
        if "npcs" in scene_context:
            for npc in scene_context["npcs"]:
                # Generate aesthetic-based adaptations
                if "aesthetic" in npc:
                    aesthetic = npc["aesthetic"]
                    if aesthetic not in self.aesthetic_preferences:
                        self.aesthetic_preferences[aesthetic] = 0.5
                        
                    # Create adaptation based on aesthetic
                    adaptation = {
                        "type": "aesthetic_adaptation",
                        "aesthetic": aesthetic,
                        "elements": self._generate_aesthetic_elements(aesthetic),
                        "intensity": self.aesthetic_preferences[aesthetic]
                    }
                    adaptations.append(adaptation)
                    
                # Generate personality-based adaptations
                if "personality" in npc:
                    personality = npc["personality"]
                    if personality in profile.get("personality_preferences", {}):
                        adaptation = {
                            "type": "personality_adaptation",
                            "personality": personality,
                            "elements": self._generate_personality_tease_phrases(personality),
                            "intensity": profile["personality_preferences"][personality]
                        }
                        adaptations.append(adaptation)
                        
        # Check for environmental elements
        if "environment" in scene_context:
            env = scene_context["environment"]
            # Generate environment-based adaptations
            if "mood" in env:
                adaptation = {
                    "type": "environment_adaptation",
                    "mood": env["mood"],
                    "elements": self._generate_environment_elements(env["mood"]),
                    "intensity": 0.6
                }
                adaptations.append(adaptation)
                
        return adaptations
        
    def _generate_environment_elements(self, mood: str) -> List[Dict[str, Any]]:
        """Generate environment-specific teasing elements"""
        elements = []
        
        # Define mood-specific elements
        mood_elements = {
            "romantic": [
                {"type": "visual", "description": "soft lighting", "intensity": 0.6},
                {"type": "mood", "description": "romantic atmosphere", "intensity": 0.7},
                {"type": "behavior", "description": "gentle touch", "intensity": 0.6}
            ],
            "tension": [
                {"type": "visual", "description": "intense eye contact", "intensity": 0.8},
                {"type": "mood", "description": "charged atmosphere", "intensity": 0.7},
                {"type": "behavior", "description": "provocative movement", "intensity": 0.7}
            ],
            "playful": [
                {"type": "visual", "description": "mischievous smile", "intensity": 0.6},
                {"type": "mood", "description": "light atmosphere", "intensity": 0.5},
                {"type": "behavior", "description": "teasing gesture", "intensity": 0.6}
            ]
        }
        
        if mood in mood_elements:
            elements.extend(mood_elements[mood])
            
        return elements
        
    def _generate_kink_tease_phrases(self, kink: str) -> List[str]:
        """Generate subtle teasing phrases for a kink"""
        phrases = {
            "bdsm": [
                "I notice you seem particularly interested in power dynamics...",
                "Your fascination with control is quite intriguing...",
                "There's something about submission that draws you in, isn't there?"
            ],
            "roleplay": [
                "Your imagination seems to run wild with fantasies...",
                "You have such a creative mind for scenarios...",
                "The way you immerse yourself in stories is fascinating..."
            ],
            "exhibitionism": [
                "You seem to enjoy being the center of attention...",
                "There's something thrilling about being watched, isn't there?",
                "Your confidence in being seen is quite attractive..."
            ],
            "voyeurism": [
                "You have such an observant nature...",
                "The way you notice details is quite impressive...",
                "Your interest in watching others is quite intriguing..."
            ]
        }
        return phrases.get(kink, [])
        
    def _generate_physical_tease_phrases(self, trait: str) -> List[str]:
        """Generate subtle teasing phrases for physical traits"""
        phrases = {
            "redhead": [
                "Your fascination with red hair is quite noticeable...",
                "There's something special about fiery locks, isn't there?",
                "The way you admire redheads is quite telling..."
            ],
            "tattooed": [
                "Your interest in ink is quite apparent...",
                "There's something about marked skin that draws you in...",
                "Your appreciation for body art is quite evident..."
            ],
            "tall": [
                "You seem to have a thing for height...",
                "There's something about towering presence that appeals to you...",
                "Your attraction to tall figures is quite noticeable..."
            ],
            "muscular": [
                "Your interest in strength is quite obvious...",
                "There's something about physical power that draws you in...",
                "Your appreciation for muscular builds is quite clear..."
            ]
        }
        return phrases.get(trait, [])
        
    def _generate_personality_tease_phrases(self, trait: str) -> List[str]:
        """Generate subtle teasing phrases for personality traits"""
        phrases = {
            "dominant": [
                "You seem drawn to commanding presence...",
                "There's something about authority that appeals to you...",
                "Your interest in control is quite evident..."
            ],
            "submissive": [
                "You have such an appreciation for yielding...",
                "There's something about surrender that draws you in...",
                "Your attraction to submission is quite noticeable..."
            ],
            "playful": [
                "Your love for fun is quite infectious...",
                "There's something about playfulness that appeals to you...",
                "Your appreciation for teasing is quite clear..."
            ],
            "serious": [
                "You seem drawn to solemnity...",
                "There's something about formality that appeals to you...",
                "Your interest in proper behavior is quite evident..."
            ]
        }
        return phrases.get(trait, [])
        
    def _generate_trigger_phrases(self, trigger: str) -> List[str]:
        """Generate subtle triggering phrases"""
        phrases = {
            "arousal": [
                "Your reactions are quite telling...",
                "There's something about this that excites you...",
                "Your interest is quite palpable..."
            ],
            "fear": [
                "Your apprehension is quite noticeable...",
                "There's something about this that makes you nervous...",
                "Your caution is quite evident..."
            ],
            "desire": [
                "Your longing is quite apparent...",
                "There's something about this that draws you in...",
                "Your craving is quite clear..."
            ],
            "submission": [
                "Your yielding nature is quite telling...",
                "There's something about this that makes you submit...",
                "Your obedience is quite evident..."
            ]
        }
        return phrases.get(trigger, [])
        
    def _calculate_strategy_intensity(self, strategy: Dict[str, Any]) -> float:
        """Calculate overall strategy intensity"""
        if not strategy["elements"]:
            return 0.5
            
        intensities = [element["intensity"] for element in strategy["elements"]]
        return sum(intensities) / len(intensities)
        
    def _calculate_success_probability(self, strategy: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate probability of strategy success"""
        base_probability = self.success_rate
        
        # Adjust based on context
        if context.get("positive_reaction", False):
            base_probability += 0.1
        if context.get("negative_reaction", False):
            base_probability -= 0.1
            
        # Adjust based on strategy elements
        if strategy["elements"]:
            element_success = sum(element["intensity"] for element in strategy["elements"]) / len(strategy["elements"])
            base_probability = (base_probability + element_success) / 2
            
        return min(1.0, max(0.0, base_probability))

class ProfilingAgent(BaseModel):
    """Agent responsible for analyzing and enhancing player profiling"""
    id: str = Field(default_factory=lambda: f"profile_{random.randint(1000, 9999)}")
    analysis_depth: float = Field(default=0.8, ge=0.0, le=1.0)
    pattern_recognition: float = Field(default=0.7, ge=0.0, le=1.0)
    observation_history: List[Dict[str, Any]] = []
    dynamic_categories: Dict[str, float] = {}
    category_confidence: Dict[str, float] = {}
    aesthetic_insights: Dict[str, Any] = {}
    pattern_evolution: Dict[str, List[Dict[str, Any]]] = {}
    autonomous_insights: Dict[str, Any] = {}
    
    async def analyze_interaction(self, event: Dict[str, Any], current_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze interaction and update profile"""
        result = {
            "new_insights": {},
            "updated_preferences": {},
            "recommendations": [],
            "dynamic_categories": {},
            "autonomous_insights": {}
        }
        
        # Analyze content
        content = event.get("content", "")
        context = event.get("context", {})
        
        # Basic analysis
        language_patterns = self._analyze_language_patterns(content)
        emotional_responses = self._analyze_emotional_responses(content, context)
        interaction_style = self._analyze_interaction_style(content, context)
        
        # Autonomous pattern discovery
        autonomous_insights = self._discover_autonomous_patterns(content, context)
        result["autonomous_insights"] = autonomous_insights
        
        # Dynamic category analysis with autonomous expansion
        dynamic_categories = self._analyze_dynamic_categories(content, context, current_profile)
        result["dynamic_categories"] = dynamic_categories
        
        # Update pattern recognition with evolution tracking
        pattern_updates = self._update_pattern_recognition(content, current_profile)
        
        # Generate new insights with autonomous elements
        new_insights = self._generate_new_insights(
            language_patterns,
            emotional_responses,
            interaction_style,
            dynamic_categories,
            pattern_updates,
            autonomous_insights
        )
        result["new_insights"] = new_insights
        
        # Update preferences with autonomous categories
        updated_preferences = self._update_preferences(
            new_insights,
            current_profile,
            autonomous_insights
        )
        result["updated_preferences"] = updated_preferences
        
        # Generate recommendations with autonomous suggestions
        recommendations = self._generate_recommendations(
            {
                "language_patterns": language_patterns,
                "emotional_responses": emotional_responses,
                "interaction_style": interaction_style,
                "dynamic_categories": dynamic_categories,
                "pattern_updates": pattern_updates,
                "autonomous_insights": autonomous_insights
            },
            current_profile
        )
        result["recommendations"] = recommendations
        
        return result
        
    def _discover_autonomous_patterns(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Autonomously discover new patterns and categories"""
        insights = {
            "new_aesthetics": [],
            "evolved_patterns": [],
            "category_connections": [],
            "confidence_metrics": {}
        }
        
        # Analyze word clusters for potential new aesthetics
        word_clusters = self._analyze_word_clusters(content)
        for cluster in word_clusters:
            if self._is_potential_aesthetic(cluster):
                new_aesthetic = self._generate_aesthetic_category(cluster)
                insights["new_aesthetics"].append(new_aesthetic)
                
        # Track pattern evolution
        for category, patterns in self.pattern_evolution.items():
            evolved_patterns = self._analyze_pattern_evolution(category, patterns)
            if evolved_patterns:
                insights["evolved_patterns"].append(evolved_patterns)
                
        # Discover category connections
        connections = self._discover_category_connections(content, context)
        insights["category_connections"] = connections
        
        # Calculate confidence metrics
        insights["confidence_metrics"] = self._calculate_confidence_metrics(insights)
        
        return insights
        
    def _analyze_word_clusters(self, content: str) -> List[Dict[str, Any]]:
        """Analyze content for word clusters that might indicate new patterns"""
        clusters = []
        words = content.lower().split()
        
        # Look for adjective-noun pairs
        for i in range(len(words) - 1):
            if self._is_adjective(words[i]) and self._is_noun(words[i + 1]):
                clusters.append({
                    "type": "adjective_noun",
                    "words": [words[i], words[i + 1]],
                    "context": self._get_context(words, i, 2)
                })
                
        # Look for descriptive phrases
        for i in range(len(words) - 2):
            if self._is_descriptive_phrase(words[i:i+3]):
                clusters.append({
                    "type": "descriptive_phrase",
                    "words": words[i:i+3],
                    "context": self._get_context(words, i, 3)
                })
                
        return clusters
        
    def _is_potential_aesthetic(self, cluster: Dict[str, Any]) -> bool:
        """Determine if a word cluster might indicate a new aesthetic"""
        # Check for aesthetic indicators
        aesthetic_indicators = [
            "style", "look", "fashion", "aesthetic", "vibe", "theme",
            "appearance", "presence", "aura", "atmosphere"
        ]
        
        # Check cluster words against indicators
        for word in cluster["words"]:
            if word in aesthetic_indicators:
                return True
                
        # Check context for aesthetic relevance
        context = cluster["context"]
        if any(indicator in context for indicator in aesthetic_indicators):
            return True
            
        return False
        
    def _generate_aesthetic_category(self, cluster: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new aesthetic category from a word cluster"""
        category = {
            "name": self._generate_aesthetic_name(cluster),
            "elements": self._generate_aesthetic_elements(cluster),
            "confidence": self._calculate_aesthetic_confidence(cluster),
            "evolution_history": [],
            "related_categories": []
        }
        
        # Add to pattern evolution tracking
        if category["name"] not in self.pattern_evolution:
            self.pattern_evolution[category["name"]] = []
            
        self.pattern_evolution[category["name"]].append({
            "timestamp": datetime.now(),
            "cluster": cluster,
            "confidence": category["confidence"]
        })
        
        return category
        
    def _generate_aesthetic_name(self, cluster: Dict[str, Any]) -> str:
        """Generate a name for a new aesthetic category"""
        # Combine relevant words from cluster
        words = cluster["words"]
        if cluster["type"] == "adjective_noun":
            return f"{words[0]}_{words[1]}"
        else:
            return "_".join(words)
            
    def _generate_aesthetic_elements(self, cluster: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate elements for a new aesthetic category"""
        elements = []
        
        # Generate visual elements
        visual_elements = self._generate_visual_elements(cluster)
        elements.extend(visual_elements)
        
        # Generate behavioral elements
        behavioral_elements = self._generate_behavioral_elements(cluster)
        elements.extend(behavioral_elements)
        
        # Generate mood elements
        mood_elements = self._generate_mood_elements(cluster)
        elements.extend(mood_elements)
        
        return elements
        
    def _calculate_aesthetic_confidence(self, cluster: Dict[str, Any]) -> float:
        """Calculate confidence score for a new aesthetic category"""
        confidence = 0.5  # Base confidence
        
        # Adjust based on word strength
        word_strength = self._calculate_word_strength(cluster["words"])
        confidence += word_strength * 0.2
        
        # Adjust based on context relevance
        context_relevance = self._calculate_context_relevance(cluster["context"])
        confidence += context_relevance * 0.2
        
        # Adjust based on pattern consistency
        pattern_consistency = self._calculate_pattern_consistency(cluster)
        confidence += pattern_consistency * 0.1
        
        return min(1.0, max(0.0, confidence))
        
    def _analyze_pattern_evolution(self, category: str, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how patterns have evolved over time"""
        if len(patterns) < 2:
            return None
            
        evolution = {
            "category": category,
            "trends": [],
            "stability": 0.0,
            "confidence_changes": []
        }
        
        # Analyze confidence trends
        confidences = [p["confidence"] for p in patterns]
        evolution["confidence_changes"] = self._calculate_confidence_changes(confidences)
        
        # Analyze pattern stability
        evolution["stability"] = self._calculate_pattern_stability(patterns)
        
        # Identify emerging trends
        evolution["trends"] = self._identify_emerging_trends(patterns)
        
        return evolution
        
    def _discover_category_connections(self, content: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover connections between different categories"""
        connections = []
        
        # Get all active categories
        categories = list(self.dynamic_categories.keys())
        
        # Analyze category co-occurrences
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i+1:]:
                connection = self._analyze_category_connection(cat1, cat2, content, context)
                if connection:
                    connections.append(connection)
                    
        return connections
        
    def _analyze_category_connection(self, cat1: str, cat2: str, content: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze connection between two categories"""
        connection = {
            "categories": [cat1, cat2],
            "strength": 0.0,
            "confidence": 0.0,
            "context": []
        }
        
        # Calculate connection strength
        strength = self._calculate_connection_strength(cat1, cat2, content)
        connection["strength"] = strength
        
        if strength > 0.5:  # Only include strong connections
            connection["confidence"] = self._calculate_connection_confidence(cat1, cat2, context)
            connection["context"] = self._get_connection_context(cat1, cat2, content)
            return connection
            
        return None
        
    def _calculate_confidence_metrics(self, insights: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive confidence metrics"""
        metrics = {
            "pattern_confidence": 0.0,
            "aesthetic_confidence": 0.0,
            "evolution_confidence": 0.0,
            "connection_confidence": 0.0
        }
        
        # Calculate pattern confidence
        if insights["evolved_patterns"]:
            metrics["pattern_confidence"] = self._calculate_pattern_confidence(insights["evolved_patterns"])
            
        # Calculate aesthetic confidence
        if insights["new_aesthetics"]:
            metrics["aesthetic_confidence"] = self._calculate_aesthetic_confidence(insights["new_aesthetics"])
            
        # Calculate evolution confidence
        metrics["evolution_confidence"] = self._calculate_evolution_confidence(insights["evolved_patterns"])
        
        # Calculate connection confidence
        if insights["category_connections"]:
            metrics["connection_confidence"] = self._calculate_connection_confidence(insights["category_connections"])
            
        return metrics
        
    def _analyze_dynamic_categories(self, content: str, context: Dict[str, Any], current_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content for potential new profiling categories"""
        categories = {}
        
        # Check for aesthetic mentions
        aesthetic_patterns = {
            "goth": r"(?i)goth|dark|mysterious|vampire|gothic",
            "punk": r"(?i)punk|rebel|edgy|alternative",
            "preppy": r"(?i)preppy|clean|sophisticated|classy"
        }
        
        for aesthetic, pattern in aesthetic_patterns.items():
            if re.search(pattern, content):
                if aesthetic not in self.dynamic_categories:
                    self.dynamic_categories[aesthetic] = 0.5
                    self.category_confidence[aesthetic] = 0.3
                else:
                    self.dynamic_categories[aesthetic] += 0.1
                    self.category_confidence[aesthetic] += 0.05
                    
                categories[aesthetic] = {
                    "confidence": self.category_confidence[aesthetic],
                    "intensity": self.dynamic_categories[aesthetic]
                }
                
        # Check for personality traits
        personality_patterns = {
            "dominant": r"(?i)dominant|assertive|controlling|powerful",
            "submissive": r"(?i)submissive|obedient|yielding|meek",
            "playful": r"(?i)playful|mischievous|teasing|fun"
        }
        
        for trait, pattern in personality_patterns.items():
            if re.search(pattern, content):
                if trait not in self.dynamic_categories:
                    self.dynamic_categories[trait] = 0.5
                    self.category_confidence[trait] = 0.3
                else:
                    self.dynamic_categories[trait] += 0.1
                    self.category_confidence[trait] += 0.05
                    
                categories[trait] = {
                    "confidence": self.category_confidence[trait],
                    "intensity": self.dynamic_categories[trait]
                }
                
        # Check for kink mentions
        kink_patterns = {
            "bdsm": r"(?i)bdsm|dominance|submission|control",
            "roleplay": r"(?i)roleplay|fantasy|scenario|scene",
            "exhibition": r"(?i)exhibition|public|exposure|show"
        }
        
        for kink, pattern in kink_patterns.items():
            if re.search(pattern, content):
                if kink not in self.dynamic_categories:
                    self.dynamic_categories[kink] = 0.5
                    self.category_confidence[kink] = 0.3
                else:
                    self.dynamic_categories[kink] += 0.1
                    self.category_confidence[kink] += 0.05
                    
                categories[kink] = {
                    "confidence": self.category_confidence[kink],
                    "intensity": self.dynamic_categories[kink]
                }
                
        return categories
        
    def _generate_new_insights(self, language_patterns: Dict[str, float],
                             emotional_responses: Dict[str, float],
                             interaction_style: Dict[str, float],
                             dynamic_categories: Dict[str, Any],
                             pattern_updates: Dict[str, float],
                             autonomous_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Generate new insights from analysis results"""
        insights = {}
        
        # Process dynamic categories
        for category, data in dynamic_categories.items():
            if data["confidence"] > 0.5:  # Only include high confidence categories
                insights[category] = {
                    "intensity": data["intensity"],
                    "confidence": data["confidence"],
                    "patterns": self._get_patterns_for_category(category, "patterns")
                }
                
        # Process language patterns
        for pattern, intensity in language_patterns.items():
            if intensity > 0.6:
                insights[f"language_{pattern}"] = {
                    "intensity": intensity,
                    "confidence": 0.7,
                    "patterns": self._get_patterns_for_category("language", pattern)
                }
                
        # Process emotional responses
        for response, intensity in emotional_responses.items():
            if intensity > 0.6:
                insights[f"emotional_{response}"] = {
                    "intensity": intensity,
                    "confidence": 0.7,
                    "patterns": self._get_patterns_for_category("emotional", response)
                }
                
        # Process interaction style
        for style, intensity in interaction_style.items():
            if intensity > 0.6:
                insights[f"style_{style}"] = {
                    "intensity": intensity,
                    "confidence": 0.7,
                    "patterns": self._get_patterns_for_category("style", style)
                }
                
        # Process autonomous insights
        for category, data in autonomous_insights.items():
            if isinstance(data, dict) and "confidence" in data:
                insights[category] = {
                    "intensity": data["intensity"],
                    "confidence": data["confidence"],
                    "patterns": self._get_patterns_for_category(category, "patterns")
                }
                
        return insights
        
    def _update_preferences(self, new_insights: Dict[str, Any], current_profile: Dict[str, Any], autonomous_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Update preferences based on new insights"""
        updated = {}
        
        # Process aesthetic preferences
        for category, data in new_insights.items():
            if category in ["goth", "punk", "preppy"]:
                if "aesthetic_preferences" not in current_profile:
                    current_profile["aesthetic_preferences"] = {}
                    
                current_profile["aesthetic_preferences"][category] = data["intensity"]
                updated["aesthetic_preferences"] = current_profile["aesthetic_preferences"]
                
        # Process personality preferences
        for category, data in new_insights.items():
            if category in ["dominant", "submissive", "playful"]:
                if "personality_preferences" not in current_profile:
                    current_profile["personality_preferences"] = {}
                    
                current_profile["personality_preferences"][category] = data["intensity"]
                updated["personality_preferences"] = current_profile["personality_preferences"]
                
        # Process kink preferences
        for category, data in new_insights.items():
            if category in ["bdsm", "roleplay", "exhibition"]:
                if "kink_preferences" not in current_profile:
                    current_profile["kink_preferences"] = {}
                    
                current_profile["kink_preferences"][category] = data["intensity"]
                updated["kink_preferences"] = current_profile["kink_preferences"]
                
        # Process autonomous insights
        for category, data in autonomous_insights.items():
            if isinstance(data, dict) and "confidence" in data:
                if category not in current_profile:
                    current_profile[category] = {}
                    
                current_profile[category] = data
                updated[category] = current_profile[category]
                
        return updated
        
    def _analyze_language_patterns(self, content: str) -> Dict[str, float]:
        """Analyze language patterns in content"""
        patterns = {
            "formal": r"\b(shall|whom|whilst|hence|thus)\b",
            "casual": r"\b(yeah|okay|cool|awesome|gonna)\b",
            "descriptive": r"\b(beautiful|amazing|incredible|wonderful)\b",
            "emotional": r"\b(love|hate|desire|crave|need)\b",
            "submissive": r"\b(yes|please|thank|sorry|apologize)\b",
            "dominant": r"\b(command|demand|require|expect|insist)\b",
            "playful": r"\b(tease|play|fun|joke|laugh)\b",
            "serious": r"\b(important|serious|proper|correct|appropriate)\b"
        }
        
        insights = {}
        for style, pattern in patterns.items():
            matches = len(re.findall(pattern, content))
            if matches > 0:
                insights[style] = min(1.0, matches * 0.1)
                
        return insights
        
    def _analyze_emotional_responses(self, content: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze emotional responses in content"""
        emotional_patterns = {
            "arousal": ["aroused", "excited", "turned on", "horny", "hot"],
            "fear": ["scared", "afraid", "terrified", "fear", "anxious"],
            "desire": ["want", "desire", "crave", "need", "long"],
            "submission": ["submit", "yield", "surrender", "obey", "please"],
            "dominance": ["control", "power", "authority", "command", "rule"],
            "playfulness": ["play", "fun", "tease", "joke", "laugh"],
            "seriousness": ["serious", "important", "proper", "correct", "appropriate"]
        }
        
        insights = {}
        for emotion, patterns in emotional_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in content)
            if matches > 0:
                insights[emotion] = min(1.0, matches * 0.1)
                
        return insights
        
    def _analyze_interaction_style(self, content: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze interaction style in content"""
        style_patterns = {
            "aggressive": ["force", "push", "demand", "command", "insist"],
            "gentle": ["gentle", "soft", "tender", "careful", "kind"],
            "playful": ["tease", "play", "fun", "game", "joke"],
            "formal": ["proper", "correct", "appropriate", "formal", "respectful"],
            "casual": ["cool", "awesome", "yeah", "okay", "gonna"],
            "submissive": ["please", "thank", "sorry", "apologize", "obey"],
            "dominant": ["command", "control", "require", "expect", "insist"]
        }
        
        insights = {}
        for style, patterns in style_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in content)
            if matches > 0:
                insights[style] = min(1.0, matches * 0.1)
                
        return insights
        
    def _update_pattern_recognition(self, content: str, current_profile: Dict[str, Any]) -> Dict[str, float]:
        """Update pattern recognition based on new content"""
        updates = {}
        
        # Check for new patterns in existing categories
        for category, values in current_profile.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    if value > 0.7:  # High confidence patterns
                        patterns = self._get_patterns_for_category(category, key)
                        matches = sum(1 for pattern in patterns if pattern in content)
                        if matches > 0:
                            updates[f"{category}_{key}"] = min(1.0, value + matches * 0.1)
                            
        return updates
        
    def _get_patterns_for_category(self, category: str, key: str) -> List[str]:
        """Get patterns for a specific category and key"""
        pattern_maps = {
            "kink_preferences": {
                "bdsm": ["whip", "chain", "collar", "submission", "dominance"],
                "roleplay": ["fantasy", "scenario", "character", "story"],
                "exhibitionism": ["public", "exposed", "watched", "seen"],
                "voyeurism": ["watching", "observing", "peeking", "spying"]
            },
            "physical_preferences": {
                "redhead": ["red hair", "ginger", "redhead"],
                "tattooed": ["tattoo", "ink", "tattooed"],
                "tall": ["tall", "height", "towering"],
                "muscular": ["muscle", "strong", "buff", "ripped"]
            },
            "personality_preferences": {
                "dominant": ["dominant", "controlling", "authoritative"],
                "submissive": ["submissive", "obedient", "yielding"],
                "playful": ["playful", "fun", "teasing"],
                "serious": ["serious", "strict", "formal"]
            }
        }
        
        return pattern_maps.get(category, {}).get(key, [])
        
    def _generate_recommendations(self, analysis: Dict[str, Any], current_profile: Dict[str, Any]) -> List[str]:
        """Generate recommendations for profile enhancement"""
        recommendations = []
        
        # Check for gaps in profile
        for category in ["kink_preferences", "physical_preferences", "personality_preferences"]:
            if not current_profile.get(category):
                recommendations.append(f"Explore {category} through targeted interactions")
                
        # Check for low confidence areas
        for category, values in current_profile.get("confidence_levels", {}).items():
            if values < 0.5:
                recommendations.append(f"Gather more data about {category}")
                
        # Check for pattern consistency
        if analysis.get("pattern_updates"):
            recommendations.append("Update pattern recognition based on new observations")
            
        return recommendations

class ResponseAnalysisAgent(BaseModel):
    """Agent responsible for analyzing player responses in detail"""
    id: str = Field(default_factory=lambda: f"analyze_{random.randint(1000, 9999)}")
    analysis_depth: float = Field(default=0.8, ge=0.0, le=1.0)
    sensitivity: float = Field(default=0.7, ge=0.0, le=1.0)
    analysis_history: List[Dict[str, Any]] = []
    
    async def analyze_response(self, event: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze player response in detail"""
        analysis = {
            "emotional_state": {},
            "reaction_intensity": 0.0,
            "triggered_preferences": [],
            "response_patterns": [],
            "suggested_actions": []
        }
        
        content = str(event.get("content", "")).lower()
        
        # Analyze emotional state
        emotional_state = self._analyze_emotional_state(content)
        if emotional_state:
            analysis["emotional_state"] = emotional_state
            
        # Calculate reaction intensity
        analysis["reaction_intensity"] = self._calculate_reaction_intensity(content, context)
        
        # Identify triggered preferences
        triggered_preferences = self._identify_triggered_preferences(content, context)
        if triggered_preferences:
            analysis["triggered_preferences"] = triggered_preferences
            
        # Identify response patterns
        response_patterns = self._identify_response_patterns(content, context)
        if response_patterns:
            analysis["response_patterns"] = response_patterns
            
        # Generate suggested actions
        suggested_actions = self._generate_suggested_actions(analysis, context)
        if suggested_actions:
            analysis["suggested_actions"] = suggested_actions
            
        return analysis
        
    def _analyze_emotional_state(self, content: str) -> Dict[str, float]:
        """Analyze emotional state from content"""
        emotional_patterns = {
            "arousal": ["aroused", "excited", "turned on", "horny", "hot", "throbbing"],
            "fear": ["scared", "afraid", "terrified", "fear", "anxious", "nervous"],
            "desire": ["want", "desire", "crave", "need", "long", "yearn"],
            "submission": ["submit", "yield", "surrender", "obey", "please", "beg"],
            "dominance": ["control", "power", "authority", "command", "rule", "dominate"],
            "playfulness": ["play", "fun", "tease", "joke", "laugh", "giggle"],
            "seriousness": ["serious", "important", "proper", "correct", "appropriate", "formal"],
            "frustration": ["frustrated", "annoyed", "irritated", "angry", "upset", "mad"],
            "satisfaction": ["satisfied", "pleased", "happy", "content", "fulfilled", "gratified"],
            "anxiety": ["anxious", "nervous", "worried", "concerned", "uneasy", "tense"]
        }
        
        emotional_state = {}
        for emotion, patterns in emotional_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in content)
            if matches > 0:
                emotional_state[emotion] = min(1.0, matches * 0.1)
                
        return emotional_state
        
    def _calculate_reaction_intensity(self, content: str, context: Dict[str, Any]) -> float:
        """Calculate intensity of player's reaction"""
        intensity = 0.5  # Base intensity
        
        # Adjust based on emotional words
        emotional_words = ["very", "extremely", "absolutely", "completely", "totally"]
        for word in emotional_words:
            if word in content:
                intensity += 0.1
                
        # Adjust based on punctuation
        if "!!!" in content or "???" in content:
            intensity += 0.2
        if "!" in content or "?" in content:
            intensity += 0.1
            
        # Adjust based on context
        if context.get("previous_intensity", 0) > 0.7:
            intensity += 0.1
            
        return min(1.0, max(0.0, intensity))
        
    def _identify_triggered_preferences(self, content: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify preferences that were triggered in the response"""
        triggered = []
        
        # Check for triggered kinks
        kink_patterns = {
            "bdsm": ["whip", "chain", "collar", "submission", "dominance"],
            "roleplay": ["fantasy", "scenario", "character", "story"],
            "exhibitionism": ["public", "exposed", "watched", "seen"],
            "voyeurism": ["watching", "observing", "peeking", "spying"]
        }
        
        for kink, patterns in kink_patterns.items():
            if any(pattern in content for pattern in patterns):
                triggered.append({
                    "type": "kink",
                    "name": kink,
                    "confidence": 0.8,
                    "triggered_by": [p for p in patterns if p in content]
                })
                
        # Check for triggered physical preferences
        physical_patterns = {
            "redhead": ["red hair", "ginger", "redhead"],
            "tattooed": ["tattoo", "ink", "tattooed"],
            "tall": ["tall", "height", "towering"],
            "muscular": ["muscle", "strong", "buff", "ripped"]
        }
        
        for trait, patterns in physical_patterns.items():
            if any(pattern in content for pattern in patterns):
                triggered.append({
                    "type": "physical",
                    "name": trait,
                    "confidence": 0.8,
                    "triggered_by": [p for p in patterns if p in content]
                })
                
        return triggered
        
    def _identify_response_patterns(self, content: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify patterns in the response"""
        patterns = []
        
        # Check for response style
        style_patterns = {
            "defensive": ["but", "however", "although", "despite", "even though"],
            "submissive": ["please", "thank", "sorry", "apologize", "obey"],
            "dominant": ["command", "control", "require", "expect", "insist"],
            "playful": ["tease", "play", "fun", "joke", "laugh"],
            "serious": ["serious", "important", "proper", "correct", "appropriate"]
        }
        
        for style, words in style_patterns.items():
            if any(word in content for word in words):
                patterns.append({
                    "type": "style",
                    "name": style,
                    "confidence": 0.7,
                    "triggered_by": [w for w in words if w in content]
                })
                
        # Check for emotional patterns
        emotional_patterns = {
            "escalation": ["more", "harder", "faster", "deeper", "stronger"],
            "resistance": ["stop", "wait", "slow", "gentle", "careful"],
            "surrender": ["yes", "please", "more", "give", "take"],
            "defiance": ["no", "won't", "can't", "don't", "refuse"]
        }
        
        for pattern, words in emotional_patterns.items():
            if any(word in content for word in words):
                patterns.append({
                    "type": "emotional",
                    "name": pattern,
                    "confidence": 0.7,
                    "triggered_by": [w for w in words if w in content]
                })
                
        return patterns
        
    def _generate_suggested_actions(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggested actions based on analysis"""
        suggestions = []
        
        # Check emotional state
        emotional_state = analysis.get("emotional_state", {})
        if emotional_state.get("frustration", 0) > 0.7:
            suggestions.append({
                "type": "adjust_intensity",
                "action": "reduce",
                "reason": "High frustration detected",
                "confidence": 0.8
            })
        elif emotional_state.get("satisfaction", 0) > 0.7:
            suggestions.append({
                "type": "maintain_course",
                "action": "continue",
                "reason": "High satisfaction detected",
                "confidence": 0.8
            })
            
        # Check triggered preferences
        triggered = analysis.get("triggered_preferences", [])
        for pref in triggered:
            if pref["confidence"] > 0.7:
                suggestions.append({
                    "type": "exploit_preference",
                    "action": "emphasize",
                    "preference": pref["name"],
                    "reason": f"Strong preference for {pref['name']} detected",
                    "confidence": pref["confidence"]
                })
                
        # Check response patterns
        patterns = analysis.get("response_patterns", [])
        for pattern in patterns:
            if pattern["confidence"] > 0.7:
                if pattern["name"] == "resistance":
                    suggestions.append({
                        "type": "adjust_approach",
                        "action": "gentle",
                        "reason": "Resistance pattern detected",
                        "confidence": pattern["confidence"]
                    })
                elif pattern["name"] == "surrender":
                    suggestions.append({
                        "type": "maintain_course",
                        "action": "continue",
                        "reason": "Surrender pattern detected",
                        "confidence": pattern["confidence"]
                    })
                    
        return suggestions 