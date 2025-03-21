# nyx/core/reflection_engine.py

import logging
import asyncio
import random
import datetime
import math
import re  # Imported so that _extract_metrics_from_description can work
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

###############################################################################
#           CODE ORIGINALLY FROM reflection_engine.py (UNCHANGED)
###############################################################################

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
        return time_since_reflection > self.reflection_intervals["min_interval"]
    
    async def generate_reflection(self, memories: List[Dict[str, Any]], topic: Optional[str] = None) -> Tuple[str, float]:
        """
        Generate a reflective insight based on memories
        """
        self.reflection_intervals["last_reflection"] = datetime.datetime.now()
        
        if not memories:
            return ("I don't have enough experiences to form a meaningful reflection on this topic yet.", 0.3)
        
        scenario_types = [self._extract_scenario_type(memory) for memory in memories]
        scenario_type = max(set(scenario_types), key=scenario_types.count) if scenario_types else "general"
        
        sentiments = [self._extract_sentiment(memory) for memory in memories]
        sentiment = max(set(sentiments), key=sentiments.count) if sentiments else "neutral"
        
        if len(memories) > 1:
            return await self._generate_complex_reflection(memories, scenario_type, sentiment, topic)
        else:
            return self._generate_simple_reflection(memories[0], scenario_type, sentiment)
    
    def _extract_scenario_type(self, memory: Dict[str, Any]) -> str:
        tags = memory.get("tags", [])
        for tag in tags:
            if tag.lower() in ["teasing", "discipline", "service", "training", "worship"]:
                return tag.lower()
        scenario_type = memory.get("metadata", {}).get("scenario_type")
        if scenario_type:
            return scenario_type.lower()
        return "general"
    
    def _extract_sentiment(self, memory: Dict[str, Any]) -> str:
        emotional_context = memory.get("metadata", {}).get("emotional_context", {})
        valence = emotional_context.get("valence", 0.0)
        if valence > 0.3:
            return "positive"
        elif valence < -0.3:
            return "negative"
        return "neutral"
    
    def _generate_simple_reflection(self, memory: Dict[str, Any], scenario_type: str, sentiment: str) -> Tuple[str, float]:
        patterns = self.reflection_patterns.get(scenario_type, self.reflection_patterns["general"])
        template_sentiment = sentiment if sentiment in ["positive", "negative"] else "positive"
        templates = patterns.get(template_sentiment, patterns["positive"])
        
        template = random.choice(templates)
        template_vars = self._generate_template_variables(memory, scenario_type, sentiment)
        reflection = template.format(**template_vars)
        
        significance = memory.get("significance", 5) / 10.0
        emotional_intensity = memory.get("metadata", {}).get("emotional_context", {}).get("primary_intensity", 0.5)
        confidence = (significance * 0.7) + (emotional_intensity * 0.3)
        
        self._record_reflection(reflection, confidence, [memory.get("id")], scenario_type, sentiment)
        return (reflection, confidence)
    
    async def _generate_complex_reflection(self,
                                           memories: List[Dict[str, Any]],
                                           scenario_type: str,
                                           sentiment: str,
                                           topic: Optional[str] = None) -> Tuple[str, float]:
        memory_texts = [m.get("memory_text", "") for m in memories]
        memory_ids = [m.get("id") for m in memories]
        
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
        
        pattern_template = random.choice(pattern_templates)
        insight_template = random.choice(insight_templates)
        
        words = []
        for memory in memories:
            words.extend(memory.get("memory_text", "").split())
        
        behavior = random.choice(["respond", "react", "behave", "engage", "participate"]) 
        situation = " ".join(random.sample(words, min(5, len(words)))) if words else "certain situations"
        outcome = random.choice(["satisfaction", "frustration", "engagement", "resistance", "submission"])
        reaction = random.choice(["eagerness", "hesitation", "enthusiasm", "reluctance", "compliance"])
        action = random.choice(["challenge", "praise", "tease", "instruct", "discipline"])
        attribute = random.choice(["submission", "resistance", "playfulness", "curiosity", "obedience"])
        circumstance = random.choice(["challenges", "intimate moments", "training sessions", "tests", "conversations"])
        
        insight = insight_template.format(
            behavior=behavior,
            situation=situation,
            outcome=outcome,
            reaction=reaction,
            action=action,
            attribute=attribute,
            circumstance=circumstance
        )
        reflection = pattern_template.format(insight=insight)
        
        avg_significance = sum(m.get("significance", 5) for m in memories) / len(memories) / 10.0
        memory_count_factor = min(1.0, len(memories) / 5.0)
        confidence = (avg_significance * 0.6) + (memory_count_factor * 0.4)
        
        self._record_reflection(reflection, confidence, memory_ids, scenario_type, sentiment, topic)
        return (reflection, confidence)
    
    def _generate_template_variables(self, memory: Dict[str, Any], scenario_type: str, sentiment: str) -> Dict[str, str]:
        subject = random.choice(self.reflection_keywords["subjects"])
        response_verbs = self.reflection_keywords["response_verbs"].get(
            scenario_type, self.reflection_keywords["response_verbs"]["general"]
        )
        response_verb = random.choice(response_verbs)
        
        quality_adj_options = self.reflection_keywords["quality_adj"].get(
            sentiment, self.reflection_keywords["quality_adj"]["neutral"]
        )
        quality_adj = random.choice(quality_adj_options)
        
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
        self.reflection_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "reflection": reflection,
            "confidence": confidence,
            "source_memory_ids": memory_ids,
            "scenario_type": scenario_type,
            "sentiment": sentiment,
            "topic": topic
        })
        if len(self.reflection_history) > 100:
            self.reflection_history = self.reflection_history[-100:]
    
    async def create_abstraction(self, 
                                 memories: List[Dict[str, Any]], 
                                 pattern_type: str = "behavior") -> Tuple[str, Dict[str, Any]]:
        if not memories:
            return ("I don't have enough experiences to form a meaningful abstraction yet.", {})
        
        entity_name = None
        entities = []
        for memory in memories:
            mem_entities = memory.get("metadata", {}).get("entities", [])
            if mem_entities:
                entities.extend(mem_entities)
        
        if entities:
            entity_name = max(set(entities), key=entities.count)
        else:
            entity_name = "the subject"
        
        pattern_desc = await self._extract_pattern_description(memories, pattern_type)
        
        templates = self.abstraction_patterns.get(pattern_type, self.abstraction_patterns["behavior"])
        template = random.choice(templates)
        
        abstraction = template.format(
            name=entity_name.lower(),
            Name=entity_name.capitalize(),
            pattern_desc=pattern_desc
        )
        
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
        memory_texts = [m.get("memory_text", "") for m in memories]
        words = []
        for text in memory_texts:
            words.extend(text.split())
        
        if pattern_type == "behavior":
            behaviors = ["responds to", "reacts when", "behaves during", "engages with"]
            behavior = random.choice(behaviors)
            context = " ".join(random.sample(words, min(3, len(words)))) if words else "certain triggers"
            return f"{behavior} {context}"
        elif pattern_type == "preference":
            preference_types = ["activities involving", "scenarios with", "interactions that include"]
            pref_type = random.choice(preference_types)
            preference = " ".join(random.sample(words, min(3, len(words)))) if words else "some elements"
            return f"{pref_type} {preference}"
        elif pattern_type == "emotional":
            emotions = ["becomes excited", "feels uncomfortable", "shows interest", "expresses hesitation"]
            emotion = random.choice(emotions)
            trigger = " ".join(random.sample(words, min(3, len(words)))) if words else "certain conditions"
            return f"{emotion} when {trigger}"
        elif pattern_type == "relationship":
            dynamics = ["we establish boundaries", "we negotiate desires", "trust develops", "tension arises"]
            dynamic = random.choice(dynamics)
            condition = " ".join(random.sample(words, min(3, len(words)))) if words else "certain situations"
            return f"{dynamic} whenever {condition}"
        
        return "exhibits consistent patterns that merit further exploration"
    
    def _calculate_pattern_confidence(self, memories: List[Dict[str, Any]]) -> float:
        count_factor = min(1.0, len(memories) / 5.0)
        avg_significance = sum(m.get("significance", 5) for m in memories) / len(memories) / 10.0
        
        now = datetime.datetime.now()
        recent_count = 0
        for memory in memories:
            ts_str = memory.get("metadata", {}).get("timestamp")
            if ts_str:
                ts = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if (now - ts).days < 14:
                    recent_count += 1
        recency_factor = recent_count / len(memories)
        
        confidence = (count_factor * 0.4) + (avg_significance * 0.4) + (recency_factor * 0.2)
        return min(1.0, max(0.1, confidence))
    
    async def generate_introspection(self, 
                                     memory_stats: Dict[str, Any], 
                                     player_model: Dict[str, Any] = None) -> Dict[str, Any]:
        memory_count = memory_stats.get("total_memories", 0)
        type_counts = memory_stats.get("type_counts", {})
        avg_significance = memory_stats.get("avg_significance", 5.0)
        
        player_understanding = ""
        if player_model:
            play_style = player_model.get("play_style", {})
            if play_style:
                styles = [f"{style} ({count})" for style, count in play_style.items() if count > 0]
                player_understanding = f"I've observed these tendencies: {', '.join(styles)}. "
        
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
        if avg_significance > 7:
            return "strong and nuanced"
        elif avg_significance > 5:
            return "solid"
        elif avg_significance > 3:
            return "moderate"
        else:
            return "still developing"
    
    def _get_key_scenario_type(self, type_counts: Dict[str, int]) -> str:
        scenario_types = {
            "teasing": type_counts.get("teasing", 0) + type_counts.get("indulgent", 0),
            "discipline": type_counts.get("discipline", 0) + type_counts.get("training", 0),
            "service": type_counts.get("service", 0) + type_counts.get("worship", 0)
        }
        if not scenario_types:
            return "various"
        return max(scenario_types.items(), key=lambda x: x[1])[0]


###############################################################################
#      CODE ORIGINALLY FROM self_reflection_system.py (NOW FOLDED IN)
###############################################################################

class ReflectionSession:
    """Represents a single reflection session"""
    def __init__(self, session_id: str, focus_areas: List[str] = None, 
                 timestamp: Optional[datetime.datetime] = None):
        self.id = session_id
        self.timestamp = timestamp or datetime.datetime.now()
        self.focus_areas = focus_areas or []
        self.insights = []
        self.action_items = []
        self.success_metrics = {}
        self.completed = False
        self.duration = 0
        self.related_data = {}
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "focus_areas": self.focus_areas,
            "insights": self.insights,
            "action_items": self.action_items,
            "success_metrics": self.success_metrics,
            "completed": self.completed,
            "duration": self.duration,
            "related_data": self.related_data
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReflectionSession':
        session = cls(
            session_id=data["id"],
            focus_areas=data["focus_areas"],
            timestamp=datetime.datetime.fromisoformat(data["timestamp"])
        )
        session.insights = data["insights"]
        session.action_items = data["action_items"]
        session.success_metrics = data["success_metrics"]
        session.completed = data["completed"]
        session.duration = data["duration"]
        session.related_data = data["related_data"]
        return session

class Hypothesis:
    """Represents a self-improvement hypothesis"""
    def __init__(self, hypothesis_id: str, statement: str, 
                 confidence: float = 0.5, source: str = "reflection"):
        self.id = hypothesis_id
        self.statement = statement
        self.confidence = confidence
        self.source = source
        self.creation_time = datetime.datetime.now()
        self.last_tested = None
        self.test_results = []
        self.supporting_evidence = []
        self.contradicting_evidence = []
        self.status = "untested"
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "confidence": self.confidence,
            "source": self.source,
            "creation_time": self.creation_time.isoformat(),
            "last_tested": self.last_tested.isoformat() if self.last_tested else None,
            "test_results": self.test_results,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "status": self.status
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Hypothesis':
        hypothesis = cls(
            hypothesis_id=data["id"],
            statement=data["statement"],
            confidence=data["confidence"],
            source=data["source"]
        )
        hypothesis.creation_time = datetime.datetime.fromisoformat(data["creation_time"])
        if data["last_tested"]:
            hypothesis.last_tested = datetime.datetime.fromisoformat(data["last_tested"])
        hypothesis.test_results = data["test_results"]
        hypothesis.supporting_evidence = data["supporting_evidence"]
        hypothesis.contradicting_evidence = data["contradicting_evidence"]
        hypothesis.status = data["status"]
        return hypothesis

class Experiment:
    """Represents a designed experiment to test a hypothesis"""
    def __init__(self, experiment_id: str, hypothesis_id: str, 
                 design: Dict[str, Any], success_criteria: Dict[str, Any]):
        self.id = experiment_id
        self.hypothesis_id = hypothesis_id
        self.design = design
        self.success_criteria = success_criteria
        self.creation_time = datetime.datetime.now()
        self.start_time = None
        self.end_time = None
        self.results = []
        self.analysis = {}
        self.conclusion = ""
        self.status = "created"
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis_id": self.hypothesis_id,
            "design": self.design,
            "success_criteria": self.success_criteria,
            "creation_time": self.creation_time.isoformat(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "results": self.results,
            "analysis": self.analysis,
            "conclusion": self.conclusion,
            "status": self.status
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experiment':
        experiment = cls(
            experiment_id=data["id"],
            hypothesis_id=data["hypothesis_id"],
            design=data["design"],
            success_criteria=data["success_criteria"]
        )
        experiment.creation_time = datetime.datetime.fromisoformat(data["creation_time"])
        if data["start_time"]:
            experiment.start_time = datetime.datetime.fromisoformat(data["start_time"])
        if data["end_time"]:
            experiment.end_time = datetime.datetime.fromisoformat(data["end_time"])
        experiment.results = data["results"]
        experiment.analysis = data["analysis"]
        experiment.conclusion = data["conclusion"]
        experiment.status = data["status"]
        return experiment

class SelfReflectionSystem:
    """System for self-reflection and continuous improvement"""
    
    def __init__(self):
        self.reflection_sessions = []
        self.hypotheses = {}   # id -> Hypothesis
        self.experiments = {}  # id -> Experiment
        self.performance_history = []
        self.decision_history = []
        self.focus_areas = [
            "decision_quality",
            "learning_effectiveness",
            "adaptation_speed",
            "creativity",
            "efficiency",
            "reasoning"
        ]
        self.config = {
            "reflection_interval": 24,
            "reflection_depth": 0.7,
            "min_sessions_for_trends": 3,
            "max_active_experiments": 5,
            "confidence_threshold": 0.7,
            "contradiction_threshold": 0.3
        }
        self.analysis_templates = {
            "decision_quality": self._analyze_decision_quality,
            "learning_effectiveness": self._analyze_learning_effectiveness,
            "adaptation_speed": self._analyze_adaptation_speed,
            "creativity": self._analyze_creativity,
            "efficiency": self._analyze_efficiency,
            "reasoning": self._analyze_reasoning
        }
        self.last_reflection_time = None
        self.next_reflection_time = None
        self.knowledge_system = None
        self.metacognition_system = None
        self.next_session_id = 1
        self.next_hypothesis_id = 1
        self.next_experiment_id = 1
    
    async def initialize(self, system_references: Dict[str, Any]) -> None:
        if "knowledge_system" in system_references:
            self.knowledge_system = system_references["knowledge_system"]
        if "metacognition_system" in system_references:
            self.metacognition_system = system_references["metacognition_system"]
        self._schedule_next_reflection()
        logger.info("Self-Reflection System initialized")
    
    def _schedule_next_reflection(self) -> None:
        now = datetime.datetime.now()
        if not self.last_reflection_time:
            self.next_reflection_time = now + datetime.timedelta(hours=4)
        else:
            self.next_reflection_time = now + datetime.timedelta(hours=self.config["reflection_interval"])
        logger.info(f"Next reflection scheduled for {self.next_reflection_time}")
    
    async def check_reflection_needed(self) -> bool:
        now = datetime.datetime.now()
        if self.next_reflection_time and now >= self.next_reflection_time:
            return True
        if self.performance_history and len(self.performance_history) >= 10:
            recent = self.performance_history[-10:]
            key_metrics = ["success_rate", "error_rate", "efficiency"]
            variance_detected = False
            for metric in key_metrics:
                values = [entry.get(metric, None) for entry in recent]
                values = [v for v in values if v is not None]
                if len(values) >= 5:
                    variance = np.var(values)
                    threshold = 0.1 * np.mean(values)
                    if variance > threshold:
                        variance_detected = True
                        break
            if variance_detected:
                logger.info("Triggering reflection due to performance variance")
                return True
        return False
    
    async def conduct_reflection_session(self, 
                                         force: bool = False, 
                                         focus_areas: List[str] = None) -> Dict[str, Any]:
        if not force and not await self.check_reflection_needed():
            return {"status": "not_needed"}
        
        logger.info("Starting reflection session")
        session_id = f"session_{self.next_session_id}"
        self.next_session_id += 1
        
        focus_areas = focus_areas or self.focus_areas
        session = ReflectionSession(session_id, focus_areas)
        
        performance_data = await self._collect_performance_data()
        decision_data = await self._collect_decision_data()
        experiment_data = await self._collect_experiment_data()
        
        session.related_data = {
            "performance": performance_data,
            "decisions": decision_data,
            "experiments": experiment_data
        }
        
        for area in focus_areas:
            if area in self.analysis_templates:
                analysis_func = self.analysis_templates[area]
                area_insights = await analysis_func(performance_data, decision_data, experiment_data)
                session.insights.extend(area_insights.get("insights", []))
                session.action_items.extend(area_insights.get("action_items", []))
                for metric, value in area_insights.get("success_metrics", {}).items():
                    session.success_metrics[f"{area}_{metric}"] = value
        
        await self._generate_hypotheses_from_insights(session.insights)
        
        session.completed = True
        session.duration = (datetime.datetime.now() - session.timestamp).total_seconds()
        self.reflection_sessions.append(session)
        
        self.last_reflection_time = session.timestamp
        self._schedule_next_reflection()
        
        if self.knowledge_system:
            await self._share_insights_with_knowledge_system(session)
        
        logger.info(f"Completed reflection session {session_id}")
        return session.to_dict()
    
    async def _collect_performance_data(self) -> Dict[str, Any]:
        performance_data = {
            "recent_performance": self.performance_history[-20:] if self.performance_history else [],
            "trends": {},
            "anomalies": []
        }
        if len(self.performance_history) >= self.config["min_sessions_for_trends"]:
            for metric in ["success_rate", "error_rate", "efficiency", "response_time"]:
                values = [entry.get(metric, None) for entry in self.performance_history]
                values = [v for v in values if v is not None]
                if len(values) >= self.config["min_sessions_for_trends"]:
                    trend = self._calculate_trend(values)
                    performance_data["trends"][metric] = trend
                    anomalies = self._detect_anomalies(values, trend)
                    if anomalies:
                        for anomaly in anomalies:
                            performance_data["anomalies"].append({
                                "metric": metric,
                                "value": anomaly["value"],
                                "expected": anomaly["expected"],
                                "deviation": anomaly["deviation"],
                                "index": anomaly["index"]
                            })
        if self.metacognition_system:
            try:
                system_metrics = await self.metacognition_system.collect_performance_metrics()
                performance_data["system_metrics"] = system_metrics
            except Exception as e:
                logger.error(f"Error collecting system metrics: {str(e)}")
        return performance_data
    
    async def _collect_decision_data(self) -> Dict[str, Any]:
        decision_data = {
            "recent_decisions": self.decision_history[-20:] if self.decision_history else [],
            "decision_types": {},
            "outcome_distribution": {},
            "confidence_accuracy": {}
        }
        if self.decision_history:
            for decision in self.decision_history:
                decision_type = decision.get("type", "unknown")
                if decision_type not in decision_data["decision_types"]:
                    decision_data["decision_types"][decision_type] = 0
                decision_data["decision_types"][decision_type] += 1
                outcome = decision.get("outcome", "unknown")
                if outcome not in decision_data["outcome_distribution"]:
                    decision_data["outcome_distribution"][outcome] = 0
                decision_data["outcome_distribution"][outcome] += 1
                if "confidence" in decision and "success" in decision:
                    confidence = decision["confidence"]
                    success = decision["success"]
                    confidence_bin = round(confidence * 10) / 10
                    if confidence_bin not in decision_data["confidence_accuracy"]:
                        decision_data["confidence_accuracy"][confidence_bin] = {"total": 0, "correct": 0}
                    decision_data["confidence_accuracy"][confidence_bin]["total"] += 1
                    if success:
                        decision_data["confidence_accuracy"][confidence_bin]["correct"] += 1
        return decision_data
    
    async def _collect_experiment_data(self) -> Dict[str, Any]:
        experiment_data = {
            "completed_experiments": [],
            "active_experiments": [],
            "success_rate": 0.0,
            "confirmation_rate": 0.0,
            "rejection_rate": 0.0
        }
        completed = []
        active = []
        for experiment in self.experiments.values():
            if experiment.status == "completed":
                completed.append(experiment.to_dict())
            elif experiment.status in ["created", "running"]:
                active.append(experiment.to_dict())
        experiment_data["completed_experiments"] = completed
        experiment_data["active_experiments"] = active
        if completed:
            successful = sum(1 for exp in completed if exp["conclusion"])
            experiment_data["success_rate"] = successful / len(completed)
            confirmed = sum(1 for exp in completed if "confirmed" in exp["conclusion"].lower())
            experiment_data["confirmation_rate"] = confirmed / len(completed) if len(completed) > 0 else 0
            rejected = sum(1 for exp in completed if "rejected" in exp["conclusion"].lower())
            experiment_data["rejection_rate"] = rejected / len(completed) if len(completed) > 0 else 0
        return experiment_data
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        if len(values) < 2:
            return {"direction": "stable", "magnitude": 0.0}
        n = len(values)
        x = list(range(n))
        mean_x = sum(x) / n
        mean_y = sum(values) / n
        numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        if denominator == 0:
            return {"direction": "stable", "magnitude": 0.0}
        slope = numerator / denominator
        if mean_y != 0:
            normalized_slope = slope / abs(mean_y)
        else:
            normalized_slope = slope
        if abs(normalized_slope) < 0.05:
            direction = "stable"
        elif normalized_slope > 0:
            direction = "improving"
        else:
            direction = "declining"
        return {
            "direction": direction,
            "magnitude": abs(normalized_slope),
            "slope": slope,
            "mean": mean_y
        }
    
    def _detect_anomalies(self, values: List[float], trend: Dict[str, Any]) -> List[Dict[str, Any]]:
        anomalies = []
        if len(values) < 5:
            return anomalies
        mean = trend["mean"]
        std_dev = np.std(values)
        slope = trend.get("slope", 0)
        for i in range(len(values)):
            expected = mean + slope * (i - len(values) / 2)
            deviation = abs(values[i] - expected) / std_dev if std_dev else 0
            if deviation > 2.0:
                anomalies.append({
                    "index": i,
                    "value": values[i],
                    "expected": expected,
                    "deviation": deviation
                })
        return anomalies
    
    async def _analyze_decision_quality(self, performance_data: Dict[str, Any],
                                        decision_data: Dict[str, Any],
                                        experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        insights = []
        action_items = []
        success_metrics = {}
        
        success_rate = 0.0
        if "outcome_distribution" in decision_data:
            outcomes = decision_data["outcome_distribution"]
            successes = outcomes.get("success", 0) + outcomes.get("correct", 0)
            total = sum(outcomes.values())
            if total > 0:
                success_rate = successes / total
        success_metrics["success_rate"] = success_rate
        
        if "confidence_accuracy" in decision_data and decision_data["confidence_accuracy"]:
            calibration_data = decision_data["confidence_accuracy"]
            calibration_error = 0.0
            calibration_points = 0
            for confidence, data in calibration_data.items():
                if data["total"] > 0:
                    accuracy = data["correct"] / data["total"]
                    error = abs(confidence - accuracy)
                    calibration_error += error
                    calibration_points += 1
            if calibration_points > 0:
                avg_calibration_error = calibration_error / calibration_points
                success_metrics["calibration_error"] = avg_calibration_error
                if avg_calibration_error > 0.2:
                    insights.append({
                        "type": "issue",
                        "area": "decision_quality",
                        "description": f"Significant confidence calibration error of {avg_calibration_error:.2f}",
                        "severity": "high" if avg_calibration_error > 0.3 else "medium",
                        "confidence": 0.9
                    })
                    action_items.append({
                        "type": "improve_calibration",
                        "description": "Implement confidence calibration training",
                        "priority": "high",
                        "expected_impact": 0.7
                    })
                else:
                    insights.append({
                        "type": "strength",
                        "area": "decision_quality",
                        "description": f"Well-calibrated confidence with error of only {avg_calibration_error:.2f}",
                        "confidence": 0.9
                    })
        
        if "recent_decisions" in decision_data and decision_data["recent_decisions"]:
            decisions_by_type = {}
            for d in decision_data["recent_decisions"]:
                decision_type = d.get("type", "unknown")
                if decision_type not in decisions_by_type:
                    decisions_by_type[decision_type] = {"count": 0, "successes": 0}
                decisions_by_type[decision_type]["count"] += 1
                if d.get("success", False):
                    decisions_by_type[decision_type]["successes"] += 1
            for dt, stats in decisions_by_type.items():
                if stats["count"] >= 5:
                    type_srate = stats["successes"] / stats["count"]
                    if type_srate < 0.5:
                        insights.append({
                            "type": "issue",
                            "area": "decision_quality",
                            "description": f"Low success rate of {type_srate:.2f} for {dt} decisions",
                            "severity": "high" if type_srate < 0.3 else "medium",
                            "confidence": min(0.5 + stats["count"] / 10, 0.9)
                        })
                        action_items.append({
                            "type": "improve_decision_type",
                            "description": f"Develop better strategy for {dt} decisions",
                            "priority": "high" if type_srate < 0.3 else "medium",
                            "expected_impact": 0.8
                        })
                    elif type_srate > 0.8:
                        insights.append({
                            "type": "strength",
                            "area": "decision_quality",
                            "description": f"High success rate of {type_srate:.2f} for {dt} decisions",
                            "confidence": min(0.5 + stats["count"] / 10, 0.9)
                        })
        
        if success_rate < 0.6:
            insights.append({
                "type": "issue",
                "area": "decision_quality",
                "description": f"Overall decision success rate is low at {success_rate:.2f}",
                "severity": "high" if success_rate < 0.4 else "medium",
                "confidence": 0.8
            })
            action_items.append({
                "type": "improve_overall_decisions",
                "description": "Implement more comprehensive decision analysis framework",
                "priority": "high",
                "expected_impact": 0.9
            })
        elif success_rate > 0.8:
            insights.append({
                "type": "strength",
                "area": "decision_quality",
                "description": f"Excellent overall decision success rate of {success_rate:.2f}",
                "confidence": 0.8
            })
        
        return {"insights": insights, "action_items": action_items, "success_metrics": success_metrics}
    
    async def _analyze_learning_effectiveness(self, performance_data: Dict[str, Any],
                                              decision_data: Dict[str, Any],
                                              experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        insights = []
        action_items = []
        success_metrics = {}
        
        if "success_rate" in experiment_data:
            srate = experiment_data["success_rate"]
            success_metrics["experiment_success"] = srate
            if srate < 0.6:
                insights.append({
                    "type": "issue",
                    "area": "learning_effectiveness",
                    "description": f"Experiment success rate is low at {srate:.2f}",
                    "severity": "medium",
                    "confidence": 0.7
                })
                action_items.append({
                    "type": "improve_experiments",
                    "description": "Revise experiment design methodology for better results",
                    "priority": "medium",
                    "expected_impact": 0.7
                })
            elif srate > 0.8:
                insights.append({
                    "type": "strength",
                    "area": "learning_effectiveness",
                    "description": f"Excellent experiment success rate of {srate:.2f}",
                    "confidence": 0.7
                })
        
        mistake_learning_rate = 0.0
        if "recent_decisions" in decision_data and decision_data["recent_decisions"]:
            mistakes = [d for d in decision_data["recent_decisions"] if not d.get("success", True)]
            repeated_mistakes = 0
            mistake_types = {}
            for m in mistakes:
                mt = m.get("type", "unknown")
                if mt not in mistake_types:
                    mistake_types[mt] = 0
                mistake_types[mt] += 1
            for mt, count in mistake_types.items():
                if count > 1:
                    repeated_mistakes += (count - 1)
            if len(mistakes) > 0:
                mistake_learning_rate = 1.0 - (repeated_mistakes / len(mistakes))
                success_metrics["mistake_learning"] = mistake_learning_rate
                if mistake_learning_rate < 0.5:
                    insights.append({
                        "type": "issue",
                        "area": "learning_effectiveness",
                        "description": f"Frequently repeating similar mistakes, learning rate only {mistake_learning_rate:.2f}",
                        "severity": "high",
                        "confidence": 0.8
                    })
                    action_items.append({
                        "type": "improve_mistake_learning",
                        "description": "Implement mistake analysis and prevention system",
                        "priority": "high",
                        "expected_impact": 0.9
                    })
                elif mistake_learning_rate > 0.8:
                    insights.append({
                        "type": "strength",
                        "area": "learning_effectiveness",
                        "description": f"Rarely repeats mistakes, excellent learning rate of {mistake_learning_rate:.2f}",
                        "confidence": 0.8
                    })
        
        if "trends" in performance_data:
            improvement_metrics = []
            for metric, trend in performance_data["trends"].items():
                if metric in ["success_rate", "accuracy", "efficiency"]:
                    if trend["direction"] == "improving" and trend["magnitude"] > 0.1:
                        improvement_metrics.append(metric)
            improvement_rate = len(improvement_metrics) / len(performance_data["trends"]) if performance_data["trends"] else 0
            success_metrics["improvement_rate"] = improvement_rate
            if improvement_rate < 0.3:
                insights.append({
                    "type": "issue",
                    "area": "learning_effectiveness",
                    "description": f"Low improvement rate across metrics: {improvement_rate:.2f}",
                    "severity": "medium",
                    "confidence": 0.7
                })
                action_items.append({
                    "type": "accelerate_learning",
                    "description": "Implement more aggressive learning rate parameters",
                    "priority": "medium",
                    "expected_impact": 0.7
                })
            elif improvement_rate > 0.7:
                insights.append({
                    "type": "strength",
                    "area": "learning_effectiveness",
                    "description": f"Strong improvement across multiple metrics: {improvement_rate:.2f}",
                    "confidence": 0.7
                })
        if "confirmation_rate" in experiment_data and "rejection_rate" in experiment_data:
            c_rate = experiment_data["confirmation_rate"]
            r_rate = experiment_data["rejection_rate"]
            if r_rate > 0:
                crr = c_rate / r_rate
                success_metrics["confirm_reject_ratio"] = crr
                if crr > 5.0:
                    insights.append({
                        "type": "issue",
                        "area": "learning_effectiveness",
                        "description": f"Confirmation bias detected: {crr:.2f} times more likely to confirm than reject hypotheses",
                        "severity": "medium",
                        "confidence": 0.8
                    })
                    action_items.append({
                        "type": "reduce_confirmation_bias",
                        "description": "Implement stronger hypothesis falsification protocols",
                        "priority": "medium",
                        "expected_impact": 0.7
                    })
                elif 0.5 <= crr <= 2.0:
                    insights.append({
                        "type": "strength",
                        "area": "learning_effectiveness",
                        "description": f"Balanced hypothesis testing with confirmation/rejection ratio of {crr:.2f}",
                        "confidence": 0.8
                    })
            elif c_rate > 0.2:
                insights.append({
                    "type": "issue",
                    "area": "learning_effectiveness",
                    "description": "Potentially biased hypothesis testing: confirming hypotheses but never rejecting any",
                    "severity": "medium",
                    "confidence": 0.7
                })
                action_items.append({
                    "type": "improve_falsification",
                    "description": "Implement explicit falsification criteria for all hypotheses",
                    "priority": "medium",
                    "expected_impact": 0.6
                })
        return {"insights": insights, "action_items": action_items, "success_metrics": success_metrics}
    
    async def _analyze_adaptation_speed(self, performance_data: Dict[str, Any],
                                        decision_data: Dict[str, Any],
                                        experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        insights = []
        action_items = []
        success_metrics = {}
        if "anomalies" in performance_data and performance_data["anomalies"]:
            anomalies = performance_data["anomalies"]
            recovery_times = []
            for anomaly in anomalies:
                index = anomaly["index"]
                metric = anomaly["metric"]
                if "recent_performance" in performance_data:
                    values = [e.get(metric) for e in performance_data["recent_performance"][index:]]
                    values = [v for v in values if v is not None]
                    if values:
                        recovery_index = None
                        expected = anomaly["expected"]
                        for i, val in enumerate(values):
                            if abs(val - expected) <= anomaly["deviation"] / 2:
                                recovery_index = i
                                break
                        if recovery_index is not None:
                            recovery_times.append(recovery_index + 1)
            if recovery_times:
                avg_recovery_time = sum(recovery_times) / len(recovery_times)
                success_metrics["anomaly_recovery_time"] = avg_recovery_time
                if avg_recovery_time > 5:
                    insights.append({
                        "type": "issue",
                        "area": "adaptation_speed",
                        "description": f"Slow recovery from anomalies, averaging {avg_recovery_time:.1f} steps",
                        "severity": "medium",
                        "confidence": 0.7
                    })
                    action_items.append({
                        "type": "improve_recovery",
                        "description": "Implement faster anomaly detection and response system",
                        "priority": "medium",
                        "expected_impact": 0.7
                    })
                elif avg_recovery_time < 2:
                    insights.append({
                        "type": "strength",
                        "area": "adaptation_speed",
                        "description": f"Excellent recovery from anomalies, averaging only {avg_recovery_time:.1f} steps",
                        "confidence": 0.7
                    })
        if len(self.reflection_sessions) >= 2:
            action_completion_times = []
            completed_actions = set()
            for session in sorted(self.reflection_sessions, key=lambda s: s.timestamp):
                for action in session.action_items:
                    action_key = f"{action['type']}_{action['description']}"
                    if action_key not in completed_actions:
                        first_session = next(
                            (s for s in sorted(self.reflection_sessions, key=lambda x: x.timestamp) 
                             if any(f"{a['type']}_{a['description']}" == action_key for a in s.action_items)),
                            None
                        )
                        if first_session and first_session.timestamp < session.timestamp:
                            completion_time = (session.timestamp - first_session.timestamp).total_seconds() / (24 * 3600)
                            action_completion_times.append(completion_time)
                            completed_actions.add(action_key)
            if action_completion_times:
                avg_completion_time = sum(action_completion_times) / len(action_completion_times)
                success_metrics["action_completion_time"] = avg_completion_time
                if avg_completion_time > 14:
                    insights.append({
                        "type": "issue",
                        "area": "adaptation_speed",
                        "description": f"Slow implementation of action items, averaging {avg_completion_time:.1f} days",
                        "severity": "high" if avg_completion_time > 30 else "medium",
                        "confidence": 0.8
                    })
                    action_items.append({
                        "type": "improve_action_implementation",
                        "description": "Create action tracking system with deadlines",
                        "priority": "high",
                        "expected_impact": 0.8
                    })
                elif avg_completion_time < 7:
                    insights.append({
                        "type": "strength",
                        "area": "adaptation_speed",
                        "description": f"Quick implementation of action items, averaging only {avg_completion_time:.1f} days",
                        "confidence": 0.8
                    })
        if "trends" in performance_data:
            trend_responsiveness = []
            for metric, trend in performance_data["trends"].items():
                if trend["direction"] == "declining" and trend["magnitude"] > 0.1:
                    recent_actions = []
                    for session in sorted(self.reflection_sessions, key=lambda s: s.timestamp, reverse=True)[:3]:
                        recent_actions.extend(session.action_items)
                    responsive = any(metric.lower() in act["description"].lower() for act in recent_actions)
                    trend_responsiveness.append(1.0 if responsive else 0.0)
            if trend_responsiveness:
                avg_responsiveness = sum(trend_responsiveness) / len(trend_responsiveness)
                success_metrics["trend_responsiveness"] = avg_responsiveness
                if avg_responsiveness < 0.5:
                    insights.append({
                        "type": "issue",
                        "area": "adaptation_speed",
                        "description": f"Poor responsiveness to negative trends, only {avg_responsiveness:.2f} response rate",
                        "severity": "high",
                        "confidence": 0.8
                    })
                    action_items.append({
                        "type": "improve_trend_response",
                        "description": "Implement automated trend alerting and response system",
                        "priority": "high",
                        "expected_impact": 0.9
                    })
                elif avg_responsiveness > 0.8:
                    insights.append({
                        "type": "strength",
                        "area": "adaptation_speed",
                        "description": f"Excellent responsiveness to negative trends, {avg_responsiveness:.2f} response rate",
                        "confidence": 0.8
                    })
        return {"insights": insights, "action_items": action_items, "success_metrics": success_metrics}
    
    async def _analyze_creativity(self, performance_data: Dict[str, Any],
                                  decision_data: Dict[str, Any],
                                  experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        insights = []
        action_items = []
        success_metrics = {}
        
        strategy_diversity = 0.0
        if "recent_decisions" in decision_data and decision_data["recent_decisions"]:
            strategies = [d.get("strategy", "unknown") for d in decision_data["recent_decisions"]]
            unique_strategies = set(strategies)
            if strategies:
                strategy_counts = {}
                for stg in strategies:
                    if stg not in strategy_counts:
                        strategy_counts[stg] = 0
                    strategy_counts[stg] += 1
                proportions = [cnt / len(strategies) for cnt in strategy_counts.values()]
                shannon_diversity = -sum(p * math.log2(p) for p in proportions)
                max_diversity = math.log2(len(unique_strategies)) if len(unique_strategies) > 1 else 1
                if max_diversity > 0:
                    strategy_diversity = shannon_diversity / max_diversity
                success_metrics["strategy_diversity"] = strategy_diversity
                if strategy_diversity < 0.5:
                    insights.append({
                        "type": "issue",
                        "area": "creativity",
                        "description": f"Low strategy diversity of {strategy_diversity:.2f}, using only {len(unique_strategies)} unique strategies",
                        "severity": "medium",
                        "confidence": 0.8
                    })
                    action_items.append({
                        "type": "increase_strategy_diversity",
                        "description": "Implement exploration phase in decision making",
                        "priority": "medium",
                        "expected_impact": 0.7
                    })
                elif strategy_diversity > 0.8:
                    insights.append({
                        "type": "strength",
                        "area": "creativity",
                        "description": f"High strategy diversity of {strategy_diversity:.2f} across {len(unique_strategies)} unique strategies",
                        "confidence": 0.8
                    })
        if "completed_experiments" in experiment_data:
            experiments = experiment_data["completed_experiments"]
            if experiments:
                hypothesis_ids = [exp["hypothesis_id"] for exp in experiments]
                novel_hypotheses = 0
                for h_id in hypothesis_ids:
                    if h_id in self.hypotheses:
                        h = self.hypotheses[h_id]
                        if h.source in ["creative", "novel"]:
                            novel_hypotheses += 1
                novelty_rate = novel_hypotheses / len(hypothesis_ids) if hypothesis_ids else 0
                success_metrics["hypothesis_novelty"] = novelty_rate
                if novelty_rate < 0.3:
                    insights.append({
                        "type": "issue",
                        "area": "creativity",
                        "description": f"Low rate of novel hypothesis generation: {novelty_rate:.2f}",
                        "severity": "medium",
                        "confidence": 0.7
                    })
                    action_items.append({
                        "type": "increase_novel_hypotheses",
                        "description": "Implement creative hypothesis generation sessions",
                        "priority": "medium",
                        "expected_impact": 0.6
                    })
                elif novelty_rate > 0.7:
                    insights.append({
                        "type": "strength",
                        "area": "creativity",
                        "description": f"High rate of novel hypothesis generation: {novelty_rate:.2f}",
                        "confidence": 0.7
                    })
        originality_score = 0.0
        if "recent_decisions" in decision_data and decision_data["recent_decisions"]:
            decisions = [d for d in decision_data["recent_decisions"] if d.get("success", False)]
            if decisions:
                original_decisions = sum(1 for d in decisions if "original" in d.get("tags", []) or "creative" in d.get("tags", []))
                originality_score = original_decisions / len(decisions)
                success_metrics["solution_originality"] = originality_score
                if originality_score < 0.2:
                    insights.append({
                        "type": "issue",
                        "area": "creativity",
                        "description": f"Low originality in successful solutions: {originality_score:.2f}",
                        "severity": "low",
                        "confidence": 0.6
                    })
                    action_items.append({
                        "type": "increase_solution_originality",
                        "description": "Implement brainstorming phase before decision making",
                        "priority": "low",
                        "expected_impact": 0.5
                    })
                elif originality_score > 0.6:
                    insights.append({
                        "type": "strength",
                        "area": "creativity",
                        "description": f"High originality in successful solutions: {originality_score:.2f}",
                        "confidence": 0.6
                    })
        if len(success_metrics) >= 2:
            avg_creativity = sum(success_metrics.values()) / len(success_metrics)
            success_metrics["overall_creativity"] = avg_creativity
            if avg_creativity < 0.4:
                insights.append({
                    "type": "issue",
                    "area": "creativity",
                    "description": f"Overall creativity is low at {avg_creativity:.2f}",
                    "severity": "medium",
                    "confidence": 0.8
                })
                action_items.append({
                    "type": "boost_overall_creativity",
                    "description": "Implement comprehensive creativity enhancement program",
                    "priority": "medium",
                    "expected_impact": 0.8
                })
            elif avg_creativity > 0.7:
                insights.append({
                    "type": "strength",
                    "area": "creativity",
                    "description": f"Overall creativity is high at {avg_creativity:.2f}",
                    "confidence": 0.8
                })
        return {"insights": insights, "action_items": action_items, "success_metrics": success_metrics}
    
    async def _analyze_efficiency(self, performance_data: Dict[str, Any],
                                  decision_data: Dict[str, Any],
                                  experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        insights = []
        action_items = []
        success_metrics = {}
        
        if "trends" in performance_data and "response_time" in performance_data["trends"]:
            trend = performance_data["trends"]["response_time"]
            success_metrics["response_time_trend"] = -trend["magnitude"] if trend["direction"] == "improving" else trend["magnitude"]
            if trend["direction"] == "declining" and trend["magnitude"] > 0.1:
                insights.append({
                    "type": "issue",
                    "area": "efficiency",
                    "description": f"Response times are getting worse, trend magnitude: {trend['magnitude']:.2f}",
                    "severity": "high" if trend["magnitude"] > 0.3 else "medium",
                    "confidence": 0.9
                })
                action_items.append({
                    "type": "optimize_response_time",
                    "description": "Profile and optimize slow operations",
                    "priority": "high",
                    "expected_impact": 0.8
                })
            elif trend["direction"] == "improving" and trend["magnitude"] > 0.1:
                insights.append({
                    "type": "strength",
                    "area": "efficiency",
                    "description": f"Response times are improving, trend magnitude: {trend['magnitude']:.2f}",
                    "confidence": 0.9
                })
        if "system_metrics" in performance_data:
            system_metrics = performance_data["system_metrics"]
            if "resource_utilization" in system_metrics:
                utilization = system_metrics["resource_utilization"]
                for resource, usage in utilization.items():
                    if usage > 0.9:
                        insights.append({
                            "type": "issue",
                            "area": "efficiency",
                            "description": f"Very high {resource} utilization at {usage:.2f}",
                            "severity": "high",
                            "confidence": 0.9
                        })
                        action_items.append({
                            "type": f"optimize_{resource}_usage",
                            "description": f"Implement {resource} optimization strategy",
                            "priority": "high",
                            "expected_impact": 0.8
                        })
                    elif usage < 0.1:
                        insights.append({
                            "type": "issue",
                            "area": "efficiency",
                            "description": f"Very low {resource} utilization at {usage:.2f}",
                            "severity": "low",
                            "confidence": 0.7
                        })
                        action_items.append({
                            "type": f"improve_{resource}_utilization",
                            "description": f"Find ways to better utilize {resource}",
                            "priority": "low",
                            "expected_impact": 0.4
                        })
                avg_utilization = sum(utilization.values()) / len(utilization)
                success_metrics["resource_utilization"] = avg_utilization
                if 0.4 <= avg_utilization <= 0.7:
                    insights.append({
                        "type": "strength",
                        "area": "efficiency",
                        "description": f"Good average resource utilization at {avg_utilization:.2f}",
                        "confidence": 0.8
                    })
            if "operation_counts" in system_metrics:
                operation_counts = system_metrics["operation_counts"]
                for operation, count in operation_counts.items():
                    if count > 1000:
                        insights.append({
                            "type": "issue",
                            "area": "efficiency",
                            "description": f"Unusually high count ({count}) of {operation} operations",
                            "severity": "medium",
                            "confidence": 0.7
                        })
                        action_items.append({
                            "type": "reduce_operation_count",
                            "description": f"Optimize or cache {operation} operations",
                            "priority": "medium",
                            "expected_impact": 0.6
                        })
        if "recent_decisions" in decision_data and decision_data["recent_decisions"]:
            decision_times = [d.get("decision_time", 0) for d in decision_data["recent_decisions"]]
            decision_times = [t for t in decision_times if t > 0]
            if decision_times:
                avg_decision_time = sum(decision_times) / len(decision_times)
                success_metrics["decision_time"] = avg_decision_time
                if avg_decision_time > 2.0:
                    insights.append({
                        "type": "issue",
                        "area": "efficiency",
                        "description": f"Slow average decision time of {avg_decision_time:.2f} seconds",
                        "severity": "medium",
                        "confidence": 0.8
                    })
                    action_items.append({
                        "type": "optimize_decision_process",
                        "description": "Streamline decision-making process",
                        "priority": "medium",
                        "expected_impact": 0.7
                    })
                elif avg_decision_time < 0.5:
                    insights.append({
                        "type": "strength",
                        "area": "efficiency",
                        "description": f"Fast average decision time of {avg_decision_time:.2f} seconds",
                        "confidence": 0.8
                    })
        return {"insights": insights, "action_items": action_items, "success_metrics": success_metrics}
    
    async def _analyze_reasoning(self, performance_data: Dict[str, Any],
                                 decision_data: Dict[str, Any],
                                 experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        insights = []
        action_items = []
        success_metrics = {}
        
        if "recent_decisions" in decision_data and decision_data["recent_decisions"]:
            logical_errors = 0
            total_decisions = len(decision_data["recent_decisions"])
            for d in decision_data["recent_decisions"]:
                if "error_type" in d and d["error_type"] in ["logical", "reasoning", "fallacy"]:
                    logical_errors += 1
            if total_decisions > 0:
                logical_error_rate = logical_errors / total_decisions
                success_metrics["logical_error_rate"] = logical_error_rate
                if logical_error_rate > 0.1:
                    insights.append({
                        "type": "issue",
                        "area": "reasoning",
                        "description": f"High rate of logical errors: {logical_error_rate:.2f}",
                        "severity": "high" if logical_error_rate > 0.2 else "medium",
                        "confidence": 0.9
                    })
                    action_items.append({
                        "type": "reduce_logical_errors",
                        "description": "Implement formal logic verification in decision process",
                        "priority": "high" if logical_error_rate > 0.2 else "medium",
                        "expected_impact": 0.9
                    })
                elif logical_error_rate < 0.02:
                    insights.append({
                        "type": "strength",
                        "area": "reasoning",
                        "description": f"Very low rate of logical errors: {logical_error_rate:.2f}",
                        "confidence": 0.9
                    })
        if "completed_experiments" in experiment_data and experiment_data["completed_experiments"]:
            methodological_issues = 0
            total_experiments = len(experiment_data["completed_experiments"])
            for exp in experiment_data["completed_experiments"]:
                design = exp["design"]
                missing_components = []
                for component in ["control", "variables", "measures", "success_criteria"]:
                    if component not in design or not design[component]:
                        missing_components.append(component)
                if missing_components:
                    methodological_issues += 1
            if total_experiments > 0:
                methodology_quality = 1.0 - (methodological_issues / total_experiments)
                success_metrics["methodology_quality"] = methodology_quality
                if methodology_quality < 0.7:
                    insights.append({
                        "type": "issue",
                        "area": "reasoning",
                        "description": f"Methodological issues in experiments, quality score: {methodology_quality:.2f}",
                        "severity": "medium",
                        "confidence": 0.8
                    })
                    action_items.append({
                        "type": "improve_methodology",
                        "description": "Implement stricter experiment design requirements",
                        "priority": "medium",
                        "expected_impact": 0.7
                    })
                elif methodology_quality > 0.9:
                    insights.append({
                        "type": "strength",
                        "area": "reasoning",
                        "description": f"Excellent methodology in experiments, quality score: {methodology_quality:.2f}",
                        "confidence": 0.8
                    })
        if "recent_decisions" in decision_data and decision_data["recent_decisions"]:
            reasoning_complexity = []
            for d in decision_data["recent_decisions"]:
                if "reasoning" in d and isinstance(d["reasoning"], list):
                    reasoning_steps = len(d["reasoning"])
                    reasoning_complexity.append(reasoning_steps)
            if reasoning_complexity:
                avg_complexity = sum(reasoning_complexity) / len(reasoning_complexity)
                success_metrics["reasoning_complexity"] = avg_complexity
                if avg_complexity < 2:
                    insights.append({
                        "type": "issue",
                        "area": "reasoning",
                        "description": f"Overly simplistic reasoning with only {avg_complexity:.1f} steps on average",
                        "severity": "medium",
                        "confidence": 0.7
                    })
                    action_items.append({
                        "type": "increase_reasoning_depth",
                        "description": "Implement multi-step reasoning process",
                        "priority": "medium",
                        "expected_impact": 0.6
                    })
                elif avg_complexity > 5:
                    insights.append({
                        "type": "strength",
                        "area": "reasoning",
                        "description": f"Thorough reasoning with {avg_complexity:.1f} steps on average",
                        "confidence": 0.7
                    })
        counterfactual_usage = 0.0
        if "recent_decisions" in decision_data and decision_data["recent_decisions"]:
            counterfactual_count = 0
            total_decisions = len(decision_data["recent_decisions"])
            for d in decision_data["recent_decisions"]:
                if "counterfactuals" in d and d["counterfactuals"]:
                    counterfactual_count += 1
            if total_decisions > 0:
                counterfactual_usage = counterfactual_count / total_decisions
                success_metrics["counterfactual_usage"] = counterfactual_usage
                if counterfactual_usage < 0.3:
                    insights.append({
                        "type": "issue",
                        "area": "reasoning",
                        "description": f"Low use of counterfactual reasoning: {counterfactual_usage:.2f}",
                        "severity": "medium",
                        "confidence": 0.7
                    })
                    action_items.append({
                        "type": "increase_counterfactuals",
                        "description": "Add counterfactual analysis to decision process",
                        "priority": "medium",
                        "expected_impact": 0.7
                    })
                elif counterfactual_usage > 0.7:
                    insights.append({
                        "type": "strength",
                        "area": "reasoning",
                        "description": f"Strong use of counterfactual reasoning: {counterfactual_usage:.2f}",
                        "confidence": 0.7
                    })
        return {"insights": insights, "action_items": action_items, "success_metrics": success_metrics}
    
    async def _generate_hypotheses_from_insights(self, insights: List[Dict[str, Any]]) -> None:
        issues = [i for i in insights if i["type"] == "issue"]
        for issue in issues:
            hypothesis_id = f"hypothesis_{self.next_hypothesis_id}"
            self.next_hypothesis_id += 1
            area = issue["area"]
            description = issue["description"]
            statement = f"Implementing improvements in {area} will address the issue: '{description}'"
            confidence = issue.get("confidence", 0.5)
            
            hypothesis = Hypothesis(
                hypothesis_id=hypothesis_id,
                statement=statement,
                confidence=confidence,
                source="reflection"
            )
            metrics = self._extract_metrics_from_description(description)
            hypothesis.supporting_evidence.append({
                "type": "insight",
                "description": description,
                "confidence": confidence,
                "metrics": metrics
            })
            self.hypotheses[hypothesis_id] = hypothesis
            await self._design_experiment_for_hypothesis(hypothesis)
    
    def _extract_metrics_from_description(self, description: str) -> Dict[str, float]:
        metrics = {}
        patterns = [
            r"(\w+) of ([\d\.]+)",
            r"(\w+): ([\d\.]+)",
            r"(\w+) is ([\d\.]+)",
            r"(\w+) rate of ([\d\.]+)"
        ]
        for pattern in patterns:
            matches = re.findall(pattern, description)
            for match in matches:
                metric, value = match
                try:
                    metrics[metric.strip().lower()] = float(value)
                except ValueError:
                    pass
        return metrics
    
    async def _design_experiment_for_hypothesis(self, hypothesis: Hypothesis) -> None:
        active_count = sum(1 for e in self.experiments.values() if e.status in ["created", "running"])
        if active_count >= self.config["max_active_experiments"]:
            logger.info(f"Not creating experiment for {hypothesis.id} due to experiment limit")
            return
        
        experiment_id = f"experiment_{self.next_experiment_id}"
        self.next_experiment_id += 1
        
        metrics_to_measure = []
        for ev in hypothesis.supporting_evidence:
            if "metrics" in ev:
                metrics_to_measure.extend(ev["metrics"].keys())
        metrics_to_measure = list(set(metrics_to_measure))
        
        success_criteria = {}
        for metric in metrics_to_measure:
            success_criteria[metric] = {
                "type": "improvement",
                "target": 0.2,
                "minimum_confidence": 0.7
            }
        
        design = {
            "hypothesis": hypothesis.statement,
            "approach": "ab_testing",
            "duration": "2 weeks",
            "metrics": metrics_to_measure,
            "variables": {
                "intervention": "Apply recommended improvements",
                "control": "Continue current approach"
            },
            "measures": metrics_to_measure
        }
        
        experiment = Experiment(
            experiment_id=experiment_id,
            hypothesis_id=hypothesis.id,
            design=design,
            success_criteria=success_criteria
        )
        self.experiments[experiment_id] = experiment
        logger.info(f"Created experiment {experiment_id} for hypothesis {hypothesis.id}")
    
    async def record_performance(self, performance_data: Dict[str, Any]) -> None:
        timestamped_data = performance_data.copy()
        timestamped_data["timestamp"] = datetime.datetime.now().isoformat()
        self.performance_history.append(timestamped_data)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        await self._update_experiments_with_performance(performance_data)
    
    async def record_decision(self, decision_data: Dict[str, Any]) -> None:
        timestamped_data = decision_data.copy()
        timestamped_data["timestamp"] = datetime.datetime.now().isoformat()
        self.decision_history.append(timestamped_data)
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    async def _update_experiments_with_performance(self, performance_data: Dict[str, Any]) -> None:
        for experiment in self.experiments.values():
            if experiment.status == "running":
                result = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "metrics": {}
                }
                for metric in experiment.design.get("metrics", []):
                    if metric in performance_data:
                        result["metrics"][metric] = performance_data[metric]
                if result["metrics"]:
                    experiment.results.append(result)
                    if len(experiment.results) >= 10:
                        await self._analyze_experiment(experiment.id)
    
    async def start_experiment(self, experiment_id: str) -> Dict[str, Any]:
        if experiment_id not in self.experiments:
            return {"error": "Experiment not found"}
        experiment = self.experiments[experiment_id]
        if experiment.status != "created":
            return {"error": f"Experiment already {experiment.status}"}
        experiment.status = "running"
        experiment.start_time = datetime.datetime.now()
        logger.info(f"Started experiment {experiment_id}")
        return {"status": "running", "experiment": experiment.to_dict()}
    
    async def _analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        if experiment_id not in self.experiments:
            return {"error": "Experiment not found"}
        experiment = self.experiments[experiment_id]
        if experiment.status != "running":
            return {"error": "Experiment not running"}
        metrics_data = {}
        for r in experiment.results:
            for metric, val in r["metrics"].items():
                if metric not in metrics_data:
                    metrics_data[metric] = []
                metrics_data[metric].append(val)
        metrics_analysis = {}
        significant_improvements = 0
        metrics_analyzed = 0
        for metric, vals in metrics_data.items():
            if len(vals) >= 5:
                metrics_analyzed += 1
                if len(vals) >= 10:
                    midpoint = len(vals) // 2
                    before = vals[:midpoint]
                    after = vals[midpoint:]
                    before_mean = sum(before) / len(before)
                    after_mean = sum(after) / len(after)
                    if before_mean != 0:
                        improvement = (after_mean - before_mean) / abs(before_mean)
                    else:
                        improvement = 0 if after_mean == 0 else 1.0
                    try:
                        t_stat, p_val = self._simple_t_test(before, after)
                        significant = p_val < 0.05
                    except:
                        significant = False
                        p_val = 1.0
                        t_stat = 0.0
                    metrics_analysis[metric] = {
                        "before_mean": before_mean,
                        "after_mean": after_mean,
                        "improvement": improvement,
                        "significant": significant,
                        "p_value": p_val,
                        "t_statistic": t_stat
                    }
                    if metric in experiment.success_criteria:
                        crit = experiment.success_criteria[metric]
                        if crit["type"] == "improvement":
                            tgt = crit["target"]
                            if improvement >= tgt and significant:
                                significant_improvements += 1
                else:
                    metrics_analysis[metric] = {
                        "mean": sum(vals) / len(vals),
                        "insufficient_data": True
                    }
        if metrics_analyzed > 0:
            success_rate = significant_improvements / metrics_analyzed
            if success_rate >= 0.7:
                conclusion = "Hypothesis confirmed with strong evidence"
                status = "confirmed"
            elif success_rate >= 0.3:
                conclusion = "Hypothesis partially supported with mixed evidence"
                status = "partially_confirmed"
            else:
                conclusion = "Hypothesis not supported by the evidence"
                status = "rejected"
        else:
            conclusion = "Insufficient data to draw conclusion"
            status = "inconclusive"
        experiment.analysis = metrics_analysis
        experiment.conclusion = conclusion
        experiment.status = "completed"
        experiment.end_time = datetime.datetime.now()
        if experiment.hypothesis_id in self.hypotheses:
            hypothesis = self.hypotheses[experiment.hypothesis_id]
            hypothesis.status = status
            hypothesis.test_results.append({
                "experiment_id": experiment.id,
                "conclusion": conclusion,
                "metrics_analysis": metrics_analysis,
                "timestamp": datetime.datetime.now().isoformat()
            })
            hypothesis.last_tested = datetime.datetime.now()
            if status == "confirmed":
                hypothesis.confidence = min(0.95, hypothesis.confidence * 1.5)
            elif status == "partially_confirmed":
                hypothesis.confidence = min(0.8, hypothesis.confidence * 1.2)
            elif status == "rejected":
                hypothesis.confidence *= 0.5
            if self.knowledge_system:
                await self._share_hypothesis_with_knowledge_system(hypothesis)
        logger.info(f"Analyzed experiment {experiment_id} with conclusion: {conclusion}")
        return {"conclusion": conclusion, "metrics_analysis": metrics_analysis, "experiment": experiment.to_dict()}
    
    def _simple_t_test(self, sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
        mean1 = sum(sample1) / len(sample1)
        mean2 = sum(sample2) / len(sample2)
        var1 = sum((x - mean1) ** 2 for x in sample1) / (len(sample1) - 1) if len(sample1) > 1 else 0
        var2 = sum((x - mean2) ** 2 for x in sample2) / (len(sample2) - 1) if len(sample2) > 1 else 0
        se = math.sqrt(var1 / len(sample1) + var2 / len(sample2)) if len(sample1) > 1 and len(sample2) > 1 else 1e-9
        t_stat = (mean1 - mean2) / se if se > 0 else 0
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
        return t_stat, p_value
    
    def _normal_cdf(self, x: float) -> float:
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        sign = 1 if x >= 0 else -1
        x = abs(x) / math.sqrt(2.0)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
        return 0.5 * (1.0 + sign * y)
    
    async def _share_insights_with_knowledge_system(self, session: ReflectionSession) -> None:
        if not self.knowledge_system:
            return
        try:
            for insight in session.insights:
                await self.knowledge_system.add_knowledge(
                    type="insight",
                    content={
                        "area": insight.get("area", "general"),
                        "description": insight["description"],
                        "type": insight["type"],
                        "severity": insight.get("severity", "medium") if insight["type"] == "issue" else None,
                        "session_id": session.id,
                        "timestamp": session.timestamp.isoformat()
                    },
                    source="reflection",
                    confidence=insight.get("confidence", 0.7)
                )
            await self.knowledge_system.add_knowledge(
                type="reflection_session",
                content={
                    "session_id": session.id,
                    "timestamp": session.timestamp.isoformat(),
                    "focus_areas": session.focus_areas,
                    "insight_count": len(session.insights),
                    "action_count": len(session.action_items),
                    "success_metrics": session.success_metrics,
                    "duration": session.duration
                },
                source="reflection",
                confidence=0.9
            )
        except Exception as e:
            logger.error(f"Error sharing insights with knowledge system: {str(e)}")
    
    async def _share_hypothesis_with_knowledge_system(self, hypothesis: Hypothesis) -> None:
        if not self.knowledge_system:
            return
        try:
            await self.knowledge_system.add_knowledge(
                type="hypothesis",
                content={
                    "statement": hypothesis.statement,
                    "status": hypothesis.status,
                    "source": hypothesis.source,
                    "creation_time": hypothesis.creation_time.isoformat(),
                    "last_tested": hypothesis.last_tested.isoformat() if hypothesis.last_tested else None,
                    "test_count": len(hypothesis.test_results),
                    "supporting_evidence_count": len(hypothesis.supporting_evidence),
                    "contradicting_evidence_count": len(hypothesis.contradicting_evidence)
                },
                source="reflection",
                confidence=hypothesis.confidence
            )
        except Exception as e:
            logger.error(f"Error sharing hypothesis with knowledge system: {str(e)}")
    
    async def get_reflection_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        sorted_sessions = sorted(self.reflection_sessions, key=lambda s: s.timestamp, reverse=True)[:limit]
        return [s.to_dict() for s in sorted_sessions]
    
    async def get_active_experiments(self) -> List[Dict[str, Any]]:
        return [
            e.to_dict() for e in self.experiments.values()
            if e.status in ["created", "running"]
        ]
    
    async def get_recent_hypotheses(self, limit: int = 10) -> List[Dict[str, Any]]:
        sorted_hypotheses = sorted(
            self.hypotheses.values(), 
            key=lambda h: h.creation_time, 
            reverse=True
        )[:limit]
        return [h.to_dict() for h in sorted_hypotheses]
    
    async def generate_counterfactuals(self, decision_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        counterfactuals = []
        decision_type = decision_data.get("type", "unknown")
        factors = decision_data.get("factors", {})
        outcome = decision_data.get("outcome", "unknown")
        success = decision_data.get("success", False)
        
        if decision_type == "resource_allocation":
            counterfactuals.append({
                "name": "Alternative allocation",
                "description": "What if resources were allocated proportionally to system load?",
                "modified_factors": {
                    "allocation_strategy": "proportional_to_load"
                },
                "expected_outcome": "Improved efficiency for high-load systems",
                "confidence": 0.7
            })
        elif decision_type == "parameter_adjustment":
            counterfactuals.append({
                "name": "More aggressive adjustment",
                "description": "What if parameter changes were twice as large?",
                "modified_factors": {
                    "adjustment_magnitude": factors.get("adjustment_magnitude", 0.1) * 2
                },
                "expected_outcome": "Faster adaptation but possible instability",
                "confidence": 0.6
            })
            counterfactuals.append({
                "name": "More conservative adjustment",
                "description": "What if parameter changes were half as large?",
                "modified_factors": {
                    "adjustment_magnitude": factors.get("adjustment_magnitude", 0.1) / 2
                },
                "expected_outcome": "More stable but slower adaptation",
                "confidence": 0.7
            })
        elif decision_type == "strategy_selection":
            current_strategy = factors.get("selected_strategy", "unknown")
            alt_strategies = factors.get("alternative_strategies", [])
            if alt_strategies:
                alt = alt_strategies[0]
                counterfactuals.append({
                    "name": "Alternative strategy",
                    "description": f"What if strategy '{alt}' was selected instead?",
                    "modified_factors": {
                        "selected_strategy": alt
                    },
                    "expected_outcome": "Different performance profile, possibly better in some areas",
                    "confidence": 0.5
                })
        
        if outcome in ["success", "failure"]:
            counterfactuals.append({
                "name": f"Alternative outcome",
                "description": f"What if this decision had resulted in {'failure' if success else 'success'}?",
                "modified_factors": {},
                "expected_outcome": f"{'Negative' if success else 'Positive'} impact on overall system performance",
                "confidence": 0.6
            })
        if len(counterfactuals) < 2:
            counterfactuals.append({
                "name": "No action taken",
                "description": "What if no decision was made and the system continued as is?",
                "modified_factors": {
                    "decision_type": "no_action"
                },
                "expected_outcome": "Continuation of current trends without intervention",
                "confidence": 0.8
            })
        return counterfactuals
    
    async def create_custom_hypothesis(self, statement: str, confidence: float = 0.5, 
                                       evidence: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        hypothesis_id = f"hypothesis_{self.next_hypothesis_id}"
        self.next_hypothesis_id += 1
        hypothesis = Hypothesis(
            hypothesis_id=hypothesis_id,
            statement=statement,
            confidence=confidence,
            source="custom"
        )
        if evidence:
            hypothesis.supporting_evidence = evidence
        self.hypotheses[hypothesis_id] = hypothesis
        await self._design_experiment_for_hypothesis(hypothesis)
        logger.info(f"Created custom hypothesis {hypothesis_id}: {statement}")
        return hypothesis.to_dict()
