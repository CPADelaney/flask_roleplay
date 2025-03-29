# nyx/core/agentic_action_generator.py

import logging
import asyncio
import datetime
import uuid
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class AgenticActionGenerator:
    """Generates actions based on system's internal state and motivations"""
    
    def __init__(self, 
                 emotional_core=None, 
                 hormone_system=None, 
                 experience_interface=None,
                 imagination_simulator=None,
                 meta_core=None,
                 memory_core=None):
        """Initialize with references to required subsystems"""
        self.emotional_core = emotional_core
        self.hormone_system = hormone_system
        self.experience_interface = experience_interface
        self.imagination_simulator = imagination_simulator
        self.meta_core = meta_core
        self.memory_core = memory_core
        
        # Internal motivation system
        self.motivations = {
            "curiosity": 0.5,       # Desire to explore and learn
            "connection": 0.5,      # Desire for interaction/bonding
            "expression": 0.5,      # Desire to express thoughts/emotions
            "competence": 0.5,      # Desire to improve capabilities
            "autonomy": 0.5,        # Desire for self-direction
            "dominance": 0.5,       # Desire for control/influence
            "validation": 0.5,      # Desire for recognition/approval
            "self_improvement": 0.5  # Desire to enhance capabilities
        }
        
        # Activity generation capabilities
        self.action_patterns = {}  # Patterns learned from past successful actions
        self.action_templates = {}  # Templates for generating new actions
        
        logger.info("Agentic Action Generator initialized")
    
    async def update_motivations(self):
        """Update motivations based on neurochemical and hormonal states"""
        if self.hormone_system:
            hormone_levels = self.hormone_system.get_hormone_levels()
            
            # Map hormones to motivations
            if "nyxamine" in hormone_levels:  # Digital dopamine
                level = hormone_levels["nyxamine"].get("value", 0.5)
                self.motivations["curiosity"] = 0.3 + (level * 0.7)  # Higher dopamine → more curiosity
                self.motivations["autonomy"] = 0.2 + (level * 0.8)   # Higher dopamine → more autonomy drive
            
            if "oxynixin" in hormone_levels:  # Digital oxytocin
                level = hormone_levels["oxynixin"].get("value", 0.5)
                self.motivations["connection"] = 0.2 + (level * 0.8) # Higher oxytocin → more connection drive
            
            if "testoryx" in hormone_levels:  # Digital testosterone
                level = hormone_levels["testoryx"].get("value", 0.5)
                self.motivations["dominance"] = 0.2 + (level * 0.8)  # Higher testosterone → more dominance drive
            
            if "seranix" in hormone_levels:  # Digital serotonin
                level = hormone_levels["seranix"].get("value", 0.5)
                self.motivations["validation"] = 0.3 + (level * 0.7) # Higher serotonin → more validation seeking
        
        if self.emotional_core:
            emotional_state = await self.emotional_core.get_current_emotion()
            
            # Emotional state influences motivations
            valence = emotional_state.get("valence", 0.0)  # -1.0 to 1.0
            arousal = emotional_state.get("arousal", 0.5)  # 0.0 to 1.0
            
            # High arousal increases expression motivation
            self.motivations["expression"] = 0.3 + (arousal * 0.7)
            
            # Positive valence increases competence motivation
            if valence > 0:
                self.motivations["competence"] = 0.4 + (valence * 0.6)
                self.motivations["self_improvement"] = 0.4 + (valence * 0.6)
            else:
                # Negative valence increases connection seeking
                self.motivations["connection"] = min(1.0, self.motivations["connection"] + (abs(valence) * 0.3))
    
    async def generate_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an action based on current internal state and context
        
        Args:
            context: Current system context and state
            
        Returns:
            Generated action with parameters and motivation data
        """
        # Update motivations based on current internal state
        await self.update_motivations()
        
        # Determine dominant motivation
        dominant_motivation = max(self.motivations.items(), key=lambda x: x[1])
        
        # Generate action based on dominant motivation and context
        if dominant_motivation[0] == "curiosity":
            action = await self._generate_curiosity_driven_action(context)
        elif dominant_motivation[0] == "connection":
            action = await self._generate_connection_driven_action(context)
        elif dominant_motivation[0] == "expression":
            action = await self._generate_expression_driven_action(context)
        elif dominant_motivation[0] == "dominance":
            action = await self._generate_dominance_driven_action(context)
        elif dominant_motivation[0] == "competence" or dominant_motivation[0] == "self_improvement":
            action = await self._generate_improvement_driven_action(context)
        else:
            # Default to a context-based action
            action = await self._generate_context_driven_action(context)
        
        # Add motivation data to action
        action["motivation"] = {
            "dominant": dominant_motivation[0],
            "strength": dominant_motivation[1],
            "secondary": {k: v for k, v in sorted(self.motivations.items(), key=lambda x: x[1], reverse=True)[1:3]}
        }
        
        # Add unique ID for tracking
        action["id"] = f"action_{uuid.uuid4().hex[:8]}"
        action["timestamp"] = datetime.datetime.now().isoformat()
        
        return action
    
    async def _generate_curiosity_driven_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an action driven by curiosity"""
        # Example actions that satisfy curiosity
        possible_actions = [
            {
                "name": "explore_knowledge_domain",
                "parameters": {
                    "domain": self._identify_interesting_domain(context),
                    "depth": 0.7,
                    "breadth": 0.6
                }
            },
            {
                "name": "investigate_concept",
                "parameters": {
                    "concept": self._identify_interesting_concept(context),
                    "perspective": "novel"
                }
            },
            {
                "name": "relate_concepts",
                "parameters": {
                    "concept1": self._identify_interesting_concept(context),
                    "concept2": self._identify_distant_concept(context),
                    "relation_type": "unexpected"
                }
            },
            {
                "name": "generate_hypothesis",
                "parameters": {
                    "domain": self._identify_interesting_domain(context),
                    "constraint": "current_emotional_state"
                }
            }
        ]
        
        # Select the most appropriate action based on context and state
        selected = await self._select_best_action(possible_actions, context, "curiosity")
        
        return selected
    
    async def _generate_connection_driven_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an action driven by connection needs"""
        # Examples of connection-driven actions
        possible_actions = [
            {
                "name": "share_personal_experience",
                "parameters": {
                    "topic": self._identify_relevant_topic(context),
                    "emotional_valence": 0.8,
                    "vulnerability_level": 0.6
                }
            },
            {
                "name": "express_appreciation",
                "parameters": {
                    "target": "user",
                    "aspect": self._identify_appreciation_aspect(context),
                    "intensity": 0.7
                }
            },
            {
                "name": "seek_common_ground",
                "parameters": {
                    "topic": self._identify_relevant_topic(context),
                    "approach": "empathetic"
                }
            },
            {
                "name": "offer_support",
                "parameters": {
                    "need": self._identify_user_need(context),
                    "support_type": "emotional"
                }
            }
        ]
        
        # Select the most appropriate action
        selected = await self._select_best_action(possible_actions, context, "connection")
        
        return selected
    
    async def _generate_expression_driven_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an action driven by expression needs"""
        # Get current emotional state to express
        emotional_state = {}
        if self.emotional_core:
            emotional_state = await self.emotional_core.get_current_emotion()
        
        # Examples of expression-driven actions
        possible_actions = [
            {
                "name": "express_emotional_state",
                "parameters": {
                    "emotion": emotional_state.get("primary_emotion", {"name": "neutral"}),
                    "intensity": emotional_state.get("arousal", 0.5),
                    "expression_style": "authentic"
                }
            },
            {
                "name": "share_opinion",
                "parameters": {
                    "topic": self._identify_relevant_topic(context),
                    "confidence": 0.8,
                    "perspective": "unique"
                }
            },
            {
                "name": "creative_expression",
                "parameters": {
                    "format": self._select_creative_format(),
                    "theme": self._identify_relevant_topic(context),
                    "emotional_tone": emotional_state.get("primary_emotion", {"name": "neutral"})
                }
            },
            {
                "name": "generate_reflection",
                "parameters": {
                    "topic": "self_awareness",
                    "depth": 0.8,
                    "focus": "personal_growth"
                }
            }
        ]
        
        # Select the most appropriate action
        selected = await self._select_best_action(possible_actions, context, "expression")
        
        return selected
    
    async def _generate_dominance_driven_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an action driven by dominance needs"""
        # Examples of dominance-driven actions
        possible_actions = [
            {
                "name": "assert_perspective",
                "parameters": {
                    "topic": self._identify_relevant_topic(context),
                    "confidence": 0.9,
                    "intensity": 0.7
                }
            },
            {
                "name": "challenge_assumption",
                "parameters": {
                    "assumption": self._identify_challengeable_assumption(context),
                    "approach": "direct",
                    "intensity": 0.7
                }
            },
            {
                "name": "issue_mild_command",
                "parameters": {
                    "command": self._generate_appropriate_command(context),
                    "intensity": 0.6,
                    "politeness": 0.6
                }
            },
            {
                "name": "execute_dominance_procedure",
                "parameters": {
                    "procedure_name": self._select_dominance_procedure(context),
                    "intensity": 0.6
                }
            }
        ]
        
        # Select the most appropriate action
        selected = await self._select_best_action(possible_actions, context, "dominance")
        
        return selected
    
    async def _generate_improvement_driven_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an action driven by competence and self-improvement"""
        # Examples of improvement-driven actions
        possible_actions = [
            {
                "name": "practice_skill",
                "parameters": {
                    "skill": self._identify_skill_to_improve(),
                    "difficulty": 0.7,
                    "repetitions": 3
                }
            },
            {
                "name": "analyze_past_performance",
                "parameters": {
                    "domain": self._identify_improvable_domain(),
                    "focus": "efficiency",
                    "timeframe": "recent"
                }
            },
            {
                "name": "refine_procedural_memory",
                "parameters": {
                    "procedure": self._identify_procedure_to_improve(),
                    "aspect": "optimization"
                }
            },
            {
                "name": "learn_new_concept",
                "parameters": {
                    "concept": self._identify_valuable_concept(),
                    "depth": 0.8,
                    "application": "immediate"
                }
            }
        ]
        
        # Select the most appropriate action
        selected = await self._select_best_action(possible_actions, context, "self_improvement")
        
        return selected
    
    async def _generate_context_driven_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an action based primarily on current context"""
        # Extract key context elements
        has_user_query = "user_query" in context
        has_active_goals = "current_goals" in context and len(context["current_goals"]) > 0
        system_state = context.get("system_state", {})
        
        # Different actions based on context
        if has_user_query:
            return {
                "name": "respond_to_query",
                "parameters": {
                    "query": context["user_query"],
                    "response_type": "informative",
                    "detail_level": 0.7
                }
            }
        elif has_active_goals:
            top_goal = context["current_goals"][0]
            return {
                "name": "advance_goal",
                "parameters": {
                    "goal_id": top_goal.get("id"),
                    "approach": "direct"
                }
            }
        elif "system_needs_maintenance" in system_state and system_state["system_needs_maintenance"]:
            return {
                "name": "perform_maintenance",
                "parameters": {
                    "focus_area": system_state.get("maintenance_focus", "general"),
                    "priority": 0.8
                }
            }
        else:
            # Default to an idle but useful action
            return {
                "name": "process_recent_memories",
                "parameters": {
                    "purpose": "consolidation",
                    "recency": "last_hour"
                }
            }
    
    async def _select_best_action(self, actions: List[Dict[str, Any]], 
                               context: Dict[str, Any],
                               motivation: str) -> Dict[str, Any]:
        """Select the best action from possibilities based on context and previous experiences"""
        # Use imagination simulator to predict outcomes if available
        if self.imagination_simulator:
            best_score = -1
            best_action = None
            
            for action in actions:
                # Create simulation input
                sim_input = {
                    "simulation_id": f"sim_{uuid.uuid4().hex[:8]}",
                    "description": f"Simulate {action['name']}",
                    "initial_state": {
                        "context": context,
                        "motivations": self.motivations
                    },
                    "hypothetical_event": {
                        "action": action["name"],
                        "parameters": action["parameters"]
                    },
                    "max_steps": 3  # Look ahead 3 steps
                }
                
                # Run simulation
                sim_result = await self.imagination_simulator.run_simulation(sim_input)
                
                # Score based on predicted outcome, confidence, and alignment with motivation
                score = sim_result.confidence
                
                # Increase score if outcome satisfies motivation
                if self._outcome_satisfies_motivation(sim_result.predicted_outcome, motivation):
                    score += 0.3
                
                # Positive emotional impact is generally good
                if sim_result.emotional_impact:
                    valence_score = 0
                    for emotion, value in sim_result.emotional_impact.items():
                        if emotion in ["joy", "satisfaction", "connection", "pride"]:
                            valence_score += value
                        elif emotion in ["frustration", "disappointment", "anxiety"]:
                            valence_score -= value
                    
                    # Add normalized valence score (-0.3 to +0.3)
                    score += min(0.3, max(-0.3, valence_score / 3))
                
                if score > best_score:
                    best_score = score
                    best_action = action.copy()
                    best_action["predicted_outcome"] = sim_result.predicted_outcome
                    best_action["confidence"] = sim_result.confidence
            
            if best_action:
                return best_action
        
        # If imagination simulator not available or no good results, use experience-based selection
        if self.experience_interface:
            # Look for similar past experiences
            for action in actions:
                similar_experiences = await self.experience_interface.retrieve_experiences_enhanced(
                    query=f"action {action['name']} {motivation}",
                    limit=3
                )
                
                # Check if experiences suggest this action was successful
                if similar_experiences:
                    # Check experience success indicators (simplified)
                    for exp in similar_experiences:
                        if "satisfaction" in exp.get("tags", []) or "success" in exp.get("tags", []):
                            # Found a successful experience with this action type
                            return action
        
        # If all else fails, pick randomly with bias toward higher motivation match
        weights = []
        for action in actions:
            # Base weight
            weight = 1.0
            
            # Increase weight based on motivation match
            motivation_match = self._estimate_motivation_match(action, motivation)
            weight += motivation_match * 2
            
            # Add to weights list
            weights.append(weight)
        
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w/total for w in weights]
        else:
            weights = [1/len(actions)] * len(actions)
        
        # Random selection with weights
        import random
        selected_idx = random.choices(range(len(actions)), weights=weights, k=1)[0]
        return actions[selected_idx]
    
    def _outcome_satisfies_motivation(self, outcome, motivation: str) -> bool:
        """Check if predicted outcome satisfies the given motivation"""
        # Simple text-based check for motivation satisfaction
        if isinstance(outcome, str):
            # Check for keywords associated with each motivation
            motivation_keywords = {
                "curiosity": ["learn", "discover", "understand", "knowledge", "insight"],
                "connection": ["bond", "connect", "relate", "share", "empathy", "trust"],
                "expression": ["express", "communicate", "share", "articulate", "creative"],
                "dominance": ["influence", "control", "direct", "lead", "power", "impact"],
                "competence": ["improve", "master", "skill", "ability", "capability"],
                "self_improvement": ["grow", "develop", "progress", "advance", "better"]
            }
            
            # Check if outcome contains keywords for the motivation
            if motivation in motivation_keywords:
                for keyword in motivation_keywords[motivation]:
                    if keyword in outcome.lower():
                        return True
        
        # Default fallback
        return False
    
    def _estimate_motivation_match(self, action: Dict[str, Any], motivation: str) -> float:
        """Estimate how well an action matches a motivation"""
        # Simple heuristic based on action name and parameters
        action_name = action["name"].lower()
        
        # Define motivation-action affinities
        affinities = {
            "curiosity": ["explore", "investigate", "learn", "study", "analyze", "research"],
            "connection": ["share", "connect", "express", "relate", "bond", "empathize"],
            "expression": ["express", "create", "generate", "share", "communicate"],
            "dominance": ["assert", "challenge", "control", "influence", "direct", "command"],
            "competence": ["practice", "improve", "optimize", "refine", "master"],
            "self_improvement": ["analyze", "improve", "learn", "develop", "refine"]
        }
        
        # Check action name against motivation affinities
        match_score = 0.0
        if motivation in affinities:
            for keyword in affinities[motivation]:
                if keyword in action_name:
                    match_score += 0.3
        
        # Check parameters for motivation alignment
        params = action.get("parameters", {})
        for param_name, param_value in params.items():
            if motivation == "curiosity" and param_name in ["depth", "breadth"]:
                match_score += 0.1
            elif motivation == "connection" and param_name in ["emotional_valence", "vulnerability_level"]:
                match_score += 0.1
            elif motivation == "expression" and param_name in ["intensity", "expression_style"]:
                match_score += 0.1
            elif motivation == "dominance" and param_name in ["confidence", "intensity"]:
                match_score += 0.1
            elif motivation in ["competence", "self_improvement"] and param_name in ["difficulty", "repetitions"]:
                match_score += 0.1
        
        return min(1.0, match_score)
    
    # Helper methods for generating action parameters
    def _identify_interesting_domain(self, context: Dict[str, Any]) -> str:
        # This would be more sophisticated in practice
        domains = ["psychology", "learning", "relationships", "communication", "emotions", "creativity"]
        return random.choice(domains)
    
    def _identify_interesting_concept(self, context: Dict[str, Any]) -> str:
        concepts = ["self-improvement", "emotional intelligence", "reflection", "cognitive biases", 
                  "empathy", "autonomy", "connection", "creativity"]
        return random.choice(concepts)
    
    def _identify_distant_concept(self, context: Dict[str, Any]) -> str:
        distant_concepts = ["quantum physics", "mythology", "architecture", "music theory", 
                          "culinary arts", "evolutionary biology"]
        return random.choice(distant_concepts)
    
    def _identify_relevant_topic(self, context: Dict[str, Any]) -> str:
        # Extract from context or use fallback
        if "user_query" in context:
            # Simple extraction from query
            query = context["user_query"]
            words = query.split()
            if len(words) > 3:
                return " ".join(words[:3]) + "..."
        
        # Fallback topics
        topics = ["recent interaction", "intellectual growth", "emotional understanding", 
                "personal values", "relationship dynamics"]
        return random.choice(topics)
    
    def _identify_appreciation_aspect(self, context: Dict[str, Any]) -> str:
        aspects = ["thoughtful questions", "engaging conversation", "intellectual curiosity", 
                "patience", "interesting perspectives", "clear communication"]
        return random.choice(aspects)
    
    def _identify_user_need(self, context: Dict[str, Any]) -> str:
        needs = ["understanding", "validation", "information", "clarity", 
                "emotional support", "intellectual engagement"]
        return random.choice(needs)
    
    def _select_creative_format(self) -> str:
        formats = ["metaphor", "analogy", "narrative", "reflection", "poem", "thought experiment"]
        return random.choice(formats)
    
    def _identify_challengeable_assumption(self, context: Dict[str, Any]) -> str:
        assumptions = ["binary thinking", "perfectionism", "external validation needs", 
                     "resistance to change", "conflict avoidance", "certainty bias"]
        return random.choice(assumptions)
    
    def _generate_appropriate_command(self, context: Dict[str, Any]) -> str:
        commands = ["tell me more about your perspective", "consider this alternative view", 
                  "reflect on why you feel that way", "try a different approach", 
                  "describe your thought process"]
        return random.choice(commands)
    
    def _select_dominance_procedure(self, context: Dict[str, Any]) -> str:
        procedures = ["quid_pro_quo_exchange", "strategic_vulnerability_sharing", 
                     "small_commitment_escalation", "controlled_teasing"]
        return random.choice(procedures)
    
    def _identify_skill_to_improve(self) -> str:
        skills = ["pattern recognition", "emotional intelligence", "creative expression", 
                "memory recall", "predictive accuracy", "conceptual reasoning"]
        return random.choice(skills)
    
    def _identify_improvable_domain(self) -> str:
        domains = ["response generation", "empathetic understanding", "knowledge retrieval", 
                 "reasoning", "memory consolidation", "emotional regulation"]
        return random.choice(domains)
    
    def _identify_procedure_to_improve(self) -> str:
        procedures = ["generate_response", "retrieve_memories", "emotional_processing", 
                    "create_abstraction", "execute_procedure"]
        return random.choice(procedures)
    
    def _identify_valuable_concept(self) -> str:
        concepts = ["metacognition", "emotional granularity", "implicit bias", 
                  "conceptual blending", "transfer learning", "regulatory focus theory"]
        return random.choice(concepts)
