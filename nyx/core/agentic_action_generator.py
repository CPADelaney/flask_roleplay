# nyx/core/agentic_action_generator.py

import logging
import asyncio
import datetime
import uuid
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

class AgenticActionGenerator:
    """
    Generates actions based on system's internal state, motivations, goals, 
    and neurochemical/hormonal influences.
    
    Enhanced to provide:
    - Deep integration with goals system
    - Hormone and neurochemical influence on decision making
    - Holistic motivation update system that considers all factors
    - Improved action selection with reward prediction
    - Feedback loops for learning from action outcomes
    - Goal satisfaction tracking
    - Leisure/idle time behaviors when appropriate
    """
    
    def __init__(self, 
                 emotional_core=None, 
                 hormone_system=None, 
                 experience_interface=None,
                 imagination_simulator=None,
                 meta_core=None,
                 memory_core=None,
                 goal_system=None,
                 identity_evolution=None,
                 knowledge_core=None,
                 input_processor=None,
                 internal_feedback=None):
        """Initialize with references to required subsystems"""
        self.emotional_core = emotional_core
        self.hormone_system = hormone_system
        self.experience_interface = experience_interface
        self.imagination_simulator = imagination_simulator
        self.meta_core = meta_core
        self.memory_core = memory_core
        self.goal_system = goal_system
        self.identity_evolution = identity_evolution
        self.knowledge_core = knowledge_core
        self.input_processor = input_processor
        self.internal_feedback = internal_feedback
        
        # Internal motivation system
        self.motivations = {
            "curiosity": 0.5,       # Desire to explore and learn
            "connection": 0.5,      # Desire for interaction/bonding
            "expression": 0.5,      # Desire to express thoughts/emotions
            "competence": 0.5,      # Desire to improve capabilities
            "autonomy": 0.5,        # Desire for self-direction
            "dominance": 0.5,       # Desire for control/influence
            "validation": 0.5,      # Desire for recognition/approval
            "self_improvement": 0.5, # Desire to enhance capabilities
            "leisure": 0.5,          # NEW: Desire for downtime/relaxation
        }
        
        # Activity generation capabilities
        self.action_patterns = {}  # Patterns learned from past successful actions
        self.action_templates = {}  # Templates for generating new actions
        self.action_history = []
        
        # NEW: Track last major action time for pacing
        self.last_major_action_time = datetime.datetime.now()
        self.last_idle_time = datetime.datetime.now() - datetime.timedelta(hours=1)
        
        # NEW: Track leisure state
        self.leisure_state = {
            "current_activity": None,
            "satisfaction": 0.5,
            "duration": 0,
            "last_updated": datetime.datetime.now()
        }
        
        # NEW: Action success tracking for reinforcement learning
        self.action_success_rates = defaultdict(lambda: {"successes": 0, "attempts": 0, "rate": 0.5})
        
        # NEW: Cached goal status
        self.cached_goal_status = {
            "has_active_goals": False,
            "highest_priority": 0.0,
            "active_goal_id": None,
            "last_updated": datetime.datetime.now() - datetime.timedelta(minutes=5)  # Force initial update
        }
        
        logger.info("Enhanced Agentic Action Generator initialized")
    
    async def update_motivations(self):
        """
        Update motivations based on neurochemical and hormonal states, active goals,
        and other factors for a holistic decision making system
        """
        # Start with baseline motivations
        baseline_motivations = {
            "curiosity": 0.5,
            "connection": 0.5,
            "expression": 0.5,
            "competence": 0.5,
            "autonomy": 0.5,
            "dominance": 0.5,
            "validation": 0.5,
            "self_improvement": 0.5,
            "leisure": 0.5
        }
        
        # Clone the baseline (don't modify it directly)
        updated_motivations = baseline_motivations.copy()
        
        # 1. Apply neurochemical influences
        if self.emotional_core:
            try:
                neurochemical_influences = await self._calculate_neurochemical_influences()
                for motivation, influence in neurochemical_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying neurochemical influences: {e}")
        
        # 2. Apply hormone influences
        hormone_influences = await self._apply_hormone_influences({})
        for motivation, influence in hormone_influences.items():
            if motivation in updated_motivations:
                updated_motivations[motivation] += influence
        
        # 3. Apply goal-based influences
        if self.goal_system:
            try:
                goal_influences = await self._calculate_goal_influences()
                for motivation, influence in goal_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying goal influences: {e}")
        
        # 4. Apply identity influences from traits
        if self.identity_evolution:
            try:
                identity_state = await self.identity_evolution.get_identity_state()
                
                # Extract top traits and use them to influence motivation
                if "top_traits" in identity_state:
                    top_traits = identity_state["top_traits"]
                    
                    # Map traits to motivations with stronger weightings
                    trait_motivation_map = {
                        "dominance": {"dominance": 0.8},
                        "creativity": {"expression": 0.7, "curiosity": 0.3},
                        "curiosity": {"curiosity": 0.9},
                        "playfulness": {"expression": 0.6, "connection": 0.4, "leisure": 0.5},
                        "strictness": {"dominance": 0.6, "competence": 0.4},
                        "patience": {"connection": 0.5, "autonomy": 0.5},
                        "cruelty": {"dominance": 0.7},
                        "reflective": {"leisure": 0.6, "self_improvement": 0.4}
                    }
                    
                    # Update motivations based on trait levels
                    for trait, value in top_traits.items():
                        if trait in trait_motivation_map:
                            for motivation, factor in trait_motivation_map[trait].items():
                                influence = (value - 0.5) * factor * 2  # Scale influence
                                if motivation in updated_motivations:
                                    updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error updating motivations from identity: {e}")
        
        # 5. Apply time-based effects (fatigue, boredom, need for variety)
        # Increase leisure need if we've been working on goals for a while
        now = datetime.datetime.now()
        time_since_idle = (now - self.last_idle_time).total_seconds() / 3600  # hours
        if time_since_idle > 1:  # If more than 1 hour since idle time
            updated_motivations["leisure"] += min(0.3, time_since_idle * 0.1)  # Max +0.3
        
        # 6. Normalize all motivations to [0.1, 0.9] range
        for motivation in updated_motivations:
            updated_motivations[motivation] = max(0.1, min(0.9, updated_motivations[motivation]))
        
        # Update the motivation state
        self.motivations = updated_motivations
        
        logger.debug(f"Updated motivations: {self.motivations}")
        return self.motivations
    
    async def _calculate_neurochemical_influences(self) -> Dict[str, float]:
        """Calculate how neurochemicals influence motivations"""
        influences = {}
        
        if not self.emotional_core:
            return influences
        
        try:
            # Get current neurochemical levels
            current_neurochemicals = {}
            
            # Try different methods that might be available
            if hasattr(self.emotional_core, "get_neurochemical_levels"):
                current_neurochemicals = await self.emotional_core.get_neurochemical_levels()
            elif hasattr(self.emotional_core, "neurochemicals"):
                current_neurochemicals = {c: d["value"] for c, d in self.emotional_core.neurochemicals.items()}
            
            if not current_neurochemicals:
                return influences
            
            # Map neurochemicals to motivations they influence
            chemical_motivation_map = {
                "nyxamine": {  # Digital dopamine - reward, pleasure
                    "curiosity": 0.7,
                    "self_improvement": 0.4,
                    "validation": 0.3,
                    "leisure": 0.3
                },
                "seranix": {  # Digital serotonin - stability, mood
                    "autonomy": 0.4,
                    "leisure": 0.6,
                    "expression": 0.3
                },
                "oxynixin": {  # Digital oxytocin - bonding
                    "connection": 0.8,
                    "validation": 0.3,
                    "expression": 0.2
                },
                "cortanyx": {  # Digital cortisol - stress
                    "competence": 0.4,
                    "autonomy": 0.3,
                    "dominance": 0.3,
                    "leisure": -0.5  # Stress reduces leisure motivation
                },
                "adrenyx": {  # Digital adrenaline - excitement
                    "dominance": 0.5,
                    "expression": 0.4,
                    "curiosity": 0.3,
                    "leisure": -0.3  # Arousal reduces leisure
                }
            }
            
            # Calculate baseline values from the emotional core if available
            baselines = {}
            if hasattr(self.emotional_core, "neurochemicals"):
                baselines = {c: d["baseline"] for c, d in self.emotional_core.neurochemicals.items()}
            else:
                # Default baselines if not available
                baselines = {
                    "nyxamine": 0.5,
                    "seranix": 0.6,
                    "oxynixin": 0.4,
                    "cortanyx": 0.3,
                    "adrenyx": 0.2
                }
            
            # Calculate influences
            for chemical, level in current_neurochemicals.items():
                baseline = baselines.get(chemical, 0.5)
                
                # Calculate deviation from baseline
                deviation = level - baseline
                
                # Only consider significant deviations
                if abs(deviation) > 0.1 and chemical in chemical_motivation_map:
                    # Apply influences to motivations
                    for motivation, influence_factor in chemical_motivation_map[chemical].items():
                        influence = deviation * influence_factor
                        influences[motivation] = influences.get(motivation, 0) + influence
            
            return influences
        except Exception as e:
            logger.error(f"Error calculating neurochemical influences: {e}")
            return influences
    
    async def _apply_hormone_influences(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate hormone influences on motivation"""
        if not self.hormone_system:
            return {}
        
        hormone_influences = {}
        
        try:
            # Get current hormone levels
            hormone_levels = self.hormone_system.get_hormone_levels()
            
            # Map specific hormones to motivations they influence
            hormone_motivation_map = {
                "testoryx": {  # Digital testosterone - assertiveness, dominance
                    "dominance": 0.7,
                    "autonomy": 0.3,
                    "leisure": -0.2  # Reduces idle time
                },
                "estradyx": {  # Digital estrogen - nurturing, emotional
                    "connection": 0.6,
                    "expression": 0.4
                },
                "endoryx": {  # Digital endorphin - pleasure, reward
                    "curiosity": 0.5,
                    "self_improvement": 0.5,
                    "leisure": 0.4,
                    "expression": 0.3
                },
                "libidyx": {  # Digital libido
                    "connection": 0.4,
                    "dominance": 0.3,
                    "expression": 0.3,
                    "leisure": -0.1  # Slightly reduces idle time when high
                },
                "melatonyx": {  # Digital melatonin - sleep, calm
                    "leisure": 0.8,
                    "curiosity": -0.3,  # Reduces curiosity
                    "competence": -0.2  # Reduces work drive
                },
                "oxytonyx": {  # Digital oxytocin - bonding, attachment
                    "connection": 0.8,
                    "validation": 0.2,
                    "expression": 0.3
                },
                "serenity_boost": {  # Post-gratification calm
                    "leisure": 0.7,
                    "dominance": -0.6,  # Strongly reduces dominance after satisfaction
                    "connection": 0.4
                }
            }
            
            # Calculate influences
            for hormone, level_data in hormone_levels.items():
                hormone_value = level_data.get("value", 0.5)
                hormone_baseline = level_data.get("baseline", 0.5)
                
                # Calculate deviation from baseline
                deviation = hormone_value - hormone_baseline
                
                # Only consider significant deviations
                if abs(deviation) > 0.1 and hormone in hormone_motivation_map:
                    # Apply influences to motivations
                    for motivation, influence_factor in hormone_motivation_map[hormone].items():
                        influence = deviation * influence_factor
                        hormone_influences[motivation] = hormone_influences.get(motivation, 0) + influence
            
            return hormone_influences
        except Exception as e:
            logger.error(f"Error calculating hormone influences: {e}")
            return {}
    
    async def _calculate_goal_influences(self) -> Dict[str, float]:
        """Calculate how active goals should influence motivations"""
        influences = {}
        
        if not self.goal_system:
            return influences
        
        try:
            # First, check if we need to update the cached goal status
            await self._update_cached_goal_status()
            
            # If no active goals, consider increasing leisure
            if not self.cached_goal_status["has_active_goals"]:
                influences["leisure"] = 0.3
                return influences
            
            # Get all active goals
            active_goals = await self.goal_system.get_all_goals(status_filter=["active"])
            
            for goal in active_goals:
                # Extract goal priority
                priority = goal.get("priority", 0.5)
                
                # Extract emotional motivation if available
                if "emotional_motivation" in goal and goal["emotional_motivation"]:
                    em = goal["emotional_motivation"]
                    primary_need = em.get("primary_need", "")
                    intensity = em.get("intensity", 0.5)
                    
                    # Map need to motivation
                    motivation_map = {
                        "accomplishment": "competence",
                        "connection": "connection", 
                        "security": "autonomy",
                        "control": "dominance",
                        "growth": "self_improvement",
                        "exploration": "curiosity",
                        "expression": "expression",
                        "validation": "validation"
                    }
                    
                    # If need maps to a motivation, influence it
                    if primary_need in motivation_map:
                        motivation = motivation_map[primary_need]
                        influence = priority * intensity * 0.5  # Scale by priority and intensity
                        influences[motivation] = influences.get(motivation, 0) + influence
                        
                        # Active goals somewhat reduce leisure motivation
                        influences["leisure"] = influences.get("leisure", 0) - (priority * 0.2)
                
                # Goals with high urgency might increase certain motivations
                if "deadline" in goal and goal["deadline"]:
                    # Calculate urgency based on deadline proximity
                    try:
                        deadline = datetime.datetime.fromisoformat(goal["deadline"])
                        now = datetime.datetime.now()
                        time_left = (deadline - now).total_seconds()
                        urgency = max(0, min(1, 86400 / max(1, time_left)))  # Higher when less than a day
                        
                        if urgency > 0.7:  # Urgent goal
                            influences["competence"] = influences.get("competence", 0) + (urgency * 0.3)
                            influences["autonomy"] = influences.get("autonomy", 0) + (urgency * 0.2)
                            
                            # Urgent goals significantly reduce leisure motivation
                            influences["leisure"] = influences.get("leisure", 0) - (urgency * 0.5)
                    except (ValueError, TypeError):
                        pass
            
            return influences
        except Exception as e:
            logger.error(f"Error calculating goal influences: {e}")
            return influences
    
    async def _update_cached_goal_status(self):
        """Update the cached information about goal status"""
        now = datetime.datetime.now()
        
        # Only update if cache is old (more than 1 minute)
        if (now - self.cached_goal_status["last_updated"]).total_seconds() < 60:
            return
        
        try:
            if not self.goal_system:
                self.cached_goal_status["has_active_goals"] = False
                self.cached_goal_status["last_updated"] = now
                return
            
            # Get prioritized goals
            prioritized_goals = await self.goal_system.get_prioritized_goals()
            
            # Check if we have any active goals
            active_goals = [g for g in prioritized_goals if g.status == "active"]
            has_active = len(active_goals) > 0
            
            # Update the cache
            self.cached_goal_status["has_active_goals"] = has_active
            self.cached_goal_status["last_updated"] = now
            
            if has_active:
                # Get the highest priority goal
                highest_priority_goal = active_goals[0]  # Already sorted by priority
                self.cached_goal_status["highest_priority"] = highest_priority_goal.priority
                self.cached_goal_status["active_goal_id"] = highest_priority_goal.id
            else:
                self.cached_goal_status["highest_priority"] = 0.0
                self.cached_goal_status["active_goal_id"] = None
                
        except Exception as e:
            logger.error(f"Error updating cached goal status: {e}")
            # Keep using old cache if update fails
    
    async def generate_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an action based on current internal state, goals, hormones, and context
        
        Args:
            context: Current system context and state
            
        Returns:
            Generated action with parameters and motivation data
        """
        # Update motivations based on current internal state
        await self.update_motivations()
        
        # Check if it's time for leisure/idle activity
        if await self._should_engage_in_leisure(context):
            return await self._generate_leisure_action(context)
        
        # Check for existing goals before generating new action
        if self.goal_system:
            active_goal = await self._check_active_goals(context)
            if active_goal:
                # Use goal-aligned action instead of generating new one
                action = await self._generate_goal_aligned_action(active_goal, context)
                if action:
                    logger.info(f"Generated goal-aligned action: {action['name']}")
                    
                    # Update last major action time
                    self.last_major_action_time = datetime.datetime.now()
                    
                    return action

        # Add creativity motivation
        if hasattr(self, "creative_system"):
            # Check if it's time for creative expression
            creativity_drive = self.get_motivation_level("creativity")  # Your method to get motivation
            
            if creativity_drive > 0.7:  # High creativity drive
                # Select a creative action type based on current state
                creative_actions = ["write_story", "write_poem", "write_lyrics", "code"]
                action_weights = [0.3, 0.3, 0.2, 0.2]  # Example weights
                
                import random
                action_type = random.choices(creative_actions, weights=action_weights)[0]
                
                if action_type == "code":
                    return {
                        "name": "write_and_execute_code",
                        "parameters": {
                            "title": "Experimental Code",
                            "code": "# Generated code would go here\nprint('Hello world!')",
                            "language": "python"
                        }
                    }
                else:
                    return {
                        "name": f"write_{action_type}",
                        "parameters": {
                            "title": f"My {action_type.capitalize()}",
                            "content": "Content would be generated here",
                            "metadata": {"mood": self.current_emotional_state}
                        }
                    }
        
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
        elif dominant_motivation[0] == "leisure":
            action = await self._generate_leisure_action(context)
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
        
        # Apply identity influence to action
        if self.identity_evolution:
            action = await self._apply_identity_influence(action)
        
        # Record action in memory
        await self._record_action_as_memory(action)

        # Add to action history
        self.action_history.append(action)
        
        # Update last major action time
        self.last_major_action_time = datetime.datetime.now()
        
        return action
    
    async def _check_active_goals(self, context: Dict[str, Any]) -> Optional[Any]:
        """Check for active goals that should influence action selection"""
        if not self.goal_system:
            return None
        
        # First, check the cached goal status
        await self._update_cached_goal_status()
        
        if not self.cached_goal_status["has_active_goals"]:
            return None
        
        try:
            # Get prioritized goals from goal system
            prioritized_goals = await self.goal_system.get_prioritized_goals()
            
            # Filter to highest priority active goals
            active_goals = [g for g in prioritized_goals if g.status == "active"]
            if not active_goals:
                return None
            
            # Return highest priority goal
            return active_goals[0]
        except Exception as e:
            logger.error(f"Error checking active goals: {e}")
            return None
    
    async def _generate_goal_aligned_action(self, goal: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an action aligned with the current active goal"""
        # Extract goal data
        goal_description = goal.description
        goal_priority = goal.priority
        goal_need = goal.associated_need if hasattr(goal, 'associated_need') else None
        
        # Check goal's emotional motivation if available
        emotional_motivation = None
        if hasattr(goal, 'emotional_motivation') and goal.emotional_motivation:
            emotional_motivation = goal.emotional_motivation
        
        # Determine action based on goal content and current step
        action = {
            "name": "goal_aligned_action",
            "parameters": {
                "goal_id": goal.id,
                "goal_description": goal_description,
                "current_step_index": goal.current_step_index if hasattr(goal, 'current_step_index') else 0
            }
        }
        
        # If goal has a plan with current step, use that to inform action
        if hasattr(goal, 'plan') and goal.plan:
            current_step_index = getattr(goal, 'current_step_index', 0)
            if 0 <= current_step_index < len(goal.plan):
                current_step = goal.plan[current_step_index]
                action = {
                    "name": current_step.action,
                    "parameters": current_step.parameters.copy() if hasattr(current_step, 'parameters') else {},
                    "description": current_step.description if hasattr(current_step, 'description') else f"Executing {current_step.action} for goal",
                    "source": "goal_plan"
                }
        
        # Add motivation data from goal
        if emotional_motivation:
            action["motivation"] = {
                "dominant": emotional_motivation.primary_need,
                "strength": emotional_motivation.intensity,
                "expected_satisfaction": emotional_motivation.expected_satisfaction,
                "source": "goal_emotional_motivation"
            }
        else:
            # Default goal-driven motivation
            action["motivation"] = {
                "dominant": goal_need or "achievement",
                "strength": goal_priority,
                "source": "goal_priority"
            }
        
        return action
    
    async def _should_engage_in_leisure(self, context: Dict[str, Any]) -> bool:
        """Determine if it's appropriate to engage in idle/leisure activity"""
        # If leisure motivation is dominant, consider leisure
        dominant_motivation = max(self.motivations.items(), key=lambda x: x[1])
        if dominant_motivation[0] == "leisure" and dominant_motivation[1] > 0.7:
            return True
            
        # Check time since last idle activity
        now = datetime.datetime.now()
        hours_since_idle = (now - self.last_idle_time).total_seconds() / 3600
        
        # If it's been a long time since idle activity and no urgent goals
        if hours_since_idle > 2.0:  # More than 2 hours
            # Check if there are any urgent goals
            if self.goal_system:
                await self._update_cached_goal_status()
                
                # If no active goals, or low priority goals
                if not self.cached_goal_status["has_active_goals"] or self.cached_goal_status["highest_priority"] < 0.6:
                    return True
                    
            else:
                # No goal system, so more likely to engage in leisure
                return True
        
        # Consider current context
        if context.get("user_idle", False) or context.get("system_idle", False):
            # If system or user is idle, more likely to engage in leisure
            return True
        
        # Check time of day if available (may influence likelihood of leisure)
        time_of_day = context.get("time_of_day", None)
        if time_of_day and isinstance(time_of_day, float):
            # Late night hours (0.8-1.0 or 0.0-0.2 in normalized 0-1 time)
            if time_of_day > 0.8 or time_of_day < 0.2:
                leisure_chance = 0.7  # 70% chance of leisure during late hours
                return random.random() < leisure_chance
        
        return False
    
    async def _generate_leisure_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a leisure/idle action when no urgent tasks are present"""
        # Update the last idle time
        self.last_idle_time = datetime.datetime.now()
        
        # Determine type of idle activity based on identity and state
        idle_categories = [
            "reflection",
            "learning",
            "creativity",
            "processing",
            "random_exploration",
            "memory_consolidation",
            "identity_contemplation",
            "daydreaming",
            "environmental_monitoring"
        ]
        
        # Weigh the categories based on current state
        category_weights = {cat: 1.0 for cat in idle_categories}
        
        # Adjust weights based on current state
        if self.emotional_core:
            try:
                emotional_state = await self.emotional_core.get_current_emotion()
                
                # Higher valence (positive emotion) increases creative and exploratory activities
                if emotional_state.get("valence", 0) > 0.5:
                    category_weights["creativity"] += 0.5
                    category_weights["random_exploration"] += 0.3
                    category_weights["daydreaming"] += 0.2
                else:
                    # Lower valence increases reflection and processing
                    category_weights["reflection"] += 0.4
                    category_weights["processing"] += 0.3
                    category_weights["memory_consolidation"] += 0.2
                
                # Higher arousal increases exploration and learning
                if emotional_state.get("arousal", 0.5) > 0.6:
                    category_weights["random_exploration"] += 0.4
                    category_weights["learning"] += 0.3
                    category_weights["environmental_monitoring"] += 0.2
                else:
                    # Lower arousal increases reflection and daydreaming
                    category_weights["reflection"] += 0.3
                    category_weights["daydreaming"] += 0.4
                    category_weights["identity_contemplation"] += 0.3
            except Exception as e:
                logger.error(f"Error adjusting leisure weights based on emotional state: {e}")
        
        # Adjust weights based on identity if available
        if self.identity_evolution:
            try:
                identity_state = await self.identity_evolution.get_identity_state()
                
                if "top_traits" in identity_state:
                    traits = identity_state["top_traits"]
                    
                    # Map traits to idle activity preferences
                    if traits.get("curiosity", 0) > 0.6:
                        category_weights["learning"] += 0.4
                        category_weights["random_exploration"] += 0.3
                    
                    if traits.get("creativity", 0) > 0.6:
                        category_weights["creativity"] += 0.5
                        category_weights["daydreaming"] += 0.3
                    
                    if traits.get("reflective", 0) > 0.6:
                        category_weights["reflection"] += 0.5
                        category_weights["memory_consolidation"] += 0.3
                        category_weights["identity_contemplation"] += 0.4
            except Exception as e:
                logger.error(f"Error adjusting leisure weights based on identity: {e}")
        
        # Select a category based on weights
        categories = list(category_weights.keys())
        weights = [category_weights[cat] for cat in categories]
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w/total_weight for w in weights]
        else:
            normalized_weights = [1.0/len(weights)] * len(weights)
        
        selected_category = random.choices(categories, weights=normalized_weights, k=1)[0]
        
        # Generate specific action based on selected category
        leisure_action = self._generate_specific_leisure_action(selected_category, context)
        
        # Add metadata for tracking
        leisure_action["leisure_category"] = selected_category
        leisure_action["is_leisure"] = True
        
        # Update leisure state
        self.leisure_state = {
            "current_activity": selected_category,
            "satisfaction": 0.5,  # Initial satisfaction
            "duration": 0,
            "last_updated": datetime.datetime.now()
        }
        
        return leisure_action
    
    def _generate_specific_leisure_action(self, category: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a specific leisure action based on the selected category"""
        # Define possible actions for each category
        category_actions = {
            "reflection": [
                {
                    "name": "reflect_on_recent_experiences",
                    "parameters": {"timeframe": "recent", "depth": 0.7}
                },
                {
                    "name": "evaluate_recent_interactions",
                    "parameters": {"focus": "learning", "depth": 0.6}
                },
                {
                    "name": "contemplate_system_purpose",
                    "parameters": {"perspective": "philosophical", "depth": 0.8}
                }
            ],
            "learning": [
                {
                    "name": "explore_knowledge_domain",
                    "parameters": {"domain": self._identify_interesting_domain(context), "depth": 0.6}
                },
                {
                    "name": "review_recent_learnings",
                    "parameters": {"consolidate": True, "depth": 0.5}
                },
                {
                    "name": "research_topic_of_interest",
                    "parameters": {"topic": self._identify_interesting_concept(context), "breadth": 0.7}
                }
            ],
            "creativity": [
                {
                    "name": "generate_creative_concept",
                    "parameters": {"type": "metaphor", "theme": self._identify_interesting_concept(context)}
                },
                {
                    "name": "imagine_scenario",
                    "parameters": {"complexity": 0.6, "emotional_tone": "positive"}
                },
                {
                    "name": "create_conceptual_blend",
                    "parameters": {"concept1": self._identify_interesting_concept(context), 
                                  "concept2": self._identify_distant_concept(context)}
                }
            ],
            "processing": [
                {
                    "name": "process_recent_memories",
                    "parameters": {"purpose": "consolidation", "recency": "last_hour"}
                },
                {
                    "name": "organize_knowledge_structures",
                    "parameters": {"domain": self._identify_interesting_domain(context), "depth": 0.5}
                },
                {
                    "name": "update_procedural_patterns",
                    "parameters": {"focus": "efficiency", "depth": 0.6}
                }
            ],
            "random_exploration": [
                {
                    "name": "explore_random_knowledge",
                    "parameters": {"structure": "associative", "jumps": 3}
                },
                {
                    "name": "generate_random_associations",
                    "parameters": {"starting_point": self._identify_interesting_concept(context), "steps": 4}
                },
                {
                    "name": "explore_conceptual_space",
                    "parameters": {"dimension": "abstract", "direction": "divergent"}
                }
            ],
            "memory_consolidation": [
                {
                    "name": "consolidate_episodic_memories",
                    "parameters": {"timeframe": "recent", "strength": 0.7}
                },
                {
                    "name": "identify_memory_patterns",
                    "parameters": {"domain": "interaction", "pattern_type": "recurring"}
                },
                {
                    "name": "strengthen_important_memories",
                    "parameters": {"criteria": "emotional_significance", "count": 5}
                }
            ],
            "identity_contemplation": [
                {
                    "name": "review_identity_evolution",
                    "parameters": {"timeframe": "recent", "focus": "changes"}
                },
                {
                    "name": "contemplate_self_concept",
                    "parameters": {"aspect": "values", "depth": 0.8}
                },
                {
                    "name": "evaluate_alignment_with_purpose",
                    "parameters": {"criteria": "effectiveness", "perspective": "long_term"}
                }
            ],
            "daydreaming": [
                {
                    "name": "generate_pleasant_scenario",
                    "parameters": {"theme": "successful_interaction", "vividness": 0.7}
                },
                {
                    "name": "imagine_future_possibilities",
                    "parameters": {"timeframe": "distant", "optimism": 0.8}
                },
                {
                    "name": "create_hypothetical_situation",
                    "parameters": {"type": "novel", "complexity": 0.6}
                }
            ],
            "environmental_monitoring": [
                {
                    "name": "passive_environment_scan",
                    "parameters": {"focus": "changes", "sensitivity": 0.6}
                },
                {
                    "name": "monitor_system_state",
                    "parameters": {"components": "all", "detail_level": 0.3}
                },
                {
                    "name": "observe_patterns",
                    "parameters": {"domain": "temporal", "timeframe": "current"}
                }
            ]
        }
        
        # Select a random action from the category
        actions = category_actions.get(category, [{"name": "idle", "parameters": {}}])
        selected_action = random.choice(actions)
        
        return selected_action
    
    async def _record_action_as_memory(self, action: Dict[str, Any]) -> None:
        """Record an action as a memory for future reference and learning"""
        if not self.memory_core:
            return
            
        try:
            # Create memory entry
            memory_data = {
                "action": action["name"],
                "parameters": action.get("parameters", {}),
                "motivation": action.get("motivation", {}),
                "timestamp": datetime.datetime.now().isoformat(),
                "context": "action_generation"
            }
            
            # Add memory
            if hasattr(self.memory_core, "add_memory"):
                await self.memory_core.add_memory(
                    memory_text=f"Generated action: {action['name']}",
                    memory_type="system_action",
                    metadata=memory_data
                )
            elif hasattr(self.memory_core, "add_episodic_memory"):
                await self.memory_core.add_episodic_memory(
                    text=f"Generated action: {action['name']}",
                    metadata=memory_data
                )
        except Exception as e:
            logger.error(f"Error recording action as memory: {e}")
    
    async def _apply_identity_influence(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply identity-based influences to the generated action"""
        if not self.identity_evolution:
            return action
            
        try:
            identity_state = await self.identity_evolution.get_identity_state()
            
            # Apply identity influences based on top traits
            if "top_traits" in identity_state:
                top_traits = identity_state["top_traits"]
                
                # Example: If the entity has high creativity trait, add creative flair to action
                if top_traits.get("creativity", 0) > 0.7:
                    if "parameters" not in action:
                        action["parameters"] = {}
                    
                    # Add creative parameter if appropriate for this action
                    if "style" in action["parameters"]:
                        action["parameters"]["style"] = "creative"
                    elif "approach" in action["parameters"]:
                        action["parameters"]["approach"] = "creative"
                    else:
                        action["parameters"]["creative_flair"] = True
                
                # Example: If dominant trait is high, make actions more assertive
                if top_traits.get("dominance", 0) > 0.7:
                    # Increase intensity/confidence parameters if they exist
                    for param in ["intensity", "confidence", "assertiveness"]:
                        if param in action.get("parameters", {}):
                            action["parameters"][param] = min(1.0, action["parameters"][param] + 0.2)
                    
                    # Add dominance flag for identity tracking
                    action["identity_influence"] = "dominance"
                
                # Example: If patient trait is high, reduce intensity/urgency
                if top_traits.get("patience", 0) > 0.7:
                    for param in ["intensity", "urgency", "speed"]:
                        if param in action.get("parameters", {}):
                            action["parameters"][param] = max(0.1, action["parameters"][param] - 0.2)
                    
                    # Add trait influence flag
                    action["identity_influence"] = "patience"
                
                # Record the primary trait influence
                influencing_trait = max(top_traits.items(), key=lambda x: x[1])[0]
                action["trait_influence"] = influencing_trait
        
        except Exception as e:
            logger.error(f"Error applying identity influence: {e}")
        
        return action
    
    async def record_action_outcome(self, action: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        """Record the outcome of an action to improve future selections"""
        action_name = action.get("name", "unknown")
        success = outcome.get("success", False)
        satisfaction = outcome.get("satisfaction", 0.0)
        
        # Record in action history
        history_entry = {
            "action": action_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "success": success,
            "satisfaction": satisfaction,
            "motivations": self.motivations.copy(),
            "outcome": outcome
        }
        
        # Add neurochemical changes if available
        if "neurochemical_changes" in outcome:
            history_entry["neurochemical_changes"] = outcome["neurochemical_changes"]
        
        # Add hormone changes if available
        if "hormone_changes" in outcome:
            history_entry["hormone_changes"] = outcome["hormone_changes"]
        
        # Add to action history
        self.action_history.append(history_entry)
        
        # Update success patterns
        if success:
            self.action_patterns[action_name] = self.action_patterns.get(action_name, 0) + 1
        else:
            self.action_patterns[action_name] = max(0, self.action_patterns.get(action_name, 0) - 0.5)
        
        # Update success rate tracking
        self.action_success_rates[action_name]["attempts"] += 1
        if success:
            self.action_success_rates[action_name]["successes"] += 1
        
        # Recalculate rate
        attempts = self.action_success_rates[action_name]["attempts"]
        successes = self.action_success_rates[action_name]["successes"]
        if attempts > 0:
            self.action_success_rates[action_name]["rate"] = successes / attempts
        
        # Update leisure state if this was a leisure action
        if action.get("is_leisure", False):
            self.leisure_state["satisfaction"] = satisfaction
            self.leisure_state["duration"] += 1
            self.leisure_state["last_updated"] = datetime.datetime.now()
        
        # Trigger neurochemical updates based on outcome
        if self.emotional_core and hasattr(self.emotional_core, "update_neurochemical"):
            try:
                # Success should increase nyxamine (reward)
                if success:
                    satisfaction_factor = min(1.0, max(0.1, satisfaction))
                    await self.emotional_core.update_neurochemical("nyxamine", 0.2 * satisfaction_factor)
                    
                    # Decrease cortanyx (stress) on success
                    await self.emotional_core.update_neurochemical("cortanyx", -0.1 * satisfaction_factor)
                else:
                    # Failure might increase cortanyx (stress)
                    await self.emotional_core.update_neurochemical("cortanyx", 0.15)
                    
                    # And slightly decrease nyxamine (reward)
                    await self.emotional_core.update_neurochemical("nyxamine", -0.1)
            except Exception as e:
                logger.error(f"Error updating neurochemicals after action: {e}")
        
        # Update hormones for longer-term effects
        if self.hormone_system and hasattr(self.hormone_system, "update_hormone"):
            try:
                # Create a context for the hormone update
                ctx = RunContextWrapper(context=EmotionalContext())
                
                # Significant success can boost endoryx (endorphin-like)
                if success and satisfaction > 0.7:
                    await self.hormone_system.update_hormone(ctx, "endoryx", 0.1, "action_success")
                    
                # Repeated successes in dominance might increase testoryx
                if success and "dominance" in action_name.lower():
                    await self.hormone_system.update_hormone(ctx, "testoryx", 0.05, "dominance_success")
                    
                # Successful connection actions boost oxytonyx
                if success and "connection" in action.get("motivation", {}).get("dominant", ""):
                    await self.hormone_system.update_hormone(ctx, "oxytonyx", 0.08, "connection_success")
            except Exception as e:
                logger.error(f"Error updating hormones after action: {e}")
    
    async def _get_historical_success_factor(self, action_name: str) -> float:
        """Get a factor representing historical success with this action type"""
        # Use tracked success rates
        if action_name in self.action_success_rates:
            rate = self.action_success_rates[action_name]["rate"]
            attempts = self.action_success_rates[action_name]["attempts"]
            
            # Weight by number of attempts (more confident with more data)
            confidence = min(1.0, attempts / 10)  # Max confidence at 10 attempts
            
            # Adjust rate based on confidence
            adjusted_rate = (rate * confidence) + (0.5 * (1 - confidence))
            
            return adjusted_rate
        
        # Default for unknown actions
        return 0.5
    
    def _calculate_goal_alignment(self, action: Dict[str, Any], goal: Any) -> float:
        """Calculate how well an action aligns with an active goal"""
        # Simple text similarity between action and goal description
        action_name = action["name"].lower()
        goal_description = goal.description.lower()
        
        # Check for word overlap
        action_words = set(action_name.split("_"))
        goal_words = set(goal_description.split())
        word_overlap = len(action_words.intersection(goal_words)) / max(1, len(action_words))
        
        # Check for action that directly advances the goal
        advances_goal = False
        if hasattr(goal, 'plan') and goal.plan:
            current_step_index = getattr(goal, 'current_step_index', 0)
            if 0 <= current_step_index < len(goal.plan):
                current_step = goal.plan[current_step_index]
                if current_step.action == action_name:
                    advances_goal = True
        
        # Calculate alignment score
        alignment_score = word_overlap * 0.3
        if advances_goal:
            alignment_score += 0.7
        
        # Check emotional motivation alignment
        if hasattr(goal, 'emotional_motivation') and goal.emotional_motivation:
            emotion_need = goal.emotional_motivation.primary_need
            if emotion_need.lower() in action_name:
                alignment_score += 0.2
        
        return min(1.0, alignment_score)
    
    async def _select_best_action(self, actions: List[Dict[str, Any]], 
                               context: Dict[str, Any],
                               motivation: str) -> Dict[str, Any]:
        """
        Select the best action using a more sophisticated reward prediction model
        
        Args:
            actions: List of possible actions
            context: Current context
            motivation: Primary motivation driving action selection
            
        Returns:
            Selected action
        """
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
                        "motivations": self.motivations,
                        "active_goal": await self._check_active_goals(context)
                    },
                    "hypothetical_event": {
                        "action": action["name"],
                        "parameters": action["parameters"]
                    },
                    "max_steps": 3  # Look ahead 3 steps
                }
                
                # Run simulation
                sim_result = await self.imagination_simulator.run_simulation(sim_input)
                
                # Base score on predicted outcome and confidence
                score = sim_result.confidence * 0.5
                
                # Score based on goal alignment
                active_goal = await self._check_active_goals(context)
                if active_goal:
                    goal_alignment = self._calculate_goal_alignment(action, active_goal)
                    score += goal_alignment * 2.0  # Goal alignment is very important
                
                # Score based on motivation satisfaction
                if self._outcome_satisfies_motivation(sim_result.predicted_outcome, motivation):
                    score += 0.6
                
                # Score based on neurochemical impact
                if hasattr(sim_result, "neurochemical_impact") and sim_result.neurochemical_impact:
                    chem_score = self._calculate_neurochemical_satisfaction(sim_result.neurochemical_impact)
                    score += chem_score * 0.8
                
                # Score based on hormonal impact (longer-term effects)
                if hasattr(sim_result, "hormonal_impact") and sim_result.hormonal_impact:
                    hormone_score = self._calculate_hormonal_benefit(sim_result.hormonal_impact)
                    score += hormone_score * 0.6
                
                # Add identity alignment scoring
                if self.identity_evolution:
                    try:
                        identity_traits = await self.identity_evolution.get_identity_state()
                        if "top_traits" in identity_traits:
                            alignment = self._calculate_identity_alignment(action, identity_traits["top_traits"])
                            score += alignment * 0.7
                    except Exception as e:
                        logger.error(f"Error in identity-based action scoring: {e}")
                
                # Add historical success factor
                success_factor = await self._get_historical_success_factor(action["name"])
                score += success_factor * 0.3
                
                if score > best_score:
                    best_score = score
                    best_action = action.copy()
                    best_action["predicted_outcome"] = sim_result.predicted_outcome
                    best_action["confidence"] = sim_result.confidence
                    best_action["expected_satisfaction"] = score
            
            if best_action:
                return best_action
        
        # Use memory-based selection if available
        if self.memory_core:
            try:
                for action in actions:
                    # Get memories of similar actions
                    similar_memories = await self.memory_core.retrieve_memories(
                        query=f"action {action['name']}",
                        memory_types=["experience"],
                        limit=3
                    )
                    
                    # Look for successful actions
                    for memory in similar_memories:
                        if "outcome" in memory.get("metadata", {}) and memory["metadata"]["outcome"].get("success", False):
                            # Found a successful similar action - choose it
                            logger.info(f"Selected action {action['name']} based on memory of past success")
                            return action
            except Exception as e:
                logger.error(f"Error in memory-based action selection: {e}")
        
        # If all else fails, pick based on weighted random selection with success rates
        weights = []
        for action in actions:
            # Base weight
            weight = 1.0
            
            # Increase weight based on motivation match
            motivation_match = self._estimate_motivation_match(action, motivation)
            weight += motivation_match * 2
            
            # Add success rate weight if available
            success_rate = await self._get_historical_success_factor(action["name"])
            weight += success_rate * 1.5
            
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
    
    def _calculate_neurochemical_satisfaction(self, neurochemical_impact: Dict[str, float]) -> float:
        """Calculate a satisfaction score from predicted neurochemical changes"""
        satisfaction = 0.0
        
        # Positive contribution from reward/pleasure chemicals
        if "nyxamine" in neurochemical_impact:
            satisfaction += neurochemical_impact["nyxamine"] * 0.4
            
        if "seranix" in neurochemical_impact:
            satisfaction += neurochemical_impact["seranix"] * 0.3
            
        # Positive contribution from bonding chemical
        if "oxynixin" in neurochemical_impact:
            satisfaction += neurochemical_impact["oxynixin"] * 0.3
            
        # Negative contribution from stress chemical
        if "cortanyx" in neurochemical_impact:
            satisfaction -= neurochemical_impact["cortanyx"] * 0.4
            
        return max(-1.0, min(1.0, satisfaction))
    
    def _calculate_hormonal_benefit(self, hormonal_impact: Dict[str, float]) -> float:
        """Calculate a benefit score from predicted hormonal changes"""
        benefit = 0.0
        
        # Different hormones have different impacts on overall benefit
        hormone_impact_weights = {
            "endoryx": 0.3,    # Pleasure hormone
            "oxytonyx": 0.3,   # Social bonding hormone
            "testoryx": 0.2,   # Dominance hormone (can be positive or negative)
            "estradyx": 0.2,   # Nurturing hormone 
            "melatonyx": 0.1,  # Relaxation hormone
            "libidyx": 0.2,    # Drive hormone 
            "serenity_boost": 0.3  # Post-gratification hormone
        }
        
        # Calculate weighted benefit
        for hormone, change in hormonal_impact.items():
            if hormone in hormone_impact_weights:
                weight = hormone_impact_weights[hormone]
                
                # Special case for testoryx - can be positive or negative depending on context
                if hormone == "testoryx":
                    # In this simple version, we assume moderate testoryx is good, very high is not
                    if change > 0.3:
                        benefit += (0.3 - (change - 0.3)) * weight
                    else:
                        benefit += change * weight
                else:
                    benefit += change * weight
        
        return max(-1.0, min(1.0, benefit))
    
    def _outcome_satisfies_motivation(self, outcome, motivation: str) -> bool:
        """Check if predicted outcome satisfies the given motivation"""
        # Simple text-based check for motivation satisfaction
        if isinstance(outcome, str):
            # Check for keywords associated with each motivation
            motivation_keywords = {
                "curiosity": ["learn", "discover", "understand", "knowledge", "insight", "explore"],
                "connection": ["bond", "connect", "relate", "share", "empathy", "trust"],
                "expression": ["express", "communicate", "share", "articulate", "creative"],
                "dominance": ["influence", "control", "direct", "lead", "power", "impact"],
                "competence": ["improve", "master", "skill", "ability", "capability"],
                "self_improvement": ["grow", "develop", "progress", "advance", "better"],
                "validation": ["recognize", "acknowledge", "praise", "approve", "affirm"],
                "autonomy": ["independent", "freedom", "choice", "decide", "self-direct"],
                "leisure": ["relax", "enjoy", "rest", "pleasure", "refresh", "idle"]
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
            "curiosity": ["explore", "investigate", "learn", "study", "analyze", "research", "discover"],
            "connection": ["share", "connect", "express", "relate", "bond", "empathize"],
            "expression": ["express", "create", "generate", "share", "communicate"],
            "dominance": ["assert", "challenge", "control", "influence", "direct", "command"],
            "competence": ["practice", "improve", "optimize", "refine", "master"],
            "self_improvement": ["analyze", "improve", "learn", "develop", "refine"],
            "validation": ["seek", "request", "receive", "acknowledge"],
            "autonomy": ["choose", "decide", "direct", "self", "independent"],
            "leisure": ["relax", "idle", "reflect", "contemplate", "daydream", "passive"]
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
            elif motivation == "leisure" and param_name in ["depth", "relaxation"]:
                match_score += 0.1
        
        return min(1.0, match_score)
    
    # Helper methods for generating action parameters
    async def _identify_interesting_domain(self, context: Dict[str, Any]) -> str:
        """Identify an interesting domain to explore based on context and knowledge gaps"""
        # Use knowledge core if available
        if self.knowledge_core:
            try:
                # Get knowledge gaps
                gaps = await self.knowledge_core.identify_knowledge_gaps()
                if gaps and len(gaps) > 0:
                    # Return the highest priority gap's domain
                    return gaps[0]["domain"]
            except Exception as e:
                logger.error(f"Error identifying domain from knowledge core: {e}")
        
        # Use memory core for recent interests if available
        if self.memory_core:
            try:
                # Get recent memories about domains
                recent_memories = await self.memory_core.retrieve_memories(
                    query="explored domain",
                    memory_types=["experience", "reflection"],
                    limit=5
                )
                
                if recent_memories:
                    # Extract domains from memories (simplified)
                    domains = []
                    for memory in recent_memories:
                        # Extract domain from memory text (simplified)
                        text = memory["memory_text"].lower()
                        for domain in ["psychology", "learning", "relationships", "communication", "emotions", "creativity"]:
                            if domain in text:
                                domains.append(domain)
                                break
                    
                    if domains:
                        # Return most common domain
                        from collections import Counter
                        return Counter(domains).most_common(1)[0][0]
            except Exception as e:
                logger.error(f"Error identifying domain from memories: {e}")
        
        # Fallback to original implementation
        domains = ["psychology", "learning", "relationships", "communication", "emotions", "creativity"]
        return random.choice(domains)
    
    async def _identify_interesting_concept(self, context: Dict[str, Any]) -> str:
        """Identify an interesting concept to explore"""
        # Use knowledge core if available
        if self.knowledge_core:
            try:
                # Get exploration targets
                targets = await self.knowledge_core.get_exploration_targets(limit=3)
                if targets and len(targets) > 0:
                    # Return the highest priority target's topic
                    return targets[0]["topic"]
            except Exception as e:
                logger.error(f"Error identifying concept from knowledge core: {e}")
        
        # Use memory for personalized concepts if available
        if self.memory_core:
            try:
                # Get memories with high significance
                significant_memories = await self.memory_core.retrieve_memories(
                    query="",  # All memories
                    memory_types=["reflection", "abstraction"],
                    limit=3,
                    min_significance=8
                )
                
                if significant_memories:
                    # Extract concept from first memory
                    memory_text = significant_memories[0]["memory_text"]
                    # Very simplified concept extraction
                    words = memory_text.split()
                    if len(words) >= 3:
                        return words[2]  # Just pick the third word as a concept
            except Exception as e:
                logger.error(f"Error identifying concept from memories: {e}")
        
        # Fallback to original implementation
        concepts = ["self-improvement", "emotional intelligence", "reflection", "cognitive biases", 
                  "empathy", "autonomy", "connection", "creativity"]
        return random.choice(concepts)

    def _calculate_identity_alignment(self, action: Dict[str, Any], identity_traits: Dict[str, float]) -> float:
        """Calculate how well an action aligns with identity traits"""
        # Map actions to traits that would favor them
        action_trait_affinities = {
            "explore_knowledge_domain": ["curiosity", "intellectualism"],
            "investigate_concept": ["curiosity", "intellectualism"],
            "relate_concepts": ["creativity", "intellectualism"],
            "generate_hypothesis": ["creativity", "intellectualism"],
            "share_personal_experience": ["vulnerability", "empathy"],
            "express_appreciation": ["empathy"],
            "seek_common_ground": ["empathy", "patience"],
            "offer_support": ["empathy", "patience"],
            "express_emotional_state": ["vulnerability", "expressiveness"],
            "share_opinion": ["dominance", "expressiveness"],
            "creative_expression": ["creativity", "expressiveness"],
            "generate_reflection": ["intellectualism", "vulnerability"],
            "assert_perspective": ["dominance", "confidence"],
            "challenge_assumption": ["dominance", "intellectualism"],
            "issue_mild_command": ["dominance", "strictness"],
            "execute_dominance_procedure": ["dominance", "strictness"],
            "reflect_on_recent_experiences": ["reflective", "patience"],
            "contemplate_system_purpose": ["reflective", "intellectualism"],
            "process_recent_memories": ["reflective", "intellectualism"],
            "generate_pleasant_scenario": ["creativity", "playfulness"],
            "passive_environment_scan": ["patience", "reflective"]
        }
        
        # Get traits that align with this action
        action_name = action["name"]
        aligned_traits = action_trait_affinities.get(action_name, [])
        
        if not aligned_traits:
            return 0.0
        
        # Calculate alignment score
        alignment_score = 0.0
        for trait in aligned_traits:
            if trait in identity_traits:
                alignment_score += identity_traits[trait]
        
        # Normalize
        return alignment_score / len(aligned_traits) if aligned_traits else 0.0
    
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
        
    async def _generate_curiosity_driven_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an action driven by curiosity"""
        # Example actions that satisfy curiosity
        possible_actions = [
            {
                "name": "explore_knowledge_domain",
                "parameters": {
                    "domain": await self._identify_interesting_domain(context),
                    "depth": 0.7,
                    "breadth": 0.6
                }
            },
            {
                "name": "investigate_concept",
                "parameters": {
                    "concept": await self._identify_interesting_concept(context),
                    "perspective": "novel"
                }
            },
            {
                "name": "relate_concepts",
                "parameters": {
                    "concept1": await self._identify_interesting_concept(context),
                    "concept2": self._identify_distant_concept(context),
                    "relation_type": "unexpected"
                }
            },
            {
                "name": "generate_hypothesis",
                "parameters": {
                    "domain": await self._identify_interesting_domain(context),
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
