"""
Creative Task Agent - Generates contextually appropriate and creative tasks/challenges 
based on the current NPCs, scenario, and player context.
"""

import random
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from utils.npc_utils import get_npc_personality_traits
from utils.story_context import get_current_scenario_context
from utils.memory_utils import get_relevant_memories

logger = logging.getLogger(__name__)

@dataclass
class TaskContext:
    """Context for task generation"""
    npc_id: str
    npc_name: str
    npc_traits: List[str]
    npc_role: str
    scenario_type: str
    location: str
    player_status: Dict
    relevant_memories: List[Dict]
    intensity_level: int  # 1-5 scale

@dataclass
class CreativeTask:
    """Represents a creative task/challenge"""
    title: str
    description: str
    duration: str  # e.g. "10 minutes", "1 hour"
    difficulty: int  # 1-5 scale
    required_items: List[str]
    success_criteria: str
    reward_type: str
    npc_involvement: str
    task_type: str  # e.g. "skill_challenge", "service", "performance"

class CreativeTaskGenerator:
    """Generates creative and contextually appropriate tasks"""
    
    def __init__(self):
        # Task type categories
        self.task_types = {
            "skill_challenge": [
                "Create something artistic that represents {npc_trait}",
                "Learn and demonstrate a new skill that would impress {npc_name}",
                "Solve a complex puzzle or riddle designed by {npc_name}",
                "Organize and optimize {location} according to {npc_name}'s standards",
                "Document and present research on a topic {npc_name} is passionate about"
            ],
            "service": [
                "Assist in organizing and cataloging {npc_name}'s collection of {items}",
                "Help prepare and execute a special event for {npc_name}",
                "Create a detailed improvement proposal for {location}",
                "Act as {npc_name}'s personal assistant for {duration}",
                "Maintain and enhance a specific aspect of {location}"
            ],
            "performance": [
                "Prepare and deliver a presentation on {topic}",
                "Demonstrate mastery of {skill} in front of {npc_name}",
                "Create and perform a creative piece expressing {emotion}",
                "Lead a group activity showcasing {player_strength}",
                "Design and execute a special performance incorporating {npc_interest}"
            ],
            "personal_growth": [
                "Develop and document progress in a new skill {npc_name} values",
                "Create a self-improvement plan focused on {trait}",
                "Keep a detailed journal tracking progress in {area}",
                "Practice and demonstrate improvement in {weakness}",
                "Study and apply {npc_name}'s expertise in {field}"
            ],
            "leadership": [
                "Organize and lead a group activity in {location}",
                "Create and implement a new system for {task}",
                "Mentor another character in {skill}",
                "Develop and present a proposal for {improvement}",
                "Coordinate a collaborative project involving {participants}"
            ]
        }
        
        # Task modifiers to increase creativity
        self.task_modifiers = {
            "time_constraints": [
                "under a strict time limit",
                "while maintaining perfect timing",
                "during specific time windows",
                "with synchronized precision",
                "following a complex schedule"
            ],
            "environmental": [
                "in a challenging environment",
                "while maintaining absolute order",
                "under specific conditions",
                "with limited resources",
                "in a designated space"
            ],
            "quality_standards": [
                "meeting exacting standards",
                "with perfect attention to detail",
                "following strict guidelines",
                "achieving specific metrics",
                "with documented quality checks"
            ],
            "presentation": [
                "with professional presentation",
                "incorporating specific themes",
                "using required formats",
                "following formal protocols",
                "with artistic elements"
            ]
        }

    def _get_task_context(self, npc_id: str, scenario_id: str) -> TaskContext:
        """Gather context for task generation"""
        npc_data = get_npc_personality_traits(npc_id)
        scenario = get_current_scenario_context(scenario_id)
        memories = get_relevant_memories(npc_id, limit=5)
        
        return TaskContext(
            npc_id=npc_id,
            npc_name=npc_data["name"],
            npc_traits=npc_data["traits"],
            npc_role=npc_data["role"],
            scenario_type=scenario["type"],
            location=scenario["location"],
            player_status=scenario["player_status"],
            relevant_memories=memories,
            intensity_level=min(5, max(1, scenario["intensity"]))
        )

    def _select_task_type(self, context: TaskContext) -> str:
        """Select appropriate task type based on context"""
        weights = {
            "skill_challenge": 1.0,
            "service": 1.0,
            "performance": 1.0,
            "personal_growth": 1.0,
            "leadership": 1.0
        }
        
        # Adjust weights based on NPC traits
        for trait in context.npc_traits:
            if "perfectionist" in trait.lower():
                weights["skill_challenge"] *= 1.5
            elif "nurturing" in trait.lower():
                weights["personal_growth"] *= 1.5
            elif "authoritative" in trait.lower():
                weights["leadership"] *= 1.5
            elif "artistic" in trait.lower():
                weights["performance"] *= 1.5
            elif "organized" in trait.lower():
                weights["service"] *= 1.5
        
        # Adjust for scenario type
        if "training" in context.scenario_type:
            weights["skill_challenge"] *= 1.3
            weights["personal_growth"] *= 1.3
        elif "social" in context.scenario_type:
            weights["performance"] *= 1.3
            weights["leadership"] *= 1.3
        
        # Select based on weights
        task_types = list(weights.keys())
        weights_list = list(weights.values())
        return random.choices(task_types, weights=weights_list, k=1)[0]

    def _generate_base_task(self, task_type: str, context: TaskContext) -> str:
        """Generate base task description"""
        task_templates = self.task_types[task_type]
        base_task = random.choice(task_templates)
        
        # Fill in template variables
        replacements = {
            "{npc_name}": context.npc_name,
            "{npc_trait}": random.choice(context.npc_traits),
            "{location}": context.location,
            "{duration}": f"{random.randint(15, 60)} minutes",
            "{items}": "relevant items",  # Would be more specific in real implementation
            "{topic}": "chosen subject",  # Would be contextual
            "{skill}": "specific skill",  # Based on scenario
            "{emotion}": "selected emotion",
            "{player_strength}": "player ability",
            "{npc_interest}": "NPC's interest",
            "{trait}": "target trait",
            "{area}": "focus area",
            "{weakness}": "improvement area",
            "{field}": "expertise field",
            "{task}": "specific task",
            "{improvement}": "target improvement",
            "{participants}": "involved parties"
        }
        
        for key, value in replacements.items():
            base_task = base_task.replace(key, value)
            
        return base_task

    def _add_modifiers(self, base_task: str, context: TaskContext) -> str:
        """Add creative modifiers to the base task"""
        # Select modifiers based on intensity level
        num_modifiers = min(3, max(1, context.intensity_level))
        modifier_types = random.sample(list(self.task_modifiers.keys()), num_modifiers)
        
        modifiers = []
        for mod_type in modifier_types:
            modifier = random.choice(self.task_modifiers[mod_type])
            modifiers.append(modifier)
        
        # Combine base task with modifiers
        full_task = f"{base_task} {', '.join(modifiers)}"
        return full_task

    def _generate_success_criteria(self, task_type: str, context: TaskContext) -> str:
        """Generate clear success criteria for the task"""
        base_criteria = {
            "skill_challenge": "Demonstrate proficiency in the required skill with measurable improvement",
            "service": "Complete all specified tasks to the established quality standards",
            "performance": "Successfully execute the performance meeting all required elements",
            "personal_growth": "Show documented progress and reflection on the growth area",
            "leadership": "Effectively coordinate and complete the project with positive team feedback"
        }
        
        return base_criteria[task_type]

    def _determine_reward_type(self, task_type: str, context: TaskContext) -> str:
        """Determine appropriate reward type for the task"""
        reward_types = {
            "skill_challenge": "skill improvement and recognition",
            "service": "increased trust and responsibility",
            "performance": "public acknowledgment and advancement",
            "personal_growth": "character development and new opportunities",
            "leadership": "enhanced status and influence"
        }
        
        return reward_types[task_type]

    def generate_task(self, npc_id: str, scenario_id: str) -> CreativeTask:
        """Generate a complete creative task"""
        # Get context
        context = self._get_task_context(npc_id, scenario_id)
        
        # Select task type
        task_type = self._select_task_type(context)
        
        # Generate base task
        base_task = self._generate_base_task(task_type, context)
        
        # Add modifiers
        full_description = self._add_modifiers(base_task, context)
        
        # Create complete task
        task = CreativeTask(
            title=f"{context.npc_name}'s {task_type.replace('_', ' ').title()} Challenge",
            description=full_description,
            duration=f"{random.randint(20, 120)} minutes",
            difficulty=context.intensity_level,
            required_items=["relevant items based on task"],  # Would be more specific
            success_criteria=self._generate_success_criteria(task_type, context),
            reward_type=self._determine_reward_type(task_type, context),
            npc_involvement=f"Direct oversight and evaluation by {context.npc_name}",
            task_type=task_type
        )
        
        return task

# Example usage:
"""
task_generator = CreativeTaskGenerator()
task = task_generator.generate_task("npc123", "scenario456")
print(f"Task: {task.title}")
print(f"Description: {task.description}")
print(f"Success Criteria: {task.success_criteria}")
print(f"Reward Type: {task.reward_type}")
""" 