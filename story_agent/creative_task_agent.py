"""
Creative Task Agent - Generates contextually appropriate and creative tasks/challenges 
based on the current NPCs, scenario, and player context.
"""

import random
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Database connection
from db.connection import get_db_connection_context

# Context retrieval 
from context.context_service import get_context_service
from context.memory_manager import get_memory_manager
from logic.aggregator_sdk import get_aggregated_roleplay_context

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

    async def _get_npc_data(self, user_id: int, conversation_id: int, npc_id: int) -> Dict:
        """Retrieve NPC data directly from the database"""
        async with get_db_connection_context() as conn:
            # Query for NPC data
            query = """
                SELECT npc_id, npc_name, dominance, cruelty, closeness, 
                       trust, respect, intensity, physical_description,
                       personality_traits, hobbies, likes, dislikes, 
                       current_location
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
            """
            
            row = await conn.fetchrow(query, user_id, conversation_id, npc_id)
            
            if not row:
                # NPC not found
                return {
                    "id": npc_id,
                    "name": f"NPC-{npc_id}",
                    "traits": [],
                    "role": "unknown",
                    "hobbies": [],
                    "likes": []
                }
            
            # Parse JSON fields
            try:
                personality_traits = json.loads(row["personality_traits"]) if isinstance(row["personality_traits"], str) else row["personality_traits"] or []
            except (json.JSONDecodeError, TypeError):
                personality_traits = []
                
            try:
                hobbies = json.loads(row["hobbies"]) if isinstance(row["hobbies"], str) else row["hobbies"] or []
            except (json.JSONDecodeError, TypeError):
                hobbies = []
                
            try:
                likes = json.loads(row["likes"]) if isinstance(row["likes"], str) else row["likes"] or []
            except (json.JSONDecodeError, TypeError):
                likes = []
                
            try:
                dislikes = json.loads(row["dislikes"]) if isinstance(row["dislikes"], str) else row["dislikes"] or []
            except (json.JSONDecodeError, TypeError):
                dislikes = []
            
            # Determine NPC role based on stats
            role = "neutral"
            if row["dominance"] > 70:
                role = "dominant"
            elif row["closeness"] > 70:
                role = "mentor"
            elif row["intensity"] > 70:
                role = "disciplinarian"
            
            return {
                "id": row["npc_id"],
                "name": row["npc_name"],
                "traits": personality_traits,
                "role": role,
                "hobbies": hobbies,
                "likes": likes,
                "dislikes": dislikes,
                "stats": {
                    "dominance": row["dominance"],
                    "cruelty": row["cruelty"],
                    "closeness": row["closeness"],
                    "trust": row["trust"],
                    "respect": row["respect"],
                    "intensity": row["intensity"]
                },
                "description": row["physical_description"],
                "current_location": row["current_location"]
            }

    async def _get_scenario_context(self, user_id: int, conversation_id: int) -> Dict:
        """Get current scenario context using context system or aggregator"""
        try:
            # Try using context service first
            context_service = await get_context_service(user_id, conversation_id)
            context_data = await context_service.get_context()
            
            # Determine scenario type
            scenario_type = "default"
            if "current_conflict" in context_data and context_data["current_conflict"]:
                scenario_type = "conflict"
            elif "current_event" in context_data and context_data["current_event"]:
                scenario_type = "event"
            elif "narrative_stage" in context_data:
                stage = context_data["narrative_stage"].get("name", "").lower()
                if "beginning" in stage:
                    scenario_type = "introduction"
                elif "revelation" in stage:
                    scenario_type = "revelation"
            
            return {
                "type": scenario_type,
                "location": context_data.get("current_location", "Unknown"),
                "intensity": context_data.get("intensity_level", 3),
                "player_status": context_data.get("player_stats", {})
            }
        except Exception as e:
            logger.warning(f"Error getting context from context service: {e}, falling back to aggregator")
            
            # Fallback to using aggregator
            aggregator_data = await get_aggregated_roleplay_context(user_id, conversation_id)
            
            # Determine scenario type
            scenario_type = "default"
            current_roleplay = aggregator_data.get("current_roleplay", {})
            if "CurrentConflict" in current_roleplay and current_roleplay["CurrentConflict"]:
                scenario_type = "conflict"
            elif "CurrentEvent" in current_roleplay and current_roleplay["CurrentEvent"]:
                scenario_type = "event"
            
            # Calculate intensity based on player stats if available
            intensity = 3  # Default medium intensity
            player_stats = aggregator_data.get("player_stats", {})
            if player_stats:
                # The more extreme the player stats are, the higher the intensity
                stat_sum = sum([
                    abs(player_stats.get("corruption", 50) - 50),
                    abs(player_stats.get("obedience", 50) - 50),
                    abs(player_stats.get("dependency", 50) - 50)
                ])
                # Convert to 1-5 scale
                intensity = max(1, min(5, 1 + stat_sum // 30))
            
            return {
                "type": scenario_type,
                "location": aggregator_data.get("current_location", "Unknown"),
                "intensity": intensity,
                "player_status": player_stats
            }

    async def _get_relevant_memories(self, user_id: int, conversation_id: int, npc_id: int, limit: int = 5) -> List[Dict]:
        """Get relevant memories for this NPC"""
        try:
            # Try to use memory manager from context system
            memory_manager = await get_memory_manager(user_id, conversation_id)
            
            # Get NPC name first
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow("""
                    SELECT npc_name FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                """, user_id, conversation_id, npc_id)
                
                npc_name = row["npc_name"] if row else f"NPC-{npc_id}"
            
            # Search memories with vector search if available
            memories = await memory_manager.search_memories(
                query_text=f"tasks with {npc_name}",
                limit=limit,
                tags=[npc_name.lower().replace(" ", "_")],
                use_vector=True
            )
            
            # Format memories
            memory_dicts = []
            for memory in memories:
                if hasattr(memory, 'to_dict'):
                    memory_dicts.append(memory.to_dict())
                else:
                    # If it's already a dict or another format
                    memory_dicts.append(memory)
            
            return memory_dicts
            
        except Exception as e:
            logger.warning(f"Error getting memories from memory manager: {e}, falling back to database")
            
            # Fallback to direct database query
            memories = []
            async with get_db_connection_context() as conn:
                # Try unified memories table first
                try:
                    rows = await conn.fetch("""
                        SELECT memory_text, memory_type, timestamp
                        FROM unified_memories
                        WHERE user_id = $1 AND conversation_id = $2 
                        AND entity_type = 'npc' AND entity_id = $3
                        ORDER BY timestamp DESC
                        LIMIT $4
                    """, user_id, conversation_id, npc_id, limit)
                    
                    for row in rows:
                        memories.append({
                            "content": row["memory_text"],
                            "type": row["memory_type"],
                            "timestamp": row["timestamp"].isoformat()
                        })
                        
                except Exception as e2:
                    logger.warning(f"Error querying unified_memories: {e2}, trying NPCMemories")
                    
                    # Try legacy NPCMemories table
                    try:
                        rows = await conn.fetch("""
                            SELECT memory_text, memory_type, timestamp
                            FROM NPCMemories
                            WHERE npc_id = $1
                            ORDER BY timestamp DESC
                            LIMIT $2
                        """, npc_id, limit)
                        
                        for row in rows:
                            memories.append({
                                "content": row["memory_text"],
                                "type": row["memory_type"],
                                "timestamp": row["timestamp"].isoformat()
                            })
                    except Exception as e3:
                        logger.error(f"Error querying NPCMemories: {e3}")
            
            return memories

    async def _get_task_context(self, user_id: int, conversation_id: int, npc_id: int) -> TaskContext:
        """Gather context for task generation"""
        # Get NPC data
        npc_data = await self._get_npc_data(user_id, conversation_id, npc_id)
        
        # Get scenario context
        scenario = await self._get_scenario_context(user_id, conversation_id)
        
        # Get relevant memories
        memories = await self._get_relevant_memories(user_id, conversation_id, npc_id, limit=5)
        
        return TaskContext(
            npc_id=str(npc_id),
            npc_name=npc_data["name"],
            npc_traits=npc_data["traits"],
            npc_role=npc_data["role"],
            scenario_type=scenario["type"],
            location=scenario["location"],
            player_status=scenario["player_status"],
            relevant_memories=memories,
            intensity_level=scenario["intensity"]
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
            trait = trait.lower()
            if trait in ["perfectionist", "meticulous", "exacting"]:
                weights["skill_challenge"] *= 1.5
            elif trait in ["nurturing", "mentoring", "supportive"]:
                weights["personal_growth"] *= 1.5
            elif trait in ["authoritative", "commanding", "dominant"]:
                weights["leadership"] *= 1.5
            elif trait in ["artistic", "creative", "expressive"]:
                weights["performance"] *= 1.5
            elif trait in ["organized", "methodical", "systematic"]:
                weights["service"] *= 1.5
        
        # Adjust for scenario type
        if "training" in context.scenario_type:
            weights["skill_challenge"] *= 1.3
            weights["personal_growth"] *= 1.3
        elif "social" in context.scenario_type or "event" in context.scenario_type:
            weights["performance"] *= 1.3
            weights["leadership"] *= 1.3
        elif "conflict" in context.scenario_type:
            weights["leadership"] *= 1.4
            weights["skill_challenge"] *= 1.2
        
        # Adjust for NPC role
        if context.npc_role == "dominant":
            weights["service"] *= 1.4
            weights["personal_growth"] *= 1.2
        elif context.npc_role == "mentor":
            weights["skill_challenge"] *= 1.3
            weights["personal_growth"] *= 1.5
        elif context.npc_role == "disciplinarian":
            weights["performance"] *= 1.3
            weights["skill_challenge"] *= 1.3
        
        # Add some randomness to avoid being too predictable
        for key in weights:
            weights[key] *= random.uniform(0.9, 1.1)
        
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
            "{npc_trait}": random.choice(context.npc_traits) if context.npc_traits else "discipline",
            "{location}": context.location,
            "{duration}": f"{random.randint(15, 60)} minutes",
            "{items}": self._get_relevant_items(context),
            "{topic}": self._get_relevant_topic(context),
            "{skill}": self._get_relevant_skill(context),
            "{emotion}": self._get_relevant_emotion(context),
            "{player_strength}": self._get_player_strength(context),
            "{npc_interest}": self._get_npc_interest(context),
            "{trait}": self._get_relevant_trait(context),
            "{area}": self._get_relevant_area(context),
            "{weakness}": self._get_player_weakness(context),
            "{field}": self._get_npc_field(context),
            "{task}": self._get_relevant_task(context),
            "{improvement}": self._get_relevant_improvement(context),
            "{participants}": self._get_relevant_participants(context)
        }
        
        for key, value in replacements.items():
            base_task = base_task.replace(key, value)
            
        return base_task
    
    def _get_relevant_items(self, context: TaskContext) -> str:
        """Get items relevant to this NPC"""
        # Check memories for mentioned items
        for memory in context.relevant_memories:
            if isinstance(memory, dict) and "content" in memory:
                content = memory["content"].lower()
                if "collection" in content and "of" in content:
                    # Try to extract collection items
                    collection_idx = content.find("collection of")
                    if collection_idx > 0:
                        collection_text = content[collection_idx + 13:]
                        end_idx = min(
                            [idx for idx in [collection_text.find("."), collection_text.find(","), 
                                           collection_text.find(" and "), collection_text.find("\n")] 
                             if idx > 0] or [30]
                        )
                        return collection_text[:end_idx]
        
        # Return based on NPC traits or location
        if "artistic" in context.npc_traits or "creative" in context.npc_traits:
            return "art supplies"
        elif "intellectual" in context.npc_traits or "scholarly" in context.npc_traits:
            return "books"
        elif "organized" in context.npc_traits or "meticulous" in context.npc_traits:
            return "documents"
        else:
            return "personal items"

    def _get_relevant_topic(self, context: TaskContext) -> str:
        """Get a topic relevant to this NPC"""
        # Check for NPC interests
        if hasattr(context, 'npc_interests') and context.npc_interests:
            return random.choice(context.npc_interests)
        
        # Based on traits
        if "intellectual" in context.npc_traits:
            return "historical research"
        elif "artistic" in context.npc_traits:
            return "artistic expression"
        elif "disciplined" in context.npc_traits:
            return "discipline methods"
        else:
            # Default topics
            topics = ["modern etiquette", "professional development", "self-improvement", 
                     "efficiency principles", "effective communication"]
            return random.choice(topics)

    def _get_relevant_skill(self, context: TaskContext) -> str:
        """Get a skill relevant to this NPC"""
        # Default skills
        skills = ["etiquette", "organization", "meticulous documentation", 
                 "focused attention", "precise communication"]
        
        # Customize based on traits or role
        if "artistic" in context.npc_traits:
            skills.extend(["drawing", "painting", "creative writing"])
        if "intellectual" in context.npc_traits:
            skills.extend(["research", "analysis", "critical thinking"])
        if context.npc_role == "dominant":
            skills.extend(["leadership", "command presence", "authoritative speech"])
        if context.npc_role == "mentor":
            skills.extend(["teaching", "guidance", "constructive feedback"])
            
        return random.choice(skills)

    def _get_relevant_emotion(self, context: TaskContext) -> str:
        """Get an emotion relevant to this scenario"""
        if context.scenario_type == "conflict":
            return random.choice(["tension", "resolution", "determination"])
        elif "revelation" in context.scenario_type:
            return random.choice(["revelation", "surprise", "understanding"])
        else:
            emotions = ["gratitude", "dedication", "admiration", "respect", "diligence"]
            return random.choice(emotions)

    def _get_player_strength(self, context: TaskContext) -> str:
        """Get a potential player strength based on player stats"""
        player_status = context.player_status
        
        strengths = ["adaptability", "attention to detail", "following instructions"]
        
        # Check player stats for high values
        if player_status.get("confidence", 0) > 60:
            strengths.append("confidence")
        if player_status.get("willpower", 0) > 60:
            strengths.append("willpower")
        if player_status.get("dependency", 0) > 60:
            strengths.append("receptiveness")
        if player_status.get("obedience", 0) > 60:
            strengths.append("obedience")
            
        return random.choice(strengths)

    def _get_npc_interest(self, context: TaskContext) -> str:
        """Get an interest relevant to this NPC"""
        # Default interests
        interests = ["structure", "discipline", "organization", "precision", "dedication"]
        
        # Add traits as potential interests
        if context.npc_traits:
            interests.extend(context.npc_traits)
            
        return random.choice(interests)

    def _get_relevant_trait(self, context: TaskContext) -> str:
        """Get a trait to focus on"""
        return random.choice(["discipline", "attention to detail", "efficiency", 
                            "organization", "obedience", "self-awareness"] + 
                           context.npc_traits)

    def _get_relevant_area(self, context: TaskContext) -> str:
        """Get a relevant area for improvement"""
        areas = ["personal organization", "time management", "communication skills", 
               "professional presentation", "self-discipline"]
        
        # Check player stats for low values
        player_status = context.player_status
        if player_status.get("physical_endurance", 0) < 50:
            areas.append("physical endurance")
        if player_status.get("mental_resilience", 0) < 50:
            areas.append("mental resilience")
            
        return random.choice(areas)

    def _get_player_weakness(self, context: TaskContext) -> str:
        """Get a potential player weakness based on player stats"""
        player_status = context.player_status
        
        weaknesses = ["inconsistency", "disorganization", "lack of focus"]
        
        # Check player stats for low values
        if player_status.get("confidence", 0) < 40:
            weaknesses.append("lack of confidence")
        if player_status.get("willpower", 0) < 40:
            weaknesses.append("insufficient willpower")
        if player_status.get("mental_resilience", 0) < 40:
            weaknesses.append("mental resilience")
            
        return random.choice(weaknesses)

    def _get_npc_field(self, context: TaskContext) -> str:
        """Get field of expertise for this NPC"""
        if "intellectual" in context.npc_traits:
            return random.choice(["research", "academia", "analysis"])
        elif "artistic" in context.npc_traits:
            return random.choice(["creative arts", "design", "aesthetics"])
        elif "disciplined" in context.npc_traits:
            return random.choice(["discipline", "efficiency", "organization"])
        else:
            return random.choice(["professional development", "personal improvement", "interpersonal dynamics"])

    def _get_relevant_task(self, context: TaskContext) -> str:
        """Get a relevant task for this context"""
        if "office" in context.location.lower() or "study" in context.location.lower():
            return random.choice(["document management", "scheduling", "correspondence", "resource allocation"])
        elif "garden" in context.location.lower() or "outdoor" in context.location.lower():
            return random.choice(["garden maintenance", "outdoor activity planning", "environmental organization"])
        else:
            return random.choice(["daily routines", "task management", "personal organization"])

    def _get_relevant_improvement(self, context: TaskContext) -> str:
        """Get a relevant improvement for this context"""
        if "office" in context.location.lower() or "study" in context.location.lower():
            return random.choice(["workspace optimization", "filing system", "scheduling improvements"])
        elif "garden" in context.location.lower() or "outdoor" in context.location.lower():
            return random.choice(["garden layout", "outdoor space utilization", "environmental aesthetics"])
        else:
            return random.choice(["efficiency initiative", "organization system", "productivity enhancement"])

    def _get_relevant_participants(self, context: TaskContext) -> str:
        """Get relevant participants for this context"""
        return "staff members" if context.npc_role == "dominant" else "peers"

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
        
        # Enhanced criteria based on NPC role and intensity
        criteria = base_criteria[task_type]
        
        if context.npc_role == "dominant":
            criteria += " while maintaining proper deference and respect"
        elif context.npc_role == "mentor":
            criteria += " while demonstrating true understanding of the principles involved"
        
        if context.intensity_level >= 4:
            criteria += ", with exceptional attention to detail and precision"
        
        return criteria

    def _determine_reward_type(self, task_type: str, context: TaskContext) -> str:
        """Determine appropriate reward type for the task"""
        base_reward_types = {
            "skill_challenge": "skill improvement and recognition",
            "service": "increased trust and responsibility",
            "performance": "public acknowledgment and advancement",
            "personal_growth": "character development and new opportunities",
            "leadership": "enhanced status and influence"
        }
        
        # Enhanced reward based on NPC and intensity
        reward = base_reward_types[task_type]
        
        if context.intensity_level >= 4:
            reward = f"significant {reward}"
        
        if context.npc_role == "dominant":
            reward += " and increased standing with " + context.npc_name
        elif context.npc_role == "mentor":
            reward += " and deeper mentoring from " + context.npc_name
        
        return reward

    async def generate_task(self, user_id: int, conversation_id: int, npc_id: int) -> CreativeTask:
        """Generate a complete creative task"""
        # Get context
        context = await self._get_task_context(user_id, conversation_id, npc_id)
        
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
            duration=f"{random.randint(20, 60 * context.intensity_level)} minutes",
            difficulty=context.intensity_level,
            required_items=self._get_required_items(task_type, context),
            success_criteria=self._generate_success_criteria(task_type, context),
            reward_type=self._determine_reward_type(task_type, context),
            npc_involvement=f"Direct oversight and evaluation by {context.npc_name}",
            task_type=task_type
        )
        
        return task
    
    def _get_required_items(self, task_type: str, context: TaskContext) -> List[str]:
        """Get required items for this task type"""
        items = []
        
        # Basic requirements based on task type
        if task_type == "skill_challenge":
            if "artistic" in context.npc_traits:
                items = ["art supplies", "paper", "reference materials"]
            else:
                items = ["practice materials", "reference guides", "tools appropriate to the skill"]
        elif task_type == "service":
            items = ["organizational materials", "cleaning supplies", "record-keeping materials"]
        elif task_type == "performance":
            items = ["presentation materials", "visual aids", "appropriate attire"]
        elif task_type == "personal_growth":
            items = ["journal", "progress tracking system", "reference materials"]
        elif task_type == "leadership":
            items = ["planning documents", "communication tools", "coordination materials"]
        
        # Add special item based on intensity
        if context.intensity_level >= 4:
            items.append("specialized equipment provided by " + context.npc_name)
            
        return items

# Example usage:
"""
task_generator = CreativeTaskGenerator()
task = await task_generator.generate_task(123, 456, 789)
print(f"Task: {task.title}")
print(f"Description: {task.description}")
print(f"Success Criteria: {task.success_criteria}")
print(f"Reward Type: {task.reward_type}")
"""
