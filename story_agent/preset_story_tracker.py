# story_agent/preset_story_tracker.py

from typing import Dict, Any, Optional, List
from story_templates.preset_stories import PresetStory, StoryBeat


class PresetStoryTracker:
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.current_story_id = None
        self.completed_beats = []
        self.current_act = 1
        self.story_variables = {}
        self.npc_mappings = {}  # Map preset NPC roles to actual NPC IDs
        self.conflict_mappings = {}  # Map preset conflicts to actual conflict IDs
        
    async def initialize_preset_story(self, story: PresetStory):
        """Initialize tracking for a preset story"""
        self.current_story_id = story.id
        
        # Store in database
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO PresetStoryProgress (
                    user_id, conversation_id, story_id, 
                    current_act, completed_beats, story_variables
                ) VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (user_id, conversation_id) 
                DO UPDATE SET story_id = $3
            """, self.user_id, self.conversation_id, story.id, 
                1, json.dumps([]), json.dumps({}))
                
    async def check_beat_triggers(self, context: Dict[str, Any]) -> Optional[StoryBeat]:
        """Check if any story beats should trigger"""
        story = await self.get_current_story()
        if not story:
            return None
            
        for beat in story.story_beats:
            if beat.id in self.completed_beats:
                continue
                
            if self.evaluate_trigger_conditions(beat.trigger_conditions, context):
                return beat
                
        return None

    async def map_preset_to_dynamic_entities(self, story: PresetStory):
        """Map preset story entities to actual game entities"""
        
        # Map NPCs based on archetypes/roles
        for preset_npc in story.required_npcs:
            actual_npc = await self.find_or_create_matching_npc(preset_npc)
            self.npc_mappings[preset_npc.id] = actual_npc.npc_id
            
        # Map locations
        for preset_location in story.required_locations:
            actual_location = await self.find_or_create_matching_location(preset_location)
            
    async def find_or_create_matching_npc(self, preset_npc):
        """Find an NPC that matches preset requirements or create one"""
        # First, try to find existing NPC with matching archetypes
        npcs = await get_available_npcs(
            self.ctx,
            min_dominance=preset_npc.min_dominance,
            gender_filter=preset_npc.gender,
            min_stage=preset_npc.min_narrative_stage
        )
        
        for npc in npcs:
            if self.npc_matches_requirements(npc, preset_npc):
                return npc
                
        # If no match, create NPC with required traits
        return await self.create_npc_for_preset(preset_npc)
        
    async def translate_beat_to_context(self, beat: StoryBeat) -> Dict[str, Any]:
        """Translate preset beat requirements to current game context"""
        return {
            "required_npcs": [self.npc_mappings.get(npc_id) for npc_id in beat.required_npcs],
            "required_location": self.location_mappings.get(beat.location_id),
            "conflict_context": self.conflict_mappings.get(beat.conflict_id),
            "narrative_stage_requirements": beat.stage_requirements
        }
        
    async def complete_story_beat(self, beat_id: str, outcomes: Dict[str, Any]):
        """Mark a story beat as completed and apply outcomes"""
        self.completed_beats.append(beat_id)
        
        # Apply outcomes to game state
        await self.apply_beat_outcomes(outcomes)
        
        # Check for act progression
        await self.check_act_progression()
        
        # Store progress
        await self.save_progress()
