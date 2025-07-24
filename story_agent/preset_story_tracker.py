# story_agent/preset_story_tracker.py

class PresetStoryTracker:
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.current_story_id = None
        self.completed_beats = []
        self.current_act = 1
        self.story_variables = {}
        
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
        
    async def complete_story_beat(self, beat_id: str, outcomes: Dict[str, Any]):
        """Mark a story beat as completed and apply outcomes"""
        self.completed_beats.append(beat_id)
        
        # Apply outcomes to game state
        await self.apply_beat_outcomes(outcomes)
        
        # Check for act progression
        await self.check_act_progression()
        
        # Store progress
        await self.save_progress()
