# story_templates/preset_story_loader.py

import json
import logging
from typing import List, Dict, Any
from db.connection import get_db_connection_context
from story_templates.preset_stories import PresetStory, EXAMPLE_PRESET_STORY

logger = logging.getLogger(__name__)

class PresetStoryLoader:
    """Loads preset stories into the database"""
    
    @staticmethod
    async def load_preset_story(story: PresetStory):
        """Load a single preset story into the database"""
        async with get_db_connection_context() as conn:
            # Check if story already exists
            existing = await conn.fetchval(
                "SELECT id FROM PresetStories WHERE story_id = $1",
                story.id
            )
            
            if existing:
                logger.info(f"Preset story {story.id} already exists, skipping")
                return
            
            # Serialize the story data
            story_data = {
                "id": story.id,
                "name": story.name,
                "theme": story.theme,
                "synopsis": story.synopsis,
                "acts": story.acts,
                "story_beats": [
                    {
                        "id": beat.id,
                        "name": beat.name,
                        "description": beat.description,
                        "trigger_conditions": beat.trigger_conditions,
                        "required_npcs": beat.required_npcs,
                        "required_locations": beat.required_locations,
                        "narrative_stage": beat.narrative_stage,
                        "outcomes": beat.outcomes,
                        "dialogue_hints": beat.dialogue_hints,
                        "can_skip": beat.can_skip
                    }
                    for beat in story.story_beats
                ],
                "required_npcs": story.required_npcs,
                "required_locations": story.required_locations,
                "required_conflicts": story.required_conflicts,
                "dynamic_elements": story.dynamic_elements,
                "player_choices_matter": story.player_choices_matter,
                "flexibility_level": story.flexibility_level,
                "enforce_ending": story.enforce_ending
            }
            
            await conn.execute(
                """
                INSERT INTO PresetStories (story_id, story_data, created_at)
                VALUES ($1, $2, NOW())
                """,
                story.id,
                json.dumps(story_data)
            )
            
            logger.info(f"Loaded preset story: {story.name}")
    
    @staticmethod
    async def load_all_preset_stories():
        """Load all available preset stories"""
        # Add all your preset stories here
        preset_stories = [
            EXAMPLE_PRESET_STORY,
            # Add more preset stories as you create them
        ]
        
        for story in preset_stories:
            await PresetStoryLoader.load_preset_story(story)
