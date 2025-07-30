# story_templates/preset_story_loader.py

import json
import logging
from typing import List, Dict, Any
from db.connection import get_db_connection_context
from story_templates.preset_stories import PresetStory
from story_templates.moth.queen_of_thorns_story import QUEEN_OF_THORNS_STORY
from story_templates.moth.poem_integrated_loader import ThornsIntegratedStoryLoader

logger = logging.getLogger(__name__)

class PresetStoryLoader:
    """Loads preset stories into the database"""
    
    @staticmethod
    async def load_preset_story(story: PresetStory, user_id: int = None, conversation_id: int = None):
        """Load a single preset story into the database"""
        async with get_db_connection_context() as conn:
            # Check if story already exists
            existing = await conn.fetchval(
                "SELECT id FROM PresetStories WHERE story_id = $1",
                story.id
            )
            
            if existing:
                logger.info(f"Preset story {story.id} already exists, updating")
                # Update existing story
                await PresetStoryLoader._update_story(conn, story)
            else:
                # Create new story
                await PresetStoryLoader._create_story(conn, story)
            
            # If user_id and conversation_id provided, also load story-specific content
            if user_id and conversation_id and story.id == 'queen_of_thorns':
                logger.info(f"Loading Queen of Thorns themes for user {user_id}")
                await ThornsIntegratedStoryLoader.load_story_with_themes(
                    story, user_id, conversation_id
                )
            
            logger.info(f"Loaded preset story: {story.name}")
    
    @staticmethod
    async def _create_story(conn, story: PresetStory):
        """Create a new preset story"""
        story_data = PresetStoryLoader._serialize_story(story)
        
        await conn.execute(
            """
            INSERT INTO PresetStories (story_id, story_data, created_at)
            VALUES ($1, $2, NOW())
            """,
            story.id,
            json.dumps(story_data)
        )
    
    @staticmethod
    async def _update_story(conn, story: PresetStory):
        """Update an existing preset story"""
        story_data = PresetStoryLoader._serialize_story(story)
        
        await conn.execute(
            """
            UPDATE PresetStories 
            SET story_data = $2, updated_at = NOW()
            WHERE story_id = $1
            """,
            story.id,
            json.dumps(story_data)
        )
    
    @staticmethod
    def _serialize_story(story: PresetStory) -> dict:
        """Serialize a PresetStory to dict for storage"""
        return {
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
            "required_conflicts": getattr(story, 'required_conflicts', []),
            "dynamic_elements": story.dynamic_elements,
            "player_choices_matter": story.player_choices_matter,
            "flexibility_level": story.flexibility_level,
            "enforce_ending": story.enforce_ending,
            "source_material": getattr(story, 'source_material', {}),
            "special_mechanics": getattr(story, 'special_mechanics', {})
        }
    
    @staticmethod
    async def load_all_preset_stories():
        """Load all available preset stories"""
        # Updated to use Queen of Thorns as the main story
        preset_stories = [
            QUEEN_OF_THORNS_STORY,
            # Add other stories here as they're created
        ]
        
        for story in preset_stories:
            await PresetStoryLoader.load_preset_story(story)
    
    @staticmethod
    async def initialize_story_for_user(
        story_id: str, 
        user_id: int, 
        conversation_id: int
    ) -> Dict[str, Any]:
        """Initialize a specific story for a user/conversation"""
        
        if story_id == 'queen_of_thorns':
            # Use the specialized initializer
            from story_templates.moth.story_initializer import QueenOfThornsStoryInitializer
            
            ctx = type('Context', (), {
                'context': {
                    'user_id': user_id,
                    'conversation_id': conversation_id
                }
            })()
            
            result = await QueenOfThornsStoryInitializer.initialize_story(
                ctx, user_id, conversation_id
            )
            
            return result
        else:
            return {
                "status": "error",
                "message": f"Unknown story ID: {story_id}"
            }
    
    @staticmethod
    async def get_available_stories() -> List[Dict[str, str]]:
        """Get list of available preset stories"""
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT story_id, story_data->>'name' as name,
                       story_data->>'theme' as theme,
                       story_data->>'synopsis' as synopsis
                FROM PresetStories
                ORDER BY created_at DESC
            """)
            
            return [
                {
                    "id": row['story_id'],
                    "name": row['name'],
                    "theme": row['theme'],
                    "synopsis": row['synopsis']
                }
                for row in rows
            ]
