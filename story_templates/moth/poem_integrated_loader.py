# story_templates/moth/poem_integrated_loader.py

import json
import logging
from typing import Dict, Any
from db.connection import get_db_connection_context
from story_templates.preset_stories import PresetStory

logger = logging.getLogger(__name__)

class PoemIntegratedStoryLoader:
    """Enhanced story loader that integrates poems into the AI's context"""
    
    @staticmethod
    async def load_story_with_poems(story: PresetStory, user_id: int, conversation_id: int):
        """Load a story and seed its poems as special memories for consistent tone"""
        
        async with get_db_connection_context() as conn:
            # First, load the basic story as before
            await PoemIntegratedStoryLoader._load_base_story(story, conn)
            
            # If story has poems, load them as special memories
            if hasattr(story, 'source_material') and 'poems' in story.source_material:
                await PoemIntegratedStoryLoader._load_poems_as_memories(
                    story, user_id, conversation_id, conn
                )
                
            # Load tone instructions if present
            if hasattr(story, 'source_material') and 'tone_prompt' in story.source_material:
                await PoemIntegratedStoryLoader._load_tone_instructions(
                    story, user_id, conversation_id, conn
                )
    
    @staticmethod
    async def _load_base_story(story: PresetStory, conn):
        """Load the base story structure"""
        existing = await conn.fetchval(
            "SELECT id FROM PresetStories WHERE story_id = $1",
            story.id
        )
        
        if existing:
            logger.info(f"Story {story.id} already exists")
            return
            
        story_data = PoemIntegratedStoryLoader._serialize_story(story)
        
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
    async def _load_poems_as_memories(story: PresetStory, user_id: int, conversation_id: int, conn):
        """Load poems as high-significance memories for the AI to reference"""
        
        poems = story.source_material.get('poems', {})
        
        for poem_id, poem_text in poems.items():
            # Check if this poem memory already exists
            exists = await conn.fetchval(
                """
                SELECT id FROM unified_memories
                WHERE entity_type = 'story_source' 
                AND entity_id = 0 
                AND user_id = $1 
                AND conversation_id = $2
                AND metadata->>'poem_id' = $3
                """,
                user_id, conversation_id, poem_id
            )
            
            if not exists:
                # Parse poem into stanzas for better retrieval
                stanzas = poem_text.strip().split('\n\n')
                poem_title = stanzas[0] if stanzas else "Untitled"
                
                # Store the full poem as a high-significance memory
                await conn.execute(
                    """
                    INSERT INTO unified_memories
                    (entity_type, entity_id, user_id, conversation_id, 
                     memory_text, memory_type, tags, significance, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    "story_source",  # Special entity type for story materials
                    0,  # Generic ID for story sources
                    user_id,
                    conversation_id,
                    poem_text,
                    "source_poem",
                    ["poetry", "mood", "tone", story.id, poem_id],
                    10,  # Maximum significance
                    {
                        "poem_id": poem_id,
                        "poem_title": poem_title,
                        "story_id": story.id,
                        "usage": "tone_reference"
                    }
                )
                
                # Also store key imagery phrases for quick access
                imagery_phrases = PoemIntegratedStoryLoader._extract_key_imagery(poem_text)
                for phrase in imagery_phrases:
                    await conn.execute(
                        """
                        INSERT INTO unified_memories
                        (entity_type, entity_id, user_id, conversation_id,
                         memory_text, memory_type, tags, significance, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """,
                        "story_source",
                        0,
                        user_id,
                        conversation_id,
                        phrase,
                        "key_imagery",
                        ["imagery", "metaphor", story.id, poem_id],
                        9,
                        {
                            "source_poem": poem_id,
                            "story_id": story.id,
                            "imagery_type": "poetic_phrase"
                        }
                    )
                
                logger.info(f"Loaded poem '{poem_title}' as memory for story {story.id}")
    
    @staticmethod
    async def _load_tone_instructions(story: PresetStory, user_id: int, conversation_id: int, conn):
        """Load tone instructions as a special directive memory"""
        
        tone_prompt = story.source_material.get('tone_prompt', '')
        
        if tone_prompt:
            # Check if already exists
            exists = await conn.fetchval(
                """
                SELECT id FROM unified_memories
                WHERE entity_type = 'story_directive'
                AND user_id = $1 
                AND conversation_id = $2
                AND metadata->>'story_id' = $3
                """,
                user_id, conversation_id, story.id
            )
            
            if not exists:
                await conn.execute(
                    """
                    INSERT INTO unified_memories
                    (entity_type, entity_id, user_id, conversation_id,
                     memory_text, memory_type, tags, significance, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    "story_directive",
                    0,
                    user_id,
                    conversation_id,
                    tone_prompt,
                    "writing_style",
                    ["directive", "tone", "style", story.id],
                    10,
                    {
                        "story_id": story.id,
                        "directive_type": "tone_and_style",
                        "apply_to": "all_story_content"
                    }
                )
                
                logger.info(f"Loaded tone instructions for story {story.id}")
    
    @staticmethod
    def _extract_key_imagery(poem_text: str) -> list:
        """Extract key imagery phrases from poems"""
        # This is a simple extraction - you could make it more sophisticated
        key_phrases = []
        
        # Look for specific imagery patterns
        imagery_indicators = [
            "like", "as", "you are", "i am", "she is",
            "beneath", "behind", "within", "through"
        ]
        
        lines = poem_text.lower().split('\n')
        for line in lines:
            line = line.strip()
            if any(indicator in line for indicator in imagery_indicators) and len(line) > 10:
                # Clean and store the original case line
                original_line = poem_text.split('\n')[lines.index(line.lower())]
                if original_line.strip():
                    key_phrases.append(original_line.strip())
        
        # Also extract short, powerful phrases
        powerful_phrases = [
            "porcelain curves", "painted smile", "queen of thorns",
            "velvet night", "fortress forged from glass", "altar of her throne",
            "bone-deep doubt", "wings of broken glass", "invisible tattoos",
            "Don't disappear", "Be mine", "velvet affliction",
            "three syllables", "tasting of burning stars", "binary stars",
            "lunar edict", "unopened letter", "sanctified ruin"
        ]
        
        for phrase in powerful_phrases:
            if phrase.lower() in poem_text.lower():
                key_phrases.append(phrase)
        
        return list(set(key_phrases))[:20]  # Limit to 20 most important phrases
    
    @staticmethod
    def _serialize_story(story: PresetStory) -> dict:
        """Serialize the full story including source material"""
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
            "required_conflicts": getattr(story, 'required_conflicts', []),
            "dynamic_elements": story.dynamic_elements,
            "player_choices_matter": story.player_choices_matter,
            "flexibility_level": story.flexibility_level,
            "enforce_ending": story.enforce_ending
        }
        
        # Include source material if present
        if hasattr(story, 'source_material'):
            story_data['source_material'] = story.source_material
            
        return story_data


# Enhanced prompt generator that uses poem memories
class PoemAwarePromptGenerator:
    """Generates prompts that reference stored poems for consistent tone"""
    
    @staticmethod
    async def get_story_context_prompt(user_id: int, conversation_id: int, story_id: str) -> str:
        """Generate a context prompt that includes poem references"""
        
        async with get_db_connection_context() as conn:
            # Fetch poem memories
            poem_memories = await conn.fetch(
                """
                SELECT memory_text, metadata
                FROM unified_memories
                WHERE user_id = $1 
                AND conversation_id = $2
                AND entity_type = 'story_source'
                AND metadata->>'story_id' = $3
                AND memory_type = 'source_poem'
                ORDER BY significance DESC
                """,
                user_id, conversation_id, story_id
            )
            
            # Fetch tone directive
            tone_directive = await conn.fetchval(
                """
                SELECT memory_text
                FROM unified_memories
                WHERE user_id = $1
                AND conversation_id = $2
                AND entity_type = 'story_directive'
                AND metadata->>'story_id' = $3
                """,
                user_id, conversation_id, story_id
            )
            
            # Fetch key imagery
            key_imagery = await conn.fetch(
                """
                SELECT memory_text
                FROM unified_memories
                WHERE user_id = $1
                AND conversation_id = $2
                AND entity_type = 'story_source'
                AND memory_type = 'key_imagery'
                AND metadata->>'story_id' = $3
                LIMIT 10
                """,
                user_id, conversation_id, story_id
            )
            
            # Build the context prompt
            context_parts = []
            
            if tone_directive:
                context_parts.append(f"=== STORY TONE AND STYLE ===\n{tone_directive}\n")
            
            if poem_memories:
                context_parts.append("=== SOURCE POEMS FOR TONE REFERENCE ===")
                for poem in poem_memories:
                    poem_title = poem['metadata'].get('poem_title', 'Untitled')
                    context_parts.append(f"\n--- {poem_title} ---\n{poem['memory_text']}\n")
            
            if key_imagery:
                imagery_phrases = [row['memory_text'] for row in key_imagery]
                context_parts.append(f"\n=== KEY IMAGERY TO ECHO ===\n" + "\n".join(f"• {phrase}" for phrase in imagery_phrases))
            
            return "\n".join(context_parts)
