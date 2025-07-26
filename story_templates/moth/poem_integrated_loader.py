# story_templates/moth/poem_integrated_loader.py
"""
Enhanced poem integration system that loads poems as AI context
Ensures consistent gothic tone throughout interactions
"""

import json
import logging
from typing import Dict, Any, List, Optional
from db.connection import get_db_connection_context
from story_templates.preset_stories import PresetStory

logger = logging.getLogger(__name__)

class PoemIntegratedStoryLoader:
    """Enhanced story loader that integrates poems into the AI's context"""
    
    @staticmethod
    async def load_story_with_poems(story: PresetStory, user_id: int, conversation_id: int):
        """Load a story and seed its poems as special memories for consistent tone"""
        
        try:
            async with get_db_connection_context() as conn:
                # First, load the basic story structure
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
                    
                # Extract and store key imagery for quick reference
                if hasattr(story, 'source_material') and 'poems' in story.source_material:
                    await PoemIntegratedStoryLoader._extract_and_store_imagery(
                        story, user_id, conversation_id, conn
                    )
                    
                logger.info(f"Successfully loaded story {story.id} with poem integration")
                
        except Exception as e:
            logger.error(f"Failed to load story with poems: {e}", exc_info=True)
            raise
    
    @staticmethod
    async def _load_base_story(story: PresetStory, conn):
        """Load the base story structure"""
        existing = await conn.fetchval(
            "SELECT id FROM PresetStories WHERE story_id = $1",
            story.id
        )
        
        if existing:
            logger.info(f"Story {story.id} already exists, updating")
            await conn.execute(
                """
                UPDATE PresetStories 
                SET story_data = $2, updated_at = NOW()
                WHERE story_id = $1
                """,
                story.id,
                json.dumps(PoemIntegratedStoryLoader._serialize_story(story))
            )
        else:
            await conn.execute(
                """
                INSERT INTO PresetStories (story_id, story_data, created_at)
                VALUES ($1, $2, NOW())
                """,
                story.id,
                json.dumps(PoemIntegratedStoryLoader._serialize_story(story))
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
            
            if exists:
                logger.info(f"Poem {poem_id} already loaded, skipping")
                continue
            
            # Parse poem for analysis
            stanzas = poem_text.strip().split('\n\n')
            poem_title = stanzas[0] if stanzas else "Untitled"
            
            # Store the full poem as a high-significance memory
            await conn.execute(
                """
                INSERT INTO unified_memories
                (entity_type, entity_id, user_id, conversation_id,
                 memory_text, memory_type, tags, significance, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                """,
                "story_source",  # Special entity type for story materials
                0,  # Generic ID for story sources
                user_id,
                conversation_id,
                poem_text,
                "source_poem",
                ["poetry", "mood", "tone", story.id, poem_id, "gothic", "romantic"],
                10,  # Maximum significance
                json.dumps({
                    "poem_id": poem_id,
                    "poem_title": poem_title,
                    "story_id": story.id,
                    "usage": "tone_reference",
                    "themes": PoemIntegratedStoryLoader._extract_themes(poem_text)
                })
            )
            
            # Also store individual stanzas for targeted retrieval
            for i, stanza in enumerate(stanzas[1:], 1):  # Skip title
                if stanza.strip():
                    await conn.execute(
                        """
                        INSERT INTO unified_memories
                        (entity_type, entity_id, user_id, conversation_id,
                         memory_text, memory_type, tags, significance, metadata, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                        """,
                        "story_source",
                        0,
                        user_id,
                        conversation_id,
                        stanza,
                        "poem_stanza",
                        ["poetry", "stanza", story.id, poem_id],
                        9,
                        json.dumps({
                            "poem_id": poem_id,
                            "stanza_number": i,
                            "story_id": story.id,
                            "mood": PoemIntegratedStoryLoader._analyze_stanza_mood(stanza)
                        })
                    )
            
            logger.info(f"Loaded poem '{poem_title}' with {len(stanzas)-1} stanzas")
    
    @staticmethod
    async def _load_tone_instructions(story: PresetStory, user_id: int, conversation_id: int, conn):
        """Load tone instructions as a special directive memory"""
        
        tone_prompt = story.source_material.get('tone_prompt', '')
        
        if not tone_prompt:
            return
        
        # Check if already exists
        exists = await conn.fetchval(
            """
            SELECT id FROM unified_memories
            WHERE entity_type = 'story_directive'
            AND user_id = $1 
            AND conversation_id = $2
            AND metadata->>'story_id' = $3
            AND memory_type = 'writing_style'
            """,
            user_id, conversation_id, story.id
        )
        
        if exists:
            # Update existing
            await conn.execute(
                """
                UPDATE unified_memories
                SET memory_text = $4, updated_at = NOW()
                WHERE id = $1
                """,
                exists, tone_prompt
            )
        else:
            # Create new
            await conn.execute(
                """
                INSERT INTO unified_memories
                (entity_type, entity_id, user_id, conversation_id,
                 memory_text, memory_type, tags, significance, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                """,
                "story_directive",
                0,
                user_id,
                conversation_id,
                tone_prompt,
                "writing_style",
                ["directive", "tone", "style", story.id, "gothic", "poetic"],
                10,
                json.dumps({
                    "story_id": story.id,
                    "directive_type": "tone_and_style",
                    "apply_to": "all_story_content",
                    "priority": "maximum"
                })
            )
        
        logger.info(f"Loaded tone instructions for story {story.id}")
    
    @staticmethod
    async def _extract_and_store_imagery(story: PresetStory, user_id: int, conversation_id: int, conn):
        """Extract key imagery and metaphors for quick access"""
        
        poems = story.source_material.get('poems', {})
        all_imagery = []
        
        for poem_id, poem_text in poems.items():
            imagery = PoemIntegratedStoryLoader._extract_key_imagery(poem_text)
            
            for phrase in imagery:
                # Store each imagery phrase
                await conn.execute(
                    """
                    INSERT INTO unified_memories
                    (entity_type, entity_id, user_id, conversation_id,
                     memory_text, memory_type, tags, significance, metadata, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                    ON CONFLICT DO NOTHING
                    """,
                    "story_source",
                    0,
                    user_id,
                    conversation_id,
                    phrase,
                    "key_imagery",
                    ["imagery", "metaphor", story.id, poem_id],
                    9,
                    json.dumps({
                        "source_poem": poem_id,
                        "story_id": story.id,
                        "imagery_type": PoemIntegratedStoryLoader._classify_imagery(phrase),
                        "associated_themes": PoemIntegratedStoryLoader._get_imagery_themes(phrase)
                    })
                )
                all_imagery.append(phrase)
        
        logger.info(f"Extracted and stored {len(all_imagery)} imagery phrases")
    
    @staticmethod
    def _extract_key_imagery(poem_text: str) -> List[str]:
        """Extract key imagery phrases from poems"""
        key_phrases = []
        
        # Direct powerful phrases
        powerful_phrases = [
            # From the poems
            "porcelain curves", "painted smile", "queen of thorns",
            "velvet night", "fortress forged from glass", "altar of her throne",
            "bone-deep doubt", "wings of broken glass", "invisible tattoos",
            "Don't disappear", "Be mine", "velvet affliction",
            "three syllables", "tasting of burning stars", "binary stars",
            "lunar edict", "unopened letter", "sanctified ruin",
            "rough geography of breaks", "moth with wings of broken glass",
            "the mask now heavy in her trembling hands",
            "fingers trace invisible tattoos", "each touch a brand",
            "your skin tastes of prayers", "carved in basalt",
            "my sweetest fall", "my silent scream"
        ]
        
        # Add found powerful phrases
        poem_lower = poem_text.lower()
        for phrase in powerful_phrases:
            if phrase.lower() in poem_lower:
                key_phrases.append(phrase)
        
        # Look for metaphorical patterns
        lines = poem_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Metaphors and similes
            if any(indicator in line.lower() for indicator in ["like", "as if", "as though"]):
                if len(line) < 100:  # Not too long
                    key_phrases.append(line)
            
            # Identity statements
            if any(phrase in line.lower() for phrase in ["i am", "you are", "she is", "we are"]):
                if "moth" in line.lower() or "flame" in line.lower() or "star" in line.lower():
                    key_phrases.append(line)
            
            # Commands and pleas
            if line.startswith(("Don't", "Be ", "Stay", "Come", "Kneel", "Remember")):
                key_phrases.append(line)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_phrases = []
        for phrase in key_phrases:
            if phrase.lower() not in seen:
                seen.add(phrase.lower())
                unique_phrases.append(phrase)
        
        return unique_phrases[:30]  # Limit to 30 most important
    
    @staticmethod
    def _extract_themes(poem_text: str) -> List[str]:
        """Extract thematic elements from a poem"""
        themes = []
        poem_lower = poem_text.lower()
        
        theme_keywords = {
            "masks": ["mask", "porcelain", "facade", "performance"],
            "vulnerability": ["trembling", "breaks", "fragile", "doubt"],
            "abandonment": ["disappear", "leave", "gone", "promise"],
            "devotion": ["worship", "kneel", "prayer", "altar"],
            "duality": ["binary", "two", "both", "beneath"],
            "control": ["command", "queen", "throne", "rule"],
            "transformation": ["moth", "flame", "burn", "transform"],
            "forbidden_love": ["three words", "unspoken", "beneath tongue"]
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in poem_lower for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    @staticmethod
    def _analyze_stanza_mood(stanza: str) -> str:
        """Analyze the emotional mood of a stanza"""
        stanza_lower = stanza.lower()
        
        if any(word in stanza_lower for word in ["trembling", "fear", "doubt", "heavy"]):
            return "vulnerable"
        elif any(word in stanza_lower for word in ["command", "queen", "throne", "worship"]):
            return "dominant"
        elif any(word in stanza_lower for word in ["disappear", "gone", "leave", "goodbye"]):
            return "desperate"
        elif any(word in stanza_lower for word in ["touch", "kiss", "skin", "close"]):
            return "intimate"
        elif any(word in stanza_lower for word in ["burn", "flame", "moth", "destroy"]):
            return "dangerous"
        else:
            return "contemplative"
    
    @staticmethod
    def _classify_imagery(phrase: str) -> str:
        """Classify the type of imagery"""
        phrase_lower = phrase.lower()
        
        if any(word in phrase_lower for word in ["mask", "porcelain", "fortress"]):
            return "protection"
        elif any(word in phrase_lower for word in ["moth", "flame", "burn"]):
            return "attraction"
        elif any(word in phrase_lower for word in ["break", "shatter", "trembl"]):
            return "vulnerability"
        elif any(word in phrase_lower for word in ["queen", "throne", "command"]):
            return "authority"
        elif any(word in phrase_lower for word in ["touch", "skin", "kiss"]):
            return "intimacy"
        else:
            return "atmospheric"
    
    @staticmethod
    def _get_imagery_themes(phrase: str) -> List[str]:
        """Get themes associated with specific imagery"""
        themes = []
        phrase_lower = phrase.lower()
        
        theme_map = {
            "masks": ["identity", "protection", "deception"],
            "moth": ["attraction", "destruction", "devotion"],
            "flame": ["danger", "illumination", "consumption"],
            "queen": ["authority", "isolation", "performance"],
            "break": ["vulnerability", "trauma", "truth"],
            "disappear": ["abandonment", "fear", "loss"],
            "three": ["love", "unspoken", "forbidden"]
        }
        
        for keyword, associated_themes in theme_map.items():
            if keyword in phrase_lower:
                themes.extend(associated_themes)
        
        return list(set(themes))  # Remove duplicates
    
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
            
        # Include special mechanics if present
        if hasattr(story, 'special_mechanics'):
            story_data['special_mechanics'] = story.special_mechanics
            
        return story_data


class PoemAwarePromptGenerator:
    """Generates prompts that reference stored poems for consistent tone"""
    
    @staticmethod
    async def get_story_context_prompt(user_id: int, conversation_id: int, story_id: str) -> str:
        """Generate a context prompt that includes poem references"""
        
        async with get_db_connection_context() as conn:
            # Fetch tone directive
            tone_directive = await conn.fetchval(
                """
                SELECT memory_text
                FROM unified_memories
                WHERE user_id = $1
                AND conversation_id = $2
                AND entity_type = 'story_directive'
                AND metadata->>'story_id' = $3
                AND memory_type = 'writing_style'
                """,
                user_id, conversation_id, story_id
            )
            
            # Fetch key imagery samples
            key_imagery = await conn.fetch(
                """
                SELECT memory_text, metadata
                FROM unified_memories
                WHERE user_id = $1
                AND conversation_id = $2
                AND entity_type = 'story_source'
                AND memory_type = 'key_imagery'
                AND metadata->>'story_id' = $3
                ORDER BY significance DESC
                LIMIT 15
                """,
                user_id, conversation_id, story_id
            )
            
            # Build the context prompt
            context_parts = []
            
            if tone_directive:
                context_parts.append(f"=== STORY TONE AND STYLE ===\n{tone_directive}\n")
            
            if key_imagery:
                imagery_by_type = {}
                for row in key_imagery:
                    img_type = row['metadata'].get('imagery_type', 'general')
                    if img_type not in imagery_by_type:
                        imagery_by_type[img_type] = []
                    imagery_by_type[img_type].append(row['memory_text'])
                
                context_parts.append("=== KEY IMAGERY TO INCORPORATE ===")
                for img_type, phrases in imagery_by_type.items():
                    context_parts.append(f"\n{img_type.upper()}:")
                    for phrase in phrases[:3]:  # Limit per type
                        context_parts.append(f"  • {phrase}")
            
            return "\n".join(context_parts)
    
    @staticmethod
    async def get_npc_dialogue_context(
        user_id: int, conversation_id: int, npc_id: int, 
        emotional_state: str = "neutral"
    ) -> str:
        """Get dialogue context for a specific NPC based on poems"""
        
        async with get_db_connection_context() as conn:
            # Get NPC data
            npc_row = await conn.fetchrow(
                """
                SELECT npc_name, current_mask, trust, dialogue_patterns
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                """,
                user_id, conversation_id, npc_id
            )
            
            if not npc_row:
                return ""
            
            # Get relevant poem stanzas based on mood
            mood_stanzas = await conn.fetch(
                """
                SELECT memory_text
                FROM unified_memories
                WHERE user_id = $1
                AND conversation_id = $2
                AND entity_type = 'story_source'
                AND memory_type = 'poem_stanza'
                AND metadata->>'mood' = $3
                LIMIT 3
                """,
                user_id, conversation_id, emotional_state
            )
            
            # Build dialogue context
            context_parts = [
                f"=== DIALOGUE CONTEXT FOR {npc_row['npc_name'].upper()} ===",
                f"Current Mask: {npc_row.get('current_mask', 'Unknown')}",
                f"Trust Level: {npc_row.get('trust', 0)}/100",
                f"Emotional State: {emotional_state}",
                ""
            ]
            
            if mood_stanzas:
                context_parts.append("MOOD REFERENCE FROM POEMS:")
                for stanza in mood_stanzas:
                    context_parts.append(f"\n{stanza['memory_text']}\n")
            
            # Add dialogue patterns if available
            if npc_row.get('dialogue_patterns'):
                patterns = json.loads(npc_row['dialogue_patterns'])
                if emotional_state in patterns:
                    context_parts.append(f"\nSUGGESTED TONE: {patterns[emotional_state]}")
            
            return "\n".join(context_parts)
    
    @staticmethod
    async def enhance_scene_description(
        user_id: int, conversation_id: int, 
        location: str, atmosphere: str = "neutral"
    ) -> str:
        """Enhance scene descriptions with poetic imagery"""
        
        async with get_db_connection_context() as conn:
            # Get atmospheric imagery
            imagery = await conn.fetch(
                """
                SELECT memory_text
                FROM unified_memories
                WHERE user_id = $1
                AND conversation_id = $2
                AND entity_type = 'story_source'
                AND memory_type = 'key_imagery'
                AND metadata->>'imagery_type' = 'atmospheric'
                LIMIT 5
                """,
                user_id, conversation_id
            )
            
            enhancements = ["ATMOSPHERIC ENHANCEMENT:"]
            
            # Add location-specific imagery
            location_lower = location.lower()
            if "sanctum" in location_lower:
                enhancements.extend([
                    "Candles gutter in silver stands, casting dancing shadows.",
                    "The air tastes of velvet night and unspoken prayers.",
                    "Every surface whispers of power and submission."
                ])
            elif "chamber" in location_lower:
                enhancements.extend([
                    "Moths dance against windows, seeking their destruction.",
                    "Masks rest heavy on mahogany stands.",
                    "The space breathes with whispered confessions."
                ])
            
            # Add imagery from poems
            if imagery:
                enhancements.append("\nPOETIC TOUCHES:")
                for img in imagery:
                    enhancements.append(f"• {img['memory_text']}")
            
            return "\n".join(enhancements)
