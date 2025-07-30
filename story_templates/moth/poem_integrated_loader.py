# story_templates/moth/poem_integrated_loader.py
"""
Enhanced poem integration system that loads poems as AI context
Ensures consistent tone for Queen of Thorns story
"""

import json
import logging
from typing import Dict, Any, List, Optional
from db.connection import get_db_connection_context
from story_templates.preset_stories import PresetStory

logger = logging.getLogger(__name__)

class ThornsIntegratedStoryLoader:
    """Enhanced story loader that integrates poems and network lore into AI context"""
    
    @staticmethod
    async def load_story_with_themes(story: PresetStory, user_id: int, conversation_id: int):
        """Load a story and seed its themes, poems, and network rules as special memories"""
        
        try:
            async with get_db_connection_context() as conn:
                # First, load the basic story structure
                await ThornsIntegratedStoryLoader._load_base_story(story, conn)
                
                # Load network consistency rules
                await ThornsIntegratedStoryLoader._load_network_rules(
                    story, user_id, conversation_id, conn
                )
                
                # If story has poems, load them as special memories
                if hasattr(story, 'source_material') and 'poems' in story.source_material:
                    await ThornsIntegratedStoryLoader._load_poems_as_memories(
                        story, user_id, conversation_id, conn
                    )
                    
                # Load tone instructions if present
                if hasattr(story, 'source_material') and 'tone_prompt' in story.source_material:
                    await ThornsIntegratedStoryLoader._load_tone_instructions(
                        story, user_id, conversation_id, conn
                    )
                    
                # Extract and store key imagery for quick reference
                if hasattr(story, 'source_material') and 'poems' in story.source_material:
                    await ThornsIntegratedStoryLoader._extract_and_store_imagery(
                        story, user_id, conversation_id, conn
                    )
                    
                logger.info(f"Successfully loaded story {story.id} with Queen of Thorns integration")
                
        except Exception as e:
            logger.error(f"Failed to load story with themes: {e}", exc_info=True)
            raise
    
    @staticmethod
    async def _load_network_rules(story: PresetStory, user_id: int, conversation_id: int, conn):
        """Load network-specific consistency rules"""
        
        # Critical rules from consistency guide
        network_rules = {
            "no_official_name": "The network has NO official name. Always refer to it as 'the network' or 'the garden'.",
            "queen_ambiguity": "The Queen's identity is ALWAYS ambiguous. Never confirm if she's one person or many.",
            "geographic_scope": "The network controls the Bay Area ONLY. Other cities have allies, not branches.",
            "transformation_time": "Transformation takes months or years, never instant.",
            "information_layers": "All information exists in four layers: PUBLIC, SEMI_PRIVATE, HIDDEN, DEEP_SECRET"
        }
        
        for rule_id, rule_text in network_rules.items():
            await conn.execute(
                """
                INSERT INTO unified_memories
                (entity_type, entity_id, user_id, conversation_id,
                 memory_text, memory_type, tags, significance, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                ON CONFLICT DO NOTHING
                """,
                "story_directive",
                0,
                user_id,
                conversation_id,
                rule_text,
                "consistency_rule",
                ["network", "rules", story.id, "critical"],
                10,  # Maximum significance
                json.dumps({
                    "rule_id": rule_id,
                    "story_id": story.id,
                    "rule_type": "critical",
                    "enforcement": "always"
                })
            )
        
        logger.info(f"Loaded {len(network_rules)} network consistency rules")
    
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
                json.dumps(ThornsIntegratedStoryLoader._serialize_story(story))
            )
        else:
            await conn.execute(
                """
                INSERT INTO PresetStories (story_id, story_data, created_at)
                VALUES ($1, $2, NOW())
                """,
                story.id,
                json.dumps(ThornsIntegratedStoryLoader._serialize_story(story))
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
                "story_source",
                0,
                user_id,
                conversation_id,
                poem_text,
                "source_poem",
                ["poetry", "mood", "tone", story.id, poem_id, "gothic", "power"],
                10,
                json.dumps({
                    "poem_id": poem_id,
                    "poem_title": poem_title,
                    "story_id": story.id,
                    "usage": "tone_reference",
                    "themes": ThornsIntegratedStoryLoader._extract_themes(poem_text)
                })
            )
            
            # Store individual stanzas for targeted retrieval
            for i, stanza in enumerate(stanzas[1:], 1):
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
                            "mood": ThornsIntegratedStoryLoader._analyze_stanza_mood(stanza)
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
                ["directive", "tone", "style", story.id, "gothic", "power"],
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
        
        # Also add network-specific imagery
        network_imagery = [
            "the network", "the garden", "roses and thorns",
            "pruning the unworthy", "seeds of power",
            "cultivation of control", "harvest of submission"
        ]
        
        for phrase in network_imagery:
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
                ["imagery", "network", story.id],
                9,
                json.dumps({
                    "source": "network_lore",
                    "story_id": story.id,
                    "imagery_type": "network",
                    "associated_themes": ["power", "control", "transformation"]
                })
            )
        
        for poem_id, poem_text in poems.items():
            imagery = ThornsIntegratedStoryLoader._extract_key_imagery(poem_text)
            
            for phrase in imagery:
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
                        "imagery_type": ThornsIntegratedStoryLoader._classify_imagery(phrase),
                        "associated_themes": ThornsIntegratedStoryLoader._get_imagery_themes(phrase)
                    })
                )
                all_imagery.append(phrase)
        
        logger.info(f"Extracted and stored {len(all_imagery) + len(network_imagery)} imagery phrases")
    
    @staticmethod
    def _extract_key_imagery(poem_text: str) -> List[str]:
        """Extract key imagery phrases from poems"""
        key_phrases = []
        
        # Network and power-focused phrases
        powerful_phrases = [
            "queen of thorns", "the network", "the garden",
            "roses", "thorns", "power", "control",
            "transformation", "cultivation", "pruning",
            "shadow matriarchy", "invisible authority"
        ]
        
        # Look for these in the poem
        poem_lower = poem_text.lower()
        for phrase in powerful_phrases:
            if phrase in poem_lower:
                # Extract the line containing this phrase
                lines = poem_text.split('\n')
                for line in lines:
                    if phrase in line.lower() and len(line) < 100:
                        key_phrases.append(line.strip())
                        break
        
        # Look for metaphorical patterns
        lines = poem_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Power dynamics language
            if any(word in line.lower() for word in ["power", "control", "authority", "submission"]):
                if len(line) < 100:
                    key_phrases.append(line)
            
            # Network metaphors
            if any(word in line.lower() for word in ["network", "garden", "rose", "thorn"]):
                if len(line) < 100:
                    key_phrases.append(line)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_phrases = []
        for phrase in key_phrases:
            if phrase.lower() not in seen:
                seen.add(phrase.lower())
                unique_phrases.append(phrase)
        
        return unique_phrases[:30]
    
    @staticmethod
    def _extract_themes(poem_text: str) -> List[str]:
        """Extract thematic elements from a poem"""
        themes = []
        poem_lower = poem_text.lower()
        
        theme_keywords = {
            "power": ["power", "control", "authority", "dominance"],
            "transformation": ["transform", "change", "remake", "cultivate"],
            "network": ["network", "garden", "roses", "thorns"],
            "protection": ["protect", "save", "guard", "shelter"],
            "duality": ["hidden", "shadow", "beneath", "mask"],
            "hierarchy": ["queen", "throne", "rule", "command"]
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in poem_lower for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    @staticmethod
    def _analyze_stanza_mood(stanza: str) -> str:
        """Analyze the emotional mood of a stanza"""
        stanza_lower = stanza.lower()
        
        if any(word in stanza_lower for word in ["command", "control", "power", "throne"]):
            return "dominant"
        elif any(word in stanza_lower for word in ["protect", "save", "guard", "shelter"]):
            return "protective"
        elif any(word in stanza_lower for word in ["transform", "change", "cultivate"]):
            return "transformative"
        elif any(word in stanza_lower for word in ["network", "garden", "thorns"]):
            return "strategic"
        else:
            return "contemplative"
    
    @staticmethod
    def _classify_imagery(phrase: str) -> str:
        """Classify the type of imagery"""
        phrase_lower = phrase.lower()
        
        if any(word in phrase_lower for word in ["network", "garden", "cultivation"]):
            return "network"
        elif any(word in phrase_lower for word in ["queen", "throne", "authority"]):
            return "power"
        elif any(word in phrase_lower for word in ["transform", "change", "remake"]):
            return "transformation"
        elif any(word in phrase_lower for word in ["rose", "thorn", "flower"]):
            return "botanical"
        elif any(word in phrase_lower for word in ["protect", "save", "shelter"]):
            return "protection"
        else:
            return "atmospheric"
    
    @staticmethod
    def _get_imagery_themes(phrase: str) -> List[str]:
        """Get themes associated with specific imagery"""
        themes = []
        phrase_lower = phrase.lower()
        
        theme_map = {
            "network": ["connection", "power", "organization"],
            "garden": ["cultivation", "growth", "control"],
            "rose": ["beauty", "danger", "duality"],
            "thorn": ["protection", "pain", "defense"],
            "queen": ["authority", "leadership", "mystery"],
            "transform": ["change", "power", "control"]
        }
        
        for keyword, associated_themes in theme_map.items():
            if keyword in phrase_lower:
                themes.extend(associated_themes)
        
        return list(set(themes))
    
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


class NetworkAwarePromptGenerator:
    """Generates prompts that enforce network consistency"""
    
    @staticmethod
    async def get_story_context_prompt(user_id: int, conversation_id: int, story_id: str) -> str:
        """Generate a context prompt that includes network rules"""
        
        async with get_db_connection_context() as conn:
            # Fetch consistency rules
            rules = await conn.fetch(
                """
                SELECT memory_text, metadata
                FROM unified_memories
                WHERE user_id = $1
                AND conversation_id = $2
                AND entity_type = 'story_directive'
                AND memory_type = 'consistency_rule'
                AND metadata->>'story_id' = $3
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
            
            # Add critical rules first
            if rules:
                context_parts.append("=== CRITICAL CONSISTENCY RULES ===")
                for rule in rules:
                    context_parts.append(f"• {rule['memory_text']}")
                context_parts.append("")
            
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
                    for phrase in phrases[:3]:
                        context_parts.append(f"  • {phrase}")
            
            # Add quick reference
            context_parts.append("\n=== QUICK REFERENCE ===")
            context_parts.append("✗ Official names → ✓ 'the network'")
            context_parts.append("✗ 'Queen [Name]' → ✓ 'The Queen, whoever she is'")
            context_parts.append("✗ 'Our Seattle chapter' → ✓ 'Allied network in Seattle'")
            context_parts.append("✗ 'Instant transformation' → ✓ 'Months of careful work'")
            
            return "\n".join(context_parts)
    
    @staticmethod
    async def get_npc_dialogue_context(
        user_id: int, conversation_id: int, npc_id: int, 
        emotional_state: str = "neutral"
    ) -> str:
        """Get dialogue context for a specific NPC based on network themes"""
        
        async with get_db_connection_context() as conn:
            # Get NPC data
            npc_row = await conn.fetchrow(
                """
                SELECT npc_name, trust, dialogue_patterns, affiliations
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                """,
                user_id, conversation_id, npc_id
            )
            
            if not npc_row:
                return ""
            
            # Check if NPC is network-affiliated
            affiliations = json.loads(npc_row.get('affiliations', '[]'))
            is_network = any('network' in aff.lower() or 'garden' in aff.lower() 
                           or 'thorn' in aff.lower() for aff in affiliations)
            
            # Build dialogue context
            context_parts = [
                f"=== DIALOGUE CONTEXT FOR {npc_row['npc_name'].upper()} ===",
                f"Trust Level: {npc_row.get('trust', 0)}/100",
                f"Emotional State: {emotional_state}",
            ]
            
            if is_network:
                context_parts.extend([
                    "",
                    "NETWORK MEMBER - Use coded language:",
                    "• Refer to 'the network' or 'the garden', never official names",
                    "• Use botanical metaphors: pruning, cultivation, growth",
                    "• Speak in layers - surface meaning and deeper meaning",
                    "• Never reveal network structure or the Queen's identity"
                ])
            
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
        """Enhance scene descriptions with network-appropriate imagery"""
        
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
                AND (metadata->>'imagery_type' = 'atmospheric' 
                     OR metadata->>'imagery_type' = 'network')
                LIMIT 5
                """,
                user_id, conversation_id
            )
            
            enhancements = ["ATMOSPHERIC ENHANCEMENT:"]
            
            # Add location-specific imagery
            location_lower = location.lower()
            if "rose garden" in location_lower:
                enhancements.extend([
                    "The scent of roses mingles with lavender and intent.",
                    "Every table holds potential recruitment or pruning.",
                    "Conversations layer meaning like petals."
                ])
            elif "safehouse" in location_lower:
                enhancements.extend([
                    "Safety has a different texture when you've known its absence.",
                    "The network's other face - protector, not predator.",
                    "Here, thorns point outward."
                ])
            elif "network" in location_lower or "garden" in location_lower:
                enhancements.extend([
                    "Power flows through invisible channels.",
                    "Every gesture carries coded meaning.",
                    "The air tastes of secrets and strength."
                ])
            
            # Add imagery from poems
            if imagery:
                enhancements.append("\nPOETIC TOUCHES:")
                for img in imagery:
                    enhancements.append(f"• {img['memory_text']}")
            
            return "\n".join(enhancements)
