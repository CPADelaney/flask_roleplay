# story_templates/moth/poem_enhanced_generation.py
"""
Text generation that references stored poems for consistent gothic tone
Integrates with the Nyx AI system for enhanced responses
"""

import asyncio
import random
from typing import Dict, Any, Optional, List, Tuple
from db.connection import get_db_connection_context
import json
import logging

logger = logging.getLogger(__name__)

class PoemEnhancedTextGenerator:
    """
    Text generation that references stored poems for consistent gothic tone.
    This integrates with your existing Nyx AI system.
    """
    
    def __init__(self, user_id: int, conversation_id: int, story_id: str = "the_moth_and_flame"):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.story_id = story_id
        self._poem_context = None
        self._key_imagery = None
        self._tone_directive = None
        self._poem_moods = None
        self._initialized = False
    
    async def initialize(self):
        """Load poem context and imagery on initialization"""
        if not self._initialized:
            await self._load_poem_context()
            self._initialized = True
    
    async def _load_poem_context(self):
        """Load poems and imagery into memory for quick access"""
        async with get_db_connection_context() as conn:
            # Get the tone directive
            self._tone_directive = await conn.fetchval(
                """
                SELECT memory_text FROM unified_memories
                WHERE user_id = $1 AND conversation_id = $2
                AND entity_type = 'story_directive'
                AND metadata->>'story_id' = $3
                """,
                self.user_id, self.conversation_id, self.story_id
            )
            
            # Get key imagery phrases
            imagery_rows = await conn.fetch(
                """
                SELECT memory_text, metadata FROM unified_memories
                WHERE user_id = $1 AND conversation_id = $2
                AND entity_type = 'story_source'
                AND memory_type = 'key_imagery'
                AND metadata->>'story_id' = $3
                """,
                self.user_id, self.conversation_id, self.story_id
            )
            
            self._key_imagery = []
            for row in imagery_rows:
                self._key_imagery.append({
                    "text": row['memory_text'],
                    "type": row['metadata'].get('imagery_type', 'general'),
                    "themes": row['metadata'].get('associated_themes', [])
                })
            
            # Get poem excerpts for different moods
            self._poem_moods = {
                'vulnerable': await self._get_poem_lines_for_mood(conn, ['doubt', 'trembling', 'fear', 'whisper']),
                'dominant': await self._get_poem_lines_for_mood(conn, ['queen', 'throne', 'command', 'worship']),
                'romantic': await self._get_poem_lines_for_mood(conn, ['kiss', 'touch', 'close', 'heart']),
                'desperate': await self._get_poem_lines_for_mood(conn, ['disappear', 'stay', 'mine', 'please']),
                'passionate': await self._get_poem_lines_for_mood(conn, ['burn', 'flame', 'moth', 'desire']),
                'contemplative': await self._get_poem_lines_for_mood(conn, ['stars', 'night', 'shadow', 'truth'])
            }
            
            logger.info(f"Loaded poem context: {len(self._key_imagery)} imagery phrases, "
                       f"{sum(len(v) for v in self._poem_moods.values())} mood lines")
    
    async def _get_poem_lines_for_mood(self, conn, keywords: List[str]) -> List[Dict[str, str]]:
        """Extract poem lines that match certain mood keywords"""
        lines = []
        
        # Query for full poems
        poem_texts = await conn.fetch(
            """
            SELECT memory_text, metadata FROM unified_memories
            WHERE user_id = $1 AND conversation_id = $2
            AND entity_type = 'story_source'
            AND memory_type = 'source_poem'
            AND metadata->>'story_id' = $3
            """,
            self.user_id, self.conversation_id, self.story_id
        )
        
        for poem_row in poem_texts:
            poem_id = poem_row['metadata'].get('poem_id', 'unknown')
            poem_lines = poem_row['memory_text'].split('\n')
            
            for i, line in enumerate(poem_lines):
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in keywords):
                    lines.append({
                        "text": line.strip(),
                        "source": poem_id,
                        "line_number": i,
                        "context": self._get_line_context(poem_lines, i)
                    })
        
        # Also query stanzas with mood tags
        mood_stanzas = await conn.fetch(
            """
            SELECT memory_text, metadata FROM unified_memories
            WHERE user_id = $1 AND conversation_id = $2
            AND entity_type = 'story_source' 
            AND memory_type = 'poem_stanza'
            AND metadata->>'story_id' = $3
            """,
            self.user_id, self.conversation_id, self.story_id
        )
        
        for stanza_row in mood_stanzas:
            stanza_text = stanza_row['memory_text']
            if any(keyword in stanza_text.lower() for keyword in keywords):
                lines.append({
                    "text": stanza_text.strip(),
                    "source": stanza_row['metadata'].get('poem_id', 'unknown'),
                    "type": "stanza",
                    "mood": stanza_row['metadata'].get('mood', 'unknown')
                })
        
        return lines[:15]  # Limit to 15 lines per mood
    
    def _get_line_context(self, poem_lines: List[str], line_index: int) -> str:
        """Get surrounding context for a line"""
        start = max(0, line_index - 1)
        end = min(len(poem_lines), line_index + 2)
        return " / ".join(poem_lines[start:end])
    
    async def generate_npc_dialogue(self, npc_name: str, context: Dict[str, Any]) -> str:
        """
        Generate dialogue that incorporates poem imagery and tone.
        This would be called by your existing dialogue system.
        
        Args:
            npc_name: Name of the NPC speaking
            context: Context including mood, trust level, scene info
            
        Returns:
            Enhanced prompt for dialogue generation
        """
        if not self._initialized:
            await self.initialize()
        
        # Build the enhanced prompt
        prompt_parts = []
        
        # Add character context
        prompt_parts.append(f"Generate dialogue for {npc_name} in this context:")
        prompt_parts.append(f"- Current mood: {context.get('npc_mood', 'dominant')}")
        prompt_parts.append(f"- Trust level: {context.get('trust_level', 0)}/100")
        prompt_parts.append(f"- Current mask: {context.get('current_mask', 'Porcelain Goddess')}")
        prompt_parts.append(f"- Scene: {context.get('scene_description', 'Velvet Sanctum')}")
        prompt_parts.append(f"- Player action: {context.get('player_action', 'observing')}")
        
        # Add tone directive if available
        if self._tone_directive:
            prompt_parts.append(f"\nTONE GUIDANCE:\n{self._tone_directive}")
        
        # Add mood-specific poem lines
        current_mood = context.get('npc_mood', 'dominant')
        if current_mood in self._poem_moods and self._poem_moods[current_mood]:
            mood_lines = self._poem_moods[current_mood]
            selected_lines = random.sample(mood_lines, min(3, len(mood_lines)))
            
            prompt_parts.append(f"\nEcho the mood and style of these lines:")
            for line in selected_lines:
                prompt_parts.append(f"- {line['text']}")
        
        # Add relevant imagery
        if self._key_imagery:
            relevant_imagery = self._select_relevant_imagery(context)
            if relevant_imagery:
                prompt_parts.append(f"\nIncorporate this imagery naturally:")
                prompt_parts.append(", ".join([img["text"] for img in relevant_imagery[:5]]))
        
        # Add specific instructions for Lilith
        if npc_name == "Lilith Ravencroft" or context.get('is_queen', False):
            prompt_parts.append(self._get_queen_specific_instructions(context))
        
        # Add dialogue requirements
        prompt_parts.append("\nDIALOGUE REQUIREMENTS:")
        prompt_parts.append("- Stay in character voice")
        prompt_parts.append("- Use poetic language when emotional")
        prompt_parts.append("- Reference masks, moths, flames as metaphors")
        prompt_parts.append("- Show don't tell internal conflict")
        
        full_prompt = "\n".join(prompt_parts)
        return full_prompt
    
    def _select_relevant_imagery(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Select imagery relevant to the current scene"""
        scene_type = context.get('scene_type', 'general')
        current_mood = context.get('npc_mood', 'neutral')
        
        relevant_imagery = []
        
        # Map scene types to imagery themes
        scene_imagery_map = {
            'mask_scene': ['protection', 'identity', 'vulnerability'],
            'vulnerable_moment': ['vulnerability', 'truth', 'fear'],
            'dominant_scene': ['authority', 'control', 'worship'],
            'intimate_scene': ['intimacy', 'connection', 'touch'],
            'abandonment_fear': ['loss', 'abandonment', 'desperation']
        }
        
        desired_themes = scene_imagery_map.get(scene_type, [])
        
        # Also add mood-based themes
        mood_theme_map = {
            'vulnerable': ['vulnerability', 'truth'],
            'dominant': ['authority', 'control'],
            'passionate': ['intimacy', 'desire'],
            'desperate': ['loss', 'fear']
        }
        
        desired_themes.extend(mood_theme_map.get(current_mood, []))
        
        # Select imagery matching themes
        for imagery in self._key_imagery:
            imagery_themes = imagery.get('themes', [])
            if any(theme in imagery_themes for theme in desired_themes):
                relevant_imagery.append(imagery)
        
        # If not enough, add by type
        if len(relevant_imagery) < 3:
            for imagery in self._key_imagery:
                if imagery['type'] == scene_type and imagery not in relevant_imagery:
                    relevant_imagery.append(imagery)
        
        return relevant_imagery
    
    def _get_queen_specific_instructions(self, context: Dict[str, Any]) -> str:
        """Special instructions for the Queen character based on poems"""
        
        trust_level = context.get('trust_level', 0)
        is_private = context.get('is_private', False)
        mask = context.get('current_mask', 'Porcelain Goddess')
        three_words_spoken = context.get('three_words_spoken', False)
        
        instructions = ["\nCHARACTER NOTES FOR THE QUEEN:"]
        
        # Trust-based instructions
        if trust_level < 30:
            instructions.extend([
                "- Maintain the mask, speak in riddles and commands",
                "- Reference: 'But masks, like mirrors, only show what's shown'",
                "- Keep emotional distance, project control",
                "- Subtle threats woven into silk words"
            ])
        elif trust_level < 60:
            instructions.extend([
                "- Show cracks in the facade, brief moments of doubt",
                "- Echo: 'The mask now heavy in her trembling hands'",
                "- Mix commands with questions she doesn't want answered",
                "- Let exhaustion show in pauses between words"
            ])
        elif trust_level < 85:
            instructions.extend([
                "- Vulnerability seeps through, fear of abandonment surfaces",
                "- Use variations of 'Don't disappear' and 'Be mine'",
                "- Poetry emerges when emotions overwhelm",
                "- Touch becomes desperate, possessive"
            ])
        else:
            instructions.extend([
                "- The woman beneath all masks, raw and true",
                "- Three words burn on her tongue",
                "- Every goodbye is a knife wound",
                "- Love and terror are the same emotion"
            ])
        
        # Mask-specific modifications
        mask_instructions = {
            "Porcelain Goddess": [
                "- Perfect control, theatrical dominance",
                "- Others are moths to command"
            ],
            "Leather Predator": [
                "- Dangerous edge, protective fury",
                "- Threats are promises"
            ],
            "Lace Vulnerability": [
                "- Soft commands that sound like pleas",
                "- Power through revealed weakness"
            ],
            "No Mask": [
                "- Just Lilith, broken and beautiful",
                "- Every word costs something"
            ]
        }
        
        instructions.extend(mask_instructions.get(mask, []))
        
        # Special state instructions
        if is_private:
            instructions.extend([
                "- Let the 'rough geography of breaks' show",
                "- Reference being 'a moth with wings of broken glass'"
            ])
        
        if three_words_spoken:
            instructions.extend([
                "- The words have been spoken, everything changes",
                "- New vulnerability, new power dynamic"
            ])
        
        return "\n".join(instructions)
    
    async def enhance_scene_description(
        self, base_description: str, scene_type: str, atmosphere: Dict[str, Any]
    ) -> str:
        """
        Enhance a scene description with poetic imagery.
        
        Args:
            base_description: Original scene description
            scene_type: Type of scene (sanctum, chambers, etc)
            atmosphere: Current atmosphere details
            
        Returns:
            Enhanced description with poetic elements
        """
        if not self._initialized:
            await self.initialize()
        
        enhancements = []
        
        # Add base description
        enhancements.append(base_description)
        
        # Add atmospheric details based on poem imagery
        scene_enhancements = {
            "velvet_sanctum": [
                "Candles gutter in silver stands, painting shadows that dance like suppliants.",
                "The air tastes of velvet night and unspoken prayers.",
                "Here, pilgrims seek their pain as benediction at the altar of her throne."
            ],
            "private_chambers": [
                "Masks rest heavy on mahogany stands, each a broken promise made porcelain.",
                "Moths dance against windows, seeking their beautiful destruction.",
                "Letters to ghosts pile on the desk, words never sent bleeding ink."
            ],
            "empty_sanctum": [
                "The temple empty, music dead, only ghosts of worship linger.",
                "Shadows stretch long where bodies writhed in reverence.",
                "The throne sits cold, awaiting its queen of thorns."
            ],
            "safehouse": [
                "Here, broken wings learn to fly again.",
                "The Moth Queen's other kingdom, where salvation wears no mask.",
                "Safety tastes different when you've known its absence."
            ]
        }
        
        # Select appropriate enhancements
        if scene_type in scene_enhancements:
            selected = random.sample(
                scene_enhancements[scene_type],
                min(2, len(scene_enhancements[scene_type]))
            )
            enhancements.extend(selected)
        
        # Add mood-specific imagery
        mood = atmosphere.get('emotional_tone', 'neutral')
        if mood in ['tense', 'fearful']:
            enhancements.append("Tension coils in the air like smoke from snuffed candles.")
        elif mood in ['intimate', 'vulnerable']:
            enhancements.append("The space breathes with confessions, walls that have heard too much.")
        elif mood in ['dominant', 'powerful']:
            enhancements.append("Power radiates from every surface, demanding genuflection.")
        
        # Add sensory details from imagery
        sensory_options = [
            "The scent of leather and roses mingles with something darker.",
            "Silk whispers against skin like promises about to break.",
            "Shadows pool in corners like spilled ink, hiding secrets."
        ]
        
        if random.random() > 0.5:
            enhancements.append(random.choice(sensory_options))
        
        return "\n\n".join(enhancements)
    
    async def get_poetic_response(
        self, trigger: str, emotion: str, trust_level: int
    ) -> Tuple[str, bool]:
        """
        Get a poetic response for special moments.
        
        Args:
            trigger: What triggered the poetic moment
            emotion: Current emotional state
            trust_level: Current trust level
            
        Returns:
            Tuple of (poetic line, requires_interpretation)
        """
        if not self._initialized:
            await self.initialize()
        
        # Select from appropriate poem lines
        if emotion in self._poem_moods:
            candidates = self._poem_moods[emotion]
        else:
            candidates = []
        
        # Add custom poetic responses based on trigger
        custom_responses = {
            "love_confession": [
                "Three syllables live beneath my tongue, tasting of burning stars.",
                "You speak of love as if it doesn't end in disappearing.",
                "That word... it's a luxury I can't afford."
            ],
            "promise_to_stay": [
                "Promises are just future ghosts. I collect them like masks.",
                "Everyone swears forever. The blue list grows longer.",
                "Stay. Just... stay. Let that be enough."
            ],
            "vulnerability_witnessed": [
                "You've seen beneath the porcelain. There's no going back.",
                "I am a moth with wings of broken glass, and you're too bright.",
                "This is who I am when the music dies."
            ],
            "intimacy": [
                "Your skin tastes of prayers I've forgotten how to speak.",
                "I trace invisible tattoos, marking you as mine.",
                "We are binary stars, destined for beautiful destruction."
            ]
        }
        
        # Combine candidates
        if trigger in custom_responses:
            candidates.extend([{"text": line} for line in custom_responses[trigger]])
        
        if not candidates:
            # Fallback to general poetic lines
            return "Some truths are better felt than spoken.", False
        
        # Select based on trust level
        if trust_level > 70:
            # More vulnerable, personal lines
            vulnerable_candidates = [c for c in candidates if 'moth' in c.get('text', '').lower() or 'break' in c.get('text', '').lower()]
            if vulnerable_candidates:
                selected = random.choice(vulnerable_candidates)
            else:
                selected = random.choice(candidates)
        else:
            # More cryptic, defensive lines
            selected = random.choice(candidates)
        
        text = selected.get('text', '') if isinstance(selected, dict) else selected
        
        # Determine if interpretation is required
        requires_interpretation = any(word in text.lower() for word in ['moth', 'flame', 'stars', 'binary'])
        
        return text, requires_interpretation
    
    async def create_story_moment(
        self, moment_type: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a complete story moment with enhanced text.
        
        Args:
            moment_type: Type of story moment
            context: Current context
            
        Returns:
            Complete moment with all text elements
        """
        if not self._initialized:
            await self.initialize()
        
        # Generate base elements
        scene_desc = await self.enhance_scene_description(
            context.get('base_description', ''),
            context.get('location_type', 'general'),
            context.get('atmosphere', {})
        )
        
        # Generate dialogue prompt
        dialogue_prompt = await self.generate_npc_dialogue(
            context.get('npc_name', 'Lilith Ravencroft'),
            context
        )
        
        # Check for special moments
        result = {
            "scene_description": scene_desc,
            "dialogue_prompt": dialogue_prompt,
            "moment_type": moment_type
        }
        
        # Add special elements based on moment type
        if moment_type == "mask_slippage":
            result["special_description"] = await self._get_mask_slippage_description(context)
        elif moment_type == "poetry_moment":
            line, interp = await self.get_poetic_response(
                context.get('trigger', 'general'),
                context.get('npc_mood', 'contemplative'),
                context.get('trust_level', 0)
            )
            result["poetry_line"] = line
            result["requires_interpretation"] = interp
        elif moment_type == "three_words":
            result["buildup"] = self._get_three_words_buildup(context)
        
        return result
    
    async def _get_mask_slippage_description(self, context: Dict[str, Any]) -> str:
        """Get description for mask slippage moment"""
        trust = context.get('trust_level', 0)
        trigger = context.get('slippage_trigger', 'emotion')
        
        descriptions = {
            "low_trust": [
                "For just a moment, the porcelain cracks. Something raw flickers beneath before she catches herself.",
                "The mask shifts, revealing a glimpse of exhaustion before snapping back into place."
            ],
            "medium_trust": [
                "The goddess facade wavers. Beneath, you see a woman holding herself together by will alone.",
                "Her carefully constructed armor shows its seams. Fear bleeds through the cracks."
            ],
            "high_trust": [
                "The mask falls away entirely. Before you stands not a queen but a wounded girl in a woman's body.",
                "All pretense crumbles. She is moth and flame both, burning and drawn to her own destruction."
            ]
        }
        
        if trust < 40:
            options = descriptions["low_trust"]
        elif trust < 70:
            options = descriptions["medium_trust"]
        else:
            options = descriptions["high_trust"]
        
        base = random.choice(options)
        
        # Add trigger-specific details
        if trigger == "abandonment":
            base += " 'Don't,' she whispers, the word escaping before she can stop it."
        elif trigger == "vulnerability":
            base += " Her hands tremble as she reaches for a mask that isn't there."
        
        return base
    
    def _get_three_words_buildup(self, context: Dict[str, Any]) -> str:
        """Get buildup description for three words moment"""
        trust = context.get('trust_level', 0)
        
        buildups = [
            "Something shifts in the air between you. Words that have lived beneath her tongue for so long rise like moths toward flame.",
            "Her lips part, trembling. Three syllables hover in the space between breath and speech.",
            "The moment crystallizes. Everything she's never said crowds behind her teeth, demanding release.",
            "Time slows. You can see the war in her eyes - the need to speak battling the fear of what comes after."
        ]
        
        buildup = random.choice(buildups)
        
        if trust > 90:
            buildup += " This time, she might not swallow them back down."
        else:
            buildup += " But old habits die hard, and some words burn too bright to speak."
        
        return buildup


# Integration functions for your existing system

async def integrate_poem_enhancement(
    user_id: int,
    conversation_id: int,
    npc_data: Dict[str, Any],
    player_input: str,
    scene_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main integration function for poem-enhanced text generation.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        npc_data: Current NPC data
        player_input: Player's input/action
        scene_context: Current scene context
        
    Returns:
        Enhanced prompts and text elements
    """
    # Initialize the generator
    generator = PoemEnhancedTextGenerator(user_id, conversation_id)
    await generator.initialize()
    
    # Determine current context
    context = {
        'npc_name': npc_data.get('npc_name', 'Unknown'),
        'npc_mood': determine_npc_mood(npc_data, player_input, scene_context),
        'trust_level': npc_data.get('trust', 0),
        'current_mask': npc_data.get('current_mask', 'Unknown'),
        'is_private': scene_context.get('is_private', False),
        'scene_description': scene_context.get('description', ''),
        'player_action': player_input,
        'scene_type': determine_scene_type(scene_context),
        'is_queen': npc_data.get('npc_name') == 'Lilith Ravencroft',
        'emotional_intensity': calculate_emotional_intensity(npc_data, player_input),
        'three_words_spoken': npc_data.get('three_words_spoken', False)
    }
    
    # Determine moment type
    moment_type = determine_moment_type(context, player_input)
    
    # Generate enhanced content
    enhanced_content = await generator.create_story_moment(moment_type, context)
    
    return enhanced_content


def determine_npc_mood(npc_data: Dict[str, Any], player_input: str, scene_context: Dict[str, Any]) -> str:
    """Determine NPC mood based on context"""
    player_input_lower = player_input.lower()
    trust = npc_data.get('trust', 0)
    
    # Check for triggers
    if any(word in player_input_lower for word in ["leave", "go", "goodbye", "farewell"]):
        return "desperate"
    elif any(word in player_input_lower for word in ["love", "adore", "devotion", "yours"]):
        return "vulnerable" if trust > 60 else "dominant"
    elif scene_context.get('location', '').lower() == 'velvet sanctum':
        return "dominant"
    elif scene_context.get('is_private') and trust > 50:
        return "vulnerable"
    elif any(word in player_input_lower for word in ["kneel", "submit", "obey"]):
        return "dominant"
    elif any(word in player_input_lower for word in ["touch", "hold", "kiss"]):
        return "passionate" if trust > 40 else "dominant"
    else:
        return "contemplative"


def determine_scene_type(scene_context: Dict[str, Any]) -> str:
    """Determine scene type for imagery selection"""
    location = scene_context.get('location', '').lower()
    recent_action = scene_context.get('recent_action', '').lower()
    
    if "mask" in recent_action or "mask" in location:
        return "mask_scene"
    elif scene_context.get('vulnerability_shown'):
        return "vulnerable_moment"
    elif "sanctum" in location and scene_context.get('is_performing'):
        return "dominant_scene"
    elif scene_context.get('intimacy_level', 0) > 50:
        return "intimate_scene"
    elif any(word in recent_action for word in ["leave", "abandon", "disappear"]):
        return "abandonment_fear"
    else:
        return "general"


def calculate_emotional_intensity(npc_data: Dict[str, Any], player_input: str) -> float:
    """Calculate emotional intensity of current moment (0-1)"""
    intensity = 0.3  # Base intensity
    
    # Trust impacts intensity
    trust = npc_data.get('trust', 0)
    if trust > 80:
        intensity += 0.3
    elif trust > 50:
        intensity += 0.2
    
    # Trigger words increase intensity
    high_intensity_triggers = [
        "love", "forever", "promise", "stay", "leave",
        "disappear", "yours", "mine", "always", "never"
    ]
    
    player_input_lower = player_input.lower()
    for trigger in high_intensity_triggers:
        if trigger in player_input_lower:
            intensity += 0.2
            break
    
    # Mask state affects intensity
    if npc_data.get('current_mask') == 'No Mask':
        intensity += 0.2
    
    # Three words proximity
    if npc_data.get('three_words_near'):
        intensity += 0.3
    
    return min(1.0, intensity)


def determine_moment_type(context: Dict[str, Any], player_input: str) -> str:
    """Determine what type of story moment this is"""
    trust = context.get('trust_level', 0)
    
    # Check for special moments
    if context.get('emotional_intensity', 0) > 0.7 and trust > 40:
        if random.random() < 0.5:
            return "poetry_moment"
    
    if trust > 85 and "love" in player_input.lower():
        return "three_words"
    
    if context.get('is_private') and random.random() < 0.3:
        return "mask_slippage"
    
    return "standard"
