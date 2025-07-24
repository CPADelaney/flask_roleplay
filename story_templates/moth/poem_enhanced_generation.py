# story_templates/moth/poem_enhanced_generation.py

import asyncio
from typing import Dict, Any, Optional
from db.connection import get_db_connection_context

class PoemEnhancedTextGenerator:
    """
    Text generation that references stored poems for consistent gothic tone.
    This would integrate with your existing Nyx AI system.
    """
    
    def __init__(self, user_id: int, conversation_id: int, story_id: str = "the_moth_and_flame"):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.story_id = story_id
        self._poem_context = None
        self._key_imagery = None
    
    async def initialize(self):
        """Load poem context and imagery on initialization"""
        await self._load_poem_context()
    
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
            
            self._key_imagery = [row['memory_text'] for row in imagery_rows]
            
            # Get poem excerpts for different moods
            self._poem_moods = {
                'vulnerable': await self._get_poem_lines_for_mood(conn, ['doubt', 'trembling', 'fear', 'whisper']),
                'dominant': await self._get_poem_lines_for_mood(conn, ['queen', 'throne', 'command', 'worship']),
                'romantic': await self._get_poem_lines_for_mood(conn, ['kiss', 'touch', 'close', 'heart']),
                'desperate': await self._get_poem_lines_for_mood(conn, ['disappear', 'stay', 'mine', 'please'])
            }
    
    async def _get_poem_lines_for_mood(self, conn, keywords: list) -> list:
        """Extract poem lines that match certain mood keywords"""
        lines = []
        
        poem_texts = await conn.fetch(
            """
            SELECT memory_text FROM unified_memories
            WHERE user_id = $1 AND conversation_id = $2
            AND entity_type = 'story_source'
            AND memory_type = 'source_poem'
            AND metadata->>'story_id' = $3
            """,
            self.user_id, self.conversation_id, self.story_id
        )
        
        for poem_row in poem_texts:
            poem_lines = poem_row['memory_text'].split('\n')
            for line in poem_lines:
                if any(keyword in line.lower() for keyword in keywords):
                    lines.append(line.strip())
        
        return lines[:10]  # Limit to 10 lines per mood
    
    async def generate_npc_dialogue(self, npc_name: str, context: Dict[str, Any]) -> str:
        """
        Generate dialogue that incorporates poem imagery and tone.
        This would be called by your existing dialogue system.
        """
        
        # Build the enhanced prompt
        prompt_parts = []
        
        # Add base character instruction
        prompt_parts.append(f"Generate dialogue for {npc_name} in this context: {context}")
        
        # Add tone directive if available
        if self._tone_directive:
            prompt_parts.append(f"\nTONE GUIDANCE:\n{self._tone_directive}")
        
        # Add mood-specific poem lines
        current_mood = context.get('npc_mood', 'dominant')
        if current_mood in self._poem_moods:
            mood_lines = self._poem_moods[current_mood]
            if mood_lines:
                prompt_parts.append(f"\nEcho the mood and style of these lines:\n" + 
                                  "\n".join(f"- {line}" for line in mood_lines[:3]))
        
        # Add relevant imagery
        if self._key_imagery:
            relevant_imagery = self._select_relevant_imagery(context)
            if relevant_imagery:
                prompt_parts.append(f"\nIncorporate this imagery naturally:\n" +
                                  ", ".join(relevant_imagery))
        
        # Add specific instructions for the queen character
        if npc_name == "Lilith Ravencroft" or context.get('is_queen', False):
            prompt_parts.append(self._get_queen_specific_instructions(context))
        
        full_prompt = "\n".join(prompt_parts)
        
        # Here you would call your actual AI generation
        # For now, returning the prompt structure
        return full_prompt
    
    def _select_relevant_imagery(self, context: Dict[str, Any]) -> list:
        """Select imagery relevant to the current scene"""
        scene_type = context.get('scene_type', 'general')
        
        imagery_map = {
            'mask_scene': ['porcelain', 'fortress', 'glass', 'mirror'],
            'vulnerable_moment': ['trembling', 'breaks', 'moth', 'wings'],
            'dominant_scene': ['throne', 'altar', 'worship', 'queen'],
            'intimate_scene': ['tattoos', 'touch', 'brand', 'skin'],
            'abandonment_fear': ['disappear', 'ghost', 'shadows', 'stay']
        }
        
        relevant_keywords = imagery_map.get(scene_type, [])
        return [img for img in self._key_imagery 
                if any(keyword in img.lower() for keyword in relevant_keywords)][:5]
    
    def _get_queen_specific_instructions(self, context: Dict[str, Any]) -> str:
        """Special instructions for the Queen character based on poems"""
        
        trust_level = context.get('trust_level', 0)
        is_private = context.get('is_private', False)
        
        instructions = ["\nCHARACTER NOTES FOR THE QUEEN:"]
        
        if trust_level < 30:
            instructions.append("- Maintain the mask, speak in riddles and commands")
            instructions.append("- Reference: 'But masks, like mirrors, only show what's shown'")
        elif trust_level < 60:
            instructions.append("- Show cracks in the facade, brief moments of doubt")
            instructions.append("- Echo: 'The mask now heavy in her trembling hands'")
        else:
            instructions.append("- Vulnerability seeps through, fear of abandonment surfaces")
            instructions.append("- Use variations of 'Don't disappear' and 'Be mine'")
        
        if is_private:
            instructions.append("- Let the 'rough geography of breaks' show")
            instructions.append("- Reference being 'a moth with wings of broken glass'")
        
        return "\n".join(instructions)
    
    async def enhance_scene_description(self, base_description: str, scene_type: str) -> str:
        """
        Enhance a scene description with poetic imagery.
        This would be called when generating location descriptions.
        """
        
        enhancements = []
        
        # Add atmospheric details based on poem imagery
        if scene_type == "velvet_sanctum":
            enhancements.extend([
                "Candles gutter in silver stands, casting dancing shadows.",
                "The air tastes of velvet night and unspoken prayers.",
                "Pilgrims seek their pain as benediction at this altar."
            ])
        elif scene_type == "private_chambers":
            enhancements.extend([
                "Here, masks rest heavy on mahogany stands.",
                "The space breathes with whispered confessions.",
                "Moths dance against windows, seeking their destruction."
            ])
        elif scene_type == "empty_sanctum":
            enhancements.extend([
                "The temple empty, music dead, ghosts of worship linger.",
                "Shadows stretch long where bodies writhed in reverence.",
                "The throne sits cold, awaiting its queen of thorns."
            ])
        
        # Select appropriate imagery
        relevant_imagery = [img for img in self._key_imagery 
                          if any(word in scene_type.lower() for word in img.lower().split())]
        
        if relevant_imagery:
            enhancements.append(f"You notice {relevant_imagery[0].lower()}.")
        
        enhanced = base_description
        if enhancements:
            enhanced += "\n\n" + " ".join(enhancements)
        
        return enhanced


# Integration example with your existing system
async def integrate_with_nyx_response(user_id: int, conversation_id: int, npc_data: Dict, player_input: str):
    """
    Example of how to integrate poem enhancement with existing Nyx response generation
    """
    
    # Initialize the poem-enhanced generator
    generator = PoemEnhancedTextGenerator(user_id, conversation_id)
    await generator.initialize()
    
    # Prepare context
    context = {
        'npc_mood': determine_npc_mood(npc_data, player_input),
        'trust_level': npc_data.get('trust', 0),
        'is_private': npc_data.get('current_location') == 'Private Chambers',
        'scene_type': determine_scene_type(npc_data, player_input),
        'is_queen': npc_data.get('name') == 'Lilith Ravencroft'
    }
    
    # Generate enhanced prompt
    dialogue_prompt = await generator.generate_npc_dialogue(
        npc_data.get('name', 'Unknown'),
        context
    )
    
    # Add to your existing Nyx prompt
    full_nyx_prompt = f"""
{dialogue_prompt}

Previous context: {npc_data.get('recent_context', '')}
Player said: {player_input}

Generate a response that:
1. Maintains the gothic poetic tone established in the source material
2. Uses imagery and metaphors consistent with the poems
3. Reflects the character's current emotional state
4. Advances the narrative toward the story beats

Response:
"""
    
    return full_nyx_prompt

def determine_npc_mood(npc_data: Dict, player_input: str) -> str:
    """Determine NPC mood based on context"""
    # Simplified example - you'd have more complex logic
    if "leave" in player_input.lower() or "go" in player_input.lower():
        return "desperate"
    elif npc_data.get('trust', 0) > 60 and "love" in player_input.lower():
        return "vulnerable"
    elif npc_data.get('scene_location') == 'velvet_sanctum':
        return "dominant"
    else:
        return "romantic"

def determine_scene_type(npc_data: Dict, player_input: str) -> str:
    """Determine scene type for imagery selection"""
    location = npc_data.get('current_location', '').lower()
    
    if "mask" in player_input.lower() or "mask" in location:
        return "mask_scene"
    elif npc_data.get('trust', 0) > 70:
        return "vulnerable_moment"
    elif "sanctum" in location and npc_data.get('is_performing'):
        return "dominant_scene"
    elif npc_data.get('intimacy_level', 0) > 50:
        return "intimate_scene"
    else:
        return "general"
