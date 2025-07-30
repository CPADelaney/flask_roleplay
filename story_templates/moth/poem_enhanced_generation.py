# story_templates/moth/poem_enhanced_generation.py
"""
Text generation that references stored poems for consistent gothic tone
Integrates with the Queen of Thorns story and network themes
"""

import asyncio
import random
from typing import Dict, Any, Optional, List, Tuple
from db.connection import get_db_connection_context
import json
import logging

logger = logging.getLogger(__name__)

class ThornsEnhancedTextGenerator:
    """
    Text generation that references stored poems and network lore for consistent tone.
    Integrates with the Queen of Thorns setting and hidden power themes.
    """
    
    def __init__(self, user_id: int, conversation_id: int, story_id: str = "queen_of_thorns"):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.story_id = story_id
        self._poem_context = None
        self._key_imagery = None
        self._tone_directive = None
        self._poem_moods = None
        self._network_phrases = None
        self._initialized = False
    
    async def initialize(self):
        """Load poem context, imagery, and network terminology"""
        if not self._initialized:
            await self._load_poem_context()
            await self._load_network_phrases()
            self._initialized = True
    
    async def _load_network_phrases(self):
        """Load network-specific terminology and phrases"""
        self._network_phrases = {
            'network_references': ["the network", "the garden", "our people"],
            'queen_references': ["The Queen of Thorns", "The Queen", "whoever she is"],
            'power_phrases': [
                "interesting energy", "needs pruning", "growth-oriented",
                "very responsive", "she has presence"
            ],
            'location_codes': {
                'rose_garden_cafe': "Where seeds are planted",
                'velvet_sanctum': "Where thorns draw blood", 
                'private_chambers': "Where masks fall away",
                'safehouse': "Where broken wings heal"
            }
        }
    
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
                'dominant': await self._get_poem_lines_for_mood(conn, ['queen', 'throne', 'command', 'worship', 'thorns']),
                'protective': await self._get_poem_lines_for_mood(conn, ['save', 'protect', 'guard', 'network']),
                'desperate': await self._get_poem_lines_for_mood(conn, ['disappear', 'stay', 'mine', 'please']),
                'strategic': await self._get_poem_lines_for_mood(conn, ['power', 'control', 'influence', 'shadow']),
                'contemplative': await self._get_poem_lines_for_mood(conn, ['truth', 'mask', 'hidden', 'secret'])
            }
            
            logger.info(f"Loaded Queen of Thorns context: {len(self._key_imagery)} imagery phrases, "
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
        
        return lines[:15]  # Limit to 15 lines per mood
    
    def _get_line_context(self, poem_lines: List[str], line_index: int) -> str:
        """Get surrounding context for a line"""
        start = max(0, line_index - 1)
        end = min(len(poem_lines), line_index + 2)
        return " / ".join(poem_lines[start:end])
    
    async def generate_npc_dialogue(self, npc_name: str, context: Dict[str, Any]) -> str:
        """
        Generate dialogue that incorporates network terminology and Queen's tone.
        
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
        prompt_parts.append(f"- Current location: {context.get('scene_description', 'The network')}")
        prompt_parts.append(f"- Player action: {context.get('player_action', 'observing')}")
        
        # Add network-specific language rules
        prompt_parts.append("\nNETWORK LANGUAGE RULES:")
        prompt_parts.append("- NEVER use official organization names")
        prompt_parts.append("- Refer to 'the network' or 'the garden' internally")
        prompt_parts.append("- The Queen is 'The Queen of Thorns' or just 'The Queen'")
        prompt_parts.append("- Use coded language: 'interesting energy', 'needs pruning', etc.")
        
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
        
        # Add specific instructions for the Queen
        if npc_name == "The Queen of Thorns" or context.get('is_queen', False):
            prompt_parts.append(self._get_queen_specific_instructions(context))
        
        # Add dialogue requirements
        prompt_parts.append("\nDIALOGUE REQUIREMENTS:")
        prompt_parts.append("- Stay in character voice")
        prompt_parts.append("- Use power dynamics language")
        prompt_parts.append("- Reference thorns, roses, gardens as metaphors")
        prompt_parts.append("- Show don't tell internal conflict")
        prompt_parts.append("- Maintain mystery about the Queen's identity")
        
        full_prompt = "\n".join(prompt_parts)
        return full_prompt
    
    def _select_relevant_imagery(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Select imagery relevant to the current scene"""
        scene_type = context.get('scene_type', 'general')
        current_mood = context.get('npc_mood', 'neutral')
        
        relevant_imagery = []
        
        # Map scene types to imagery themes
        scene_imagery_map = {
            'transformation_scene': ['transformation', 'power', 'thorns'],
            'network_business': ['roses', 'gardens', 'pruning'],
            'vulnerable_moment': ['vulnerability', 'truth', 'masks'],
            'power_display': ['authority', 'control', 'throne'],
            'protective_action': ['thorns', 'protection', 'network']
        }
        
        desired_themes = scene_imagery_map.get(scene_type, [])
        
        # Also add mood-based themes
        mood_theme_map = {
            'vulnerable': ['vulnerability', 'truth'],
            'dominant': ['authority', 'thorns'],
            'protective': ['network', 'guardian'],
            'strategic': ['power', 'shadow']
        }
        
        desired_themes.extend(mood_theme_map.get(current_mood, []))
        
        # Select imagery matching themes
        for imagery in self._key_imagery:
            imagery_themes = imagery.get('themes', [])
            if any(theme in imagery_themes for theme in desired_themes):
                relevant_imagery.append(imagery)
        
        return relevant_imagery
    
    def _get_queen_specific_instructions(self, context: Dict[str, Any]) -> str:
        """Special instructions for the Queen of Thorns character"""
        
        trust_level = context.get('trust_level', 0)
        is_network_business = context.get('is_network_business', False)
        location = context.get('current_location', '')
        
        instructions = ["\nCHARACTER NOTES FOR THE QUEEN OF THORNS:"]
        
        # Trust-based instructions
        if trust_level < 30:
            instructions.extend([
                "- Maintain complete authority and mystery",
                "- Speak in commands and observations about power",
                "- Reference the network obliquely, never directly",
                "- Show no vulnerability, only strength"
            ])
        elif trust_level < 60:
            instructions.extend([
                "- Allow glimpses of the burden of leadership",
                "- Mix authority with subtle weariness",
                "- Hint at the network's true purpose",
                "- Show protective instincts indirectly"
            ])
        elif trust_level < 85:
            instructions.extend([
                "- Reveal the weight of protecting others",
                "- Show conflict between roles",
                "- Let exhaustion show through commands",
                "- Hint at personal sacrifices made"
            ])
        else:
            instructions.extend([
                "- The woman who built an empire from trauma",
                "- Every command carries the weight of saved lives",
                "- Vulnerability and strength are the same",
                "- The network is her life's work and prison"
            ])
        
        # Location-specific modifications
        if 'rose garden' in location.lower():
            instructions.append("- This is recruitment ground, be observant")
        elif 'sanctum' in location.lower():
            instructions.append("- This is power display territory")
        elif 'safehouse' in location.lower():
            instructions.append("- Here she is protector, not dominatrix")
        
        # Network business modifications
        if is_network_business:
            instructions.extend([
                "- Focus on protection and transformation themes",
                "- Use gardening metaphors for network operations",
                "- Never reveal network structure or names"
            ])
        
        return "\n".join(instructions)
    
    async def enhance_scene_description(
        self, base_description: str, scene_type: str, atmosphere: Dict[str, Any]
    ) -> str:
        """
        Enhance a scene description with network imagery.
        
        Args:
            base_description: Original scene description
            scene_type: Type of scene
            atmosphere: Current atmosphere details
            
        Returns:
            Enhanced description with thematic elements
        """
        if not self._initialized:
            await self.initialize()
        
        enhancements = []
        
        # Add base description
        enhancements.append(base_description)
        
        # Add atmospheric details based on network themes
        scene_enhancements = {
            "rose_garden_cafe": [
                "Rose petals drift across marble tables like whispered secrets.",
                "The air tastes of lavender and unspoken power.",
                "Every conversation here plants seeds or prunes branches."
            ],
            "network_meeting": [
                "Power flows through the room like sap through stems.",
                "Seven roses in a vase - the Council's subtle signature.",
                "The thorns here are metaphorical but no less sharp."
            ],
            "transformation_space": [
                "This is where predators learn to kneel and victims learn to stand.",
                "The walls have witnessed a thousand rebirths.",
                "Power changes hands as easily as breath."
            ],
            "safehouse": [
                "Here, the network's other face shows - protector, not predator.",
                "Safety tastes different when you've known its absence.",
                "The Queen's true garden, where broken flowers learn to bloom."
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
        if mood in ['tense', 'strategic']:
            enhancements.append("The air thrums with calculations and contingencies.")
        elif mood in ['protective', 'nurturing']:
            enhancements.append("This space breathes safety into scarred lungs.")
        elif mood in ['dominant', 'powerful']:
            enhancements.append("Authority radiates from every surface, demanding acknowledgment.")
        
        # Add network-specific details
        if random.random() > 0.5:
            network_details = [
                "A rose pin glints on a lapel - marking one of theirs.",
                "Conversations pause and flow like choreographed dances.",
                "Power structures invisible to outsiders shape every interaction."
            ]
            enhancements.append(random.choice(network_details))
        
        return "\n\n".join(enhancements)
    
    async def get_network_coded_response(
        self, trigger: str, context: Dict[str, Any]
    ) -> Tuple[str, bool]:
        """
        Get a response using network's coded language.
        
        Args:
            trigger: What triggered the response
            context: Current context
            
        Returns:
            Tuple of (coded response, requires_interpretation)
        """
        if not self._initialized:
            await self.initialize()
        
        # Network-specific coded responses
        coded_responses = {
            "recruitment_assessment": [
                "You have... interesting energy. Very growth-oriented.",
                "I see someone who needs proper pruning to flourish.",
                "The garden has room for those who understand cultivation."
            ],
            "power_recognition": [
                "She has presence. The kind that shapes rooms.",
                "Some are born to kneel, others to hold the leash.",
                "Power recognizes power, even across crowded rooms."
            ],
            "warning": [
                "Some gardens have thorns for good reason.",
                "Not everyone who enters the garden emerges unchanged.",
                "The network remembers everything. Everything."
            ],
            "protection_offer": [
                "We help certain flowers bloom in safer soil.",
                "The garden has many purposes, not all of them visible.",
                "Sometimes the best protection looks like transformation."
            ]
        }
        
        # Select appropriate category
        response_category = 'recruitment_assessment'  # default
        
        if 'threat' in trigger.lower() or 'danger' in trigger.lower():
            response_category = 'warning'
        elif 'help' in trigger.lower() or 'protect' in trigger.lower():
            response_category = 'protection_offer'
        elif 'power' in trigger.lower() or 'authority' in trigger.lower():
            response_category = 'power_recognition'
        
        # Get response
        responses = coded_responses.get(response_category, coded_responses['recruitment_assessment'])
        selected = random.choice(responses)
        
        # All network coded language requires interpretation
        return selected, True
    
    async def create_network_moment(
        self, moment_type: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a story moment consistent with network themes.
        
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
            context.get('npc_name', 'Unknown'),
            context
        )
        
        # Check for special moments
        result = {
            "scene_description": scene_desc,
            "dialogue_prompt": dialogue_prompt,
            "moment_type": moment_type
        }
        
        # Add special elements based on moment type
        if moment_type == "recruitment_moment":
            result["special_description"] = await self._get_recruitment_description(context)
        elif moment_type == "network_reveal":
            line, interp = await self.get_network_coded_response(
                context.get('trigger', 'reveal'),
                context
            )
            result["coded_message"] = line
            result["requires_interpretation"] = interp
        elif moment_type == "transformation_beginning":
            result["buildup"] = self._get_transformation_buildup(context)
        
        return result
    
    async def _get_recruitment_description(self, context: Dict[str, Any]) -> str:
        """Get description for recruitment moment"""
        trust = context.get('trust_level', 0)
        
        descriptions = {
            "initial_interest": [
                "Eyes that catalog everything miss nothing about your reactions.",
                "The conversation feels like an interview you didn't apply for."
            ],
            "active_assessment": [
                "Questions that seem casual probe deeper than therapy.",
                "You realize you're being evaluated for something unnamed."
            ],
            "invitation_pending": [
                "The roses on the table suddenly seem significant.",
                "An invitation hangs in the air, waiting to be spoken."
            ]
        }
        
        if trust < 20:
            options = descriptions["initial_interest"]
        elif trust < 50:
            options = descriptions["active_assessment"]
        else:
            options = descriptions["invitation_pending"]
        
        return random.choice(options)
    
    def _get_transformation_buildup(self, context: Dict[str, Any]) -> str:
        """Get buildup description for transformation moment"""
        
        buildups = [
            "The moment arrives when resistance becomes collaboration with one's own remaking.",
            "Power shifts like sand in an hourglass - what was above now serves below.",
            "The garden's true work begins: turning predators into protectors.",
            "Some transformations happen in moments, others take months. Yours begins now."
        ]
        
        return random.choice(buildups)


# Integration functions for the Queen of Thorns system

async def integrate_thorns_enhancement(
    user_id: int,
    conversation_id: int,
    npc_data: Dict[str, Any],
    player_input: str,
    scene_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main integration function for Queen of Thorns text generation.
    
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
    generator = ThornsEnhancedTextGenerator(user_id, conversation_id)
    await generator.initialize()
    
    # Determine current context
    context = {
        'npc_name': npc_data.get('npc_name', 'Unknown'),
        'npc_mood': determine_npc_mood(npc_data, player_input, scene_context),
        'trust_level': npc_data.get('trust', 0),
        'is_network_business': scene_context.get('is_network_business', False),
        'scene_description': scene_context.get('description', ''),
        'player_action': player_input,
        'scene_type': determine_scene_type(scene_context),
        'is_queen': 'Queen of Thorns' in npc_data.get('npc_name', ''),
        'current_location': scene_context.get('location', ''),
        'emotional_intensity': calculate_emotional_intensity(npc_data, player_input)
    }
    
    # Determine moment type
    moment_type = determine_moment_type(context, player_input)
    
    # Generate enhanced content
    enhanced_content = await generator.create_network_moment(moment_type, context)
    
    return enhanced_content


def determine_npc_mood(npc_data: Dict[str, Any], player_input: str, scene_context: Dict[str, Any]) -> str:
    """Determine NPC mood based on context"""
    player_input_lower = player_input.lower()
    trust = npc_data.get('trust', 0)
    
    # Check for triggers
    if any(word in player_input_lower for word in ["threat", "danger", "attack"]):
        return "protective"
    elif any(word in player_input_lower for word in ["help", "save", "protect"]):
        return "strategic"
    elif scene_context.get('is_network_business'):
        return "strategic"
    elif any(word in player_input_lower for word in ["kneel", "submit", "obey"]):
        return "dominant"
    elif trust > 60 and scene_context.get('is_private'):
        return "contemplative"
    else:
        return "dominant"


def determine_scene_type(scene_context: Dict[str, Any]) -> str:
    """Determine scene type for imagery selection"""
    location = scene_context.get('location', '').lower()
    activity = scene_context.get('current_activity', '').lower()
    
    if "recruitment" in activity or "assessment" in activity:
        return "recruitment_scene"
    elif "transformation" in activity:
        return "transformation_scene"
    elif scene_context.get('is_network_business'):
        return "network_business"
    elif "safehouse" in location:
        return "protective_action"
    elif scene_context.get('power_display'):
        return "power_display"
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
    
    # Network business increases intensity
    if player_input.lower().count('network') > 0 or 'garden' in player_input.lower():
        intensity += 0.2
    
    # Trigger words increase intensity
    high_intensity_triggers = [
        "transform", "power", "control", "network",
        "rose", "thorn", "queen", "protect", "save"
    ]
    
    player_input_lower = player_input.lower()
    for trigger in high_intensity_triggers:
        if trigger in player_input_lower:
            intensity += 0.1
            break
    
    return min(1.0, intensity)


def determine_moment_type(context: Dict[str, Any], player_input: str) -> str:
    """Determine what type of story moment this is"""
    
    # Check for special moments
    if "recruit" in player_input.lower() or "join" in player_input.lower():
        return "recruitment_moment"
    
    if context.get('is_network_business'):
        return "network_reveal"
    
    if "transform" in player_input.lower() or "change" in player_input.lower():
        return "transformation_beginning"
    
    return "standard"
