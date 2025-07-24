# story_templates/moth/story_runner.py
"""
Main story runner that coordinates all components of The Moth and Flame
Handles initialization, progression, and special mechanics
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from db.connection import get_db_connection_context
from story_templates.moth.story_initializer import MothFlameStoryInitializer, MothFlameStoryProgression
from story_templates.moth.poem_enhanced_generation import PoemEnhancedTextGenerator, integrate_poem_enhancement
from npcs.lilith_mechanics import LilithMechanicsHandler
from memory.wrapper import MemorySystem
from lore.core import canon

logger = logging.getLogger(__name__)

class MothFlameStoryRunner:
    """
    Main coordinator for The Moth and Flame story.
    Manages story state, progression, and special mechanics.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.story_id = "the_moth_and_flame"
        
        # Component handlers
        self.poem_generator = None
        self.lilith_mechanics = None
        self.memory_system = None
        
        # Story state
        self.current_act = 1
        self.current_beat = None
        self.story_flags = {}
        self.lilith_npc_id = None
        
        # Tracking
        self._initialized = False
        self._last_beat_check = None
    
    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize or resume the story.
        
        Returns:
            Dict with initialization status
        """
        try:
            # Check if story already exists
            story_exists = await self._check_story_exists()
            
            if not story_exists:
                # Initialize new story
                logger.info(f"Initializing new Moth and Flame story for user {self.user_id}")
                
                ctx = type('Context', (), {
                    'context': {
                        'user_id': self.user_id,
                        'conversation_id': self.conversation_id
                    }
                })()
                
                init_result = await MothFlameStoryInitializer.initialize_story(
                    ctx, self.user_id, self.conversation_id
                )
                
                if init_result['status'] != 'success':
                    return init_result
                
                self.lilith_npc_id = init_result['main_npc_id']
                logger.info(f"Story initialized with Lilith ID: {self.lilith_npc_id}")
            else:
                # Load existing story state
                await self._load_story_state()
                logger.info(f"Resumed existing story at Act {self.current_act}, Beat: {self.current_beat}")
            
            # Initialize components
            await self._initialize_components()
            
            self._initialized = True
            
            return {
                "status": "success",
                "message": "Story ready",
                "new_story": not story_exists,
                "current_act": self.current_act,
                "current_beat": self.current_beat
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize story: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to initialize story"
            }
    
    async def _check_story_exists(self) -> bool:
        """Check if story already exists for this user/conversation"""
        async with get_db_connection_context() as conn:
            exists = await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM story_states
                    WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                )
                """,
                self.user_id, self.conversation_id, self.story_id
            )
            return exists
    
    async def _load_story_state(self):
        """Load existing story state from database"""
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                """
                SELECT current_act, current_beat, story_flags, progress,
                       story_flags->>'lilith_npc_id' as lilith_id
                FROM story_states
                WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                """,
                self.user_id, self.conversation_id, self.story_id
            )
            
            if row:
                self.current_act = row['current_act']
                self.current_beat = row['current_beat']
                self.story_flags = json.loads(row['story_flags'])
                self.lilith_npc_id = int(row['lilith_id']) if row['lilith_id'] else None
    
    async def _initialize_components(self):
        """Initialize all component handlers"""
        # Poem generator
        self.poem_generator = PoemEnhancedTextGenerator(
            self.user_id, self.conversation_id, self.story_id
        )
        await self.poem_generator.initialize()
        
        # Lilith mechanics handler
        if self.lilith_npc_id:
            self.lilith_mechanics = LilithMechanicsHandler(
                self.user_id, self.conversation_id, self.lilith_npc_id
            )
            await self.lilith_mechanics.initialize()
        
        # Memory system
        self.memory_system = await MemorySystem.get_instance(
            self.user_id, self.conversation_id
        )
    
    async def process_player_action(
        self, 
        player_input: str, 
        current_location: str,
        scene_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a player action and generate appropriate response.
        
        Args:
            player_input: What the player said/did
            current_location: Current location name
            scene_context: Additional context
            
        Returns:
            Dict with response elements
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Build complete context
            context = await self._build_action_context(
                player_input, current_location, scene_context
            )
            
            # Check for story beat triggers
            beat_trigger = await self._check_beat_triggers(context)
            if beat_trigger:
                beat_result = await self._trigger_story_beat(beat_trigger, context)
                if beat_result.get('interrupt_action'):
                    return beat_result
            
            # Check Lilith's special mechanics
            mechanics_results = await self._check_special_mechanics(context)
            
            # Generate enhanced response
            response = await self._generate_response(
                context, mechanics_results
            )
            
            # Update story state
            await self._update_story_state(context, response)
            
            # Check for story progression
            progression = await self._check_story_progression()
            if progression:
                response['story_progression'] = progression
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing player action: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to process action"
            }
    
    async def _build_action_context(
        self, 
        player_input: str,
        current_location: str,
        scene_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build complete context for action processing"""
        # Get Lilith's current state
        lilith_data = None
        if self.lilith_npc_id:
            async with get_db_connection_context() as conn:
                lilith_row = await conn.fetchrow(
                    """
                    SELECT npc_name, trust, dominance, cruelty, affection, intensity,
                           current_mask, three_words_spoken, current_location
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                    """,
                    self.user_id, self.conversation_id, self.lilith_npc_id
                )
                
                if lilith_row:
                    lilith_data = dict(lilith_row)
        
        # Get location details
        location_data = await self._get_location_data(current_location)
        
        # Build context
        context = {
            'player_input': player_input,
            'player_action': player_input,  # Alias for compatibility
            'current_location': current_location,
            'location_data': location_data,
            'lilith_data': lilith_data,
            'trust_level': lilith_data.get('trust', 0) if lilith_data else 0,
            'current_mask': lilith_data.get('current_mask', 'Unknown') if lilith_data else 'Unknown',
            'story_act': self.current_act,
            'story_beat': self.current_beat,
            'story_flags': self.story_flags,
            'timestamp': datetime.now()
        }
        
        # Merge with provided scene context
        if scene_context:
            context.update(scene_context)
        
        # Determine derived context
        context['is_private'] = self._is_private_location(current_location)
        context['lilith_present'] = (
            lilith_data and 
            lilith_data.get('current_location') == current_location
        )
        
        return context
    
    async def _get_location_data(self, location_name: str) -> Dict[str, Any]:
        """Get location details from database"""
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                """
                SELECT location_name, description, metadata
                FROM Locations
                WHERE user_id = $1 AND conversation_id = $2 
                AND LOWER(location_name) = LOWER($3)
                """,
                self.user_id, self.conversation_id, location_name
            )
            
            if row:
                return {
                    'name': row['location_name'],
                    'description': row['description'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                }
            
            return {'name': location_name, 'description': '', 'metadata': {}}
    
    def _is_private_location(self, location: str) -> bool:
        """Check if location is private"""
        location_lower = location.lower()
        private_keywords = ['private', 'chambers', 'bedroom', 'personal', 'hidden']
        return any(keyword in location_lower for keyword in private_keywords)
    
    async def _check_beat_triggers(self, context: Dict[str, Any]) -> Optional[str]:
        """Check if any story beats should trigger"""
        # Use the story progression checker
        beat_id = await MothFlameStoryProgression.check_beat_triggers(
            self.user_id, self.conversation_id
        )
        
        if beat_id and beat_id != self.current_beat:
            logger.info(f"Story beat triggered: {beat_id}")
            return beat_id
        
        return None
    
    async def _trigger_story_beat(
        self, beat_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Trigger a specific story beat"""
        result = await MothFlameStoryProgression.trigger_story_beat(
            self.user_id, self.conversation_id, beat_id
        )
        
        if result.get('status') == 'success':
            self.current_beat = beat_id
            
            # Get beat details
            from story_templates.moth.the_moth_and_flame import THE_MOTH_AND_FLAME
            beat = next((b for b in THE_MOTH_AND_FLAME.story_beats if b.id == beat_id), None)
            
            if beat:
                # Generate beat introduction
                beat_intro = await self._generate_beat_introduction(beat, context)
                result['introduction'] = beat_intro
                result['dialogue_hints'] = beat.dialogue_hints
                
                # Check if this beat interrupts current action
                if beat.narrative_stage in ['Full Revelation', 'Veil Thinning']:
                    result['interrupt_action'] = True
        
        return result
    
    async def _generate_beat_introduction(
        self, beat: Any, context: Dict[str, Any]
    ) -> str:
        """Generate introduction text for a story beat"""
        # Use poem generator to create atmospheric introduction
        intro_context = {
            'beat_name': beat.name,
            'beat_description': beat.description,
            'narrative_stage': beat.narrative_stage,
            'location': context.get('current_location'),
            'atmosphere': {
                'emotional_tone': self._get_beat_tone(beat.narrative_stage)
            }
        }
        
        enhanced_desc = await self.poem_generator.enhance_scene_description(
            beat.description,
            context.get('current_location', 'general'),
            intro_context['atmosphere']
        )
        
        return enhanced_desc
    
    def _get_beat_tone(self, narrative_stage: str) -> str:
        """Get emotional tone for narrative stage"""
        tone_map = {
            'Innocent Beginning': 'curious',
            'First Doubts': 'uncertain',
            'Creeping Realization': 'tense',
            'Veil Thinning': 'vulnerable',
            'Full Revelation': 'intense'
        }
        return tone_map.get(narrative_stage, 'neutral')
    
    async def _check_special_mechanics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check all of Lilith's special mechanics"""
        if not self.lilith_mechanics or not context.get('lilith_present'):
            return {}
        
        results = {}
        
        # Check mask state
        mask_result = await self.lilith_mechanics.check_mask_state(context)
        if mask_result.get('slipped') or mask_result.get('mask_change'):
            results['mask_event'] = mask_result
        
        # Check for poetry moment
        poetry_result = await self.lilith_mechanics.check_poetry_moment(context)
        if poetry_result.get('poetry_moment'):
            results['poetry_moment'] = poetry_result
        
        # Check for three words moment
        three_words_result = await self.lilith_mechanics.check_three_words_moment(context)
        if three_words_result.get('three_words_possible'):
            results['three_words_moment'] = three_words_result
        
        # Check for trauma triggers
        trauma_result = await self.lilith_mechanics.check_trauma_trigger(context)
        if trauma_result.get('trauma_triggered'):
            results['trauma_trigger'] = trauma_result
        
        # Check for dual identity reveal
        identity_result = await self.lilith_mechanics.check_dual_identity_reveal(context)
        if identity_result.get('reveal_possible'):
            results['identity_reveal'] = identity_result
        
        return results
    
    async def _generate_response(
        self, 
        context: Dict[str, Any],
        mechanics_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate the complete response"""
        response = {
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }
        
        # Determine NPC mood and scene type
        npc_mood = self._determine_npc_mood(context, mechanics_results)
        scene_type = self._determine_scene_type(context)
        
        # Update context with mood
        context['npc_mood'] = npc_mood
        context['scene_type'] = scene_type
        context['emotional_intensity'] = self._calculate_emotional_intensity(
            context, mechanics_results
        )
        
        # Get dialogue style from mechanics
        if self.lilith_mechanics and context.get('lilith_present'):
            dialogue_style = await self.lilith_mechanics.get_dialogue_style(context)
            response['dialogue_style'] = dialogue_style
        
        # Generate enhanced content
        enhanced_content = await integrate_poem_enhancement(
            self.user_id,
            self.conversation_id,
            context.get('lilith_data', {}),
            context['player_input'],
            context
        )
        
        response.update(enhanced_content)
        
        # Add special mechanics results
        if mechanics_results:
            response['special_events'] = mechanics_results
            
            # Handle special responses
            if 'mask_event' in mechanics_results:
                response['mask_event'] = mechanics_results['mask_event']
            
            if 'poetry_moment' in mechanics_results:
                response['poetry_lines'] = mechanics_results['poetry_moment']['suggested_lines']
            
            if 'three_words_moment' in mechanics_results:
                response['three_words_buildup'] = mechanics_results['three_words_moment']['buildup_description']
                response['requires_player_choice'] = True
            
            if 'trauma_trigger' in mechanics_results:
                response['trauma_response'] = mechanics_results['trauma_trigger']['description']
                response['mood_override'] = 'defensive'
        
        # Add current story context
        response['story_context'] = {
            'act': self.current_act,
            'beat': self.current_beat,
            'trust_level': context.get('trust_level', 0),
            'current_mask': context.get('current_mask', 'Unknown')
        }
        
        return response
    
    def _determine_npc_mood(
        self, context: Dict[str, Any], mechanics: Dict[str, Any]
    ) -> str:
        """Determine Lilith's current mood"""
        # Override for trauma
        if mechanics.get('trauma_trigger'):
            return 'defensive'
        
        # Check player input
        player_input = context.get('player_input', '').lower()
        
        if any(word in player_input for word in ['leave', 'goodbye', 'go away']):
            return 'desperate'
        elif any(word in player_input for word in ['love', 'adore', 'mine']):
            trust = context.get('trust_level', 0)
            return 'vulnerable' if trust > 60 else 'dominant'
        elif context.get('is_private') and context.get('trust_level', 0) > 50:
            return 'vulnerable'
        elif 'sanctum' in context.get('current_location', '').lower():
            return 'dominant'
        else:
            return 'contemplative'
    
    def _determine_scene_type(self, context: Dict[str, Any]) -> str:
        """Determine the type of scene"""
        location = context.get('current_location', '').lower()
        
        if context.get('story_beat') in ['glimpse_beneath', 'the_confession']:
            return 'vulnerable_moment'
        elif context.get('story_beat') in ['the_performance', 'the_test']:
            return 'dominant_scene'
        elif 'mask' in context.get('player_input', '').lower():
            return 'mask_scene'
        elif context.get('is_private'):
            return 'intimate_scene'
        else:
            return 'general'
    
    def _calculate_emotional_intensity(
        self, context: Dict[str, Any], mechanics: Dict[str, Any]
    ) -> float:
        """Calculate current emotional intensity"""
        intensity = 0.3
        
        # Trust increases intensity
        trust = context.get('trust_level', 0)
        intensity += (trust / 100) * 0.3
        
        # Special events increase intensity
        if mechanics.get('three_words_moment'):
            intensity += 0.4
        if mechanics.get('mask_event', {}).get('mask_change'):
            intensity += 0.3
        if mechanics.get('trauma_trigger'):
            intensity += 0.5
        
        # Story beat intensity
        beat_intensities = {
            'the_confession': 0.4,
            'breaking_point': 0.5,
            'eternal_dance': 0.6
        }
        
        if context.get('story_beat') in beat_intensities:
            intensity += beat_intensities[context['story_beat']]
        
        return min(1.0, intensity)
    
    async def _update_story_state(
        self, context: Dict[str, Any], response: Dict[str, Any]
    ):
        """Update story state based on action results"""
        updates = {}
        
        # Update story flags
        if response.get('special_events'):
            if 'identity_reveal' in response['special_events']:
                self.story_flags['dual_identity_revealed'] = True
                updates['dual_identity_revealed'] = True
            
            if 'three_words_moment' in response['special_events']:
                self.story_flags['three_words_near'] = True
                updates['three_words_near'] = True
        
        # Update trust and other metrics
        if response.get('trust_change'):
            current_trust = context.get('trust_level', 0)
            new_trust = max(-100, min(100, current_trust + response['trust_change']))
            
            # Update through canon
            if self.lilith_npc_id:
                canon_ctx = type('CanonicalContext', (), {
                    'user_id': self.user_id,
                    'conversation_id': self.conversation_id
                })()
                
                async with get_db_connection_context() as conn:
                    await canon.update_entity_canonically(
                        canon_ctx, conn, "NPCStats", self.lilith_npc_id,
                        {'trust': new_trust},
                        f"Trust changed from {current_trust} to {new_trust}"
                    )
        
        # Update story state in database
        if updates:
            async with get_db_connection_context() as conn:
                await conn.execute(
                    """
                    UPDATE story_states
                    SET story_flags = story_flags || $4,
                        updated_at = NOW()
                    WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                    """,
                    self.user_id, self.conversation_id, self.story_id,
                    json.dumps(updates)
                )
    
    async def _check_story_progression(self) -> Optional[Dict[str, Any]]:
        """Check if story should progress to next act"""
        # Get completed beats
        completed_beats = self.story_flags.get('completed_beats', [])
        
        from story_templates.moth.the_moth_and_flame import THE_MOTH_AND_FLAME
        
        # Check act completion
        act_beats = [b.id for b in THE_MOTH_AND_FLAME.story_beats if b.id in completed_beats]
        
        act_requirements = {
            1: ['first_glimpse', 'invitation', 'first_session'],
            2: ['after_hours', 'glimpse_beneath', 'the_confession'],
            3: ['the_test', 'breaking_point']
        }
        
        current_requirements = act_requirements.get(self.current_act, [])
        if all(beat in completed_beats for beat in current_requirements):
            # Progress to next act
            if self.current_act < 3:
                self.current_act += 1
                
                async with get_db_connection_context() as conn:
                    await conn.execute(
                        """
                        UPDATE story_states
                        SET current_act = $4
                        WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                        """,
                        self.user_id, self.conversation_id, self.story_id,
                        self.current_act
                    )
                
                return {
                    'act_complete': self.current_act - 1,
                    'new_act': self.current_act,
                    'message': f"Act {self.current_act} begins..."
                }
        
        return None
    
    async def handle_special_choice(
        self, choice_type: str, player_choice: str
    ) -> Dict[str, Any]:
        """
        Handle special story choices (like three words response).
        
        Args:
            choice_type: Type of choice
            player_choice: Player's choice
            
        Returns:
            Result of the choice
        """
        if choice_type == "three_words_response":
            if not self.lilith_mechanics:
                return {"error": "Lilith not present"}
            
            result = await self.lilith_mechanics.speak_three_words(player_choice)
            
            # Update story state
            if result.get('words_spoken'):
                self.story_flags['three_words_spoken'] = True
                self.story_flags['three_words_outcome'] = result['outcome']
                
                async with get_db_connection_context() as conn:
                    await conn.execute(
                        """
                        UPDATE story_states
                        SET story_flags = story_flags || $4
                        WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                        """,
                        self.user_id, self.conversation_id, self.story_id,
                        json.dumps({
                            'three_words_spoken': True,
                            'three_words_outcome': result['outcome']
                        })
                    )
            
            return result
        
        return {"error": f"Unknown choice type: {choice_type}"}
    
    async def get_story_status(self) -> Dict[str, Any]:
        """Get current story status and progress"""
        if not self._initialized:
            await self.initialize()
        
        from story_templates.moth.the_moth_and_flame import THE_MOTH_AND_FLAME
        
        # Calculate progress
        total_beats = len(THE_MOTH_AND_FLAME.story_beats)
        completed_beats = len(self.story_flags.get('completed_beats', []))
        progress_percentage = (completed_beats / total_beats) * 100
        
        # Get Lilith's current state
        lilith_state = {}
        if self.lilith_npc_id:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT trust, current_mask, three_words_spoken
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                    """,
                    self.user_id, self.conversation_id, self.lilith_npc_id
                )
                if row:
                    lilith_state = dict(row)
        
        return {
            'story_id': self.story_id,
            'current_act': self.current_act,
            'current_beat': self.current_beat,
            'progress_percentage': progress_percentage,
            'completed_beats': self.story_flags.get('completed_beats', []),
            'lilith_state': lilith_state,
            'key_flags': {
                'dual_identity_revealed': self.story_flags.get('dual_identity_revealed', False),
                'three_words_spoken': self.story_flags.get('three_words_spoken', False),
                'moth_flame_established': self.story_flags.get('moth_flame_dynamic') != 'unestablished'
            }
        }
