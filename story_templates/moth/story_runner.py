# story_templates/moth/story_runner.py
"""
Main story runner that coordinates all components of Queen of Thorns
Handles initialization, progression, and special mechanics
Fully integrated with SF Bay preset
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from db.connection import get_db_connection_context
from story_templates.moth.story_initializer import QueenOfThornsStoryInitializer, QueenOfThornsStoryProgression
from story_templates.moth.poem_enhanced_generation import ThornsEnhancedTextGenerator, integrate_thorns_enhancement
from story_templates.moth.npcs.queen_mechanics import QueenMechanicsHandler
from story_templates.moth.lore import SFBayQueenOfThornsPreset, QueenOfThornsLoreAccess
from memory.wrapper import MemorySystem
from lore.core import canon

logger = logging.getLogger(__name__)

class QueenOfThornsStoryRunner:
    """
    Main coordinator for Queen of Thorns story.
    Manages story state, progression, special mechanics, and preset lore.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.story_id = "queen_of_thorns"
        
        # Component handlers
        self.text_generator = None
        self.queen_mechanics = None
        self.memory_system = None
        self.lore_access = None
        
        # Story state
        self.current_act = 1
        self.current_beat = None
        self.story_flags = {}
        self.queen_npc_id = None
        
        # Network state
        self.network_awareness = 0
        self.information_layer = "public"
        self.player_rank = "outsider"
        
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
                logger.info(f"Initializing new Queen of Thorns story for user {self.user_id}")
                
                ctx = type('Context', (), {
                    'context': {
                        'user_id': self.user_id,
                        'conversation_id': self.conversation_id
                    }
                })()
                
                # Initialize the story with SF preset
                init_result = await QueenOfThornsStoryInitializer.initialize_story(
                    ctx, self.user_id, self.conversation_id
                )
                
                if init_result['status'] != 'success':
                    return init_result
                
                self.queen_npc_id = init_result['main_npc_id']
                
                logger.info(f"Story initialized with Queen ID: {self.queen_npc_id}")
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
                "current_beat": self.current_beat,
                "network_awareness": self.network_awareness,
                "information_layer": self.information_layer,
                "player_rank": self.player_rank,
                "setting": "San Francisco Bay Area, 2025"
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
                       story_flags->>'queen_npc_id' as queen_id,
                       story_flags->>'network_awareness' as awareness,
                       story_flags->>'information_layer' as info_layer,
                       story_flags->>'player_rank' as rank
                FROM story_states
                WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                """,
                self.user_id, self.conversation_id, self.story_id
            )
            
            if row:
                self.current_act = row['current_act']
                self.current_beat = row['current_beat']
                self.story_flags = json.loads(row['story_flags'])
                self.queen_npc_id = int(row['queen_id']) if row['queen_id'] else None
                self.network_awareness = int(row['awareness'] or 0)
                self.information_layer = row['info_layer'] or 'public'
                self.player_rank = row['rank'] or 'outsider'
    
    async def _initialize_components(self):
        """Initialize all component handlers"""
        # Text generator
        self.text_generator = ThornsEnhancedTextGenerator(
            self.user_id, self.conversation_id, self.story_id
        )
        await self.text_generator.initialize()
        
        # Queen mechanics handler
        if self.queen_npc_id:
            self.queen_mechanics = QueenMechanicsHandler(
                self.user_id, self.conversation_id, self.queen_npc_id
            )
            await self.queen_mechanics.initialize()
        
        # Memory system
        self.memory_system = await MemorySystem.get_instance(
            self.user_id, self.conversation_id
        )
        
        # Lore access
        self.lore_access = QueenOfThornsLoreAccess(
            self.user_id, self.conversation_id
        )
    
    async def get_current_location_lore(self) -> Dict[str, Any]:
        """Get lore for current location"""
        from lore.managers.local_lore import get_location_lore
        
        # Get player's current location
        async with get_db_connection_context() as conn:
            current_loc = await conn.fetchval(
                """
                SELECT value FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentLocation'
                """,
                self.user_id, self.conversation_id
            )
        
        if not current_loc:
            return {}
        
        # Map to location ID
        ctx = self.create_context()
        location_id = await self._ensure_location_exists(ctx, current_loc)
        
        # Get all lore for this location
        lore_result = await get_location_lore(ctx, location_id)
        
        return lore_result.model_dump() if hasattr(lore_result, 'model_dump') else lore_result
    
    async def _ensure_location_exists(self, ctx, location_name: str) -> int:
        """Ensure a location exists and return its ID"""
        from lore.core import canon
        
        async with get_db_connection_context() as conn:
            location_id = await canon.find_or_create_location(
                ctx, conn, location_name
            )
            
        return location_id
    
    def create_context(self):
        """Create a context object for function calls"""
        return type('Context', (), {
            'context': {
                'user_id': self.user_id,
                'conversation_id': self.conversation_id
            }
        })()
    
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
            
            # Enhance context with location lore
            location_lore = self.lore_access.get_location_by_name(current_location)
            if location_lore:
                context['location_lore'] = location_lore
                
            # Get current location lore from the lore system
            context['full_location_lore'] = await self.get_current_location_lore()
                
            # Check if we're in a special network location
            context['network_location'] = self._check_network_location(current_location)
            
            # Check for story beat triggers
            beat_trigger = await self._check_beat_triggers(context)
            if beat_trigger:
                beat_result = await self._trigger_story_beat(beat_trigger, context)
                if beat_result.get('interrupt_action'):
                    return beat_result
            
            # Check network mechanics
            network_results = await self._check_network_mechanics(context)
            
            # Check Queen's special mechanics if she's present
            queen_mechanics_results = {}
            if context.get('queen_present') and self.queen_mechanics:
                queen_mechanics_results = await self._check_queen_special_mechanics(context)
            
            # Combine all mechanics results
            all_mechanics = {**network_results, **queen_mechanics_results}
            
            # Generate enhanced response
            response = await self._generate_response(
                context, all_mechanics
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
    
    def _check_network_location(self, location: str) -> Optional[Dict[str, Any]]:
        """Check if we're in a network-related location"""
        location_lower = location.lower()
        
        if 'rose garden' in location_lower:
            return {
                "type": "recruitment_hub",
                "network_layer": "public",
                "special_rules": ["assessment_possible", "coded_language"],
                "npc_present": "Lily Chen (Gardener)"
            }
        elif 'thornfield' in location_lower:
            return {
                "type": "power_brokerage",
                "network_layer": "semi_private",
                "special_rules": ["contracts_available", "transformation_legal"],
                "atmosphere": "professional_power"
            }
        elif 'inner garden' in location_lower:
            return {
                "type": "queen_sanctuary",
                "network_layer": "deep_secret",
                "special_rules": ["queen_presence", "ultimate_authority"],
                "atmosphere": "mysterious_power"
            }
        elif 'safehouse' in location_lower or 'butterfly house' in location_lower:
            return {
                "type": "protection_space",
                "network_layer": "hidden",
                "special_rules": ["no_violence", "healing_priority"],
                "atmosphere": "protective_nurturing"
            }
        elif 'montenegro' in location_lower:
            return {
                "type": "transformation_assessment",
                "network_layer": "semi_private",
                "special_rules": ["art_reveals_nature", "psychological_evaluation"],
                "npc_present": "Isabella Montenegro"
            }
        
        return None
    
    async def _build_action_context(
        self, 
        player_input: str,
        current_location: str,
        scene_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build complete context for action processing"""
        # Get Queen's current state if relevant
        queen_data = None
        if self.queen_npc_id:
            async with get_db_connection_context() as conn:
                # Check if Queen is present
                queen_row = await conn.fetchrow(
                    """
                    SELECT npc_name, trust, dominance, network_role, current_location,
                           mystery_mechanics, information_layers
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                    """,
                    self.user_id, self.conversation_id, self.queen_npc_id
                )
                
                if queen_row:
                    queen_data = dict(queen_row)
                    # Parse JSON fields
                    if queen_data.get('mystery_mechanics'):
                        queen_data['mystery_mechanics'] = json.loads(queen_data['mystery_mechanics'])
                    if queen_data.get('information_layers'):
                        queen_data['information_layers'] = json.loads(queen_data['information_layers'])
        
        # Get location details
        location_data = await self._get_location_data(current_location)
        
        # Build context
        context = {
            'player_input': player_input,
            'player_action': player_input,
            'current_location': current_location,
            'location_data': location_data,
            'queen_data': queen_data,
            'queen_present': queen_data is not None and queen_data.get('current_location') == current_location,
            'network_awareness': self.network_awareness,
            'information_layer': self.information_layer,
            'player_rank': self.player_rank,
            'story_act': self.current_act,
            'story_beat': self.current_beat,
            'story_flags': self.story_flags,
            'timestamp': datetime.now()
        }
        
        # Merge with provided scene context
        if scene_context:
            context.update(scene_context)
        
        # Determine derived context
        context['is_network_business'] = self._is_network_business(context)
        context['is_private'] = self._is_private_location(current_location)
        
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
    
    def _is_network_business(self, context: Dict[str, Any]) -> bool:
        """Check if current action involves network business"""
        indicators = [
            'network' in context.get('player_input', '').lower(),
            'garden' in context.get('player_input', '').lower(),
            'rose' in context.get('player_input', '').lower(),
            'thorn' in context.get('player_input', '').lower(),
            'transform' in context.get('player_input', '').lower(),
            'queen' in context.get('player_input', '').lower(),
            context.get('network_location') is not None,
            self.information_layer != 'public'
        ]
        
        return any(indicators)
    
    def _is_private_location(self, location: str) -> bool:
        """Check if location is private"""
        location_lower = location.lower()
        private_keywords = ['private', 'chambers', 'bedroom', 'personal', 'hidden', 'inner']
        
        return any(keyword in location_lower for keyword in private_keywords)
    
    async def _check_beat_triggers(self, context: Dict[str, Any]) -> Optional[str]:
        """Check if any story beats should trigger"""
        # Use the story progression checker
        beat_id = await QueenOfThornsStoryProgression.check_beat_triggers(
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
        result = await QueenOfThornsStoryProgression.trigger_story_beat(
            self.user_id, self.conversation_id, beat_id
        )
        
        if result.get('status') == 'success':
            self.current_beat = beat_id
            
            # Get beat details
            from story_templates.moth.queen_of_thorns_story import QUEEN_OF_THORNS_STORY
            beat = next((b for b in QUEEN_OF_THORNS_STORY.story_beats if b.id == beat_id), None)
            
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
        # Use text generator to create atmospheric introduction
        intro_context = {
            'beat_name': beat.name,
            'beat_description': beat.description,
            'narrative_stage': beat.narrative_stage,
            'location': context.get('current_location'),
            'atmosphere': {
                'emotional_tone': self._get_beat_tone(beat.narrative_stage)
            }
        }
        
        # Add network-specific atmosphere
        if context.get('network_location'):
            intro_context['atmosphere']['network_type'] = context['network_location']['type']
            intro_context['atmosphere']['special_rules'] = context['network_location'].get('special_rules', [])
        
        enhanced_desc = await self.text_generator.enhance_scene_description(
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
            'Veil Thinning': 'revealing',
            'Full Revelation': 'intense'
        }
        return tone_map.get(narrative_stage, 'neutral')
    
    async def _check_network_mechanics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check network-specific mechanics"""
        results = {}
        
        # Check for recruitment moment
        if self.player_rank == "outsider" and self.network_awareness > 10:
            if self.lore_access.check_recognition_code(context['player_input']):
                results['recruitment_possible'] = {
                    'type': 'subtle_assessment',
                    'description': 'Your words carry weight they might not realize'
                }
        
        # Check for information layer progression
        layer_thresholds = {
            "public": {"next": "semi_private", "threshold": 30},
            "semi_private": {"next": "hidden", "threshold": 60},
            "hidden": {"next": "deep_secret", "threshold": 90}
        }
        
        current_threshold = layer_thresholds.get(self.information_layer, {})
        if current_threshold and self.network_awareness >= current_threshold.get('threshold', 999):
            results['layer_progression'] = {
                'from': self.information_layer,
                'to': current_threshold['next'],
                'revelation': self._get_layer_revelation(current_threshold['next'])
            }
        
        # Check for transformation witnessing
        if self.information_layer in ['semi_private', 'hidden'] and self.network_awareness >= 40:
            if context.get('network_location', {}).get('type') in ['power_brokerage', 'transformation_assessment']:
                results['transformation_possible'] = {
                    'type': 'witness_opportunity',
                    'description': 'You might see how the network reshapes predators'
                }
        
        # Check for Queen revelation based on context
        if context.get('queen_present') and self.network_awareness >= 70:
            results['queen_nature_glimpse'] = {
                'type': 'ambiguous_reveal',
                'description': 'Is she one or many? The mystery deepens',
                'trust_requirement': 85
            }
        
        return results
    
    def _get_layer_revelation(self, new_layer: str) -> str:
        """Get revelation text for information layer progression"""
        revelations = {
            "semi_private": "The network exists and has purpose beyond charity",
            "hidden": "They save trafficking victims and transform predators",
            "deep_secret": "The true nature of the Queen begins to reveal itself"
        }
        return revelations.get(new_layer, "New understanding dawns")
    
    async def _check_queen_special_mechanics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check Queen-specific mechanics if she's present"""
        if not self.queen_mechanics or not context.get('queen_present'):
            return {}
        
        results = {}
        
        # Check identity mystery mechanics
        identity_check = await self.queen_mechanics.check_identity_mystery(context)
        if identity_check.get('mystery_deepens'):
            results['identity_mystery'] = identity_check
        
        # Check for power display
        power_check = await self.queen_mechanics.check_power_display(context)
        if power_check.get('power_demonstrated'):
            results['power_display'] = power_check
        
        # Check for coded language moment
        coded_check = await self.queen_mechanics.check_coded_language(context)
        if coded_check.get('coded_message'):
            results['coded_language'] = coded_check
        
        # Check for transformation opportunity
        if self.information_layer != 'public':
            transform_check = await self.queen_mechanics.check_transformation_moment(context)
            if transform_check.get('transformation_possible'):
                results['transformation_moment'] = transform_check
        
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
        
        # Get dialogue style from Queen mechanics if applicable
        if self.queen_mechanics and context.get('queen_present'):
            dialogue_style = await self.queen_mechanics.get_dialogue_style(context)
            response['dialogue_style'] = dialogue_style
        
        # Generate enhanced content
        enhanced_content = await integrate_thorns_enhancement(
            self.user_id,
            self.conversation_id,
            context.get('queen_data', {}),
            context['player_input'],
            context
        )
        
        response.update(enhanced_content)
        
        # Add mechanics results
        if mechanics_results:
            response['special_events'] = mechanics_results
            
            # Handle special responses
            if 'recruitment_possible' in mechanics_results:
                response['coded_language'] = True
                response['hidden_assessment'] = True
            
            if 'layer_progression' in mechanics_results:
                response['revelation'] = mechanics_results['layer_progression']['revelation']
                response['new_understanding'] = True
            
            if 'identity_mystery' in mechanics_results:
                response['mystery_deepens'] = mechanics_results['identity_mystery']
            
            if 'transformation_possible' in mechanics_results:
                response['witness_opportunity'] = True
            
            if 'coded_language' in mechanics_results:
                response['coded_message'] = mechanics_results['coded_language']['message']
        
        # Add current story context
        response['story_context'] = {
            'act': self.current_act,
            'beat': self.current_beat,
            'network_awareness': self.network_awareness,
            'information_layer': self.information_layer,
            'player_rank': self.player_rank,
            'setting': 'San Francisco Bay Area, 2025'
        }
        
        # Add network-specific elements
        if context.get('network_location'):
            response['location_significance'] = context['network_location']
        
        # Add available actions based on rank
        response['available_actions'] = self._get_available_actions()
        
        return response
    
    def _determine_npc_mood(
        self, context: Dict[str, Any], mechanics: Dict[str, Any]
    ) -> str:
        """Determine NPC's current mood"""
        player_input = context.get('player_input', '').lower()
        
        if 'threat' in player_input or 'expose' in player_input:
            return 'protective'
        elif 'help' in player_input or 'save' in player_input:
            return 'strategic'
        elif context.get('is_network_business'):
            return 'strategic'
        elif 'power' in player_input or 'control' in player_input:
            return 'dominant'
        elif mechanics.get('transformation_moment'):
            return 'transformative'
        elif context.get('is_private') and self.network_awareness > 50:
            return 'contemplative'
        else:
            return 'observant'
    
    def _determine_scene_type(self, context: Dict[str, Any]) -> str:
        """Determine the type of scene"""
        if context.get('network_location'):
            location_type = context['network_location']['type']
            if location_type == 'recruitment_hub':
                return 'assessment_scene'
            elif location_type == 'power_brokerage':
                return 'transformation_scene'
            elif location_type == 'queen_sanctuary':
                return 'revelation_scene'
            elif location_type == 'protection_space':
                return 'nurturing_scene'
            elif location_type == 'transformation_assessment':
                return 'evaluation_scene'
        
        # Check story beat
        if context.get('story_beat') in ['witnessing_transformation', 'the_greenhouse']:
            return 'transformation_scene'
        elif context.get('story_beat') in ['queen_or_queens', 'the_inner_garden']:
            return 'revelation_scene'
        
        return 'general'
    
    def _calculate_emotional_intensity(
        self, context: Dict[str, Any], mechanics: Dict[str, Any]
    ) -> float:
        """Calculate current emotional intensity"""
        intensity = 0.3
        
        # Network awareness increases intensity
        intensity += (self.network_awareness / 100) * 0.3
        
        # Special events increase intensity
        if mechanics.get('queen_nature_glimpse'):
            intensity += 0.3
        if mechanics.get('layer_progression'):
            intensity += 0.2
        if mechanics.get('transformation_moment'):
            intensity += 0.25
        if mechanics.get('identity_mystery'):
            intensity += 0.2
        
        # Information layer depth
        layer_intensity = {
            'public': 0.0,
            'semi_private': 0.1,
            'hidden': 0.2,
            'deep_secret': 0.4
        }
        intensity += layer_intensity.get(self.information_layer, 0)
        
        # Story beat intensity
        beat_intensities = {
            'witnessing_transformation': 0.3,
            'queen_or_queens': 0.4,
            'your_place_determined': 0.5
        }
        
        if context.get('story_beat') in beat_intensities:
            intensity += beat_intensities[context['story_beat']]
        
        return min(1.0, intensity)
    
    async def _update_story_state(
        self, context: Dict[str, Any], response: Dict[str, Any]
    ):
        """Update story state based on action results"""
        updates = {}
        
        # Update network awareness
        awareness_gain = 0
        if response.get('coded_language'):
            awareness_gain += 5
        if response.get('special_events'):
            if 'transformation_moment' in response['special_events']:
                awareness_gain += 15
            if 'identity_mystery' in response['special_events']:
                awareness_gain += 10
            if 'layer_progression' in response['special_events']:
                awareness_gain += 20
        if context.get('network_location'):
            awareness_gain += 2
        
        if awareness_gain > 0:
            self.network_awareness = min(100, self.network_awareness + awareness_gain)
            updates['network_awareness'] = self.network_awareness
        
        # Update information layer
        if response.get('special_events', {}).get('layer_progression'):
            new_layer = response['special_events']['layer_progression']['to']
            self.information_layer = new_layer
            updates['information_layer'] = new_layer
        
        # Update player rank based on awareness
        new_rank = self._calculate_player_rank()
        if new_rank != self.player_rank:
            self.player_rank = new_rank
            updates['player_rank'] = new_rank
        
        # Update special flags
        if response.get('witness_opportunity'):
            self.story_flags['transformation_witnessed'] = True
            updates['transformation_witnessed'] = True
        
        if context.get('queen_present'):
            if not self.story_flags.get('queen_encountered'):
                self.story_flags['queen_encountered'] = True
                updates['queen_encountered'] = True
        
        # Update network members known
        known_members = self.story_flags.get('known_network_members', [])
        for npc_name in ['Lily Chen', 'Victoria Chen', 'Isabella Montenegro']:
            if npc_name.lower() in context.get('player_input', '').lower() or \
               npc_name in context.get('location_data', {}).get('metadata', {}).get('npc_present', ''):
                if npc_name not in known_members:
                    known_members.append(npc_name)
                    updates['known_network_members'] = known_members
        
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
    
    def _calculate_player_rank(self) -> str:
        """Calculate player's rank in the network"""
        if self.network_awareness < 20:
            return "outsider"
        elif self.network_awareness < 40:
            return "noticed"
        elif self.network_awareness < 60:
            return "seedling"
        elif self.network_awareness < 80:
            return "rose"
        elif self.network_awareness < 90:
            return "thorn"
        elif self.network_awareness < 95:
            return "gardener"
        else:
            return "council_candidate"
    
    async def _check_story_progression(self) -> Optional[Dict[str, Any]]:
        """Check if story should progress"""
        # Network rank progression is the main driver
        rank_progression = {
            "noticed": "The network has taken notice of your potential",
            "seedling": "You've been planted in the garden's soil",
            "rose": "You bloom as part of the network",
            "thorn": "You've proven yourself a protector",
            "gardener": "You now cultivate others' growth",
            "council_candidate": "The inner circle considers your worth"
        }
        
        if self.player_rank in rank_progression:
            completed_ranks = self.story_flags.get('completed_ranks', [])
            if self.player_rank not in completed_ranks:
                completed_ranks.append(self.player_rank)
                
                async with get_db_connection_context() as conn:
                    await conn.execute(
                        """
                        UPDATE story_states
                        SET story_flags = story_flags || $4
                        WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                        """,
                        self.user_id, self.conversation_id, self.story_id,
                        json.dumps({'completed_ranks': completed_ranks})
                    )
                
                return {
                    'rank_achieved': self.player_rank,
                    'message': rank_progression[self.player_rank],
                    'new_opportunities': self._get_rank_opportunities()
                }
        
        # Check act progression
        from story_templates.moth.queen_of_thorns_story import QUEEN_OF_THORNS_STORY
        
        completed_beats = self.story_flags.get('completed_beats', [])
        act_beats = [b.id for b in QUEEN_OF_THORNS_STORY.story_beats if b.id in completed_beats]
        
        act_requirements = {
            1: ['the_garden_gate', 'interesting_energy', 'first_pruning'],
            2: ['deeper_soil', 'the_greenhouse', 'witnessing_transformation'],
            3: ['the_inner_garden', 'queen_or_queens']
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
                    'message': f"Act {self.current_act} begins: {QUEEN_OF_THORNS_STORY.acts[self.current_act-1]['name']}"
                }
        
        return None
    
    def _get_rank_opportunities(self) -> List[str]:
        """Get opportunities available at current rank"""
        opportunities = {
            "outsider": ["Observe the garden's edges"],
            "noticed": ["Coded conversations", "Garden invitations"],
            "seedling": ["Network meetings", "Basic cultivation training"],
            "rose": ["Protection assignments", "Transformation witnessing"],
            "thorn": ["Enforcement duties", "Safehouse access", "Direct action"],
            "gardener": ["Recruitment responsibility", "Council observation", "Shape others"],
            "council_candidate": ["Queen audiences", "Network direction", "Deep secrets"]
        }
        
        return opportunities.get(self.player_rank, [])
    
    def _get_available_actions(self) -> List[str]:
        """Get actions available to player based on current state"""
        actions = ["Observe", "Speak", "Question"]
        
        # Add rank-based actions
        if self.player_rank in ["seedling", "rose", "thorn", "gardener"]:
            actions.append("Use coded language")
        
        if self.player_rank in ["rose", "thorn", "gardener"]:
            actions.append("Request network assistance")
        
        if self.player_rank in ["thorn", "gardener"]:
            actions.append("Identify transformation candidates")
        
        if self.player_rank == "gardener":
            actions.append("Recruit new members")
        
        # Add location-based actions
        if self.story_flags.get('network_location', {}).get('type') == 'recruitment_hub':
            actions.append("Signal interest")
        elif self.story_flags.get('network_location', {}).get('type') == 'transformation_assessment':
            actions.append("Submit to evaluation")
        
        return actions
    
    async def handle_special_choice(
        self, choice_type: str, player_choice: str
    ) -> Dict[str, Any]:
        """
        Handle special story choices.
        
        Args:
            choice_type: Type of choice
            player_choice: Player's choice
            
        Returns:
            Result of the choice
        """
        if choice_type == "rose_or_thorn":
            # Player choosing their path in the network
            if player_choice.lower() == "rose":
                self.story_flags['chosen_path'] = 'cultivator'
                role_desc = "You will help others grow and transform"
            else:
                self.story_flags['chosen_path'] = 'protector'
                role_desc = "You will protect the vulnerable and enforce justice"
            
            async with get_db_connection_context() as conn:
                await conn.execute(
                    """
                    UPDATE story_states
                    SET story_flags = story_flags || $4
                    WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                    """,
                    self.user_id, self.conversation_id, self.story_id,
                    json.dumps({'chosen_path': self.story_flags['chosen_path']})
                )
            
            return {
                'choice_made': choice_type,
                'path_chosen': self.story_flags['chosen_path'],
                'description': role_desc,
                'new_responsibilities': True
            }
        
        elif choice_type == "queen_theory":
            # Player's theory about the Queen's nature
            theories = {
                "single": "You believe she is one extraordinary woman",
                "multiple": "You believe the role passes between queens",
                "collective": "You believe it's a shared identity",
                "mystery": "You accept the mystery may never be solved"
            }
            
            if player_choice.lower() in theories:
                self.story_flags['queen_theory'] = player_choice.lower()
                
                return {
                    'choice_made': choice_type,
                    'theory': player_choice.lower(),
                    'description': theories[player_choice.lower()],
                    'queen_response': "An interesting theory..."
                }
        
        elif choice_type == "transformation_ethics":
            # Player's stance on forced transformation
            stances = {
                "support": "You believe transformation serves justice",
                "question": "You have doubts about the methods",
                "oppose": "You believe in choice, not force"
            }
            
            if player_choice.lower() in stances:
                self.story_flags['transformation_stance'] = player_choice.lower()
                
                return {
                    'choice_made': choice_type,
                    'stance': player_choice.lower(),
                    'description': stances[player_choice.lower()],
                    'network_reaction': 'varies_by_member'
                }
        
        return {"error": f"Unknown choice type: {choice_type}"}
    
    async def get_story_status(self) -> Dict[str, Any]:
        """Get current story status and progress"""
        if not self._initialized:
            await self.initialize()
        
        from story_templates.moth.queen_of_thorns_story import QUEEN_OF_THORNS_STORY
        
        # Calculate progress
        total_beats = len(QUEEN_OF_THORNS_STORY.story_beats)
        completed_beats = len(self.story_flags.get('completed_beats', []))
        progress_percentage = (completed_beats / total_beats) * 100 if total_beats > 0 else 0
        
        # Get Queen's current state if relevant
        queen_state = {}
        if self.queen_npc_id:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT trust, mystery_mechanics
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                    """,
                    self.user_id, self.conversation_id, self.queen_npc_id
                )
                if row:
                    queen_state = {
                        'trust': row['trust'],
                        'mystery_active': bool(row['mystery_mechanics'])
                    }
        
        status = {
            'story_id': self.story_id,
            'current_act': self.current_act,
            'current_beat': self.current_beat,
            'progress_percentage': progress_percentage,
            'completed_beats': self.story_flags.get('completed_beats', []),
            'network_awareness': self.network_awareness,
            'information_layer': self.information_layer,
            'player_rank': self.player_rank,
            'queen_state': queen_state,
            'key_flags': {
                'network_member': self.player_rank not in ['outsider', 'noticed'],
                'queen_met': self.story_flags.get('queen_encountered', False),
                'transformation_witnessed': self.story_flags.get('transformation_witnessed', False),
                'safehouse_known': self.information_layer in ['hidden', 'deep_secret'],
                'chosen_path': self.story_flags.get('chosen_path'),
                'queen_theory': self.story_flags.get('queen_theory')
            },
            'setting': 'San Francisco Bay Area, 2025',
            'network_elements': {
                'known_members': self.story_flags.get('known_network_members', []),
                'visited_locations': self.story_flags.get('network_locations_visited', []),
                'rank_opportunities': self._get_rank_opportunities(),
                'available_actions': self._get_available_actions()
            }
        }
        
        return status
