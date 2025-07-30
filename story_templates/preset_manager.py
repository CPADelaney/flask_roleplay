# story_templates/preset_manager.py

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

class PresetStoryManager:
    """Manages preset story integration with dynamic generation"""
    
    @staticmethod
    async def check_preset_story(conversation_id: int) -> Optional[Dict[str, Any]]:
        """Check if conversation is using a preset story"""
        async with get_db_connection_context() as conn:
            # Check story_states table
            story_row = await conn.fetchrow("""
                SELECT story_id, story_flags, current_act, current_beat, progress
                FROM story_states 
                WHERE conversation_id = $1 
                AND story_id IN ('queen_of_thorns')  -- Updated story ID
                ORDER BY started_at DESC
                LIMIT 1
            """, conversation_id)
            
            if story_row:
                flags = json.loads(story_row['story_flags']) if story_row['story_flags'] else {}
                return {
                    'story_id': story_row['story_id'],
                    'uses_sf_preset': True,  # Queen of Thorns always uses SF preset
                    'preset_active': True,
                    'current_act': story_row['current_act'],
                    'current_beat': story_row['current_beat'],
                    'progress': story_row['progress'],
                    'story_flags': flags,
                    'network_awareness': flags.get('network_awareness', 0),
                    'information_layer': flags.get('information_layer', 'public'),
                    'player_rank': flags.get('player_rank', 'outsider')
                }
            
            # Also check CurrentRoleplay for preset marker
            preset_marker = await conn.fetchval("""
                SELECT value FROM CurrentRoleplay
                WHERE conversation_id = $1 AND key = 'preset_story_id'
            """, conversation_id)
            
            if preset_marker:
                story_id = json.loads(preset_marker) if isinstance(preset_marker, str) else preset_marker
                
                # Get additional info if story_states exists
                story_info = await conn.fetchrow("""
                    SELECT current_act, current_beat, story_flags
                    FROM story_states
                    WHERE conversation_id = $1 AND story_id = $2
                """, conversation_id, story_id)
                
                result = {
                    'story_id': story_id,
                    'preset_active': True,
                    'uses_sf_preset': story_id == 'queen_of_thorns'
                }
                
                if story_info:
                    flags = json.loads(story_info['story_flags']) if story_info['story_flags'] else {}
                    result.update({
                        'current_act': story_info['current_act'],
                        'current_beat': story_info['current_beat'],
                        'story_flags': flags,
                        'network_awareness': flags.get('network_awareness', 0),
                        'information_layer': flags.get('information_layer', 'public'),
                        'player_rank': flags.get('player_rank', 'outsider')
                    })
                
                return result
            
            return None
    
    @staticmethod
    async def get_preset_constraints(story_id: str) -> Dict[str, Any]:
        """Get constraints for a specific preset story"""
        constraints = {
            'story_id': story_id,
            'rules': [],
            'system_prompt_additions': '',
            'validator': None,
            'quick_reference': ''
        }
        
        if story_id == 'queen_of_thorns':
            from story_templates.moth.lore.consistency_guide import QueenOfThornsConsistencyGuide
            
            constraints.update({
                'rules': QueenOfThornsConsistencyGuide.get_critical_rules(),
                'system_prompt_additions': QueenOfThornsConsistencyGuide.get_complete_system_prompt(),
                'validator': QueenOfThornsConsistencyGuide.validate_content,
                'quick_reference': QueenOfThornsConsistencyGuide.get_quick_reference(),
                'forbidden_phrases': [
                    "The Rose & Thorn Society announced",
                    "The Garden's official",
                    "The Shadow Matriarchy decreed",
                    "Queen [Name]",
                    "our Seattle chapter",
                    "branches in other cities",
                    "instantly transformed",
                    "The Moth Queen"  # Old reference
                ],
                'correct_usage': {
                    'network_reference': 'the network',
                    'organization_reference': 'the garden',
                    'queen_reference': 'The Queen of Thorns, whoever she is',
                    'other_cities': 'allied networks in [city]',
                    'transformation_time': 'months of careful work',
                    'information': 'exists in four layers'
                }
            })
        
        return constraints
    
    @staticmethod
    async def validate_preset_content(
        content: str, 
        story_id: str,
        conversation_id: int,
        content_type: str = 'narrative'
    ) -> Dict[str, Any]:
        """Validate content against preset story rules"""
        
        constraints = await PresetStoryManager.get_preset_constraints(story_id)
        
        if not constraints.get('validator'):
            return {'valid': True, 'message': 'No validator for this story'}
        
        # Run the validator
        validation_result = constraints['validator'](content)
        
        # Log violations
        if not validation_result['valid']:
            logger.error(
                f"Preset story violations in conversation {conversation_id}: "
                f"Story: {story_id}, Type: {content_type}, "
                f"Violations: {validation_result['violations']}"
            )
            
            # Store violation for analysis
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    INSERT INTO preset_story_violations 
                    (conversation_id, story_id, content_type, violations, content_sample, created_at)
                    VALUES ($1, $2, $3, $4, $5, NOW())
                """, conversation_id, story_id, content_type, 
                    json.dumps(validation_result['violations']), 
                    content[:500])  # Store first 500 chars for review
        
        return validation_result
    
    @staticmethod
    async def inject_preset_context(
        base_prompt: str, 
        conversation_id: int,
        include_validation: bool = True
    ) -> str:
        """Inject preset story constraints into prompts"""
        preset_info = await PresetStoryManager.check_preset_story(conversation_id)
        
        if not preset_info:
            return base_prompt
        
        constraints = await PresetStoryManager.get_preset_constraints(preset_info['story_id'])
        
        enhanced_prompt = f"""{base_prompt}

==== ACTIVE PRESET STORY: {preset_info['story_id']} ====
{constraints['system_prompt_additions']}

Current Story State:
- Act: {preset_info.get('current_act', 1)}
- Beat: {preset_info.get('current_beat', 'unknown')}
- Progress: {preset_info.get('progress', 0)}%
- Network Awareness: {preset_info.get('network_awareness', 0)}
- Information Layer: {preset_info.get('information_layer', 'public')}
- Player Rank: {preset_info.get('player_rank', 'outsider')}

Quick Reference:
{constraints['quick_reference']}
"""
        
        if include_validation:
            enhanced_prompt += """

VALIDATION REQUIREMENTS:
Before generating ANY response:
1. Check all consistency rules are followed
2. NEVER use official organization names
3. The Queen's identity is a secret
4. Maintain the four-layer information system
5. Remember the network controls Bay Area ONLY
6. Transformation takes months/years
7. Use coded garden language appropriately

If you detect any violations in your planned response, revise before outputting.
"""
        
        return enhanced_prompt
    
    @staticmethod
    async def get_location_lore(
        conversation_id: int,
        location_name: str
    ) -> Dict[str, Any]:
        """Get location-specific lore for preset stories"""
        preset_info = await PresetStoryManager.check_preset_story(conversation_id)
        
        if not preset_info:
            return {}
        
        if preset_info['story_id'] == 'queen_of_thorns':
            from story_templates.moth.lore import SFBayQueenOfThornsPreset
            
            # Get all locations from preset
            all_locations = SFBayQueenOfThornsPreset.get_specific_locations()
            
            # Find matching location
            location_lower = location_name.lower()
            matching_location = None
            
            for loc in all_locations:
                if loc['name'].lower() in location_lower or location_lower in loc['name'].lower():
                    matching_location = loc
                    break
            
            if not matching_location:
                # Check districts
                districts = SFBayQueenOfThornsPreset.get_districts()
                for district in districts:
                    if district['name'].lower() in location_lower:
                        return {
                            'district': district,
                            'type': 'district',
                            'hidden_elements': district.get('hidden_elements'),
                            'danger_level': district.get('danger_level')
                        }
                return {}
            
            # Get relevant myths
            all_myths = SFBayQueenOfThornsPreset.get_urban_myths()
            relevant_myths = [
                myth for myth in all_myths
                if 'queen' in myth.get('name', '').lower() 
                or 'rose' in myth.get('name', '').lower()
                or matching_location['name'].lower() in myth.get('origin_location', '').lower()
            ]
            
            return {
                'location': matching_location,
                'type': 'specific_location',
                'myths': relevant_myths,
                'public_function': matching_location.get('public_function'),
                'hidden_function': matching_location.get('hidden_function'),
                'recognition_signs': matching_location.get('recognition_signs', []),
                'operational_details': matching_location.get('operational_details', {})
            }
        
        return {}
    
    @staticmethod
    async def should_apply_special_mechanics(
        conversation_id: int,
        mechanic_type: str,
        context: Dict[str, Any]
    ) -> bool:
        """Check if special story mechanics should apply"""
        preset_info = await PresetStoryManager.check_preset_story(conversation_id)
        
        if not preset_info:
            return False
        
        if preset_info['story_id'] == 'queen_of_thorns':
            story_flags = preset_info.get('story_flags', {})
            
            if mechanic_type == 'network_assessment':
                # Assessment happens at cafes and galleries
                location = context.get('current_location', '').lower()
                return (
                    'rose garden' in location or 
                    'gallery' in location or
                    'thornfield' in location
                )
            
            elif mechanic_type == 'transformation_witnessing':
                # Can witness transformations at higher awareness
                return (
                    preset_info.get('network_awareness', 0) >= 40 and
                    preset_info.get('information_layer') != 'public'
                )
            
            elif mechanic_type == 'coded_language':
                # Always active in network spaces
                return context.get('npc_is_network_member', False)
            
            elif mechanic_type == 'queen_ambiguity':
                # Always active when Queen is referenced
                return 'queen' in context.get('dialogue', '').lower()
        
        return False
    
    @staticmethod
    async def get_character_constraints(
        conversation_id: int,
        character_name: str
    ) -> Dict[str, Any]:
        """Get character-specific constraints for preset stories"""
        preset_info = await PresetStoryManager.check_preset_story(conversation_id)
        
        if not preset_info:
            return {}
        
        if preset_info['story_id'] == 'queen_of_thorns':
            char_lower = character_name.lower()
            
            if 'queen' in char_lower and 'thorns' in char_lower:
                return {
                    'dialogue_rules': [
                        'NEVER reveal your true identity',
                        'Speak in layers - surface meaning and deeper meaning',
                        'Use garden/cultivation metaphors',
                        'Maintain absolute ambiguity about singular/plural nature',
                        'Command through presence, not volume'
                    ],
                    'behavior_constraints': [
                        'Never appear weak or uncertain',
                        'Show different faces to different awareness levels',
                        'Protect the network above all',
                        'Transform predators, protect victims'
                    ],
                    'information_layers': True,
                    'identity_mystery': True
                }
            
            elif 'lily chen' in char_lower:
                return {
                    'dialogue_rules': [
                        'Friendly but observant',
                        'Assess everyone for "interesting energy"',
                        'Never mention the network directly to outsiders',
                        'Use coffee orders as personality reading'
                    ],
                    'role': 'network_recruiter'
                }
            
            elif 'victoria chen' in char_lower:
                return {
                    'dialogue_rules': [
                        'Professional with hidden edge',
                        'Discuss "founder coaching" euphemistically',
                        'Reference portfolio companies as examples',
                        'Never reveal transformation methods'
                    ],
                    'role': 'network_transformer'
                }
        
        return {}
    
    @staticmethod
    async def log_generation_metrics(
        conversation_id: int,
        generation_type: str,
        success: bool,
        violations: List[str] = None,
        response_time: float = None
    ):
        """Log metrics for preset story generation"""
        preset_info = await PresetStoryManager.check_preset_story(conversation_id)
        
        if not preset_info:
            return
        
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO preset_story_metrics
                (conversation_id, story_id, generation_type, success, 
                 violations, response_time, network_awareness, information_layer, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
            """, conversation_id, preset_info['story_id'], generation_type,
                success, json.dumps(violations) if violations else None,
                response_time, preset_info.get('network_awareness', 0),
                preset_info.get('information_layer', 'public'))
