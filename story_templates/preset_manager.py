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
                AND story_id IN ('the_moth_and_flame')  -- Add more preset story IDs here
                ORDER BY started_at DESC
                LIMIT 1
            """, conversation_id)
            
            if story_row:
                flags = json.loads(story_row['story_flags']) if story_row['story_flags'] else {}
                return {
                    'story_id': story_row['story_id'],
                    'uses_sf_preset': flags.get('uses_sf_preset', False),
                    'preset_active': True,
                    'current_act': story_row['current_act'],
                    'current_beat': story_row['current_beat'],
                    'progress': story_row['progress'],
                    'story_flags': flags
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
                    'preset_active': True
                }
                
                if story_info:
                    result.update({
                        'current_act': story_info['current_act'],
                        'current_beat': story_info['current_beat'],
                        'story_flags': json.loads(story_info['story_flags']) if story_info['story_flags'] else {}
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
        
        if story_id == 'the_moth_and_flame':
            from story_templates.moth.lore.consistency_guide import QueenOfThornsConsistencyGuide
            
            constraints.update({
                'rules': QueenOfThornsConsistencyGuide.get_critical_rules(),
                'system_prompt_additions': QueenOfThornsConsistencyGuide.get_complete_system_prompt(),
                'validator': QueenOfThornsConsistencyGuide.validate_content,
                'quick_reference': QueenOfThornsConsistencyGuide.get_quick_reference(),
                'forbidden_phrases': [
                    "The Rose & Thorn Society announced",
                    "The Garden's official",
                    "Queen [Name]",
                    "our Seattle chapter",
                    "instantly transformed"
                ],
                'correct_usage': {
                    'network_reference': 'the network',
                    'queen_reference': 'The Queen, whoever she is',
                    'other_cities': 'allied networks in [city]',
                    'transformation_time': 'months of careful work'
                }
            })
        
        # Add more preset stories here as they're created
        
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

Quick Reference:
{constraints['quick_reference']}
"""
        
        if include_validation:
            enhanced_prompt += """

VALIDATION REQUIREMENTS:
Before generating ANY response:
1. Check all consistency rules are followed
2. Verify no forbidden phrases are used
3. Ensure correct terminology is applied
4. Validate character behaviors match established patterns

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
        
        if preset_info['story_id'] == 'the_moth_and_flame':
            from story_templates.moth.lore import SFBayMothFlamePreset
            
            # Get all locations from preset
            all_locations = SFBayMothFlamePreset.get_specific_locations()
            
            # Find matching location
            location_lower = location_name.lower()
            matching_location = None
            
            for loc in all_locations:
                if loc['name'].lower() in location_lower or location_lower in loc['name'].lower():
                    matching_location = loc
                    break
            
            if not matching_location:
                # Check districts
                districts = SFBayMothFlamePreset.get_districts()
                for district in districts:
                    if district['name'].lower() in location_lower:
                        return {
                            'district': district,
                            'type': 'district',
                            'special_rules': district.get('special_rules', []),
                            'atmosphere': district.get('atmosphere', {})
                        }
                return {}
            
            # Get relevant myths
            all_myths = SFBayMothFlamePreset.get_urban_myths()
            relevant_myths = [
                myth for myth in all_myths
                if any(keyword in location_lower for keyword in 
                      ['sanctum', 'garden', 'underground', matching_location['name'].lower()])
            ]
            
            return {
                'location': matching_location,
                'type': 'specific_location',
                'myths': relevant_myths,
                'access_level': matching_location.get('access_level', 'public'),
                'special_mechanics': matching_location.get('special_mechanics', []),
                'atmosphere': matching_location.get('atmosphere', {}),
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
        
        if preset_info['story_id'] == 'the_moth_and_flame':
            story_flags = preset_info.get('story_flags', {})
            
            if mechanic_type == 'fog_protection':
                # Fog protection applies at night in outdoor locations
                return (
                    context.get('time_of_day', '').lower() in ['night', 'late night'] and
                    'outdoor' in context.get('current_location', '').lower()
                )
            
            elif mechanic_type == 'safehouse_sanctuary':
                # Safehouse rules apply in specific locations
                location = context.get('current_location', '').lower()
                return any(safe in location for safe in ['safehouse', 'butterfly house'])
            
            elif mechanic_type == 'queen_presence':
                # Queen presence mechanics in her domains
                location = context.get('current_location', '').lower()
                return 'velvet sanctum' in location or 'inner garden' in location
        
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
        
        if preset_info['story_id'] == 'the_moth_and_flame' and character_name.lower() == 'lilith':
            return {
                'dialogue_rules': [
                    'Always speak in character as the Queen of Thorns',
                    'Use poetic, gothic language',
                    'Reference roses, flames, and transformation',
                    'Never directly say "I love you" unless trust > 95'
                ],
                'behavior_constraints': [
                    'Maintain dominant persona',
                    'Show vulnerability only in private and high trust',
                    'React strongly to abandonment triggers',
                    'Protect trafficking victims fiercely'
                ],
                'mask_system': True,
                'three_words_mechanic': True
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
                 violations, response_time, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, NOW())
            """, conversation_id, preset_info['story_id'], generation_type,
                success, json.dumps(violations) if violations else None,
                response_time)
