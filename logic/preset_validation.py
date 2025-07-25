# logic/preset_validation.py

async def validate_preset_response(
    response_text: str, 
    story_id: str,
    conversation_id: int
) -> Dict[str, Any]:
    """Validate response against preset story rules"""
    
    if story_id == 'the_moth_and_flame':
        from story_templates.moth.lore.consistency_guide import QueenOfThornsConsistencyGuide
        
        validation_result = QueenOfThornsConsistencyGuide.validate_content(response_text)
        
        if not validation_result['valid']:
            # Log violations
            logger.error(f"Preset story violations in conversation {conversation_id}: {validation_result['violations']}")
            
            # Optionally, retry generation with stricter prompt
            return {
                'valid': False,
                'violations': validation_result['violations'],
                'warnings': validation_result['warnings'],
                'needs_regeneration': True
            }
        
        return validation_result
    
    return {'valid': True}

# Use in your response pipeline
async def process_llm_response(response_data: Dict[str, Any], conversation_id: int) -> Dict[str, Any]:
    """Process and validate LLM response"""
    
    preset_info = await check_preset_story(conversation_id)
    
    if preset_info:
        # Validate the response
        narrative = response_data.get('function_args', {}).get('narrative', '')
        validation = await validate_preset_response(
            narrative, 
            preset_info['story_id'],
            conversation_id
        )
        
        if not validation['valid']:
            # Handle violations - could retry or fix
            logger.warning(f"Response violated preset rules: {validation['violations']}")
            
            # Option 1: Request regeneration with stricter prompt
            # Option 2: Post-process to fix violations
            # Option 3: Return error to user
    
    return response_data
