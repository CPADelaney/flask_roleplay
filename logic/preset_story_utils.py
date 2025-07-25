# logic/preset_story_utils.py

async def check_preset_story(conversation_id: int) -> Optional[Dict[str, Any]]:
    """Check if conversation is using a preset story"""
    async with get_db_connection_context() as conn:
        # Check story_states table
        story_row = await conn.fetchrow("""
            SELECT story_id, story_flags 
            FROM story_states 
            WHERE conversation_id = $1 
            AND story_id IN ('the_moth_and_flame')
        """, conversation_id)
        
        if story_row:
            flags = json.loads(story_row['story_flags'])
            return {
                'story_id': story_row['story_id'],
                'uses_sf_preset': flags.get('uses_sf_preset', False),
                'preset_active': True
            }
        
        # Also check CurrentRoleplay for preset marker
        preset_marker = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE conversation_id = $1 AND key = 'preset_story_id'
        """, conversation_id)
        
        if preset_marker:
            return {
                'story_id': json.loads(preset_marker),
                'preset_active': True
            }
        
        return None
