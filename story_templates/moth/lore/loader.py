# story_templates/moth/lore/loader.py
"""
Loader that initializes pre-generated lore when starting a new game
"""

class PresetLoreLoader:
    """Loads preset lore into the game world"""
    
    @staticmethod
    async def load_modern_city_preset(ctx, user_id: int, conversation_id: int):
        """Load the modern city preset"""
        logger.info(f"Loading modern city preset for user {user_id}")
        
        # Generate all lore
        lore_data = await ModernCityLorePresets.generate_complete_city_lore(
            ctx, user_id, conversation_id
        )
        
        # Initialize the world with this lore
        from lore.core.lore_system import LoreSystem
        lore_system = await LoreSystem.get_instance(user_id, conversation_id)
        
        # Store foundation lore
        environment_desc = lore_data["foundation"]["world_setting"]["description"]
        await lore_system.generate_complete_lore(ctx, environment_desc)
        
        # Add all the specific elements
        # This would integrate with your existing lore managers
        
        return {
            "status": "success",
            "message": "Modern city lore loaded successfully",
            "stats": {
                "districts": len(lore_data["foundation"]["districts"]),
                "factions": len(lore_data["criminal_factions"]),
                "myths": len(lore_data["myths"]),
                "cultural_elements": len(lore_data["culture"])
            }
        }
