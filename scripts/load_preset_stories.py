import asyncio
import logging
from story_templates.preset_story_loader import PresetStoryLoader
from story_templates.the_velvet_rebellion import THE_VELVET_REBELLION
from db.connection import initialize_connection_pool, close_connection_pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Load all preset stories into the database"""
    # Initialize database connection
    await initialize_connection_pool()
    
    try:
        # Load your preset story
        logger.info("Loading The Velvet Rebellion preset story...")
        await PresetStoryLoader.load_preset_story(THE_VELVET_REBELLION)
        
        logger.info("Successfully loaded all preset stories!")
        
    except Exception as e:
        logger.error(f"Error loading preset stories: {e}", exc_info=True)
    finally:
        await close_connection_pool()

if __name__ == "__main__":
    asyncio.run(main())
